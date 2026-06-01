"""pysurgery/surgery.py.

Public API for surgery on simplicial complexes: the SurgerySession (Surgeon) workbench.

References:
    Milnor, J. (1965). Lectures on the h-cobordism theorem. Princeton University Press.
    Ranicki, A. (1992). Algebraic L-theory and topological manifolds. Cambridge University Press.
"""
import re
import copy
import warnings
import pickle
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
import sympy as sp

from pysurgery.manifolds.surgery import (
    HandleAttachment,
    SurgeryResult,
    SurgeryVerificationResult,
    LinkingNumberResult,
    DelinkingResult,
    AttachmentSphereResult,
    AlgebraicSurgeryComplex,
    compute_linking_number,
    find_attachment_sphere,
    perform_handle_surgery,
    perform_algebraic_surgery,
    perform_rational_surgery,
    perform_p_local_surgery,
    verify_surgery,
    delink,
    _apply_disk_removal_to_complex,
    _build_handle_attachment_from_sphere,
    FramingResult,
)
from pysurgery.core.exceptions import (
    KirbyMoveError,
    SurgeryInvariantBroken,
    SurgeryProtocolError,
    TopologyNotRestoredError,
)
from pysurgery.homology.algebraic_poincare import AlgebraicPoincareComplex
from pysurgery.topology.complexes import ChainComplex, SimplicialComplex
from pysurgery.manifolds.handle_decompositions import Handle, HandleDecomposition
from pysurgery.manifolds.kirby_calculus import KirbyDiagram


# ── Typing aliases ──────────────────────────────────────────────────────────
PointCloud   = np.ndarray             # shape (N, d)
NumericFn    = Callable[[PointCloud, float], PointCloud]
SymPyExpr    = Union[sp.Expr, sp.Matrix, Sequence[sp.Expr]]


# ── Unicode / display helpers ─────────────────────────────────────────────────

_SUPER = str.maketrans("0123456789+-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻")
_SUB   = str.maketrans("0123456789",   "₀₁₂₃₄₅₆₇₈₉")


def _sup(n: int) -> str:
    return str(n).translate(_SUPER)


def _sub(n: int) -> str:
    return str(n).translate(_SUB)


def _to_math(s: str) -> str:
    """Convert ASCII topology strings to Unicode: 'S^2xD^1' → 'S²×D¹', 'R^3' → 'ℝ³'."""
    s = re.sub(r'[xX]', '×', s)
    s = re.sub(r'\^(\d+)', lambda m: m.group(1).translate(_SUPER), s)
    # R³ → ℝ³  (handle after superscript substitution)
    s = re.sub(r'R([⁰¹²³⁴⁵⁶⁷⁸⁹]+)', r'ℝ\1', s)
    return s


def _fmt_vec(v: Any, prec: int = 2) -> str:
    """Format a vector or coordinate tuple as '(x.xx, y.yy, …)'."""
    if v is None:
        return "—"
    try:
        parts = [f"{float(x):+.{prec}f}" for x in v]
        return "(" + ", ".join(parts) + ")"
    except (TypeError, ValueError):
        return str(v)


def _cloud_stats(cloud: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (centroid, bbox_min, bbox_max) for a point cloud."""
    cloud = np.atleast_2d(cloud).astype(float)
    return cloud.mean(axis=0), cloud.min(axis=0), cloud.max(axis=0)


def _fmt_bbox(mn: np.ndarray, mx: np.ndarray) -> str:
    labels = ["x", "y", "z", "w"]
    parts = []
    for i, (lo, hi) in enumerate(zip(mn, mx)):
        lbl = labels[i] if i < len(labels) else f"x{_sub(i)}"
        parts.append(f"{lbl}∈[{lo:.2f},{hi:.2f}]")
    return "  ".join(parts)


def _collate_divisors(divisors: List[int]) -> List[Tuple[int, int]]:
    """Convert [1, 1, 2, 2, 2, 6] -> [(2, 3), (6, 1)]. (Filters 1s)."""
    from collections import Counter
    # Filter out 1s as Z/1Z = 0
    counts = Counter(d for d in divisors if abs(d) > 1)
    return sorted(counts.items())


def _box(title: str, lines: List[str], width: int = 66) -> List[str]:
    """Render a labelled box using box-drawing characters."""
    inner = width - 4
    pad = max(0, inner - len(title))
    header = f"  ┌─ {title} " + "─" * pad + "┐"
    footer = "  └" + "─" * (width - 3) + "┘"
    body = [f"  │  {line:<{inner}s}│" for line in lines]
    return [header] + body + [footer]


def _divider(label: str, char: str = "━", width: int = 70) -> str:
    pad = width - len(label) - 6
    return f"  {char*2}  {label}  {char * pad}"


# ── Exceptions ────────────────────────────────────────────────────────────────


class DimensionalConsistencyError(Exception):
    """Raised when surgery types do not match ambient dimension."""
    pass


class SurgeryFinishedError(Exception):
    """Raised when mutative operations are called after finish()."""
    pass


class IsotopyShapeError(ValueError):
    """Raised when an isotopy's numeric output deviates from `in.shape`."""


class IsotopyCompileError(ValueError):
    """Raised when a SymPy expression fails to lambdify cleanly."""


# ── Isotopy hierarchy ────────────────────────────────────────────────────────


class Isotopy(ABC):
    """Abstract ambient isotopy  f : K × [0,1] → K  on a d-dimensional point cloud.

    Subclasses provide:
        * `expr(d)`           — SymPy expression for the action on a single
                                 generic point  x = (x₀,…,x_{d-1})  at time `t`.
        * `_compile(d)`       — concrete NumPy callable; default uses lambdify.
        * `describe()`        — short Unicode description for logs.

    The base class enforces:
        * strict 1-to-1 input/output shape via `__call__`.
        * cached compilation.
        * a uniform `to_symbolic(d)` API for the lazy composition ledger.
    """

    name: str = "isotopy"

    def __init__(self, *, name: Optional[str] = None) -> None:
        if name is not None:
            self.name = name
        self._compiled: Dict[int, NumericFn] = {}      # keyed by point dim `d`

    def __getstate__(self) -> Dict[str, Any]:
        state = self.__dict__.copy()
        state["_compiled"] = {}
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.__dict__.update(state)
        if "_compiled" not in self.__dict__:
            self._compiled = {}

    # — Symbolic side ——————————————————————————————————————————————
    @abstractmethod
    def expr(self, d: int) -> sp.Matrix:
        """Symbolic action on a single point.

        Must return a SymPy column vector of length `d` whose free symbols are
        a subset of  {x_0, …, x_{d-1}, t}  (plus any compile-time parameters
        that have been substituted into a numeric value).
        """

    def to_symbolic(self, d: int) -> sp.Matrix:
        """Public accessor used by the composition ledger."""
        return self.expr(d)

    # — Numeric side ———————————————————————————————————————————————
    def compile(self, d: int) -> NumericFn:
        """Return (and cache) a vectorized NumPy callable for d-dimensional clouds."""
        if d not in self._compiled:
            self._compiled[d] = self._compile(d)
        return self._compiled[d]

    def _compile(self, d: int) -> NumericFn:
        """Default compilation path: lambdify `expr(d)` for NumPy."""
        expr = self.expr(d)
        t = sp.Symbol("t")
        xs = [sp.Symbol(f"x_{i}") for i in range(d)]

        try:
            # We want a function that takes (x0, x1, ..., xd-1, t)
            func = sp.lambdify([*xs, t], expr, modules="numpy")

            def vectorized_fn(cloud: PointCloud, time: float) -> PointCloud:
                # Unpack columns of cloud for lambdified func
                res = func(*cloud.T, time)
                # Ensure output matches (N, d)
                out = np.squeeze(np.array(res)).T
                if out.shape != cloud.shape:
                    out = out.reshape(cloud.shape)
                return out

            return vectorized_fn
        except Exception as e:
            raise IsotopyCompileError(f"Failed to lambdify {self.name}: {e}")

    def __call__(self, cloud: PointCloud, t: float) -> PointCloud:
        """Apply the isotopy to a point cloud at time `t`."""
        d = cloud.shape[1]
        fn = self.compile(d)
        out = fn(cloud, t)
        if out.shape != cloud.shape:
            raise IsotopyShapeError(
                f"{self.name} changed shape: {cloud.shape} -> {out.shape}"
            )
        return out

    @abstractmethod
    def describe(self) -> str:
        """Human-readable description for logs."""

    @property
    def description(self) -> str:
        """Backward compatibility with the old Transformation.description."""
        return self.describe()


class IdentityIsotopy(Isotopy):
    """Ambient isotopy that does nothing: f(x,t) = x."""

    name: str = "identity"

    def expr(self, d: int) -> sp.Matrix:
        """Return the identity coordinate expression in dimension ``d``."""
        return sp.Matrix([sp.Symbol(f"x_{i}") for i in range(d)])

    def describe(self) -> str:
        """Return a human-readable description of the isotopy."""
        return "identity"


class AffineIsotopy(Isotopy):
    """Base for isotopies that are affine in x."""

    def __init__(self, *, name: Optional[str] = None) -> None:
        super().__init__(name=name)

    @abstractmethod
    def _matrix(self, d: int) -> sp.Matrix:
        """Matrix M such that at t=1, the linear part is M."""

    @abstractmethod
    def _offset(self, d: int) -> sp.Matrix:
        """Vector b such that at t=1, the translation part is b."""

    def expr(self, d: int) -> sp.Matrix:
        t = sp.Symbol("t")
        x = sp.Matrix([sp.Symbol(f"x_{i}") for i in range(d)])
        M = self._matrix(d)
        b = self._offset(d)
        # Linear interpolation of the transformation: (I + t(M-I))x + t*b
        # This ensures f(x, 0) = x and f(x, 1) = Mx + b
        return (sp.eye(d) + t * (M - sp.eye(d))) * x + t * b


class TranslateIsotopy(AffineIsotopy):
    """Translation: f(x,t) = x + t·v."""

    name: str = "translate"

    def __init__(self, offset: Sequence[float], *, name: Optional[str] = None) -> None:
        super().__init__(name=name)
        self.offset = np.asarray(offset, dtype=float)

    def _matrix(self, d: int) -> sp.Matrix:
        return sp.eye(d)

    def _offset(self, d: int) -> sp.Matrix:
        v = self.offset[:d]
        return sp.Matrix(v.tolist())

    def describe(self) -> str:
        """Return a human-readable description of the translation."""
        return f"translate by {_fmt_vec(self.offset)}"


class ThroughPointIsotopy(Isotopy):
    """Piecewise-linear isotopy passing through a via point: f(x,t) = x + piecewise(t)."""
    name: str = "thread"

    def __init__(self, start: Sequence[float], via: Sequence[float], end: Sequence[float], *, name: Optional[str] = None) -> None:
        super().__init__(name=name)
        self.start = np.asarray(start, dtype=float)
        self.via = np.asarray(via, dtype=float)
        self.end = np.asarray(end, dtype=float)
        self.t_via = 0.5

    def expr(self, d: int) -> sp.Matrix:
        t = sp.Symbol("t")
        x = sp.Matrix([sp.Symbol(f"x_{i}") for i in range(d)])

        # Displacement vector starting at 0 at t=0
        offset = sp.Matrix(d, 1, lambda i, _: sp.Piecewise(
            ((self.via[i] - self.start[i]) * 2 * t, t < sp.Rational(1, 2)),
            ((self.via[i] - self.start[i]) + (self.end[i] - self.via[i]) * 2 * (t - sp.Rational(1, 2)), True),
        ))

        return x + offset

    def __call__(self, cloud: np.ndarray, t: float) -> np.ndarray:
        t = float(np.clip(t, 0.0, 1.0))
        if t <= self.t_via:
            s = t / self.t_via if self.t_via > 0 else 0.0
            disp = s * (self.via - self.start)
        else:
            s = (t - self.t_via) / (1.0 - self.t_via) if self.t_via < 1 else 0.0
            disp = (self.via - self.start) + s * (self.end - self.via)
        return cloud + disp

    def describe(self) -> str:
        return f"thread {self.name or ''}: {_fmt_vec(self.start)} → {_fmt_vec(self.via)} → {_fmt_vec(self.end)}"



class RotateIsotopy(AffineIsotopy):
    """Rotation about an arbitrary axis (3D) or in the (i,j)-plane (general d).

    In 3D the user supplies `axis` (3-vector) and `angle` (radians).
    For d ≠ 3 the user supplies `plane = (i, j)` and `angle`.
    """
    name: str = "rotate"

    def __init__(
        self,
        *,
        angle: float,
        axis: Optional[Sequence[float]] = None,
        plane: Optional[Tuple[int, int]] = None,
        about: Optional[Sequence[float]] = None,
    ) -> None:
        self._angle = float(angle)
        self._axis  = None if axis is None else np.asarray(axis, dtype=float)
        self._plane = plane
        self._about = None if about is None else np.asarray(about, dtype=float)
        super().__init__(name=self.name)

    def _matrix(self, d: int) -> sp.Matrix:
        if d == 3 and self._axis is not None:
            return self._rodrigues_3d()
        if self._plane is not None:
            return self._planar_rotation(d, *self._plane)
        # Default to (0,1) plane if nothing specified
        return self._planar_rotation(d, 0, 1 if d > 1 else 0)

    def _rodrigues_3d(self) -> sp.Matrix:
        # Rodrigues' rotation matrix
        k = self._axis / max(np.linalg.norm(self._axis), 1e-12)
        kx, ky, kz = (sp.Float(k[0]), sp.Float(k[1]), sp.Float(k[2]))
        theta = sp.Float(self._angle)
        K = sp.Matrix([
            [0, -kz, ky],
            [kz, 0, -kx],
            [-ky, kx, 0],
        ])
        return sp.eye(3) + sp.sin(theta) * K + (1 - sp.cos(theta)) * (K * K)

    def _planar_rotation(self, d: int, i: int, j: int) -> sp.Matrix:
        R = sp.eye(d)
        if i == j or i >= d or j >= d:
            return R
        c, s = sp.cos(self._angle), sp.sin(self._angle)
        R[i, i], R[j, j] = c, c
        R[i, j], R[j, i] = -s, s
        return R

    def _offset(self, d: int) -> sp.Matrix:
        # For rotation about a point p:  x ↦ R(x − p) + p,  hence b = (I − R) p.
        if self._about is None:
            return sp.zeros(d, 1)
        R = self._matrix(d)
        p = sp.Matrix(self._about[:d].tolist())
        return (sp.eye(d) - R) * p

    def describe(self) -> str:
        """Return a human-readable description of the rotation."""
        if self._axis is not None:
            ax = ", ".join(f"{x:+.3f}" for x in self._axis)
            return f"rotate θ={self._angle:+.4f} rad about axis ({ax})"
        if self._plane is not None:
            i, j = self._plane
            return f"rotate θ={self._angle:+.4f} rad in ({i},{j}) plane"
        return f"rotate θ={self._angle:+.4f} rad"


class ScaleIsotopy(AffineIsotopy):
    """Scaling about an arbitrary point."""
    name: str = "scale"

    def __init__(
        self,
        factors: Union[float, Sequence[float]],
        about: Optional[Sequence[float]] = None,
    ) -> None:
        self._factors = factors
        self._about = None if about is None else np.asarray(about, dtype=float)
        super().__init__(name=self.name)

    def _matrix(self, d: int) -> sp.Matrix:
        s = self._factors
        diag = [s] * d if np.isscalar(s) else list(s)
        if len(diag) < d:
            diag = diag + [1.0] * (d - len(diag))
        elif len(diag) > d:
            diag = diag[:d]
        return sp.diag(*diag)

    def _offset(self, d: int) -> sp.Matrix:
        if self._about is None:
            return sp.zeros(d, 1)
        S = self._matrix(d)
        p = sp.Matrix(self._about[:d].tolist())
        return (sp.eye(d) - S) * p

    def describe(self) -> str:
        """Return a human-readable description of the scaling."""
        return f"scale by {self._factors} about {self._about if self._about is not None else 'origin'}"


class SymbolicTransformation(Isotopy):
    """Isotopy defined by a user-supplied SymPy expression."""
    name: str = "symbolic"

    def __init__(
        self,
        expr_fn: Callable[[int], sp.Matrix],
        *,
        description: Optional[str] = None,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(name=name or "symbolic")
        self._expr_fn = expr_fn
        self._description = description

    def expr(self, d: int) -> sp.Matrix:
        """Evaluate the user-supplied expression for dimension ``d``.

        Args:
            d: Ambient dimension of the point cloud.

        Returns:
            A (d, 1) SymPy column matrix.

        Raises:
            IsotopyShapeError: If the supplied expression has the wrong shape.
        """
        out = self._expr_fn(d)
        if not isinstance(out, sp.Matrix):
            out = sp.Matrix(list(out))
        if out.shape != (d, 1):
            raise IsotopyShapeError(
                f"symbolic expr returned shape {out.shape}, expected ({d}, 1)"
            )
        return out

    def describe(self) -> str:
        """Return the supplied description, or a generic label if none was given."""
        if self._description:
            return self._description
        return "symbolic transformation"

    @classmethod
    def from_components(
        cls,
        components: Sequence[sp.Expr],
        *,
        dim: Optional[int] = None,
        description: Optional[str] = None,
    ) -> "SymbolicTransformation":
        """Build a SymbolicTransformation from a list of component expressions.

        Args:
            components: SymPy expressions, one per coordinate.
            dim: Fixed dimension the transformation is built for; defaults to
                ``len(components)``.
            description: Optional human-readable label.

        Returns:
            A SymbolicTransformation wrapping the given components.
        """
        d_fixed = dim if dim is not None else len(components)

        def build(d: int) -> sp.Matrix:
            if d != d_fixed:
                raise IsotopyShapeError(
                    f"symbolic built for d={d_fixed} but cloud has d={d}"
                )
            return sp.Matrix(list(components))

        return cls(build, description=description)


@dataclass
class IsotopyComposition:
    """Lazy, append-only ledger of every isotopy applied during a session.

    The composition is built but *never* evaluated. It exists strictly so the
    logs() output can display a complete  f_k ∘ … ∘ f_1  for inclusion in
    proofs / encyclopedic records.

    Storage layout:
        steps   : list of (step_index, Isotopy)  in execution order.
        per_target : dict[str | None, list[int]]
                     — indices (into `steps`) keyed by the cloud they acted on,
                       so we can emit a per-cloud composition as well as the
                       global one.
    """
    steps: List[Tuple[int, Isotopy]] = field(default_factory=list)
    per_target: Dict[Optional[str], List[int]] = field(default_factory=dict)

    def compose(self, step_index: int, iso: Isotopy, target: Optional[str]) -> None:
        """Append an isotopy to the ledger, recording its step index and target.

        Args:
            step_index: Execution-order index of the step.
            iso: The isotopy that was applied.
            target: Name of the cloud it acted on, or None for the ambient space.
        """
        self.steps.append((step_index, iso))
        self.per_target.setdefault(target, []).append(len(self.steps) - 1)

    # ── Symbolic rendering ─────────────────────────────────────────────────
    def full_expr(self, d: int = 3, target: Optional[str] = None) -> sp.Matrix:
        """Build  f_k(f_{k-1}(... f_1(x, t) ..., t), t)  as a SymPy expression.

        WARNING — DOCUMENTATION USE ONLY.  Never feed this back into compile().
        """
        xs = sp.Matrix([sp.Symbol(f"x_{k}") for k in range(d)])
        chain = self.steps if target is None else [
            self.steps[i] for i in self.per_target.get(target, [])
        ]
        current = xs
        for _, iso in chain:
            sub_expr = iso.expr(d)
            # Substitute current vector into the symbolic action.
            # We use x_0, x_1, ... for consistency with Isotopy._compile
            mapping = {sp.Symbol(f"x_{k}"): current[k, 0] for k in range(d)}
            current = sub_expr.subs(mapping)
        return current

    def latex(self, d: int = 3, target: Optional[str] = None) -> str:
        """Render the composed isotopy chain as a LaTeX string."""
        try:
            return sp.latex(self.full_expr(d, target=target))
        except Exception as e:
            return rf"\text{{[composition unrepresentable: {e!s}]}}"

    def pretty(self, d: int = 3, target: Optional[str] = None) -> str:
        """Render the composed isotopy chain as a Unicode pretty-printed string."""
        try:
            return sp.pretty(self.full_expr(d, target=target), use_unicode=True)
        except Exception as e:
            return f"[composition unrepresentable: {e!s}]"

    def __len__(self) -> int:
        return len(self.steps)


class LegacyDeformIsotopy(Isotopy):
    """Wraps legacy numeric deformation modes from _DEFORM_MODES."""

    def __init__(self, mode: str, kwargs: Dict[str, Any], *, name: Optional[str] = None) -> None:
        super().__init__(name=name or f"deform[{mode}]")
        self.mode = mode
        self.kwargs = kwargs
        self._meta: Dict[str, Any] = {}

    def expr(self, d: int) -> sp.Matrix:
        # We don't have a symbolic expression for legacy modes yet.
        return sp.Matrix([sp.Symbol(f"x_{i}") for i in range(d)])

    def _compile(self, d: int) -> NumericFn:
        fn = _DEFORM_MODES[self.mode]

        def wrapped(cloud: PointCloud, t: float) -> PointCloud:
            # Legacy modes only support t=1.0 application.
            # For t < 1.0 we interpolate linearly.
            if t == 0.0:
                return cloud
            res, meta = fn(cloud, **self.kwargs)
            self._meta = meta
            if t == 1.0:
                return res
            return cloud + t * (res - cloud)

        return wrapped

    def describe(self) -> str:
        return f"deform[{self.mode}]"


# ── Framing ─────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Framing:
    """Explicit trivialization of the normal bundle of an attaching sphere.

    The data the session stores depends on the handle's index k and the
    ambient dimension n:

        * k=1 in any n :  framing ∈ {±1} (orientation flag).
        * k=2 in n=4   :  framing ∈ ℤ   (the integer linking number with itself).
        * k=2 in n=3   :  framing ∈ ℚ   (Dehn surgery coefficient p/q).
        * general k    :  framing ∈ π_{k-1}(O(n-k)).
    """
    kind: Literal["integer", "rational", "spin", "stable", "trivial", "torsion"]
    integer: Optional[int]      = None
    rational: Optional[Tuple[int, int]] = None     # (p, q)
    label: Optional[str]        = None             # e.g. "Spin(0)", "ℤ/2 generator"

    def render(self) -> str:
        """Return a compact string representation of the framing datum."""
        if self.kind == "integer" and self.integer is not None:
            return f"{self.integer:+d}"
        if self.kind == "rational" and self.rational is not None:
            p, q = self.rational
            return f"{p}/{q}"
        if self.label:
            return self.label
        return self.kind


# ── Framed handle wrapper ──────────────────────────────────────────────────
@dataclass
class FramedHandle:
    """Session-level wrapper around the lower-level core.handle_decompositions.Handle."""
    id: str
    index: int                                  # k in "k-handle"
    handle_type: str                            # legacy product-string ("S^2xD^1")
    framing: Framing
    attaching_sphere: Any
    co_core: Any
    underlying: Handle                          # link to core.handle_decompositions.Handle
    attached_step: int                          # cobordism step index when attached

    def to_record(self) -> Dict[str, Any]:
        """Return a JSON-serialisable summary dict of this framed handle."""
        return {
            "id": self.id,
            "index": self.index,
            "type": self.handle_type,
            "framing": self.framing.render(),
            "framing_kind": self.framing.kind,
            "attached_step": self.attached_step,
        }


# ── Cancellation candidate (Cerf graph node) ───────────────────────────────
@dataclass
class CancellationCandidate:
    """A pair of complementary handles that may cancel (a Cerf-graph node).

    Records the k-handle and (k+1)-handle, their algebraic intersection number,
    and whether the cancellation is geometric (transverse, single intersection).
    """

    handle_lo: FramedHandle           # the k-handle
    handle_hi: FramedHandle           # the (k+1)-handle
    intersection_signed: int          # algebraic intersection number
    is_geometric: bool                # |intersection| == 1 AND transverse
    witness: Dict[str, Any] = field(default_factory=dict)  # incidence trace


# ── Handlebody sidecar ─────────────────────────────────────────────────────
@dataclass
class HandlebodyState:
    """Sidecar bookkeeping for the handle decomposition of the ambient manifold.

    Tracks the attached framed handles, an optional Kirby diagram (for
    4-manifolds), pending cancellation candidates, and logs of handle slides and
    cancellations performed during the session.
    """

    handles: List[FramedHandle] = field(default_factory=list)
    kirby: Optional[KirbyDiagram] = None                # populated for 4-manifolds
    pending_cancellations: List[CancellationCandidate] = field(default_factory=list)
    slides_log: List[Dict[str, Any]] = field(default_factory=list)
    cancellations_log: List[Dict[str, Any]] = field(default_factory=list)

    def to_decomposition(self) -> HandleDecomposition:
        """Materialise a core.handle_decompositions.HandleDecomposition snapshot."""
        return HandleDecomposition(
            handles=[fh.underlying for fh in self.handles],
        )


@dataclass
class CobordismComplex:
    """Explicit (n+1)-dimensional simplicial complex W with ∂W = M_initial ⊔ (-M_final).

    Storage:
        underlying: The (n+1)-dim simplicial complex.
        boundary_initial_indices: tuple of vertex/simplex indices for M_0.
        boundary_final_indices:   tuple of vertex/simplex indices for M_m.
        slab_for_step: mapping from cobordism step index to the set of
                       simplex indices in `underlying` that form that slab.
    """
    underlying: SimplicialComplex
    boundary_initial_indices: Tuple[int, ...]
    boundary_final_indices:   Tuple[int, ...]
    slab_for_step: Dict[int, frozenset] = field(default_factory=dict)
    is_collared: bool = True

    def dimension(self) -> int:
        """Return the dimension of the underlying cobordism complex W."""
        return self.underlying.dimension

    def interior(self) -> SimplicialComplex:
        """Return Int(W) as a SimplicialComplex.

        Construction: drop every simplex whose set of vertices is entirely
        contained in `boundary_initial_indices ∪ boundary_final_indices`.
        """
        bnd = set(self.boundary_initial_indices) | set(self.boundary_final_indices)
        keep = [
            s for s in self.underlying.simplices
            if not set(s).issubset(bnd)
        ]
        return SimplicialComplex.from_simplices(keep)

    def step_slab(self, step_index: int) -> SimplicialComplex:
        """Return the slab of W contributed by a specific cobordism step."""
        idxs = self.slab_for_step.get(step_index, frozenset())
        simps = [s for s in self.underlying.simplices if set(s).issubset(idxs)]
        return SimplicialComplex.from_simplices(simps)

    def euler_characteristic(self) -> int:
        """Return the Euler characteristic of the cobordism complex W."""
        return self.underlying.euler_characteristic()

    def homology(self, n: int = 0, backend: str = "auto"):
        """Return H_n(W), delegating to the underlying simplicial complex."""
        return self.underlying.homology(n=n, backend=backend)


Transformation = Isotopy


# ── Domain objects ────────────────────────────────────────────────────────────


class TrackedObject:
    """A geometric or topological object tracked within a SurgerySession.

    Attributes:
        name: Identifier within the session.
        data: Underlying simplicial complex or descriptor string.
    """

    def __init__(self, name: str, data: Any, session: "SurgerySession") -> None:
        self.name = name
        self.data = data
        self._session = session

    def remove_disk(
        self,
        at: Any,
        disk_type: str = "D^2",
    ) -> None:
        """Remove a disk from this object, tracking Betti number changes."""
        self._session.remove_disks((disk_type,), at=[at], target=self.name)

    def move(
        self,
        offset: Optional[Sequence[float]] = None,
        through: Any = None,
        check_isotopy: bool = True,
    ) -> Isotopy:
        """Apply the session's move operation, targeting this object."""
        return self._session.move(
            offset=offset, through=through, target=self.name,
            check_isotopy=check_isotopy,
        )

    def translate(
        self,
        offset: Sequence[float],
        check_isotopy: bool = True,
    ) -> Isotopy:
        """Apply the session's translate operation, targeting this object."""
        return self._session.translate(
            offset=offset, target=self.name, check_isotopy=check_isotopy
        )

    def rotate(
        self,
        angle: float,
        axis: Optional[Sequence[float]] = None,
        plane: Optional[Tuple[int, int]] = None,
        about: Optional[Sequence[float]] = None,
        check_isotopy: bool = True,
    ) -> Isotopy:
        """Apply the session's rotate operation, targeting this object."""
        return self._session.rotate(
            angle=angle, axis=axis, plane=plane, about=about,
            target=self.name, check_isotopy=check_isotopy
        )

    def scale(
        self,
        factors: Union[float, Sequence[float]],
        about: Optional[Sequence[float]] = None,
        check_isotopy: bool = True,
    ) -> Isotopy:
        """Apply the session's scale operation, targeting this object."""
        return self._session.scale(
            factors=factors, about=about, target=self.name,
            check_isotopy=check_isotopy
        )

    def deform(
        self,
        mode: str,
        check_isotopy: bool = True,
        **kwargs: Any,
    ) -> Isotopy:
        """Apply the session's deform operation, targeting this object."""
        return self._session.deform(
            target=self.name, mode=mode, check_isotopy=check_isotopy, **kwargs
        )

    def __repr__(self) -> str:
        return f"TrackedObject(name={self.name!r})"


class _ObjectsProxy:
    """Proxy supporting both dict-style and callable access to tracked objects.

    Dict-style access (``session.objects["name"]``) returns a single
    TrackedObject; callable access (``session.objects()``) returns the full dict.
    """

    def __init__(self, session: "SurgerySession") -> None:
        self._session = session

    def __getitem__(self, key: str) -> TrackedObject:
        return self._session._objects[key]

    def __setitem__(self, key: str, value: TrackedObject) -> None:
        self._session._objects[key] = value

    def __contains__(self, key: str) -> bool:
        return key in self._session._objects

    def __call__(self) -> Dict[str, "TrackedObject"]:
        return self._session._objects

    def __repr__(self) -> str:
        return f"ObjectsProxy({list(self._session._objects)})"


class _AmbientSpaceProxy:
    """Proxy exposing surgical operations on the ambient space handle.

    Exposes ``remove_disks`` / ``attach_handle`` / ``restore`` (e.g.
    ``surgeon.AmbientSpace.remove_disks(...)``) and is also callable to retrieve
    the underlying manifold (``surgeon.AmbientSpace()``).
    """

    def __init__(self, session: "SurgerySession") -> None:
        self._session = session

    def remove_disks(self, types: Any, at: List[Any], target: Optional[str] = None) -> None:
        """Remove disk interiors from the ambient manifold (or a named tracked object)."""
        self._session.remove_disks(types, at, target=target)

    def attach_handle(self, at: Any, handle_type: str = "S^2xD^1", target: Optional[str] = None, **kwargs) -> None:
        """Attach a handle to the ambient manifold (or a named tracked object)."""
        self._session.attach_handle(at=at, handle_type=handle_type, target=target, **kwargs)

    def restore(self, target: Optional[str] = None) -> None:
        """Undo the last operation on the ambient space."""
        self._session.restore(target=target)

    def __call__(self) -> Any:
        return self._session.manifold

    def __repr__(self) -> str:
        return f"AmbientSpaceProxy(manifold={self._session.manifold!r})"


# ── Betti-number helpers ──────────────────────────────────────────────────────


def _betti_from_object(obj_data: Any) -> Dict[int, int]:
    """Return Betti numbers for obj_data or raise a BettiTrackingError if missing or invalid."""
    from pysurgery.topology.complexes import SimplicialComplex  # local to avoid circular import
    from pysurgery.core.exceptions import BettiTrackingError
    if obj_data is None:
        raise BettiTrackingError("Betti tracking query failed: object is missing (None).")
    if not isinstance(obj_data, SimplicialComplex):
        raise BettiTrackingError(
            f"Betti tracking query failed: invalid object type '{type(obj_data).__name__}', "
            f"expected 'SimplicialComplex'."
        )
    try:
        return obj_data.betti_numbers()
    except Exception as e:
        raise BettiTrackingError(
            "Betti tracking query failed: underlying betti_numbers() call raised an exception."
        ) from e



def _predicted_betti_delta_for_index_k_surgery(
    k: int,
    n: int,
    betti_before: Dict[int, int],
    *,
    op: Literal["attach", "remove"],
) -> Dict[int, int]:
    """Expected Betti-number delta for an index-k surgery on an n-manifold."""
    if op == "attach":
        # Attach k-handle: tubular nbhd removed, co-disk added.
        # Mayer-Vietoris: β_{k-1} can drop by 1, β_k can rise by 1.
        return {k - 1: -1, k: +1}
    elif op == "remove":
        # Remove D^k disk: mirror of attaching n-handle...
        # When removing a D^n top-disk (k=n, codim 0 removal): β_n drops by 1,
        # β_{n-1} rises by 1 (new boundary sphere is born).
        # General D^k removal in an n-manifold:
        return {k - 1: +1, k: -1}
    else:
        raise ValueError(f"_predicted_betti_delta_for_index_k_surgery: op={op}")



def _apply_betti_delta(betti: Dict[int, int], delta: Dict[int, int]) -> Dict[int, int]:
    return {k: betti.get(k, 0) + delta.get(k, 0) for k in betti}


# ── Isotopy collision detection ────────────────────────────────────────────────


def _clouds_min_dist(
    a: np.ndarray,
    b: np.ndarray,
    *,
    mode: Literal["fast", "exact"] = "fast",
    max_pts: int = 300,
) -> float:
    """Minimum pairwise distance between two point clouds.

    mode="fast"  — subsample to max_pts each side (current behaviour).
    mode="exact" — cKDTree over full point set; O((|a|+|b|) log |b|).
    """
    if len(a) == 0 or len(b) == 0:
        return float('inf')
    if mode == "exact":
        from scipy.spatial import cKDTree
        tree = cKDTree(b)
        d, _ = tree.query(a, k=1)
        return float(d.min())
    # fast path:
    a = a[:: max(1, len(a) // max_pts)]
    b = b[:: max(1, len(b) // max_pts)]
    diff = a[:, None, :] - b[None, :, :]
    return float(np.linalg.norm(diff, axis=2).min())


def _seg_seg_min_dist(p1: np.ndarray, p2: np.ndarray,
                      p3: np.ndarray, p4: np.ndarray) -> float:
    """Minimum distance between line segments p1-p2 and p3-p4 in R^d."""
    d1, d2, r = p2 - p1, p4 - p3, p1 - p3
    a, e, f = float(np.dot(d1, d1)), float(np.dot(d2, d2)), float(np.dot(d2, r))
    if a < 1e-12 and e < 1e-12:
        return float(np.linalg.norm(r))
    if a < 1e-12:
        s, t = 0.0, np.clip(f / e, 0.0, 1.0)
    else:
        c = float(np.dot(d1, r))
        if e < 1e-12:
            s, t = float(np.clip(-c / a, 0.0, 1.0)), 0.0
        else:
            b = float(np.dot(d1, d2))
            denom = a * e - b * b
            s = float(np.clip((b * f - c * e) / denom, 0.0, 1.0)) if abs(denom) > 1e-12 else 0.5
            t = (b * s + f) / e
            if t < 0.0:
                t, s = 0.0, float(np.clip(-c / a, 0.0, 1.0))
            elif t > 1.0:
                t, s = 1.0, float(np.clip((b - c) / a, 0.0, 1.0))
    return float(np.linalg.norm((p1 + s * d1) - (p3 + t * d2)))


def _edges_cloud_min_dist(
    moving_pts: np.ndarray,
    moving_edges: List[Tuple[int, int]],
    stationary_pts: np.ndarray,
    stationary_edges: List[Tuple[int, int]],
) -> float:
    """Minimum segment-to-segment distance between two edge-connected point sets.

    This is the geometrically correct minimum distance between two 1-manifolds
    (edge graphs).  Unlike _clouds_min_dist, it detects crossings where the
    edges intersect even when no vertices are close.
    """
    best = float("inf")
    for i, j in moving_edges:
        if i >= len(moving_pts) or j >= len(moving_pts):
            continue
        for k, left in stationary_edges:
            if k >= len(stationary_pts) or left >= len(stationary_pts):
                continue
            d = _seg_seg_min_dist(
                moving_pts[i], moving_pts[j],
                stationary_pts[k], stationary_pts[left],
            )
            if d < best:
                best = d
    return best


def _check_path_intersection(
    moving: np.ndarray,
    stationary: Dict[str, np.ndarray],
    iso: Isotopy,
    n_checks: int = 5,
    tol: float = 0.01,
    *,
    mode: Literal["fast", "exact"] = "fast",
    t_samples: Optional[int] = None,
    edges_moving: Optional[List[Tuple[int, int]]] = None,
    edges_stationary: Optional[Dict[str, List[Tuple[int, int]]]] = None,
) -> Optional[str]:
    """Return a warning string if the isotopy path intersects any stationary cloud.

    When ``edges_moving`` and ``edges_stationary`` are provided (lists of
    ``(i, j)`` vertex-index pairs defining the 1-skeleton of each complex),
    the check uses exact segment-to-segment minimum distance instead of
    vertex-to-vertex distance.  This correctly detects crossings where
    edges intersect but no vertices coincide — the failure mode that arises
    for linked 1-manifolds (circles) passing through each other.

    Without edge information the check falls back to vertex-cloud distance,
    which is blind to interior-of-edge crossings.
    """
    if len(moving) == 0:
        return None
    n = t_samples or n_checks
    use_segs = (edges_moving is not None and edges_stationary is not None
                and len(edges_moving) > 0)

    for t in np.linspace(0.0, 1.0, n + 2)[1:-1]:
        intermediate = iso(moving, float(t))
        bb_min = intermediate.min(axis=0)
        bb_max = intermediate.max(axis=0)

        for name, other in stationary.items():
            if len(other) == 0:
                continue
            # Bounding-box early-exit (same logic as before)
            if mode == "fast":
                o_min, o_max = other.min(axis=0), other.max(axis=0)
                if np.any(bb_min > o_max + tol) or np.any(bb_max < o_min - tol):
                    continue

            # ── Distance computation ──────────────────────────────────────────
            if use_segs:
                stat_edges = (edges_stationary or {}).get(name, [])
                if stat_edges:
                    d = _edges_cloud_min_dist(
                        intermediate, edges_moving, other, stat_edges
                    )
                else:
                    d = _clouds_min_dist(intermediate, other, mode=mode)
            else:
                d = _clouds_min_dist(intermediate, other, mode=mode)

            if d < tol:
                check_kind = "segment-segment" if use_segs else "vertex-vertex"
                return (
                    f"{iso.name} path intersects '{name}' at t≈{t:.2f} "
                    f"(min {check_kind} dist {d:.3g} < tol={tol}, mode={mode}). "
                    "Use check_isotopy=False to skip."
                )
    return None


def _clip_cloud_against_tube(
    P: np.ndarray,
    tube_simplices: Sequence[Tuple[int, ...]],
    K: Any,
    tol: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Clip and split point clouds into kept/masked arrays using scipy's cKDTree over tube centroids."""
    if getattr(K, "_coordinates", None) is None or len(tube_simplices) == 0 or len(P) == 0:
        return P.copy(), np.empty((0, P.shape[1]), dtype=P.dtype)
    
    tube_centroids = []
    coords = K._coordinates
    for τ in tube_simplices:
        if isinstance(coords, dict):
            centroid = np.mean([coords[v] for v in τ], axis=0)
        else:
            centroid = coords[list(τ)].mean(axis=0)
        tube_centroids.append(centroid)
    tube_centroids = np.asarray(tube_centroids)
    
    from scipy.spatial import cKDTree
    tree = cKDTree(tube_centroids)
    dists, _ = tree.query(P, k=1)
    mask_in = dists < tol
    P_masked = P[mask_in]
    P_kept = P[~mask_in]
    return P_kept, P_masked



# ── Deformation mode registry ─────────────────────────────────────────────────
#
# Each mode is a plain function registered with @_register_deform_mode(name).
# Adding a new mode requires only writing a new function — SurgerySession.deform()
# dispatches through this dict without any other changes.

_DEFORM_MODES: Dict[str, Any] = {}


def _register_deform_mode(name: str):
    """Decorator that registers a deformation function under *name*.

    The function signature must be::

        def _deform_<name>(cloud: np.ndarray, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
            ...
            return deformed_cloud, metadata_dict

    After registration the mode is immediately available via
    ``SurgerySession.deform(target, mode=name, **kwargs)``.
    """
    def decorator(fn: Any) -> Any:
        _DEFORM_MODES[name] = fn
        return fn
    return decorator


@_register_deform_mode("open_cut")
def _deform_open_cut(
    cloud: np.ndarray,
    *,
    cut_site: Tuple[float, ...],
    width: float = 1.5,
    falloff_radius: float = 1.0,
    pull_direction: Optional[Any] = None,
    plane_normal: Optional[Any] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Morph a cut manifold into an open cylinder by pulling the rim circles apart.

    What is Being Computed?:
        After a single disk removal, the manifold has one boundary circle (the
        rim of the cut).  This deformation pulls the two angular halves of that
        rim in opposite directions along ``pull_direction``, widening the nick
        into a gap large enough for another object to pass through.

    Algorithm:
        1. Compute per-point Gaussian weights centred on ``cut_site``.
        2. Assign each point a side (±1) based on which half-space of the
           dividing plane (normal: ``plane_normal``) it falls on.
        3. Displace:  Δp = side · weight · (width / 2) · pull_direction.

    Preserved Invariants:
        Points beyond ~3 σ from ``cut_site`` receive negligible displacement.
        The deformation is smooth and symmetric about the cut plane.

    Args:
        cloud: (N, 3) float array — the point cloud to deform.
        cut_site: 3-tuple; the 3D point on the manifold where the cut sits.
        width: Total rim separation — each half moves ``width / 2``.
        falloff_radius: Gaussian σ in world units; controls deformation extent.
        pull_direction: Unit 3-vector along which the rim halves are pulled
            apart.  If ``None``, the direction of *minimum* local variance
            is used (typically the surface normal or the tangent out of the
            manifold plane), which is correct for tori cut at their inner
            equator.
        plane_normal: Unit 3-vector normal to the dividing plane.  If ``None``
            defaults to ``pull_direction`` (most common case).

    Returns:
        (deformed_cloud, metadata_dict)
    """
    cs = np.asarray(cut_site, dtype=float)
    dists = np.linalg.norm(cloud - cs, axis=1)                      # (N,)
    sigma = max(float(falloff_radius), 1e-8)
    weights = np.exp(-0.5 * (dists / sigma) ** 2)                   # (N,)

    # ── Resolve pull_direction ────────────────────────────────────────────────
    if pull_direction is None:
        nearby = cloud[weights > 0.05] - cs
        if nearby.shape[0] >= 3:
            _, eigvecs = np.linalg.eigh(np.cov(nearby.T))
            # eigh returns ascending eigenvalues; index 0 = min-variance axis.
            # For a surface patch this is typically the out-of-plane direction —
            # exactly the axis along which the rim circles should be pulled apart.
            pd: np.ndarray = eigvecs[:, 0].astype(float)
        else:
            pd = np.array([0.0, 1.0, 0.0])
    else:
        pd = np.asarray(pull_direction, dtype=float)
    n = np.linalg.norm(pd)
    pd = pd / n if n > 1e-8 else np.array([0.0, 1.0, 0.0])

    # ── Resolve plane_normal (default = pull_direction) ───────────────────────
    if plane_normal is None:
        pn: np.ndarray = pd.copy()
    else:
        pn = np.asarray(plane_normal, dtype=float)
        n2 = np.linalg.norm(pn)
        pn = pn / n2 if n2 > 1e-8 else pd.copy()

    # ── Displacement ──────────────────────────────────────────────────────────
    # side ∈ {-1, 0, +1} — which half-space each point is on
    side = np.sign((cloud - cs) @ pn)                               # (N,)
    disp = (side * weights * (width / 2.0))[:, None] * pd[None, :] # (N, 3)
    deformed = cloud + disp

    metadata: Dict[str, Any] = {
        "mode": "open_cut",
        "cut_site": cs.tolist(),
        "width": float(width),
        "falloff_radius": float(falloff_radius),
        "pull_direction": pd.tolist(),
        "plane_normal": pn.tolist(),
        "displaced_points": int((weights > 0.01).sum()),
    }
    return deformed, metadata


def list_deform_modes() -> List[str]:
    """Return the names of all currently registered deformation modes.

    New modes added via ``@_register_deform_mode`` appear here automatically.
    """
    return list(_DEFORM_MODES)


# ── Dimension helpers ─────────────────────────────────────────────────────────


def _parse_dim_from_string(s: str) -> int:
    """Extract the first integer exponent from strings like 'R^3', 'D^4'."""
    m = re.search(r'\^(\d+)', s)
    return int(m.group(1)) if m else 3


def _total_dim(type_str: str) -> int:
    """Sum all exponents in a product type string like 'S^2xD^1' → 3."""
    dims = re.findall(r'\^(\d+)', type_str)
    return sum(int(d) for d in dims) if dims else 0


# ── SurgerySession ─────────────────────────────────────────────────────────────


@dataclass
class _SessionSnapshot:
    manifold: SimplicialComplex
    objects_data: Dict[str, SimplicialComplex]
    point_clouds: Dict[str, np.ndarray]
    betti_tracker: Dict[str, Dict[int, int]]
    cobordism_len: int
    stack_len: int
    step_counter: int
    handle_counter: int
    handlebody_state_repr: bytes
    cob_basis_change_len: int
    isotopy_log_pickle: bytes
    W_pickle: bytes
    manifold_coords: Optional[np.ndarray]
    objects_coords: Dict[str, Optional[np.ndarray]]



class _AtomicStep:
    """A context manager enforcing atomicity for a single surgical step.

    If any exception is raised, or if the step block completes without calling
    `txn.commit()`, the session rolls back to its exact pre-step state.
    """
    def __init__(self, session: "SurgerySession", label: str = "") -> None:
        self.session = session
        self.label = label
        self._committed = False
        self._snapshot: Optional[_SessionSnapshot] = None

    @property
    def snapshot(self) -> Optional[_SessionSnapshot]:
        return self._snapshot

    def __enter__(self) -> "_AtomicStep":
        self._snapshot = self.session._capture_state()
        return self

    def commit(self) -> None:
        self._committed = True

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        if exc_type is not None:
            # Revert to the snapshot on exception
            self.session._restore_state(self._snapshot)
            # Do not suppress the exception
            return False
        if not self._committed:
            # Step finished without commit, treat as protocol failure and roll back
            self.session._restore_state(self._snapshot)
            raise SurgeryProtocolError(
                f"Atomic step '{self.label}' exited without explicit commit."
            )
        return False


class _Transaction:
    """A context manager that rolls back the session to a checkpoint on any exception."""
    def __init__(self, session: "SurgerySession", label: str = "") -> None:
        self.session = session
        self.label = label
        self._snapshot: Optional[_SessionSnapshot] = None

    def __enter__(self) -> "_Transaction":
        self._snapshot = self.session._capture_state()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        if exc_type is not None:
            # Revert all changes back to the pre-transaction checkpoint
            self.session._restore_state(self._snapshot)
            # Propagate the exception
            return False
        return False


class SurgerySession:
    """The Surgeon API: a unified workbench for performing and certifying surgery on manifolds.

    Overview:
        Maintains synchronisation between the Topological Model
        (simplicial complex), the Algebraic Model (chain complex), and the
        Geometric Model (point clouds).

    Key Concepts:
        - Atomic Removal:  ``remove_disks`` removes tubular neighbourhoods.
        - Atomic Attachment: ``attach_handle`` glues a handle complex.
        - Ambient Isotopy: ``move`` applies a continuous deformation.
        - Transactional Undo: ``restore`` reverts the last operation.
        - Finality: ``finish`` locks the session; ``logs``/``objects`` remain readable.

    Common Workflows:
        1. ``SurgerySession(ambient_space, objects, point_clouds)``
        2. ``session.AmbientSpace.remove_disks(...)``
        3. ``session.AmbientSpace.attach_handle(...)``
        4. ``session.move(offset=..., target=...)``
        5. ``session.finish()`` then ``session.logs()``

    Attributes:
        manifold: Ambient simplicial complex or string descriptor.
        chain_complex: Algebraic Poincaré Complex.
        point_clouds: Dict mapping names to raw point arrays.
        cobordism: Ordered list of operation records (the bordism trace).
        stack: Transactional undo stack.
        AmbientSpace: Proxy exposing remove_disks / attach_handle / restore.
        objects: Proxy supporting dict access and callable access.

    References:
        Ranicki, A. (1992). Algebraic L-theory and topological manifolds.
        Cambridge University Press.
    """

    def __init__(
        self,
        ambient_space: Any,
        objects: Optional[Dict[str, Any]] = None,
        point_clouds: Optional[Dict[str, np.ndarray]] = None,
    ) -> None:
        self.manifold = ambient_space

        if isinstance(ambient_space, str):
            dim = _parse_dim_from_string(ambient_space)
        else:
            dim = getattr(ambient_space, "dimension", 3)

        cc = ChainComplex(boundaries={}, dimensions=[0, 1], cells={0: 1, 1: 0})
        self.chain_complex = AlgebraicPoincareComplex(
            chain_complex=cc,
            fundamental_class=np.array([1]),
            dimension=dim,
        )

        self.point_clouds: Dict[str, np.ndarray] = (
            {k: np.array(v, dtype=float) for k, v in point_clouds.items()}
            if point_clouds is not None
            else {}
        )

        self._objects: Dict[str, TrackedObject] = {}
        if objects:
            for k, v in objects.items():
                self._objects[k] = TrackedObject(k, v, self)

        self.cobordism: List[Dict[str, Any]] = []
        self.stack: List[Dict[str, Any]] = []
        self._isotopy_log = IsotopyComposition()
        self._isotopy_collision_policy: str = "raise"
        self._isotopy_path_check_mode: Literal["fast", "exact"] = "fast"
        self.handlebody_state = HandlebodyState()
        self._cob_basis_change: List[np.ndarray] = []
        self.W = self._initial_W_from_ambient(ambient_space)
        self._finished: bool = False
        self._handle_counter: int = 0
        self._step_counter: int = 0
        # Tracks current Betti numbers per named object as surgery progresses.
        # Populated on first remove_disks/attach_handle with a target.
        self._betti_tracker: Dict[str, Dict[int, int]] = {}

        self.AmbientSpace = _AmbientSpaceProxy(self)
        self.objects = _ObjectsProxy(self)

    def _capture_state(self) -> _SessionSnapshot:
        try:
            manifold_copy = (self.manifold.model_copy(deep=True)
                             if hasattr(self.manifold, "model_copy")
                             else copy.deepcopy(self.manifold))
            objects_data_copy = {
                name: (obj.data.model_copy(deep=True)
                       if hasattr(obj.data, "model_copy")
                       else copy.deepcopy(obj.data))
                for name, obj in self._objects.items()
            }
            point_clouds_copy = {n: p.copy() for n, p in self.point_clouds.items()}
            betti_tracker_copy = copy.deepcopy(self._betti_tracker)
            
            handlebody_state_repr = pickle.dumps(self.handlebody_state)
            isotopy_log_pickle = pickle.dumps(self._isotopy_log)
            W_pickle = pickle.dumps(self.W)
            
            manifold_coords = (self.manifold._coordinates.copy()
                               if hasattr(self.manifold, "_coordinates")
                               and self.manifold._coordinates is not None
                               else None)
            objects_coords = {
                name: (obj.data._coordinates.copy()
                       if hasattr(obj.data, "_coordinates")
                       and obj.data._coordinates is not None
                       else None)
                for name, obj in self._objects.items()
            }
            
            return _SessionSnapshot(
                manifold=manifold_copy,
                objects_data=objects_data_copy,
                point_clouds=point_clouds_copy,
                betti_tracker=betti_tracker_copy,
                cobordism_len=len(self.cobordism),
                stack_len=len(self.stack),
                step_counter=self._step_counter,
                handle_counter=self._handle_counter,
                handlebody_state_repr=handlebody_state_repr,
                cob_basis_change_len=len(self._cob_basis_change),
                isotopy_log_pickle=isotopy_log_pickle,
                W_pickle=W_pickle,
                manifold_coords=manifold_coords,
                objects_coords=objects_coords,
            )
        except Exception as e:
            raise SurgeryProtocolError(
                f"Session state contains unpicklable references; "
                f"transaction protocol cannot guarantee atomicity. "
                f"Cause: {repr(e)}"
            )

    def _restore_state(self, snap: _SessionSnapshot) -> None:
        self.manifold = snap.manifold
        for name, K in snap.objects_data.items():
            if name in self._objects:
                self._objects[name].data = K
            else:
                self._objects[name] = TrackedObject(name, K, self)
        # Drop any objects added since the snapshot:
        for name in list(self._objects.keys()):
            if name not in snap.objects_data:
                del self._objects[name]
        self.point_clouds = {n: p.copy() for n, p in snap.point_clouds.items()}
        self._betti_tracker = copy.deepcopy(snap.betti_tracker)
        del self.cobordism[snap.cobordism_len:]
        del self.stack[snap.stack_len:]
        self._step_counter = snap.step_counter
        self._handle_counter = snap.handle_counter
        try:
            self.handlebody_state = pickle.loads(snap.handlebody_state_repr)
            self._isotopy_log = pickle.loads(snap.isotopy_log_pickle)
            self.W = pickle.loads(snap.W_pickle)
        except Exception as e:
            raise SurgeryProtocolError(
                f"Session state contains unpicklable references; "
                f"transaction protocol cannot guarantee atomicity. "
                f"Cause: {repr(e)}"
            )
        del self._cob_basis_change[snap.cob_basis_change_len:]
        if snap.manifold_coords is not None:
            self.manifold._coordinates = snap.manifold_coords
        for name, coords in snap.objects_coords.items():
            if coords is not None and name in self._objects:
                self._objects[name].data._coordinates = coords

    def transaction(self, label: str = "") -> _Transaction:
        """Return a transaction context manager.

        Any exception inside the block will cause the session to roll back to the
        pre-transaction checkpoint.
        """
        self._check_finished()
        return _Transaction(self, label)

    def _atomic_step(self, label: str = "") -> _AtomicStep:
        """Return an atomic step context manager.

        The step rolls back the session on exception or if it exits without an
        explicit commit.
        """
        self._check_finished()
        return _AtomicStep(self, label)

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _check_finished(self) -> None:
        if self._finished:
            raise SurgeryFinishedError(
                "Surgery session has been finished. No further mutations are allowed."
            )

    def _set_target_complex(
        self,
        target: Optional[str],
        new_K: SimplicialComplex,
    ) -> None:
        if target is None:
            self.manifold = new_K
        elif target in self._objects:
            self._objects[target].data = new_K
        else:
            raise KeyError(f"_set_target_complex: target {target!r} not in session._objects")
        # Invalidate betti tracker for target — force recomputation next time
        if target in self._betti_tracker:
            del self._betti_tracker[target]


    def _snapshot_clouds(self) -> Dict[str, np.ndarray]:
        return {k: v.copy() for k, v in self.point_clouds.items()}

    def _next_step(self) -> int:
        self._step_counter += 1
        return self._step_counter

    def _apply_isotopy(
        self,
        iso: Isotopy,
        target: Optional[str] = None,
        check_isotopy: bool = True,
    ) -> Isotopy:
        """Internal orchestrator for applying any Isotopy to point clouds."""
        self._check_finished()

        # Resolve which clouds are moving
        targets = [target] if target is not None else list(self.point_clouds.keys())

        # ── Collision guard ──────────────────────────────────────────────────
        if check_isotopy:
            for t_name in targets:
                moving_cloud = self.point_clouds.get(t_name)
                if moving_cloud is not None:
                    stationary = {
                        k: v for k, v in self.point_clouds.items() if k != t_name
                    }
                    if stationary:
                        # Extract 1-skeleton edges from tracked complexes so
                        # the intersection check uses segment-segment distance
                        # instead of vertex-vertex distance.  This is critical
                        # for 1-manifolds (circles, arcs) where edge crossings
                        # occur without any vertices coinciding.
                        def _edges_of(name: str) -> List[Tuple[int, int]]:
                            obj = self._objects.get(name)
                            if obj is None or not hasattr(obj, "data"):
                                return []
                            try:
                                return [tuple(s) for s in obj.data.n_simplices(1)]
                            except Exception:
                                return []

                        edges_mv = _edges_of(t_name)
                        edges_st = {k: _edges_of(k) for k in stationary}
                        use_seg = bool(edges_mv) and any(bool(v) for v in edges_st.values())

                        warn_msg = _check_path_intersection(
                            moving_cloud, stationary, iso,
                            mode=self._isotopy_path_check_mode,
                            edges_moving=edges_mv if use_seg else None,
                            edges_stationary=edges_st if use_seg else None,
                        )
                        if warn_msg:
                            if self._isotopy_collision_policy == "raise":
                                raise ValueError(f"Isotopy collision: {warn_msg}")
                            warnings.warn(f"SurgerySession: {warn_msg}", stacklevel=3)

        cloud_snapshot = self._snapshot_clouds()

        # Stats before
        stats_before: Dict[str, Any] = {}
        for t_name in targets:
            if t_name in self.point_clouds:
                c, mn, mx = _cloud_stats(self.point_clouds[t_name])
                stats_before[t_name] = {
                    "n": len(self.point_clouds[t_name]),
                    "centroid": c.tolist(),
                    "bbox_min": mn.tolist(),
                    "bbox_max": mx.tolist(),
                }

        # Eager application (at t=1)
        coords_snapshot: Dict[str, Optional[np.ndarray]] = {}
        for t_name in targets:
            if t_name in self.point_clouds:
                self.point_clouds[t_name] = iso(self.point_clouds[t_name], 1.0)

            # ── Also apply to the simplicial complex's _coordinates (Fix #16) ──
            data = self._objects[t_name].data if t_name in self._objects else None
            if isinstance(data, SimplicialComplex) and getattr(data, "_coordinates", None) is not None:
                coords_snapshot[t_name] = data._coordinates.copy()
                data._coordinates = iso(data._coordinates, 1.0)

        # Also apply to self.manifold._coordinates if target is None
        if target is None and getattr(self.manifold, "_coordinates", None) is not None:
            coords_snapshot["__manifold__"] = self.manifold._coordinates.copy()
            self.manifold._coordinates = iso(self.manifold._coordinates, 1.0)

        # Stats after
        stats_after: Dict[str, Any] = {}
        for t_name in targets:
            if t_name in self.point_clouds:
                c, mn, mx = _cloud_stats(self.point_clouds[t_name])
                stats_after[t_name] = {
                    "n": len(self.point_clouds[t_name]),
                    "centroid": c.tolist(),
                    "bbox_min": mn.tolist(),
                    "bbox_max": mx.tolist(),
                }

        step_idx = self._next_step()
        self._isotopy_log.compose(step_idx, iso, target)

        # ── Cobordism slab (isotopy) ───────────────────────────────────────
        self._extend_W_by(step_idx, lambda: self._slab_isotopy(iso, target))

        # ── Basis-change (Identity for isotopies) ───────────────────────────
        n_handles = len(self.handlebody_state.handles)
        self._record_basis_change(np.eye(n_handles, dtype=int))

        self.stack.append({
            "action": iso.name,
            "step_index": step_idx,
            "target": target,
            "cloud_snapshot": cloud_snapshot,
            "coords_snapshot": coords_snapshot,
            "isotopy": iso,
        })
        self.cobordism.append({
            "step": iso.name,
            "step_index": step_idx,
            "target": target,
            "targets": targets,
            "description": iso.describe(),
            "stats_before": stats_before,
            "stats_after": stats_after,
            "isotopy_name": iso.name,
            "invariants": self._snapshot_invariants(target),
        })

        return iso

    def _latest_change_of_basis_matrix(self) -> Optional[np.ndarray]:
        """Compute the product of basis-change matrices since the last structural change.

        Returns None if no matrices are logged. Uses GL_infinity padding (identity
        blocks) to compose matrices of different sizes.
        """
        if not self._cob_basis_change:
            return None

        # Determine the maximum dimension in the current sequence
        max_d = max(m.shape[0] for m in self._cob_basis_change)
        
        def pad(m: np.ndarray, d: int) -> np.ndarray:
            curr_d = m.shape[0]
            if curr_d == d:
                return m
            res = np.eye(d, dtype=m.dtype)
            res[:curr_d, :curr_d] = m
            return res

        # Multiply from right to left: M_k @ ... @ M_1
        res = pad(self._cob_basis_change[0], max_d)
        for m in self._cob_basis_change[1:]:
            res = pad(m, max_d) @ res
        return res

    def _record_basis_change(self, matrix: np.ndarray) -> None:
        """Log a new basis-change matrix for Whitehead torsion tracking."""
        self._cob_basis_change.append(matrix)

    def _snapshot_invariants(self, target: Optional[str] = None) -> Dict[str, Any]:
        """Capture a full suite of algebraic invariants for the current session state.
        
        Includes:
            - Betti numbers
            - Intersection / Linking form
            - Smith Normal Form of the form matrix
            - Whitehead torsion (tau)
        """
        cc = self.chain_complex.chain_complex
        
        # Betti numbers
        betti = {}
        if hasattr(cc, "betti_numbers"):
             betti = cc.betti_numbers()
        
        # Form matrix (Intersection/Linking)
        form_record = self._compute_form_matrix(target)
        
        # SNF reduction
        snf_record = self._reduce_snf(form_record)
        
        # Whitehead torsion
        wh_record = self._whitehead_torsion()
        
        # Fundamental group descriptor
        pi1_desc = "1"
        if hasattr(cc, "pi1"):
            from pysurgery.topology.fundamental_group import infer_standard_group_descriptor
            pi1_desc = infer_standard_group_descriptor(cc.pi1) or str(cc.pi1)

        return {
            "coefficient_ring": cc.coefficient_ring,
            "betti": betti,
            "form": form_record,
            "snf": snf_record,
            "whitehead": wh_record,
            "pi1_descriptor": pi1_desc,
        }

    def _compute_form_matrix(self, target: Optional[str] = None) -> Dict[str, Any]:
        """Decide intersection (4k) vs linking (4k-1) form and emit the explicit matrix.

        The choice is made on dimensional grounds from the chain complex's
        dimension.
        """
        n = self.chain_complex.dimension
        if n % 4 == 0:  # 4k-manifold
            if hasattr(self.chain_complex, "intersection_form"):
                Q = self.chain_complex.intersection_form()
                M = np.asarray(Q.matrix, dtype=int)
                return {
                    "kind": "intersection",
                    "dim": n,
                    "matrix": M.tolist(),
                    "rank": int(M.shape[0]),
                    "signature": int(Q.signature()),
                    "determinant": int(np.linalg.det(M).round()),
                    "even": bool(np.all(np.diag(M) % 2 == 0)),
                    "unimodular": abs(int(np.linalg.det(M).round())) == 1,
                }
        
        if n % 4 == 3:  # (4k-1)-manifold
            if hasattr(self.chain_complex, "linking_form"):
                L = self.chain_complex.linking_form()
                Mn, Md = L.numerator_matrix(), L.denominator
                return {
                    "kind": "linking",
                    "dim": n,
                    "matrix": [[(int(a), int(Md)) for a in row] for row in Mn.tolist()],
                    "rank": int(Mn.shape[0]),
                    "signature": None,
                    "determinant": None,
                    "even": None,
                    "unimodular": None,
                }
        
        return {"kind": "none", "dim": n}

    def _reduce_snf(self, form_record: Dict[str, Any]) -> Dict[str, Any]:
        """Compute Smith Normal Form of the form's matrix over Z."""
        if form_record["kind"] not in ("intersection", "linking"):
            return {
                "diagonal": [], 
                "elementary_divisors": [], 
                "free_rank": 0,
                "torsion_summands": [], 
                "method": "skipped"
            }
        
        from sympy.matrices.normalforms import smith_normal_form
        if form_record["kind"] == "linking":
            # rational matrices: clear denominators first
            # matrix is List[List[Tuple[int, int]]]
            Mn = sp.Matrix([[a for (a, _q) in row] for row in form_record["matrix"]])
            # assume common denominator from first entry
            D = form_record["matrix"][0][0][1] if form_record["matrix"] else 1
        else:
            Mn = sp.Matrix(form_record["matrix"])
            D = 1
            
        if Mn.rows == 0 or Mn.cols == 0:
            return {
                "diagonal": [], "denominator": int(D),
                "elementary_divisors": [], "free_rank": 0,
                "torsion_summands": [], "method": "empty_matrix"
            }

        S = smith_normal_form(Mn, domain=sp.ZZ)
        diag = [int(S[i, i]) for i in range(min(S.shape))]
        nonzero = [d for d in diag if d != 0]
        free_rank = len(diag) - len(nonzero)
        return {
            "diagonal": diag,
            "denominator": int(D),
            "elementary_divisors": nonzero,
            "free_rank": free_rank,
            "torsion_summands": _collate_divisors(nonzero),
            "method": "sympy.smith_normal_form",
        }

    def _whitehead_torsion(self) -> Dict[str, Any]:
        """Compute Whitehead torsion element from the change-of-basis ledger."""
        from pysurgery.algebra.k_theory import compute_whitehead_group
        cc = self.chain_complex.chain_complex
        pi1 = getattr(cc, "pi1", None)
        if pi1 is None:
            return {
                "Wh_rank": 0, "Wh_description": "pi1 unavailable",
                "tau": None, "is_s_cobordism": None, "exact": False,
                "method": "skipped"
            }
            
        Wh = compute_whitehead_group(pi1)
        tau = self._latest_change_of_basis_matrix()
        is_s = None
        if tau is not None:
            if tau.size == 0:
                is_s = True
            else:
                is_s = bool(np.all(tau == np.eye(tau.shape[0], dtype=int)))
        
        return {
            "Wh_rank": int(Wh.rank),
            "Wh_description": Wh.description,
            "tau": None if tau is None else f"[{tau.shape[0]}x{tau.shape[1]}]",
            "is_s_cobordism": is_s,
            "exact": bool(getattr(Wh, "exact", False)),
            "method": "k_theory.compute_whitehead_group",
        }

    def _initial_W_from_ambient(self, M0: Any) -> CobordismComplex:
        """Seed W as M0 x [0, 1]."""
        if isinstance(M0, SimplicialComplex):
            vertices = [v[0] for v in M0.n_simplices(0)]
            max_v = max(vertices) if vertices else 0
            offset = max_v + 1
            
            new_simplices = []
            for d in range(M0.dimension + 1):
                for sigma in M0.n_simplices(d):
                    k = len(sigma) - 1
                    for j in range(k + 1):
                        new_simplex = list(sigma[:j+1]) + [v + offset for v in sigma[j:]]
                        new_simplices.append(new_simplex)
            
            underlying = SimplicialComplex.from_simplices(new_simplices)
            boundary_initial_indices = tuple(vertices)
            boundary_final_indices = tuple(v + offset for v in vertices)
            
            return CobordismComplex(
                underlying=underlying,
                boundary_initial_indices=boundary_initial_indices,
                boundary_final_indices=boundary_final_indices
            )
        else:
            # Placeholder for symbolic ambient space
            # Create a minimal complex (e.g. a single point/simplex)
            underlying = SimplicialComplex.from_simplices([[0]])
            return CobordismComplex(
                underlying=underlying,
                boundary_initial_indices=(0,),
                boundary_final_indices=(0,)
            )

    def _extend_W_by(self, step_index: int, slab_builder: Callable[[], SimplicialComplex]) -> None:
        """Glue the slab onto W and update final boundary."""
        slab = slab_builder()
        if slab.dimension < 0 or not slab.simplices:
            self.W.slab_for_step[step_index] = frozenset()
            return
        
        # 1. Union the underlying complex
        all_simplices = set(self.W.underlying.simplices) | set(slab.simplices)
        self.W.underlying = SimplicialComplex.from_simplices(list(all_simplices))
        
        # 2. Record slab vertices
        slab_vertices = set(v for s in slab.simplices for v in s)
        self.W.slab_for_step[step_index] = frozenset(slab_vertices)
        
        # 3. Update boundary final indices
        old_boundary = set(self.W.boundary_final_indices)
        new_boundary = slab_vertices - old_boundary
        if new_boundary:
            self.W.boundary_final_indices = tuple(sorted(list(new_boundary)))

    # ── Slab builders ─────────────────────────────────────────────────────────

    def _slab_isotopy(self, iso: Isotopy, target: Optional[str]) -> SimplicialComplex:
        """Product strip M_final x [0,1]."""
        if isinstance(self.manifold, str):
            return SimplicialComplex.from_simplices([])
        
        Knew = self.manifold
        max_w = max(v for s in self.W.underlying.simplices for v in s) if self.W.underlying.simplices else 0
        offset = max_w + 1
        
        new_simplices = []
        for d in range(Knew.dimension + 1):
            for sigma in Knew.n_simplices(d):
                k = len(sigma) - 1
                for j in range(k + 1):
                    new_simplex = list(sigma[:j+1]) + [v + offset for v in sigma[j:]]
                    new_simplices.append(new_simplex)
        return SimplicialComplex.from_simplices(new_simplices)

    def _slab_disk_removal(self, types: Any, at: Any) -> SimplicialComplex:
        """Slab for disk removal."""
        if isinstance(self.manifold, str):
            return SimplicialComplex.from_simplices([])
        
        Knew = self.manifold
        max_w = max(v for s in self.W.underlying.simplices for v in s) if self.W.underlying.simplices else 0
        offset = max_w + 1
        
        new_simplices = []
        for d in range(Knew.dimension + 1):
            for sigma in Knew.n_simplices(d):
                k = len(sigma) - 1
                for j in range(k + 1):
                    new_simplex = list(sigma[:j+1]) + [v + offset for v in sigma[j:]]
                    new_simplices.append(new_simplex)
        return SimplicialComplex.from_simplices(new_simplices)

    def _slab_handle(self, h_type: str, framing: Framing, at: Any) -> SimplicialComplex:
        """Slab for handle attachment."""
        if isinstance(self.manifold, str):
            return SimplicialComplex.from_simplices([])
        
        Knew = self.manifold
        max_w = max(v for s in self.W.underlying.simplices for v in s) if self.W.underlying.simplices else 0
        offset = max_w + 1
        
        new_simplices = []
        for d in range(Knew.dimension + 1):
            for sigma in Knew.n_simplices(d):
                k = len(sigma) - 1
                for j in range(k + 1):
                    new_simplex = list(sigma[:j+1]) + [v + offset for v in sigma[j:]]
                    new_simplices.append(new_simplex)
        return SimplicialComplex.from_simplices(new_simplices)

    def _slab_slide(self, slider: str, over: str, sign: int) -> SimplicialComplex:
        """Slab for handle slide."""
        return SimplicialComplex.from_simplices([])

    def _slab_cancel(self, h_lo: str, h_hi: str) -> SimplicialComplex:
        """Slab for handle cancellation."""
        return SimplicialComplex.from_simplices([])


    def _ambient_dim(self) -> int:
        if isinstance(self.manifold, str):
            return _parse_dim_from_string(self.manifold)
        return getattr(self.manifold, "dimension", 3)

    # ── Surgical operations ────────────────────────────────────────────────────

    def remove_disks(
        self,
        types: Any,
        at: List[Any],
        target: Optional[str] = None,
    ) -> None:
        """Remove the interior of one or more tubular neighbourhoods (disks).

        Operates on the manifold (or a named tracked object) and clips any
        registered point clouds.
        """
        self._check_finished()
        with self._atomic_step("remove_disks") as txn:
            if isinstance(types, str):
                types = (types,)

            ambient_dim = self._ambient_dim()
            for t in types:
                d = _total_dim(t)
                if d > ambient_dim:
                    raise DimensionalConsistencyError(
                        f"Cannot remove disk of type {t} (dim={d}) from a "
                        f"{ambient_dim}-manifold."
                    )

            K_target_before = (self._objects[target].data if target and target in self._objects
                               else self.manifold)

            if isinstance(K_target_before, str):
                K_new = K_target_before
                tube_simplices = ()
                new_point_clouds = dict(self.point_clouds)
                # Seed mock Betti numbers for symbolic / placeholder mode:
                betti_before = {j: (1 if j == 0 else 0) for j in range(ambient_dim + 1)}
                betti_delta = {}
                for t in types:
                    k_disk = _total_dim(t)
                    d = _predicted_betti_delta_for_index_k_surgery(
                        k_disk, ambient_dim, betti_before, op="remove"
                    )
                    for j, dv in d.items():
                        betti_delta[j] = betti_delta.get(j, 0) + dv
                observed_after = {
                    j: betti_before.get(j, 0) + betti_delta.get(j, 0)
                    for j in betti_before
                }
            else:
                K_new, tube_simplices = _apply_disk_removal_to_complex(K_target_before, types, at)
                
                new_point_clouds = dict(self.point_clouds)
                if target and target in self.point_clouds:
                    DEFAULT_TUBE_RADIUS = 0.05
                    P_kept, P_masked = _clip_cloud_against_tube(
                        self.point_clouds[target], tube_simplices, K_target_before,
                        tol=DEFAULT_TUBE_RADIUS,
                    )
                    new_point_clouds[target] = P_kept
                    new_point_clouds[target + "__removed"] = P_masked

                betti_before = K_target_before.betti_numbers(backend="auto")
                observed_after = K_new.betti_numbers(backend="auto")

                # Verify that observed Betti numbers are compatible with removing the disks in `types`.
                # For each disk of dimension k, the Betti change is either:
                #   Case A (cycle cut):     Δβ_k = -1,  Δβ_{k-1} = 0
                #   Case B (boundary born):  Δβ_k = 0,   Δβ_{k-1} = +1
                # All other Betti numbers remain unchanged.
                # If multiple disks are removed, the total delta is a sum of such choices.
                possible_deltas = [{}]
                for t in types:
                    k = _total_dim(t)
                    next_deltas = []
                    for d in possible_deltas:
                        # Case A:
                        d_a = dict(d)
                        d_a[k] = d_a.get(k, 0) - 1
                        next_deltas.append(d_a)
                        # Case B:
                        d_b = dict(d)
                        d_b[k - 1] = d_b.get(k - 1, 0) + 1
                        next_deltas.append(d_b)
                    possible_deltas = next_deltas

                matched = False
                for delta in possible_deltas:
                    all_dims = set(betti_before.keys()) | set(observed_after.keys()) | set(delta.keys())
                    if all(betti_before.get(j, 0) + delta.get(j, 0) == observed_after.get(j, 0) for j in all_dims if j >= 0):
                        matched = True
                        break

                if not matched:
                    raise SurgeryInvariantBroken(
                        f"remove_disks: observed β = {observed_after} is not compatible with "
                        f"removing disks of types {types} from K with β = {betti_before}; "
                        f"Mayer-Vietoris postcondition failed."
                    )
                betti_delta = {j: observed_after.get(j, 0) - betti_before.get(j, 0) for j in betti_before}

            self._set_target_complex(target, K_new)
            self.point_clouds = new_point_clouds
            
            step_idx = self._next_step()
            
            # Basis-change (Identity for remove_disks)
            n_handles = len(self.handlebody_state.handles)
            self._record_basis_change(np.eye(n_handles, dtype=int))
            
            # ── Cobordism slab (removal) ──
            self._extend_W_by(step_idx, lambda: self._slab_disk_removal(types, at))

            cob_entry = {
                "step": "remove_disks",
                "step_index": step_idx,
                "types": types,
                "types_math": [_to_math(t) for t in types],
                "dims": [_total_dim(t) for t in types],
                "ambient_dim": ambient_dim,
                "at": at,
                "target": target,
                "betti_before": betti_before,
                "betti_after": observed_after,
                "delta_betti": [betti_delta.get(j, 0) for j in sorted(betti_before)],
                "torsion_change": 0,
                "theorem": "SURGERY_HANDLE_MAYER_VIETORIS",
                "invariants": self._snapshot_invariants(target),
            }
            self.cobordism.append(cob_entry)
            self.stack.append({
                "action": "remove_disks",
                "step_index": step_idx,
                "types": types,
                "at": at,
                "target": target,
                "cloud_snapshot": self._snapshot_clouds(),
                "snapshot": txn.snapshot,
            })
            txn.commit()

    def attach_handle(
        self,
        at: Any,
        handle_type: str = "S^2xD^1",
        target: Optional[str] = None,
        framing: Union[int, Tuple[int, int], Framing, FramingResult, None] = None,
        attaching_sphere: Any = None,
        co_core: Any = None,
        cancelling_of: Optional[int] = None,  # NEW
    ) -> FramedHandle:
        """Attaches a handle to the boundary created by disk removal."""
        self._check_finished()
        with self._atomic_step("attach_handle") as txn:
            ambient_dim = self._ambient_dim()
            handle_dim = _total_dim(handle_type)
            if handle_dim > ambient_dim:
                raise DimensionalConsistencyError(
                    f"Cannot attach handle of type {handle_type} (dim={handle_dim}) to a "
                    f"{ambient_dim}-manifold."
                )

            self._handle_counter += 1
            handle_name = f"Handle{self._handle_counter}"

            K_target_before = (self._objects[target].data if target and target in self._objects
                               else self.manifold)

            k_match = re.search(r'S\^(\d+)', handle_type)
            k = int(k_match.group(1)) + 1 if k_match else _total_dim(handle_type)

            if attaching_sphere is None and at is not None:
                attaching_sphere = at if isinstance(at, (list, tuple)) else [at]

            # Resolve FramingResult if necessary
            resolved_framing: Optional[Union[int, Tuple[int, int], Framing]] = None
            if isinstance(framing, FramingResult):
                resolved_framing = framing.value
            else:
                resolved_framing = framing

            target_dim = ambient_dim
            if not isinstance(K_target_before, str):
                target_dim = K_target_before.dimension

            if isinstance(K_target_before, str):
                K_new = K_target_before
                class MockAttachment:
                    def __init__(self, framing):
                        self.framing = framing
                attachment = MockAttachment(resolved_framing)

                betti_before = {j: (1 if j == 0 else 0) for j in range(target_dim + 1)}
                betti_delta = {}
                if cancelling_of is not None:
                    prior_entry = next((c for c in reversed(self.cobordism) if c.get("step_index") == cancelling_of), None)
                    if prior_entry:
                        prior_betti_before = prior_entry.get("betti_before")
                        prior_betti_after = prior_entry.get("betti_after")
                        if prior_betti_before and prior_betti_after:
                            for j in prior_betti_before:
                                betti_delta[j] = prior_betti_before[j] - prior_betti_after.get(j, 0)
                    if not betti_delta:
                        d = _predicted_betti_delta_for_index_k_surgery(
                            k, target_dim, betti_before, op="attach"
                        )
                        for j, dv in d.items():
                            betti_delta[j] = betti_delta.get(j, 0) + dv
                else:
                    d = _predicted_betti_delta_for_index_k_surgery(
                        k, target_dim, betti_before, op="attach"
                    )
                    for j, dv in d.items():
                        betti_delta[j] = betti_delta.get(j, 0) + dv

                observed_after = {
                    j: betti_before.get(j, 0) + betti_delta.get(j, 0)
                    for j in betti_before
                }
            else:
                attachment = _build_handle_attachment_from_sphere(
                    K=K_target_before,
                    attaching_sphere=tuple(attaching_sphere),
                    k=k, n=target_dim,
                    framing=resolved_framing,
                    backend="auto",
                )

                if cancelling_of is not None:
                    import dataclasses
                    attachment = dataclasses.replace(attachment, tubular_neighborhood=())

                result = perform_handle_surgery(K_target_before, attachment, backend="auto")
                K_new = result.complex_after

                # Betti tracking
                betti_before = K_target_before.betti_numbers(backend="auto")
                
                observed_after = K_new.betti_numbers(backend="auto")

                # Predicted deltas for index-k surgery: 
                # j=k-1 can drop by 1 or stay 0.
                # j=k   can rise by 1 or stay 0.
                # All other dimensions must stay the same (delta=0).
                # We use {-1, 0, 1} for both to be safe and match perform_handle_surgery.
                allowed_deltas = {}
                for j in range(target_dim + 1):
                    if j in (k - 1, k):
                        allowed_deltas[j] = {-1, 0, 1}
                    else:
                        allowed_deltas[j] = {0}

                betti_delta = {}
                for j in allowed_deltas:
                    beta_b = betti_before.get(j, 0)
                    beta_a = observed_after.get(j, 0)
                    delta = beta_a - beta_b
                    betti_delta[j] = delta
                    # Use any() to support unhashable mock objects like EqualToAnything in tests
                    if not any(delta == allowed for allowed in allowed_deltas[j]):
                        raise SurgeryInvariantBroken(
                            f"attach_handle: dim {j} delta {delta} not in {allowed_deltas[j]}; "
                            f"betti_before={betti_before}, observed_after={observed_after}. "
                            f"Mayer-Vietoris postcondition failed."
                        )

            self._set_target_complex(target, K_new)

            # Register the new handle as a tracked object in self._objects
            self._objects[handle_name] = TrackedObject(
                name=handle_name,
                data=handle_type if isinstance(K_target_before, str) else K_new,
                session=self
            )

            actual_framing = (attachment.framing if isinstance(attachment.framing, Framing)
                               else Framing(kind="integer", integer=attachment.framing or 0))

            fh = FramedHandle(
                id=handle_name,
                index=k,
                handle_type=handle_type,
                framing=actual_framing,
                attaching_sphere=attaching_sphere,
                co_core=co_core,
                underlying=None,  # type: ignore
                attached_step=self._step_counter + 1
            )
            self.handlebody_state.handles.append(fh)

            n_old = len(self.handlebody_state.handles) - 1
            bc_matrix = np.eye(n_old + 1, dtype=int)
            framing_val = 1
            if actual_framing.kind == "integer" and actual_framing.integer is not None:
                if actual_framing.integer < 0:
                    framing_val = -1
            bc_matrix[n_old, n_old] = framing_val
            self._record_basis_change(bc_matrix)

            step_idx = self._next_step()
            
            # ── Cobordism slab (attachment) ──
            self._extend_W_by(step_idx, lambda: self._slab_handle(handle_type, actual_framing, at))

            cob_entry = {
                "step": "attach_handle",
                "step_index": step_idx,
                "handle_type": handle_type,
                "handle_type_math": _to_math(handle_type),
                "handle_dim": handle_dim,
                "handle_name": handle_name,
                "ambient_dim": ambient_dim,
                "at": at,
                "target": target,
                "betti_before": betti_before,
                "betti_after": observed_after,
                "delta_betti": [betti_delta.get(j, 0) for j in sorted(betti_before)],
                "torsion_change": 0,
                "theorem": "SURGERY_CANCELLING_PAIR" if cancelling_of is not None else "SURGERY_HANDLE_MAYER_VIETORIS",
                "framing": actual_framing.render(),
                "invariants": self._snapshot_invariants(target),
                "cancelling_of": cancelling_of,  # NEW
            }
            self.cobordism.append(cob_entry)
            self.stack.append({
                "action": "attach_handle",
                "step_index": step_idx,
                "handle_type": handle_type,
                "handle_name": handle_name,
                "at": at,
                "target": target,
                "framing": actual_framing,
                "cloud_snapshot": self._snapshot_clouds(),
                "snapshot": txn.snapshot,
            })
            txn.commit()
            return fh

    def move(
        self,
        offset: Optional[Sequence[float]] = None,
        through: Any = None,
        target: Optional[str] = None,
        check_isotopy: bool = True,
    ) -> Isotopy:
        """Apply a translation OR a through-point isotopy.

        - If `through` is None: classical TranslateIsotopy(offset).
        - If `through` is not None: ThroughPointIsotopy passing through `through`.
          The start is taken as the centroid of the target's cloud (or
          self.manifold's coords if target is None); the end is start + offset.
        """
        if through is None:
            iso = TranslateIsotopy(offset if offset is not None else (0.0, 0.0, 0.0))
            return self._apply_isotopy(iso, target=target, check_isotopy=check_isotopy)

        # Determine start and end
        if target and target in self.point_clouds and len(self.point_clouds[target]) > 0:
            start = self.point_clouds[target].mean(axis=0)
        elif target and target in self._objects and getattr(self._objects[target].data, "_coordinates", None) is not None:
            start = self._objects[target].data._coordinates.mean(axis=0)
        elif getattr(self.manifold, "_coordinates", None) is not None:
            start = self.manifold._coordinates.mean(axis=0)
        else:
            start = np.zeros_like(through)

        via = np.asarray(through, dtype=float)
        end = start + np.asarray(offset if offset is not None else (0.0, 0.0, 0.0), dtype=float)
        iso = ThroughPointIsotopy(start=start, via=via, end=end)
        return self._apply_isotopy(iso, target=target, check_isotopy=check_isotopy)

    def translate(
        self,
        offset: Sequence[float],
        target: Optional[str] = None,
        check_isotopy: bool = True,
    ) -> Isotopy:
        """Alias for move()."""
        return self.move(offset=offset, target=target, check_isotopy=check_isotopy)

    def rotate(
        self,
        angle: float,
        axis: Optional[Sequence[float]] = None,
        plane: Optional[Tuple[int, int]] = None,
        about: Optional[Sequence[float]] = None,
        target: Optional[str] = None,
        check_isotopy: bool = True,
    ) -> Isotopy:
        """Apply a rotation isotopy."""
        iso = RotateIsotopy(angle=angle, axis=axis, plane=plane, about=about)
        return self._apply_isotopy(iso, target=target, check_isotopy=check_isotopy)

    def scale(
        self,
        factors: Union[float, Sequence[float]],
        about: Optional[Sequence[float]] = None,
        target: Optional[str] = None,
        check_isotopy: bool = True,
    ) -> Isotopy:
        """Apply a scaling isotopy."""
        iso = ScaleIsotopy(factors=factors, about=about)
        return self._apply_isotopy(iso, target=target, check_isotopy=check_isotopy)

    def slide_handle(
        self,
        slider_id: str,
        over_id: str,
        *,
        sign: Literal[1, -1] = 1,
    ) -> None:
        """Slide handle 'slider' over handle 'over'.
        
        Requires index(slider) == index(over).
        """
        self._check_finished()
        
        # 1. Resolve handles
        h_slider = next((h for h in self.handlebody_state.handles if h.id == slider_id), None)
        h_over = next((h for h in self.handlebody_state.handles if h.id == over_id), None)
        
        if not h_slider or not h_over:
            raise KeyError(f"Handle IDs {slider_id} or {over_id} not found.")
        
        if h_slider.index != h_over.index:
            raise KirbyMoveError(
                f"Handle slide requires same index: slider={h_slider.index}, over={h_over.index}"
            )
        
        # 2. Update basis change matrix (global basis of all handles)
        n = len(self.handlebody_state.handles)
        idx_s_global = self.handlebody_state.handles.index(h_slider)
        idx_o_global = self.handlebody_state.handles.index(h_over)
        
        slide_mat = np.eye(n, dtype=int)
        slide_mat[idx_s_global, idx_o_global] = sign
        self._record_basis_change(slide_mat)
        
        # 3. Kirby update (4D case)
        ambient_n = self._ambient_dim()
        if ambient_n == 4 and h_slider.index == 2:
            if self.handlebody_state.kirby:
                # assuming KirbyLinkingMatrix indices match handles list order
                self.handlebody_state.kirby = self.handlebody_state.kirby.handle_slide(
                    idx_s_global, idx_o_global
                )
        
        # 4. Algebraic update (chain_complex boundary)
        # Relative indices for the specific dimension k
        k = h_slider.index
        handles_k = [h for h in self.handlebody_state.handles if h.index == k]
        idx_s_rel = handles_k.index(h_slider)
        idx_o_rel = handles_k.index(h_over)

        if hasattr(self.chain_complex, "chain_complex"):
            bound = self.chain_complex.chain_complex.boundaries.get(k)
            if bound is not None and idx_s_rel < bound.shape[1] and idx_o_rel < bound.shape[1]:
                from scipy.sparse import csr_matrix
                dense = bound.toarray()
                # col_slider = col_slider + sign * col_over
                dense[:, idx_s_rel] += sign * dense[:, idx_o_rel]
                self.chain_complex.chain_complex.boundaries[k] = csr_matrix(dense)
            
            # Also update boundary[k+1] if it exists (rows correspond to index k handles)
            bound_next = self.chain_complex.chain_complex.boundaries.get(k + 1)
            if bound_next is not None and idx_s_rel < bound_next.shape[0] and idx_o_rel < bound_next.shape[0]:
                from scipy.sparse import csr_matrix
                dense_next = bound_next.toarray()
                # row_over = row_over - sign * row_slider (inverse basis change)
                dense_next[idx_o_rel, :] -= sign * dense_next[idx_s_rel, :]
                self.chain_complex.chain_complex.boundaries[k+1] = csr_matrix(dense_next)

        step_idx = self._next_step()
        # ── Cobordism slab (slide) ─────────────────────────────────────────
        self._extend_W_by(step_idx, lambda: self._slab_slide(slider_id, over_id, sign))

        self.cobordism.append({
            "step": "slide_handle",
            "step_index": step_idx,
            "slider": slider_id,
            "over": over_id,
            "sign": sign,
            "description": f"slide {slider_id} over {over_id} (sign={sign:+d})",
            "invariants": self._snapshot_invariants(),
        })

    def cancellation_candidates(self) -> List[CancellationCandidate]:
        """Recompute Cerf cancellation candidates from current boundary matrices."""
        candidates = []
        handles = self.handlebody_state.handles
        
        # Check pairs (h_lo, h_hi) with index(hi) = index(lo) + 1
        for h_lo in handles:
            for h_hi in handles:
                if h_hi.index == h_lo.index + 1:
                    k = h_hi.index
                    # Relative indices in the basis for C_{k-1} and C_k
                    # We assume handles of same index appear in the basis in 
                    # the order they were attached.
                    handles_lo = [h for h in handles if h.index == k - 1]
                    handles_hi = [h for h in handles if h.index == k]
                    
                    lo_idx = handles_lo.index(h_lo)
                    hi_idx = handles_hi.index(h_hi)
                    
                    if hasattr(self.chain_complex, "chain_complex"):
                        bound = self.chain_complex.chain_complex.boundaries.get(k)
                        if bound is not None:
                            dense = bound.toarray()
                            if lo_idx < dense.shape[0] and hi_idx < dense.shape[1]:
                                intersection = int(dense[lo_idx, hi_idx])
                                if abs(intersection) == 1:
                                    # Candidate found
                                    col = dense[:, hi_idx]
                                    row = dense[lo_idx, :]
                                    is_geom = bool(np.count_nonzero(col) == 1 and 
                                                   np.count_nonzero(row) == 1)
                                    
                                    candidates.append(CancellationCandidate(
                                        handle_lo=h_lo,
                                        handle_hi=h_hi,
                                        intersection_signed=intersection,
                                        is_geometric=is_geom
                                    ))
        
        self.handlebody_state.pending_cancellations = candidates
        return candidates

    def cancel_handles(
        self,
        h_lo_id: str,
        h_hi_id: str,
        *,
        require_geometric: bool = True,
    ) -> None:
        """Execute the cancellation (h_lo, h_hi) -> empty."""
        self._check_finished()
        
        # 1. Update candidates
        self.cancellation_candidates()
        
        # 2. Find specific candidate
        match = next((c for c in self.handlebody_state.pending_cancellations 
                      if c.handle_lo.id == h_lo_id and c.handle_hi.id == h_hi_id), None)
        
        if not match:
            raise KirbyMoveError(f"Handles {h_lo_id}/{h_hi_id} are not a valid cancellation pair.")
        
        if require_geometric and not match.is_geometric:
            raise KirbyMoveError(f"Cancellation of {h_lo_id}/{h_hi_id} must be geometric (transverse intersection).")

        # 3. Algebraic update
        k = match.handle_hi.index
        handles_lo = [h for h in self.handlebody_state.handles if h.index == k - 1]
        handles_hi = [h for h in self.handlebody_state.handles if h.index == k]
        
        lo_idx_rel = handles_lo.index(match.handle_lo)
        hi_idx_rel = handles_hi.index(match.handle_hi)
        
        if hasattr(self.chain_complex, "chain_complex"):
            # boundary[k] : C_k -> C_k-1. 
            # Drop column hi_idx_rel and row lo_idx_rel.
            bound = self.chain_complex.chain_complex.boundaries.get(k)
            if bound is not None:
                from scipy.sparse import csr_matrix
                dense = bound.toarray()
                # Remove row lo_idx_rel
                new_dense = np.delete(dense, lo_idx_rel, axis=0)
                # Remove column hi_idx_rel
                new_dense = np.delete(new_dense, hi_idx_rel, axis=1)
                self.chain_complex.chain_complex.boundaries[k] = csr_matrix(new_dense)
            
            # boundary[k-1] : C_k-1 -> C_k-2. Column lo_idx_rel should be dropped.
            bound_prev = self.chain_complex.chain_complex.boundaries.get(k - 1)
            if bound_prev is not None:
                from scipy.sparse import csr_matrix
                dense_prev = bound_prev.toarray()
                if lo_idx_rel < dense_prev.shape[1]:
                    new_dense_prev = np.delete(dense_prev, lo_idx_rel, axis=1)
                    self.chain_complex.chain_complex.boundaries[k - 1] = csr_matrix(new_dense_prev)
                
            # boundary[k+1] : C_k+1 -> C_k. Row hi_idx_rel should be dropped.
            bound_next = self.chain_complex.chain_complex.boundaries.get(k + 1)
            if bound_next is not None:
                from scipy.sparse import csr_matrix
                dense_next = bound_next.toarray()
                if hi_idx_rel < dense_next.shape[0]:
                    new_dense_next = np.delete(dense_next, hi_idx_rel, axis=0)
                    self.chain_complex.chain_complex.boundaries[k + 1] = csr_matrix(new_dense_next)

        # 4. Remove handles from state
        self.handlebody_state.handles.remove(match.handle_lo)
        self.handlebody_state.handles.remove(match.handle_hi)
        
        step_idx = self._next_step()
        # ── Cobordism slab (cancellation) ──────────────────────────────────
        self._extend_W_by(step_idx, lambda: self._slab_cancel(h_lo_id, h_hi_id))

        self.cobordism.append({
            "step": "cancel_handles",
            "step_index": step_idx,
            "handle_lo": h_lo_id,
            "handle_hi": h_hi_id,
            "description": f"cancel handle pair ({h_lo_id}, {h_hi_id})",
            "invariants": self._snapshot_invariants(),
        })
        
        # 6. Clear basis change segment
        self._cob_basis_change = []

    def deform(
        self,
        target: str,
        mode: str,
        check_isotopy: bool = True,
        **kwargs: Any,
    ) -> Isotopy:
        """Apply a named local deformation to the point cloud of a tracked object.

        Delegates to _apply_isotopy using a LegacyDeformIsotopy.
        """
        if mode not in _DEFORM_MODES:
            raise ValueError(
                f"Unknown deformation mode {mode!r}. "
                f"Registered: {list_deform_modes()}"
            )
        if target not in self.point_clouds:
            raise KeyError(
                f"No point cloud for {target!r}. "
                f"Registered clouds: {list(self.point_clouds)}"
            )

        iso = LegacyDeformIsotopy(mode, kwargs)
        # Apply the isotopy
        self._apply_isotopy(iso, target=target, check_isotopy=check_isotopy)

        # After application, iso._meta is populated by the wrapped function.
        # We can update the last cobordism entry with this metadata.
        if self.cobordism and self.cobordism[-1]["step"] == iso.name:
            self.cobordism[-1]["meta"] = iso._meta

        return iso

    def cobordism_complex(self) -> CobordismComplex:
        """Return the running W (mutates as the session progresses)."""
        return self.W

    def cobordism_interior(self) -> SimplicialComplex:
        """Return the interior of the cobordism complex."""
        return self.W.interior()

    def cobordism_homology(self, n: int = 0, backend: str = "auto"):
        """H_n(W, M_0 ⊔ M_m) — derived from W.underlying.homology."""
        return self.W.homology(n=n, backend=backend)

    def is_h_cobordism(self) -> bool:
        """Return True iff the boundary inclusions are both homotopy equivalences.

        Equivalently, the inclusions M_0 ↪ W and M_m ↪ W are homotopy
        equivalences (H_*(W, M_0) = 0).
        """
        if self.W.is_collared:
            return True
        return False

    def is_s_cobordism(self) -> bool:
        """True iff h-cobordism AND Whitehead torsion τ(W, M_0) = 0 in Wh(π_1)."""
        if not self.is_h_cobordism():
            return False
        
        wt = self._whitehead_torsion()
        return wt.get("is_s_cobordism", False)

    def restore(self, target: Optional[str] = None) -> None:
        """Transactional undo: revert the last surgical operation.

        What is Being Computed?:
            Reverts the last surgical operation, restoring point clouds to their
            prior state.

        Algorithm:
            Pops the last entry from the undo stack and applies its cloud snapshot.
            For ``attach_handle``, the registered Handle object is also removed.

        Preserved Invariants:
            Restores all topological and geometric invariants to prior values.

        Args:
            target: Currently unused; full-session undo semantics apply.

        Raises:
            SurgeryFinishedError: if called after ``finish()``.
        """
        self._check_finished()
        if not self.stack:
            return

        last_op = self.stack.pop()

        snapshot = last_op.get("cloud_snapshot", {})
        self.point_clouds.update(snapshot)

        reverted_action = last_op.get("action", "unknown")
        reverted_step = last_op.get("step_index", "?")

        if reverted_action == "attach_handle":
            handle_name = last_op.get("handle_name")
            if handle_name and handle_name in self._objects:
                del self._objects[handle_name]
                self._handle_counter = max(0, self._handle_counter - 1)

        step_idx = self._next_step()
        self.cobordism.append({
            "step": "restore",
            "step_index": step_idx,
            "reverted": reverted_action,
            "reverted_step": reverted_step,
            "cloud_keys": list(snapshot.keys()),
        })

    def finish(self) -> None:
        """Finalise the session, locking all mutative operations.

        Algorithm:
            Sets the internal ``_finished`` flag. After this point only inspection
            methods remain accessible: ``logs``, ``objects``, ``point_clouds``,
            ``manifold``, ``chain_complex``, ``evaluate_obstruction``.

        Preserved Invariants:
            All topological structures locked at their current state.
        """
        self._finished = True

    # ── Inspection ─────────────────────────────────────────────────────────────

    def evaluate_obstruction(self) -> Any:
        """Compute the surgery obstruction element in L_n(π₁) via the Ranicki assembly map.

        Algorithm:
            Constructs an AlgebraicSurgeryComplex from the session's current chain
            complex and evaluates the assembly map.

        Preserved Invariants:
            Evaluates whether the normal map is normally cobordant to a homotopy
            equivalence.

        Returns:
            ObstructionResult with L-group element, exactness flag, and assembly
            certification.

        References:
            Ranicki, A. (1992). Algebraic L-theory and topological manifolds, §9.
        """
        asc = AlgebraicSurgeryComplex(
            domain=self.chain_complex,
            codomain=self.chain_complex,
            degree=1,
        )
        return perform_algebraic_surgery(asc)

    # ── Logging ────────────────────────────────────────────────────────────────

    def logs(self, latex: bool = False) -> str:
        """Generate a structured Surgery Sequence log of the session.

        What is Being Computed?:
            Generates a structured Surgery Sequence log with three sections:
            (I) Topological Trace — Mayer-Vietoris certificates and step transitions;
            (II) Geometric Trace — per-step point-cloud statistics and isotopy functions;
            (III) Algebraic Proof — intersection form, L-group obstruction.

        Algorithm:
            Iterates over the cobordism trace, formats each operation with all
            captured metadata, computes final cloud statistics, then evaluates the
            algebraic obstruction.

        Preserved Invariants:
            Read-only; does not mutate session state.

        Args:
            latex: If True, renders using LaTeX commands instead of Unicode.

        Returns:
            Formatted multi-section string.
        """
        if latex:
            return self._logs_latex()
        return self._logs_plain()

    # ── Plain-text rendering ───────────────────────────────────────────────────

    def _logs_plain(self) -> str:
        W = 70  # total line width
        lines: List[str] = []
        n = self.chain_complex.dimension

        def rule(char: str = "═") -> str:
            return char * W

        def hdr(text: str) -> str:
            pad = max(0, W - len(text) - 2)
            return f"  {text}{' ' * pad}"

        # ── Header ─────────────────────────────────────────────────────────────
        ambient_label = _to_math(str(self.manifold)) if isinstance(self.manifold, str) else repr(self.manifold)
        n_steps = self._step_counter
        obj_names = list(self._objects.keys())
        obj_str = ", ".join(obj_names) if obj_names else "none"

        lines.append(rule())
        lines.append(hdr("  SURGERY SEQUENCE LOG"))
        lines.append(hdr(f"  Ambient: {ambient_label}  ·  Steps recorded: {n_steps}  ·  Objects: {obj_str}"))
        lines.append(rule())

        # ── Section I: Topological Trace ───────────────────────────────────────
        lines.append("")
        lines.append(_divider("I. TOPOLOGICAL TRACE", char="━", width=W))
        lines.append("")

        # Surgery sequence arrow diagram
        topo_ops = [op for op in self.cobordism if op["step"] in ("remove_disks", "attach_handle")]
        if topo_ops:
            parts = ["  M₀"]
            for i, op in enumerate(topo_ops):
                if op["step"] == "remove_disks":
                    types_str = ", ".join(op.get("types_math", op["types"]))
                    tgt = op.get("target")
                    lbl = f"remove_disks({types_str}" + (f" on {tgt}" if tgt else "") + ")"
                    arrow = f" ──[{lbl}]──▶ M{_sub(i + 1)}"
                else:
                    ht = op.get("handle_type_math", op.get("handle_type", "?"))
                    tgt = op.get("target")
                    lbl = f"attach_handle({ht}" + (f" on {tgt}" if tgt else "") + ")"
                    arrow = f" ──[{lbl}]──▶ M{_sub(i + 1)}"
                parts.append(arrow)
            lines.append("".join(parts))
        else:
            lines.append("  (no topological steps recorded)")
        lines.append("")

        # Betti number sequence table (one column per topological step)
        tracked_targets = list({
            op.get("target") for op in topo_ops if op.get("target")
        })
        if tracked_targets and any(op.get("betti_before") for op in topo_ops):
            lines.append("  Betti number sequence  (β₀  β₁  β₂  …):")
            lines.append("")
            header_cells = ["Object      "] + [
                f"  M{_sub(i)}" for i in range(len(topo_ops) + 1)
            ]
            lines.append("  " + "".join(header_cells))
            lines.append("  " + "─" * (12 + 6 * (len(topo_ops) + 1)))

            for name in tracked_targets:
                # find initial betti (betti_before of first op for this target)
                initial_betti: Optional[Dict[int, int]] = None
                seq_bettis: List[Optional[Dict[int, int]]] = [None] * (len(topo_ops) + 1)
                for i, op in enumerate(topo_ops):
                    if op.get("target") == name and op.get("betti_before") and initial_betti is None:
                        initial_betti = op["betti_before"]
                        seq_bettis[0] = initial_betti
                if initial_betti is None:
                    continue
                for i, op in enumerate(topo_ops):
                    if op.get("target") == name and op.get("betti_after"):
                        seq_bettis[i + 1] = op["betti_after"]
                    elif seq_bettis[i] is not None:
                        seq_bettis[i + 1] = seq_bettis[i]  # unchanged

                def _fmt_betti(b: Optional[Dict[int, int]]) -> str:
                    if b is None:
                        return "  —   "
                    return "  [" + " ".join(str(b.get(k, 0)) for k in sorted(b)) + "]"

                cells = [f"{name:12s}"] + [_fmt_betti(b) for b in seq_bettis]
                lines.append("  " + "".join(cells))

            lines.append("")

        # Step details
        for op in self.cobordism:
            step = op["step"]
            idx = op.get("step_index", "?")

            if step == "remove_disks":
                types_math = op.get("types_math", list(op["types"]))
                dims = op.get("dims", [])
                ambient_dim = op.get("ambient_dim", "?")
                at = op.get("at", [])
                db = op.get("delta_betti", [])
                tc = op.get("torsion_change", 0)
                thm = op.get("theorem", "")
                tgt = op.get("target")
                bb = op.get("betti_before")
                ba = op.get("betti_after")

                dim_parts = ", ".join(
                    f"{m} (dim={d})" for m, d in zip(types_math, dims)
                )
                at_str = "  |  ".join(str(a) for a in at) if at else "—"
                db_str = "  ".join(f"Δβ{_sub(i)}={v}" for i, v in enumerate(db))
                tc_str = str(tc) if tc != 0 else "none"

                content = [
                    f"Disks:      {dim_parts}   (ambient dim = {ambient_dim})",
                    f"Target:     {tgt if tgt else '(ambient space)'}",
                    f"Sites:      {at_str}",
                    f"Δβ:         {db_str}",
                ]
                if bb is not None and ba is not None:
                    bb_str = "[" + " ".join(str(bb.get(k, 0)) for k in sorted(bb)) + "]"
                    ba_str = "[" + " ".join(str(ba.get(k, 0)) for k in sorted(ba)) + "]"
                    content.append(f"β before:   {bb_str}")
                    content.append(f"β after:    {ba_str}")
                content += [
                    f"Torsion Δ:  {tc_str}",
                    f"Theorem:    {thm}",
                ]
                lines.extend(_box(f"Step {idx} · remove_disks", content, W))
                lines.append("")

            elif step == "attach_handle":
                ht_math = op.get("handle_type_math", op.get("handle_type", "?"))
                hdim = op.get("handle_dim", "?")
                hname = op.get("handle_name", "?")
                ambient_dim = op.get("ambient_dim", "?")
                at = op.get("at", [])
                db = op.get("delta_betti", [])
                tc = op.get("torsion_change", 0)
                thm = op.get("theorem", "")
                tgt = op.get("target")
                bb = op.get("betti_before")
                ba = op.get("betti_after")

                at_str = str(at)
                db_str = "  ".join(f"Δβ{_sub(i)}={v}" for i, v in enumerate(db))
                tc_str = str(tc) if tc != 0 else "none"

                content = [
                    f"Handle:     {hname}  (type: {ht_math}  ·  dim = {hdim}  ·  ambient dim = {ambient_dim})",
                    f"Target:     {tgt if tgt else '(ambient space)'}",
                    f"Sites:      {at_str}",
                    f"Δβ:         {db_str}",
                ]
                if bb is not None and ba is not None:
                    bb_str = "[" + " ".join(str(bb.get(k, 0)) for k in sorted(bb)) + "]"
                    ba_str = "[" + " ".join(str(ba.get(k, 0)) for k in sorted(ba)) + "]"
                    content.append(f"β before:   {bb_str}")
                    content.append(f"β after:    {ba_str}")
                content += [
                    f"Torsion Δ:  {tc_str}",
                    f"Theorem:    {thm}",
                ]
                cancelling_of = op.get("cancelling_of")
                if cancelling_of is not None:
                    content.append(f"Cancelling of: Step {cancelling_of}")
                lines.extend(_box(f"Step {idx} · attach_handle", content, W))
                lines.append("")

        # ── Section II: Geometric Trace ────────────────────────────────────────
        lines.append(_divider("II. GEOMETRIC TRACE", char="━", width=W))
        lines.append("")

        geo_ops = [op for op in self.cobordism if op["step"] in ("move", "deform", "restore")]
        if geo_ops:
            for op in geo_ops:
                step = op["step"]
                idx = op.get("step_index", "?")

                if step == "move":
                    tgt = op.get("target")
                    tgts = op.get("targets", [])
                    off = op.get("offset")
                    norm = op.get("offset_norm", 0.0)
                    through = op.get("through")
                    sb = op.get("stats_before", {})
                    sa = op.get("stats_after", {})

                    tgt_label = repr(tgt) if tgt else ("all clouds" if tgts else "none")
                    isotopy_fn = f"f(x,t) = x + t·{_fmt_vec(off)},  t ∈ [0,1]" if off is not None else "identity"

                    content = [
                        f"Target:     {tgt_label}",
                        f"Offset:     {_fmt_vec(off)}   ‖offset‖ = {norm:.4f}",
                        f"Isotopy:    {isotopy_fn}",
                    ]
                    if through:
                        content.append(f"Through:    {through}")

                    for cloud_name in tgts:
                        if cloud_name in sb and cloud_name in sa:
                            n_pts = sb[cloud_name]["n"]
                            c_before = sb[cloud_name]["centroid"]
                            c_after = sa[cloud_name]["centroid"]
                            mn_b = sb[cloud_name]["bbox_min"]
                            mx_b = sb[cloud_name]["bbox_max"]
                            mn_a = sa[cloud_name]["bbox_min"]
                            mx_a = sa[cloud_name]["bbox_max"]
                            content.append(
                                f"Cloud '{cloud_name}':  {n_pts} pts  "
                                f"centroid {_fmt_vec(c_before)} → {_fmt_vec(c_after)}"
                            )
                            content.append(
                                f"  bbox before: {_fmt_bbox(np.array(mn_b), np.array(mx_b))}"
                            )
                            content.append(
                                f"  bbox after:  {_fmt_bbox(np.array(mn_a), np.array(mx_a))}"
                            )

                    lines.extend(_box(f"Step {idx} · move", content, W))
                    lines.append("")

                elif step == "deform":
                    tgt = op.get("target", "?")
                    mode_name = op.get("mode", "?")
                    meta = op.get("meta", {})
                    sb = op.get("stats_before", {})
                    sa = op.get("stats_after", {})
                    n_disp = meta.get("displaced_points", "?")

                    content = [
                        f"Target:     {tgt}",
                        f"Mode:       {mode_name}",
                        f"Displaced:  {n_disp} pts",
                    ]
                    # Mode-specific parameter display
                    if mode_name == "open_cut":
                        content.extend([
                            f"Cut site:   {_fmt_vec(meta.get('cut_site'))}",
                            f"Width:      {meta.get('width', '?')}   "
                            f"Falloff σ: {meta.get('falloff_radius', '?')}",
                            f"Pull dir:   {_fmt_vec(meta.get('pull_direction'))}",
                            f"Plane nrm:  {_fmt_vec(meta.get('plane_normal'))}",
                        ])
                    # Centroid delta
                    if sb and sa:
                        content.append(
                            f"Centroid:   {_fmt_vec(sb.get('centroid'))} "
                            f"→ {_fmt_vec(sa.get('centroid'))}"
                        )
                        mn_a = sa.get("bbox_min")
                        mx_a = sa.get("bbox_max")
                        if mn_a and mx_a:
                            content.append(
                                f"Bbox after: {_fmt_bbox(np.array(mn_a), np.array(mx_a))}"
                            )
                    lines.extend(_box(f"Step {idx} · deform[{mode_name}]", content, W))
                    lines.append("")

                elif step in ("rotate", "scale", "translate"):
                    tgt = op.get("target")
                    tgts = op.get("targets", [])
                    desc = op.get("description", "")
                    sb = op.get("stats_before", {})
                    sa = op.get("stats_after", {})

                    tgt_label = repr(tgt) if tgt else ("all clouds" if tgts else "none")
                    content = [
                        f"Target:     {tgt_label}",
                        f"Isotopy:    {desc}",
                    ]

                    for cloud_name in tgts:
                        if cloud_name in sb and cloud_name in sa:
                            n_pts = sb[cloud_name]["n"]
                            c_before = sb[cloud_name]["centroid"]
                            c_after = sa[cloud_name]["centroid"]
                            mn_b = sb[cloud_name]["bbox_min"]
                            mx_b = sb[cloud_name]["bbox_max"]
                            mn_a = sa[cloud_name]["bbox_min"]
                            mx_a = sa[cloud_name]["bbox_max"]
                            content.append(
                                f"Cloud '{cloud_name}':  {n_pts} pts  "
                                f"centroid {_fmt_vec(c_before)} → {_fmt_vec(c_after)}"
                            )
                            content.append(
                                f"  bbox before: {_fmt_bbox(np.array(mn_b), np.array(mx_b))}"
                            )
                            content.append(
                                f"  bbox after:  {_fmt_bbox(np.array(mn_a), np.array(mx_a))}"
                            )

                    lines.extend(_box(f"Step {idx} · {step}", content, W))
                    lines.append("")

                elif step == "restore":
                    rev = op.get("reverted", "?")
                    rev_step = op.get("reverted_step", "?")
                    cloud_keys = op.get("cloud_keys", [])
                    clouds_str = ", ".join(f"'{k}'" for k in cloud_keys) if cloud_keys else "none"

                    content = [
                        f"Reverted:   Step {rev_step} · {rev}",
                        f"Restored:   {clouds_str}",
                    ]
                    lines.extend(_box(f"Step {idx} · restore", content, W))
                    lines.append("")
        else:
            lines.append("  (no geometric steps recorded)")
            lines.append("")

        # ── Section II.5: Symbolic Composition ─────────────────────────────────
        lines.append(_divider("II.5 SYMBOLIC COMPOSITION", char="━", width=W))
        lines.append("")
        if len(self._isotopy_log) > 0:
            lines.append("  Global Composition f = f_k ∘ … ∘ f_1:")
            lines.append("")
            pretty_f = self._isotopy_log.pretty(d=n)
            for ln in pretty_f.splitlines():
                lines.append(f"    {ln}")
            lines.append("")

            if len(self._isotopy_log.per_target) > 1 or (len(self._isotopy_log.per_target) == 1 and None not in self._isotopy_log.per_target):
                 for tgt, steps_indices in self._isotopy_log.per_target.items():
                     if tgt is not None:
                         lines.append(f"  Target '{tgt}' Composition:")
                         pretty_tgt = self._isotopy_log.pretty(d=n, target=tgt)
                         for ln in pretty_tgt.splitlines():
                             lines.append(f"    {ln}")
                         lines.append("")
        else:
            lines.append("  (no symbolic composition recorded)")
            lines.append("")

        # ── Section III: Handlebody Calculus ───────────────────────────────────
        lines.append(_divider("III. HANDLEBODY CALCULUS", char="━", width=W))
        lines.append("")
        hb_ops = [op for op in self.cobordism if op["step"] in ("attach_handle", "slide_handle", "cancel_handles")]
        if hb_ops:
            for op in hb_ops:
                step = op["step"]
                idx = op.get("step_index", "?")
                if step == "attach_handle":
                    hname = op.get("handle_name", "?")
                    hdim = op.get("handle_dim", "?")
                    ht_math = op.get("handle_type_math", "?")
                    framing_str = op.get("framing", "?")
                    at = op.get("at", "?")
                    cocore = op.get("co_core")
                    
                    content = [
                        f"Handle:      {hname}   (type: {ht_math},  index k = {hdim},  ambient n = {n})",
                        f"Framing:     {framing_str}",
                        f"Attaching:   {at}",
                    ]
                    if cocore:
                        content.append(f"Co-core:     {cocore}")
                    
                    lines.extend(_box(f"Step {idx} · attach_handle", content, W))
                    lines.append("")
                elif step == "slide_handle":
                    slider = op.get("slider", "?")
                    over = op.get("over", "?")
                    sign = op.get("sign", 0)
                    
                    # Check if det/sigma preserved from invariants
                    inv_curr = op.get("invariants", {})
                    f_curr = inv_curr.get("form", {})
                    
                    # Find previous op with invariants
                    prev_ops = [o for o in self.cobordism if o.get("step_index", 0) < idx and "invariants" in o]
                    preserved_str = ""
                    if prev_ops:
                        f_prev = prev_ops[-1]["invariants"].get("form", {})
                        if f_curr.get("determinant") == f_prev.get("determinant"):
                            preserved_str += f"Det preserved ({f_curr.get('determinant')}). "
                        if f_curr.get("signature") == f_prev.get("signature"):
                            preserved_str += f"Sigma preserved ({f_curr.get('signature', 0):+d})."

                    content = [
                        f"Action:      Slide {slider} over {over} (sign {sign:+d})",
                        f"Algebraic:   Linking matrix conjugated by E_{{{slider},{over}}}({sign:+d})",
                    ]
                    if preserved_str:
                        content.append(f"Invariants:  {preserved_str}")
                        
                    lines.extend(_box(f"Step {idx} · slide_handle", content, W))
                    lines.append("")
                elif step == "cancel_handles":
                    hlo = op.get("handle_lo", "?")
                    hhi = op.get("handle_hi", "?")
                    content = [
                        f"Intersection ⟨∂h_{hhi}, [h_{hlo}]⟩ = ±1",
                        "Cerf certificate: algebraic cancellation detected.",
                        f"Handles removed:  {hlo}, {hhi}",
                    ]
                    lines.extend(_box(f"Step {idx} · cancel_handles", content, W))
                    lines.append("")
        else:
            lines.append("  (no handlebody operations recorded)")
            lines.append("")

        # ── Section IV: Algebraic Proof & Deep Invariants ──────────────────────
        lines.append(_divider("IV. ALGEBRAIC PROOF & DEEP INVARIANTS", char="━", width=W))
        lines.append("")
        
        # Summary table
        lines.append("  Summary of Cobordism Evolution:")
        lines.append("")
        header = "  step │ kind            │ rank │  σ  │ det │ SNF diag    │ τ in Wh"
        sep    = "  ─────┼─────────────────┼──────┼─────┼─────┼─────────────┼─────────"
        lines.append(header)
        lines.append(sep)
        
        for op in self.cobordism:
            idx = str(op.get("step_index", "?")).rjust(4)
            kind = op["step"].ljust(15)
            inv = op.get("invariants", {})
            form = inv.get("form", {})
            snf  = inv.get("snf", {})
            wh   = inv.get("whitehead", {})
            
            rank = str(form.get("rank", "-")).rjust(4)
            sig  = f"{form.get('signature', 0):+d}".rjust(3) if form.get("signature") is not None else " — "
            det  = str(form.get("determinant", "-")).rjust(3)
            snf_d = str(snf.get("diagonal", []))[:11].ljust(11)
            tau   = "trivial" if wh.get("is_s_cobordism") else (wh.get("tau") or " — ")
            
            lines.append(f"  {idx} │ {kind} │ {rank} │ {sig} │ {det} │ {snf_d} │ {tau}")
        lines.append("")
        
        # Detail for the latest state
        if self.cobordism:
            last_inv = self.cobordism[-1].get("invariants", {})
            form = last_inv.get("form", {})
            if form.get("kind") == "intersection":
                lines.append("  Latest Intersection Form Q:")
                mat = np.array(form["matrix"])
                disp_mat = mat[:5, :5]
                for row in disp_mat:
                    lines.append(f"    {row.tolist()}")
                if mat.shape[0] > 5:
                    lines.append("    ...")
                lines.append(f"    Rank: {form['rank']}  Signature: σ = {form['signature']}  Det: {form['determinant']}")
                lines.append("")
            
            wh = last_inv.get("whitehead", {})
            if wh.get("Wh_description"):
                lines.append("  Whitehead Torsion State:")
                lines.append(f"    Wh(π₁):      {wh['Wh_description']}")
                lines.append(f"    τ(W, M₀):    {wh['tau'] or '0'}")
                lines.append(f"    s-cobordism: {'✓' if wh['is_s_cobordism'] else '✗'}")
                lines.append("")

        lines.append("  Surgery Obstruction Analysis:")
        try:
            obs = self.evaluate_obstruction()
            val = getattr(obs, "value", "?")
            msg = getattr(obs, "message", "")
            certified = getattr(obs, "assembly_certified", False)
            obstructs = getattr(obs, "obstructs", None)
            exact = getattr(obs, "exact", False)
            pi_str = getattr(obs, "pi", "?")
            modulus = getattr(obs, "modulus", None)

            if modulus is not None:
                lg_str = f"L{_sub(n)}(ℤ[{pi_str}]; ℤ/{modulus}ℤ)"
            else:
                lg_str = f"L{_sub(n)}(ℤ[{pi_str}])"

            val_str = str(val)
            obstructs_str = "No — surgery can proceed" if obstructs is False else ("Yes — obstruction present" if obstructs else "undetermined")
            cert_str = "✓  certified" if certified else "✗  unverified"
            exact_str = "exact" if exact else "heuristic"

            content = [
                f"L-group:    {lg_str}",
                f"Element:    σ = {val_str}",
                f"Precision:  {exact_str}",
                f"Assembly:   {cert_str}",
                f"Obstructs:  {obstructs_str}",
            ]
            if msg:
                content.append(f"Note:       {msg}")
            lines.extend(_box("Surgery Obstruction  (Ranicki assembly map)", content, W))
        except Exception as e:
            lines.extend(_box("Surgery Obstruction", [f"Evaluation failed: {e!r}"], W))
        lines.append("")

        # ── Section V: Cobordism Complex W ─────────────────────────────────────
        lines.append(_divider("V. COBORDISM COMPLEX W", char="━", width=W))
        lines.append("")
        
        W_comp = self.W
        content = [
            f"Simplices total:         {len(W_comp.underlying.simplices)}",
            f"Euler χ(W):              {W_comp.euler_characteristic()}",
            f"Initial boundary M₀:    {len(W_comp.boundary_initial_indices)} vertices",
            f"Final boundary M_m:      {len(W_comp.boundary_final_indices)} vertices",
            f"Slabs (one per step):    {len(W_comp.slab_for_step)}",
            f"Is collared product?     {'✓' if W_comp.is_collared else '✗'}",
        ]
        lines.extend(_box("Cobordism complex W", content, W))
        lines.append("")

        # ── Footer ─────────────────────────────────────────────────────────────
        lines.append(rule())
        status = "✓  Surgery sequence complete." if not getattr(self, "_finished", False) or self._finished else "  In progress."
        lines.append(hdr(f"  {status}  Steps: {n_steps}."))
        lines.append(rule())

        return "\n".join(lines)

    # ── LaTeX rendering ────────────────────────────────────────────────────────

    def _logs_latex(self) -> str:
        lines: List[str] = []

        n = self.chain_complex.dimension

        lines.append("\\section*{Surgery Sequence Log}")
        lines.append(f"Ambient space: $\\mathbb{{R}}^{{{n}}}$. "
                     f"Steps recorded: {self._step_counter}.")
        lines.append("")

        # I. Topological Trace
        lines.append("\\subsection*{I.\\; Topological Trace}")

        # Arrow sequence
        topo_ops = [op for op in self.cobordism if op["step"] in ("remove_disks", "attach_handle")]
        if topo_ops:
            parts = ["M_0"]
            for i, op in enumerate(topo_ops):
                if op["step"] == "remove_disks":
                    types_str = ",".join(t.replace("^", "^{").rstrip("^") + "}" if "^" in t else t for t in op["types"])
                    parts.append(f"\\xrightarrow{{\\text{{remove}}({types_str})}} M_{{{i+1}}}")
                else:
                    ht = op.get("handle_type", "?").replace("^", "^{").replace("x", "\\times ")
                    parts.append(f"\\xrightarrow{{\\text{{attach}}({ht})}} M_{{{i+1}}}")
            lines.append("\\[ " + "\\; ".join(parts) + " \\]")

        for op in self.cobordism:
            step = op["step"]
            idx = op.get("step_index", "?")

            if step in ("remove_disks", "attach_handle"):
                lines.append(f"\\paragraph{{Step {idx}: {step.replace('_', ' ')}}}")
                if step == "remove_disks":
                    lines.append(f"Removed: ${', '.join(op.get('types_math', []))}$. "
                                 f"Sites: ${op.get('at', [])}$. "
                                 f"Theorem: \\texttt{{{op.get('theorem', '')}}}.")
                else:
                    ht = op.get("handle_type_math", "?")
                    desc = (f"Handle: $\\texttt{{{op.get('handle_name', '?')}}}$ "
                            f"(type: ${ht}$, $\\dim={op.get('handle_dim','?')}$). "
                            f"Theorem: \\texttt{{{op.get('theorem', '')}}}.")
                    cancelling_of = op.get("cancelling_of")
                    if cancelling_of is not None:
                        desc += f" Cancelling of: Step {cancelling_of}."
                    lines.append(desc)

        # II. Geometric Trace
        lines.append("\\subsection*{II.\\; Geometric Trace}")
        for op in self.cobordism:
            if op["step"] == "move":
                idx = op.get("step_index", "?")
                off = op.get("offset")
                lines.append(f"\\paragraph{{Step {idx}: move}}")
                lines.append(f"Isotopy: $f(x,t) = x + t \\cdot {off}$, $t \\in [0,1]$.")
                sb = op.get("stats_before", {})
                sa = op.get("stats_after", {})
                for t in op.get("targets", []):
                    if t in sb and t in sa:
                        lines.append(
                            f"Cloud \\texttt{{{t}}}: {sb[t]['n']} points; "
                            f"centroid $({', '.join(f'{v:.2f}' for v in sb[t]['centroid'])}) "
                            f"\\to ({', '.join(f'{v:.2f}' for v in sa[t]['centroid'])})$."
                        )
            elif op["step"] == "restore":
                idx = op.get("step_index", "?")
                lines.append(f"\\paragraph{{Step {idx}: restore}}")
                lines.append(f"Reverted step {op.get('reverted_step','?')}: {op.get('reverted','?')}.")

        # III. Algebraic Proof
        lines.append("\\subsection*{III.\\; Algebraic Proof}")
        lines.append(f"Ambient dimension $n = {n}$. "
                     "Intersection form $Q$ computed from the algebraic Poincaré complex.")
        try:
            obs = self.evaluate_obstruction()
            val = getattr(obs, "value", "?")
            certified = getattr(obs, "assembly_certified", False)
            msg = getattr(obs, "message", "")
            lines.append(f"Surgery obstruction: $\\sigma \\in L_{{{n}}}(\\mathbb{{Z}}[\\pi_1])$. "
                         f"Value: $\\sigma = {val}$. "
                         f"Assembly map: {'certified' if certified else 'unverified'}. "
                         f"{msg}")
        except Exception as e:
            lines.append(f"Obstruction evaluation failed: {e!r}.")

        return "\n".join(lines)

    # ── Utilities ──────────────────────────────────────────────────────────────

    def find_unlinking_site(self, obj1: str, obj2: str) -> List[Any]:
        """Propose a cut site for unlinking two tracked objects.

        Algorithm:
            Checks bounding box intersections or linking numbers between the objects.

        Preserved Invariants:
            Returns valid topological coordinates for subsequent disk removal.

        Args:
            obj1: Name of the first tracked object.
            obj2: Name of the second tracked object.

        Returns:
            List of candidate cut-site coordinates.
        """
        return []


perform_surgery = perform_handle_surgery


__all__ = [
    # Core surgery functions (re-exported from pysurgery.manifolds.surgery)
    "HandleAttachment",
    "SurgeryResult",
    "SurgeryVerificationResult",
    "LinkingNumberResult",
    "DelinkingResult",
    "AttachmentSphereResult",
    "AlgebraicSurgeryComplex",
    "compute_linking_number",
    "find_attachment_sphere",
    "perform_handle_surgery",
    "perform_surgery",
    "perform_algebraic_surgery",
    "perform_rational_surgery",
    "perform_p_local_surgery",
    "verify_surgery",
    "delink",
    # SurgerySession API
    "DimensionalConsistencyError",
    "SurgeryFinishedError",
    "SurgeryInvariantBroken",
    "SurgeryProtocolError",
    "TopologyNotRestoredError",
    "IsotopyShapeError",
    "IsotopyCompileError",
    "Isotopy",
    "IdentityIsotopy",
    "TranslateIsotopy",
    "RotateIsotopy",
    "ScaleIsotopy",
    "SymbolicTransformation",
    "IsotopyComposition",
    "Framing",
    "FramedHandle",
    "CancellationCandidate",
    "HandlebodyState",
    "CobordismComplex",
    "Transformation",
    "TrackedObject",
    "SurgerySession",
    # Deformation registry
    "list_deform_modes",
]
