from typing import List, Optional, Union
from math import comb
import warnings
from collections import Counter
from pydantic import BaseModel, Field
from .core.intersection_forms import IntersectionForm
from .core.quadratic_forms import QuadraticForm
from .core.fundamental_group import GroupPresentation
from .core.exceptions import SurgeryObstructionError
from .bridge.julia_bridge import julia_engine


class ObstructionResult(BaseModel):
    """Typed result for Wall obstruction evaluations."""

    dimension: int
    pi: str
    computable: bool
    exact: bool
    value: Optional[int] = None
    modulus: Optional[int] = None
    message: str = ""
    assumptions: List[str] = Field(default_factory=list)
    factor_analysis: List[str] = Field(default_factory=list)
    summands: List[dict] = Field(default_factory=list)
    decomposition_kind: str = "scalar"
    assembly_certified: bool = False
    obstructs: Optional[bool] = None
    zero_certified: bool = False

    def model_post_init(self, __context) -> None:
        if (
            not self.assembly_certified
            and self.computable
            and self.exact
            and self.value is not None
            and self.decomposition_kind in {"scalar", "single_factor"}
        ):
            self.assembly_certified = True
        # If a caller supplied explicit state, preserve it.
        if self.obstructs is not None:
            self.zero_certified = bool(self.zero_certified)
            return
        if not self.computable or not self.exact:
            self.obstructs = None
            self.zero_certified = False
            return
        if self.value is None:
            self.obstructs = None
            self.zero_certified = False
            return

        value = int(self.value)
        if self.modulus is None:
            self.zero_certified = value == 0
        else:
            mod = abs(int(self.modulus))
            self.zero_certified = mod != 0 and (value % mod) == 0
        self.obstructs = not self.zero_certified

    def legacy_output(self) -> Union[int, str]:
        if self.computable and self.value is not None:
            return int(self.value)
        return self.message

    def to_direct_sum_element(self) -> "LDirectSumElement":
        return LDirectSumElement.from_obstruction(self)

    @classmethod
    def from_direct_sum_element(
        cls,
        element: "LDirectSumElement",
        *,
        collapse_integral: bool = False,
    ) -> "ObstructionResult":
        return element.to_obstruction_result(collapse_integral=collapse_integral)


class LDirectSummand(BaseModel):
    """Typed representation of one direct-sum component in a Wall obstruction element."""

    shift: int = 0
    multiplicity: int = 1
    dimension: int
    pi: str
    symbol: str
    computable: bool
    exact: bool
    value: Optional[int] = None
    modulus: Optional[int] = None
    obstructs: Optional[bool] = None
    zero_certified: bool = False
    message: str = ""

    def group_key(self) -> tuple[int, int, str, str, Optional[int]]:
        return (self.shift, self.dimension, self.pi, self.symbol, self.modulus)


class LDirectSumElement(BaseModel):
    """Formal direct-sum element in decomposed L-group coordinates."""

    dimension: int
    pi: str
    summands: List[LDirectSummand] = Field(default_factory=list)
    computable: bool = False
    exact: bool = False

    def normalized(self) -> "LDirectSumElement":
        if not self.computable or not self.exact:
            return self
        contrib = self._normalized_contributions()
        out: List[LDirectSummand] = []
        for key in sorted(contrib.keys()):
            shift, dim, pi, symbol, modulus = key
            v = int(contrib[key])
            if modulus is not None:
                mod = abs(int(modulus))
                if mod != 0:
                    v %= mod
            zero_certified = (
                (v == 0) if modulus is None else (int(v) % int(modulus) == 0)
            )
            out.append(
                LDirectSummand(
                    shift=shift,
                    multiplicity=1,
                    dimension=dim,
                    pi=pi,
                    symbol=symbol,
                    computable=True,
                    exact=True,
                    value=v,
                    modulus=modulus,
                    obstructs=not zero_certified,
                    zero_certified=zero_certified,
                    message="",
                )
            )
        return LDirectSumElement(
            dimension=self.dimension,
            pi=self.pi,
            summands=out,
            computable=True,
            exact=True,
        )

    @classmethod
    def from_obstruction(cls, obstruction: ObstructionResult) -> "LDirectSumElement":
        typed_summands: List[LDirectSummand] = []
        if obstruction.summands:
            for s in obstruction.summands:
                typed_summands.append(
                    LDirectSummand(
                        shift=int(s.get("shift", 0)),
                        multiplicity=int(s.get("multiplicity", 1)),
                        dimension=int(s.get("dimension", obstruction.dimension)),
                        pi=str(s.get("pi", obstruction.pi)),
                        symbol=str(
                            s.get(
                                "symbol",
                                l_group_symbol(
                                    int(s.get("dimension", obstruction.dimension)),
                                    str(s.get("pi", obstruction.pi)),
                                ),
                            )
                        ),
                        computable=bool(s.get("computable", obstruction.computable)),
                        exact=bool(s.get("exact", obstruction.exact)),
                        value=None if s.get("value") is None else int(s.get("value")),
                        modulus=None
                        if s.get("modulus") is None
                        else int(s.get("modulus")),
                        obstructs=s.get("obstructs"),
                        zero_certified=bool(s.get("zero_certified", False)),
                        message=str(s.get("message", "")),
                    )
                )
        else:
            typed_summands.append(
                LDirectSummand(
                    shift=0,
                    multiplicity=1,
                    dimension=obstruction.dimension,
                    pi=obstruction.pi,
                    symbol=l_group_symbol(obstruction.dimension, obstruction.pi),
                    computable=obstruction.computable,
                    exact=obstruction.exact,
                    value=obstruction.value,
                    modulus=obstruction.modulus,
                    obstructs=obstruction.obstructs,
                    zero_certified=obstruction.zero_certified,
                    message=obstruction.message,
                )
            )
        return cls(
            dimension=obstruction.dimension,
            pi=obstruction.pi,
            summands=typed_summands,
            computable=obstruction.computable,
            exact=obstruction.exact,
        )

    def _normalized_contributions(
        self,
    ) -> dict[tuple[int, int, str, str, Optional[int]], int]:
        contrib: dict[tuple[int, int, str, str, Optional[int]], int] = {}
        for s in self.summands:
            if not s.computable or not s.exact or s.value is None:
                raise ValueError(
                    "Direct-sum arithmetic requires computable exact summands with explicit values."
                )
            key = s.group_key()
            term = int(s.multiplicity) * int(s.value)
            if s.modulus is not None:
                mod = abs(int(s.modulus))
                if mod != 0:
                    term %= mod
            contrib[key] = contrib.get(key, 0) + term
            if s.modulus is not None:
                mod = abs(int(s.modulus))
                if mod != 0:
                    contrib[key] %= mod
        return contrib

    def _combine(self, other: "LDirectSumElement", sign: int) -> "LDirectSumElement":
        if self.dimension != other.dimension or self.pi != other.pi:
            raise ValueError(
                "Can only combine direct-sum elements with matching (dimension, pi)."
            )
        c1 = self._normalized_contributions()
        c2 = other._normalized_contributions()
        keys = set(c1.keys()) | set(c2.keys())
        out: List[LDirectSummand] = []
        for key in sorted(keys):
            shift, dim, pi, symbol, modulus = key
            v = c1.get(key, 0) + sign * c2.get(key, 0)
            if modulus is not None:
                mod = abs(int(modulus))
                if mod != 0:
                    v %= mod
            zero_certified = (
                (v == 0) if modulus is None else (int(v) % int(modulus) == 0)
            )
            out.append(
                LDirectSummand(
                    shift=shift,
                    multiplicity=1,
                    dimension=dim,
                    pi=pi,
                    symbol=symbol,
                    computable=True,
                    exact=True,
                    value=int(v),
                    modulus=modulus,
                    obstructs=not zero_certified,
                    zero_certified=zero_certified,
                    message="",
                )
            )
        return LDirectSumElement(
            dimension=self.dimension,
            pi=self.pi,
            summands=out,
            computable=True,
            exact=True,
        )

    def __add__(self, other: "LDirectSumElement") -> "LDirectSumElement":
        return self._combine(other, sign=1)

    def __sub__(self, other: "LDirectSumElement") -> "LDirectSumElement":
        return self._combine(other, sign=-1)

    def __neg__(self) -> "LDirectSumElement":
        return (-1) * self

    def __mul__(self, scalar: int) -> "LDirectSumElement":
        if not isinstance(scalar, int):
            raise ValueError(
                "Direct-sum scalar multiplication requires an integer scalar."
            )
        if not self.computable or not self.exact:
            raise ValueError(
                "Direct-sum scalar multiplication requires computable exact summands."
            )
        out: List[LDirectSummand] = []
        for s in self.summands:
            if s.value is None:
                raise ValueError(
                    "Direct-sum scalar multiplication requires explicit summand values."
                )
            v = int(s.value) * scalar
            if s.modulus is not None:
                mod = abs(int(s.modulus))
                if mod != 0:
                    v %= mod
            zero_certified = (
                (v == 0) if s.modulus is None else (int(v) % int(s.modulus) == 0)
            )
            out.append(
                LDirectSummand(
                    shift=s.shift,
                    multiplicity=s.multiplicity,
                    dimension=s.dimension,
                    pi=s.pi,
                    symbol=s.symbol,
                    computable=True,
                    exact=True,
                    value=v,
                    modulus=s.modulus,
                    obstructs=not zero_certified,
                    zero_certified=zero_certified,
                    message=s.message,
                )
            )
        return LDirectSumElement(
            dimension=self.dimension,
            pi=self.pi,
            summands=out,
            computable=True,
            exact=True,
        ).normalized()

    def __rmul__(self, scalar: int) -> "LDirectSumElement":
        return self.__mul__(scalar)

    def equivalent(self, other: "LDirectSumElement") -> bool:
        if self.dimension != other.dimension or self.pi != other.pi:
            return False
        if not (self.computable and self.exact and other.computable and other.exact):
            return False
        return (
            self.normalized()._normalized_contributions()
            == other.normalized()._normalized_contributions()
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, LDirectSumElement):
            return False
        return self.equivalent(other)

    def to_obstruction_result(
        self, *, collapse_integral: bool = False
    ) -> ObstructionResult:
        ns = self.normalized() if self.computable and self.exact else self
        summands = []
        for s in ns.summands:
            summands.append(
                {
                    "shift": s.shift,
                    "multiplicity": s.multiplicity,
                    "dimension": s.dimension,
                    "pi": s.pi,
                    "symbol": s.symbol,
                    "computable": s.computable,
                    "exact": s.exact,
                    "value": s.value,
                    "modulus": s.modulus,
                    "obstructs": s.obstructs,
                    "zero_certified": s.zero_certified,
                    "message": s.message,
                }
            )

        scalar_value: Optional[int] = None
        scalar_modulus: Optional[int] = None
        if (
            collapse_integral
            and ns.computable
            and ns.exact
            and all(s.value is not None and s.modulus is None for s in ns.summands)
        ):
            scalar_value = int(sum(int(s.value) for s in ns.summands))

        known_obstructs = any(s.obstructs is True for s in ns.summands)
        all_zero_certified = bool(ns.summands) and all(
            s.zero_certified for s in ns.summands
        )
        return ObstructionResult(
            dimension=ns.dimension,
            pi=ns.pi,
            computable=ns.computable,
            exact=ns.exact,
            value=scalar_value,
            modulus=scalar_modulus,
            message="Direct-sum obstruction element",
            factor_analysis=["typed_direct_sum_conversion"],
            summands=summands,
            obstructs=True
            if known_obstructs
            else (False if all_zero_certified else None),
            zero_certified=all_zero_certified,
        )


class WallGroupL(BaseModel):
    """
    Interface for computing Wall's surgery obstruction groups L_n(pi).
    Extends beyond the simply-connected case into finite groups and Z.
    """

    dimension: int
    pi: Union[str, GroupPresentation] = "1"

    @staticmethod
    def _normalize_pi(pi: Union[str, GroupPresentation]) -> str:
        if isinstance(pi, GroupPresentation):
            return pi.normalized()
        return str(pi)

    @staticmethod
    def _signature_over_8(form: IntersectionForm, context: str) -> int:
        sig = form.signature()
        if sig % 8 != 0:
            raise SurgeryObstructionError(
                f"{context}: signature {sig} is not divisible by 8, so the obstruction is not integral in this model."
            )
        return sig // 8

    def compute_obstruction(
        self, form: Optional[IntersectionForm] = None
    ) -> Union[int, str]:
        """
        Backward-compatible output (int or diagnostic string).
        """
        return self.compute_obstruction_result(form).legacy_output()

    def compute_obstruction_element(
        self, form: Optional[IntersectionForm] = None
    ) -> LDirectSumElement:
        return self.compute_obstruction_result(form).to_direct_sum_element()

    def _compute_for_single_factor(
        self, n: int, pi: str, form: Optional[IntersectionForm]
    ) -> ObstructionResult:
        assumptions: List[str] = []
        if pi == "1":
            assumptions.append("Classical simply-connected Wall L-group model")
            if n % 4 == 0:
                if form is None:
                    raise SurgeryObstructionError(
                        "Intersection form required to compute Wall group L_{4k}(1) obstruction. "
                        "Hint: Provide an IntersectionForm instance derived from the manifold's H_{2k} basis."
                    )
                return ObstructionResult(
                    dimension=n,
                    pi=pi,
                    computable=True,
                    exact=True,
                    value=self._signature_over_8(form, "L_{4k}(1)"),
                    modulus=None,
                    decomposition_kind="single_factor",
                    assembly_certified=True,
                    assumptions=assumptions,
                )
            if n % 4 == 2:
                if form is None or not isinstance(form, QuadraticForm):
                    raise SurgeryObstructionError(
                        "L_{4k+2}(1) obstruction requires the Arf Invariant. "
                        "Hint: Symmetric bilinear forms are insufficient. Provide a QuadraticForm with explicit Z_2 quadratic refinement q: H -> Z_2."
                    )
                return ObstructionResult(
                    dimension=n,
                    pi=pi,
                    computable=True,
                    exact=True,
                    value=form.arf_invariant(),
                    modulus=2,
                    decomposition_kind="single_factor",
                    assembly_certified=True,
                    assumptions=assumptions,
                )
            return ObstructionResult(
                dimension=n,
                pi=pi,
                computable=True,
                exact=True,
                value=0,
                modulus=None,
                decomposition_kind="single_factor",
                assembly_certified=True,
                assumptions=assumptions,
            )

        if pi == "Z":
            assumptions.append("Shaneson splitting model for L_n(Z)")
            if n % 4 == 0:
                return ObstructionResult(
                    dimension=n,
                    pi=pi,
                    computable=True,
                    exact=True,
                    value=self._signature_over_8(form, "L_{4k}(Z)") if form else 0,
                    decomposition_kind="single_factor",
                    assembly_certified=True,
                    assumptions=assumptions,
                )
            if n % 4 == 1:
                return ObstructionResult(
                    dimension=n,
                    pi=pi,
                    computable=True,
                    exact=True,
                    value=self._signature_over_8(form, "L_{4k+1}(Z)") if form else 0,
                    decomposition_kind="single_factor",
                    assembly_certified=True,
                    assumptions=assumptions,
                )
            if n % 4 == 2:
                if form and isinstance(form, QuadraticForm):
                    return ObstructionResult(
                        dimension=n,
                        pi=pi,
                        computable=True,
                        exact=True,
                        value=form.arf_invariant(),
                        modulus=2,
                        decomposition_kind="single_factor",
                        assembly_certified=True,
                        assumptions=assumptions,
                    )
                return ObstructionResult(
                    dimension=n,
                    pi=pi,
                    computable=False,
                    exact=False,
                    message="Requires Arf invariant form input",
                    assumptions=assumptions,
                )
            if n % 4 == 3:
                if form and isinstance(form, QuadraticForm):
                    return ObstructionResult(
                        dimension=n,
                        pi=pi,
                        computable=True,
                        exact=True,
                        value=form.arf_invariant(),
                        modulus=2,
                        decomposition_kind="single_factor",
                        assembly_certified=True,
                        assumptions=assumptions,
                    )
                return ObstructionResult(
                    dimension=n,
                    pi=pi,
                    computable=False,
                    exact=False,
                    message="Requires Arf invariant form input",
                    assumptions=assumptions,
                )

        if pi.startswith("Z_"):
            p = int(pi.split("_")[1])
            assumptions.append(
                "Finite cyclic group ring obstruction via multisignature"
            )
            if n % 4 != 0:
                return ObstructionResult(
                    dimension=n,
                    pi=pi,
                    computable=False,
                    exact=False,
                    message=f"L_{n}(Z_{p}) needs representation-theoretic data beyond current API",
                    assumptions=assumptions,
                )
            if form is None:
                raise SurgeryObstructionError(
                    f"Intersection form required to compute Wall group L_{{{n}}}(Z_{p}) multisignature obstruction."
                )
            if julia_engine.available:
                return ObstructionResult(
                    dimension=n,
                    pi=pi,
                    computable=True,
                    exact=True,
                    value=int(julia_engine.compute_multisignature(form.matrix, p)),
                    decomposition_kind="single_factor",
                    assembly_certified=True,
                    assumptions=assumptions,
                )
            return ObstructionResult(
                dimension=n,
                pi=pi,
                computable=False,
                exact=False,
                message=(
                    "JuliaBridge backend unavailable for exact multisignature in "
                    "WallGroupL.compute_obstruction_result; install/enable Julia for this calculation."
                ),
                assumptions=assumptions,
            )

        return ObstructionResult(
            dimension=n,
            pi=pi,
            computable=False,
            exact=False,
            message=f"Unsupported group descriptor '{pi}' for current Wall obstruction API",
            assumptions=["Group-ring representation data not fully specified"],
        )

    def compute_obstruction_result(
        self, form: Optional[IntersectionForm] = None
    ) -> ObstructionResult:
        """Compute surgery obstruction as a typed result with exactness metadata."""
        n = self.dimension
        pi = self._normalize_pi(self.pi)
        if "x" not in pi.lower():
            return self._compute_for_single_factor(n, pi, form)

        factors = [f.strip() for f in pi.split("x") if f.strip()]
        nontrivial = [f for f in factors if f != "1"]
        if len(nontrivial) == 0:
            return self._compute_for_single_factor(n, "1", form)
        if len(nontrivial) == 1:
            return self._compute_for_single_factor(n, nontrivial[0], form)

        # Generalized Shaneson splitting for products with multiple Z factors:
        # L_n(pi x Z^k) ≅ ⊕_{j=0}^k binom(k,j) L_{n-j}(pi).
        z_count = sum(1 for f in nontrivial if f == "Z")
        if z_count > 0:
            rest = [f for f in nontrivial if f != "Z"]
            rest_pi = " x ".join(rest) if rest else "1"
            summands: List[dict] = []
            component_results: List[ObstructionResult] = []
            try:
                for j in range(z_count + 1):
                    multiplicity = comb(z_count, j)
                    base_res = WallGroupL(
                        dimension=n - j, pi=rest_pi
                    ).compute_obstruction_result(form)
                    component_results.append(base_res)
                    summands.append(
                        {
                            "shift": j,
                            "multiplicity": multiplicity,
                            "dimension": n - j,
                            "pi": rest_pi,
                            "symbol": l_group_symbol(n - j, rest_pi),
                            "computable": base_res.computable,
                            "exact": base_res.exact,
                            "value": base_res.value,
                            "modulus": base_res.modulus,
                            "obstructs": base_res.obstructs,
                            "zero_certified": base_res.zero_certified,
                            "message": base_res.message,
                        }
                    )
            except SurgeryObstructionError as e:
                return ObstructionResult(
                    dimension=n,
                    pi=pi,
                    computable=False,
                    exact=False,
                    message=(
                        f"Shaneson splitting for '{pi}' reduced to factor '{rest_pi}', but required form data is missing: {e}"
                    ),
                    assumptions=[
                        "Shaneson splitting applied",
                        "Insufficient form input for reduced factors",
                    ],
                    summands=summands,
                    decomposition_kind="shaneson",
                    assembly_certified=False,
                )

            all_computable = all(r.computable for r in component_results)
            all_exact = all(r.exact for r in component_results)
            all_integral = all(
                (r.value is not None and r.modulus is None) for r in component_results
            )
            if all_computable and all_integral:
                total = 0
                for j, r in enumerate(component_results):
                    if r.value is None:
                        continue
                    total += comb(z_count, j) * int(r.value)
                return ObstructionResult(
                    dimension=n,
                    pi=pi,
                    computable=True,
                    exact=all_exact,
                    value=total,
                    assumptions=["Generalized Shaneson splitting over Z-factors"],
                    factor_analysis=[
                        f"factorized={factors}",
                        f"reduced_rest={rest_pi}",
                        f"z_factor_count={z_count}",
                        "All summands are integral and were aggregated",
                    ],
                    summands=summands,
                    decomposition_kind="shaneson_integral_sum",
                    assembly_certified=True,
                )

            if all_computable:
                known_obstructs = any(r.obstructs is True for r in component_results)
                all_zero_certified = all(r.zero_certified for r in component_results)
                if all_zero_certified:
                    decomposition_message = f"Shaneson decomposition for '{pi}' is computable as a direct-sum element and all summands are certified zero."
                elif known_obstructs:
                    decomposition_message = f"Shaneson decomposition for '{pi}' is computable as a direct-sum element and contains a certified non-zero summand."
                else:
                    decomposition_message = (
                        f"Shaneson decomposition for '{pi}' is computable as a direct-sum element, "
                        "but cannot be collapsed to a single integer due to modular/mixed summands."
                    )
                return ObstructionResult(
                    dimension=n,
                    pi=pi,
                    computable=True,
                    exact=all_exact,
                    value=None,
                    message=decomposition_message,
                    assumptions=["Generalized Shaneson splitting over Z-factors"],
                    factor_analysis=[
                        f"factorized={factors}",
                        f"reduced_rest={rest_pi}",
                        f"z_factor_count={z_count}",
                        "Use `summands` for the full obstruction element",
                    ],
                    summands=summands,
                    decomposition_kind="shaneson_direct_sum",
                    assembly_certified=True,
                    obstructs=True
                    if known_obstructs
                    else (False if all_zero_certified else None),
                    zero_certified=all_zero_certified,
                )

            return ObstructionResult(
                dimension=n,
                pi=pi,
                computable=False,
                exact=False,
                message=(
                    f"Partial product decomposition for '{pi}' succeeded, but one or more Shaneson summands "
                    "is not computable with current inputs/backends."
                ),
                assumptions=[
                    "Generalized Shaneson splitting applied",
                    "At least one summand uncomputable",
                ],
                factor_analysis=[
                    f"factorized={factors}",
                    f"reduced_rest={rest_pi}",
                    f"z_factor_count={z_count}",
                ],
                summands=summands,
                decomposition_kind="shaneson_partial",
                assembly_certified=False,
            )

        counts = Counter(nontrivial)
        reduced_factors = sorted(counts.items(), key=lambda item: item[0])
        component_results: List[ObstructionResult] = []
        summands: List[dict] = []
        for factor, multiplicity in reduced_factors:
            try:
                base = WallGroupL(dimension=n, pi=factor).compute_obstruction_result(
                    form
                )
            except Exception as exc:
                base = ObstructionResult(
                    dimension=n,
                    pi=factor,
                    computable=False,
                    exact=False,
                    value=None,
                    message=f"Factor solver failed: {exc}",
                    assumptions=[
                        "Single-factor obstruction unavailable with current inputs"
                    ],
                )
            component_results.append(base)
            summands.append(
                {
                    "shift": 0,
                    "multiplicity": int(multiplicity),
                    "dimension": n,
                    "pi": factor,
                    "symbol": l_group_symbol(n, factor),
                    "computable": base.computable,
                    "exact": base.exact,
                    "value": base.value,
                    "modulus": base.modulus,
                    "obstructs": base.obstructs,
                    "zero_certified": base.zero_certified,
                    "message": base.message,
                }
            )

        all_computable = all(r.computable for r in component_results)
        all_exact = all(r.exact for r in component_results)
        any_obstructs = any(r.obstructs is True for r in component_results)
        all_zero = bool(component_results) and all(
            r.zero_certified for r in component_results
        )

        warnings.warn(
            "Wall obstruction fallback in `WallGroupL.compute_obstruction_result`: non-Z multi-factor product "
            "returned as a factor-wise surrogate direct-sum certificate; full assembly-map evaluation is not implemented."
        )
        return ObstructionResult(
            dimension=n,
            pi=pi,
            computable=all_computable,
            exact=False,
            value=None,
            message=(
                f"Product group '{pi}' uses a phase-7 factor-wise surrogate decomposition. "
                "This is structurally informative but not a full assembly-map obstruction."
            ),
            assumptions=[
                "No complete product-group assembly map implemented yet",
                "Factor-wise surrogate decomposition over non-Z factors",
            ],
            factor_analysis=[
                f"factors={factors}",
                f"nontrivial={nontrivial}",
                f"reduced_counts={dict(reduced_factors)}",
            ],
            summands=summands,
            decomposition_kind="factor_surrogate",
            assembly_certified=False,
            obstructs=True
            if any_obstructs
            else (False if all_zero and all_computable else None),
            zero_certified=bool(all_zero and all_computable),
        )


def l_group_symbol(n: int, pi: Union[str, GroupPresentation] = "1") -> str:
    """
    Returns the mathematical symbol/structure of L_n(pi).
    """
    if isinstance(pi, GroupPresentation):
        pi = pi.normalized()

    if pi == "1":
        if n % 4 == 0:
            return "Z"
        if n % 4 == 2:
            return "Z_2"
        return "0"
    elif pi == "Z":
        # By Shaneson splitting: L_n(Z) = L_n(1) + L_{n-1}(1)
        if n % 4 == 0:
            return "Z"
        if n % 4 == 1:
            return "Z"
        if n % 4 == 2:
            return "Z_2"
        if n % 4 == 3:
            return "Z_2"
    pi_s = str(pi)
    if "x" in pi_s.lower():
        factors = [f.strip() for f in pi_s.split("x") if f.strip()]
        nontrivial = [f for f in factors if f != "1"]
        z_count = sum(1 for f in nontrivial if f == "Z")
        if z_count > 0 and all(f == "Z" for f in nontrivial):
            rest = [f for f in nontrivial if f != "Z"]
            rest_pi = " x ".join(rest) if rest else "1"
            terms: List[str] = []
            for j in range(z_count + 1):
                mult = comb(z_count, j)
                base = l_group_symbol(n - j, rest_pi)
                if mult == 1:
                    terms.append(base)
                else:
                    terms.append(f"{mult}*({base})")
            return " + ".join(terms)
        return f"L_{n}({pi}) (product-group decomposition required)"
    return f"L_{n}({pi})"
