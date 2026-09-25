r"""Fundamental cycles of closed pseudomanifolds, by coherent orientation, exactly.

Overview:
    If K is a closed, orientable, strongly connected p-pseudomanifold -- pure of
    dimension p, every (p-1)-simplex in exactly two p-simplices, the dual graph
    connected -- then ``H_p(K; Z) = Z`` and its generator is the sum of all p-simplices
    with coherently chosen signs. Coherent means: whenever p-simplices sigma, tau share
    a (p-1)-face f,

        s_sigma [sigma : f] + s_tau [tau : f] = 0,

    which fixes ``s_tau`` from ``s_sigma``. Propagating that over the dual graph is a
    breadth-first search, and it either closes up consistently (orientable -- take the
    cycle) or meets a contradiction (non-orientable -- the component carries no integer
    p-cycle). Both outcomes are answers, and ``d z = 0`` is then verified in exact
    integer arithmetic, independently of how the signs were found.

    This is the input ``pysurgery.homology.poincare_duality_verification`` expects
    (``compute_poincare_duality_map(sc, n, fundamental_class, k)``): ``as_chain(K)``
    returns the coefficient vector over ``K.n_simplices(p)``.

Key Concepts:
    - **Why nothing may sit above dimension p.** The 2-simplices of a solid tetrahedron
      form a closed orientable 2-pseudomanifold whose sum is a cycle -- and a BOUNDARY in
      K. Summing p-simplices is a generator of ``H_p`` only when there is nothing above
      them; with nothing above, ``H_p(K) = Z_p(K)`` (no boundaries).
    - **One cycle per strong component.** A p-cycle has constant absolute coefficient
      along the dual graph, so ``H_p(K; Z) = Z^{c_or}`` (c_or the orientable strong
      components) and ``H_p(K; F_2) = F_2^c``. Adding several components' classes with
      arbitrary relative signs would hide a sign choice inside every linking number or
      intersection number computed from the sum, so they are returned separately.
    - **Refusals, by name** (``NoFundamentalClassError``): simplices above dimension p;
      branching or boundary (p-1)-faces; several components (for the single-cycle
      function); non-orientability over Z.

Common Workflows:
    1. **The fundamental class of a closed manifold** -> ``fundamental_cycle(K, n)``.
    2. **A basis of top homology** -> ``top_homology_basis(K, p)``.
    3. **Orientability of a closed pseudomanifold** -> ``is_orientable_pseudomanifold``.
    4. **Mod-2 fundamental classes** -> ``fundamental_cycles(K, p, coefficient_ring="Z2")``.

Coefficient Ring:
    Z (coherent signs, exact verification) and Z2 (every coefficient 1).
"""

from __future__ import annotations

import warnings
from collections import deque
from typing import TYPE_CHECKING, Dict, List, Tuple

import numpy as np
from pydantic import BaseModel, ConfigDict

from ..bridge.julia_bridge import julia_engine
from ..core.exceptions import NoFundamentalClassError

if TYPE_CHECKING:  # pragma: no cover
    from .complexes import SimplicialComplex

Simplex = Tuple[int, ...]

__all__ = [
    "FundamentalCycle",
    "CoherentOrientation",
    "coherent_orientation",
    "fundamental_cycles",
    "fundamental_cycle",
    "top_homology_basis",
    "is_orientable_pseudomanifold",
]


class FundamentalCycle(BaseModel):
    """A fundamental cycle of one strong component of a closed p-pseudomanifold.

    Attributes:
        p (int): The dimension.
        simplices (list[tuple[int, ...]]): The p-simplices of ONE strong component.
        signs (list[int]): The coefficient of each simplex (+-1 over Z, 1 over Z2),
            relative to its sorted vertex order.
        component (int): Index of the strong component.
        n_components (int): Number of strong components of K.
        coefficient_ring (str): ``"Z"`` or ``"Z2"``.
        verified (bool): ``d z = 0`` checked in exact integer (resp. mod-2) arithmetic.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    p: int
    simplices: List[Tuple[int, ...]]
    signs: List[int]
    component: int
    n_components: int
    coefficient_ring: str = "Z"
    verified: bool = False

    def as_pairs(self) -> List[Tuple[Simplex, int]]:
        """``[(simplex, coefficient), ...]`` -- the chain as explicit pairs.

        Returns:
            The chain as (sorted simplex, coefficient) pairs.
        """
        return [(s, int(c)) for s, c in zip(self.simplices, self.signs)]

    def as_chain(self, K: "SimplicialComplex") -> np.ndarray:
        """Coefficient vector over ``K.n_simplices(p)`` (pySurgery's chain convention).

        Args:
            K: The complex the cycle lives in.

        Returns:
            An int64 array aligned with ``K.n_simplices(p)``.
        """
        index = K.simplex_to_index(self.p)
        out = np.zeros(K.count_simplices(self.p), dtype=np.int64)
        for s, c in zip(self.simplices, self.signs):
            out[index[tuple(s)]] = int(c)
        return out

    def __str__(self) -> str:
        return (
            f"fundamental {self.p}-cycle over {self.coefficient_ring} of component "
            f"{self.component + 1}/{self.n_components}: {len(self.simplices)} simplices, "
            f"{'d z = 0 verified' if self.verified else 'NOT VERIFIED'}"
        )


class CoherentOrientation(BaseModel):
    """Coherent signs on the top simplices of a closed p-pseudomanifold.

    Attributes:
        p (int): The dimension.
        simplices (list[tuple[int, ...]]): All p-simplices of K.
        signs (list[int]): The propagated sign of each simplex (meaningful on orientable
            components; on a non-orientable one, the signs of the BFS tree).
        component (list[int]): Strong-component index of each simplex.
        orientable (list[bool]): Per component, whether propagation closed up.
        n_components (int): Number of strong components.
    """

    p: int
    simplices: List[Tuple[int, ...]]
    signs: List[int]
    component: List[int]
    orientable: List[bool]
    n_components: int

    @property
    def is_orientable(self) -> bool:
        """Every strong component is orientable."""
        return all(self.orientable)


def _top_structure(K: "SimplicialComplex", p: int):
    """The p-simplices and the owners of every (p-1)-face, hypotheses checked.

    Refuses unless K is a closed p-pseudomanifold with nothing above dimension p.
    """
    if p <= 0:
        raise NoFundamentalClassError(
            "p = 0: H_0 is generated by one vertex per component, not by a fundamental class"
        )
    top = [tuple(int(v) for v in s) for s in K.n_simplices(p)]
    if not top:
        raise NoFundamentalClassError(f"K has no {p}-simplices (dimension {K.dimension})")
    if K.dimension > p:
        raise NoFundamentalClassError(
            f"K has simplices of dimension {K.dimension} > p = {p}. The p-simplices may "
            f"still sum to a cycle, but it can be a BOUNDARY in K (the 2-simplices of a "
            f"solid tetrahedron do); a fundamental class needs K to be p-dimensional."
        )
    owners: Dict[Simplex, List[int]] = {}
    for i, s in enumerate(top):
        for j in range(len(s)):
            owners.setdefault(s[:j] + s[j + 1:], []).append(i)
    n_bd = sum(1 for v in owners.values() if len(v) == 1)
    n_branch = sum(1 for v in owners.values() if len(v) > 2)
    if n_branch:
        raise NoFundamentalClassError(
            f"branching: {n_branch} of {len(owners)} ({p - 1})-faces lie in more than two "
            f"{p}-simplices. Not a pseudomanifold."
        )
    if n_bd:
        raise NoFundamentalClassError(
            f"not closed: {n_bd} of {len(owners)} ({p - 1})-faces lie in a single "
            f"{p}-simplex. A complex with boundary has no fundamental class in H_{p}."
        )
    return top, owners


def _propagate_python(top: List[Simplex], owners: Dict[Simplex, List[int]]):
    """Breadth-first coherent-sign propagation over the dual graph."""
    signs = np.zeros(len(top), dtype=np.int64)
    comp = np.full(len(top), -1, dtype=np.int64)
    orientable: List[bool] = []
    for seed in range(len(top)):
        if signs[seed]:
            continue
        cid = len(orientable)
        signs[seed] = 1
        comp[seed] = cid
        q = deque([seed])
        ok = True
        while q:
            i = q.popleft()
            s = top[i]
            for j in range(len(s)):
                f = s[:j] + s[j + 1:]
                inc_s = -1 if j % 2 else 1
                for nb in owners[f]:
                    if nb == i:
                        continue
                    t = top[nb]
                    jn = next(k for k in range(len(t)) if t[k] not in f)
                    inc_t = -1 if jn % 2 else 1
                    want = -signs[i] * inc_s * inc_t
                    if signs[nb] == 0:
                        signs[nb] = want
                        comp[nb] = cid
                        q.append(nb)
                    elif signs[nb] != want:
                        ok = False
        orientable.append(ok)
    return signs.tolist(), comp.tolist(), orientable


def _verify(simplices: List[Simplex], coeffs: List[int], modulus: int = 0) -> bool:
    """``d z = 0`` in exact integer (or mod-``modulus``) arithmetic."""
    acc: Dict[Simplex, int] = {}
    for s, c in zip(simplices, coeffs):
        for j in range(len(s)):
            f = s[:j] + s[j + 1:]
            acc[f] = acc.get(f, 0) + int(c) * (-1 if j % 2 else 1)
    if modulus:
        return all(v % modulus == 0 for v in acc.values())
    return all(v == 0 for v in acc.values())


def coherent_orientation(
    K: "SimplicialComplex", p: int | None = None, backend: str = "auto"
) -> CoherentOrientation:
    """Propagate coherent signs over the dual graph of a closed p-pseudomanifold.

    What is Being Computed?:
        Signs ``s_sigma`` with ``s_sigma [sigma:f] + s_tau [tau:f] = 0`` across every
        shared (p-1)-face, one BFS per strong component; a component is orientable iff
        the propagation closes up without contradiction.

    Args:
        K: A simplicial complex.
        p: The dimension (default ``K.dimension``).
        backend: 'auto', 'julia' or 'python'.

    Returns:
        A ``CoherentOrientation``.

    Raises:
        NoFundamentalClassError: If K is not a closed p-pseudomanifold with nothing above
            dimension p.
    """
    p = K.dimension if p is None else int(p)
    top, owners = _top_structure(K, p)
    backend_norm = str(backend).lower().strip()
    use_julia = backend_norm == "julia" or (backend_norm == "auto" and julia_engine.available)
    result = None
    if use_julia:
        try:
            result = julia_engine.coherent_orientation(top)
        except Exception as e:  # pragma: no cover - depends on the Julia runtime
            if backend_norm == "julia":
                raise
            warnings.warn(f"Julia coherent orientation failed ({e!r}); falling back to Python.")
    if result is None:
        result = _propagate_python(top, owners)
    signs, comp, orientable = result
    return CoherentOrientation(
        p=p, simplices=top, signs=[int(x) for x in signs], component=[int(x) for x in comp],
        orientable=[bool(x) for x in orientable], n_components=len(orientable),
    )


def fundamental_cycles(
    K: "SimplicialComplex",
    p: int | None = None,
    coefficient_ring: str = "Z",
    backend: str = "auto",
) -> List[FundamentalCycle]:
    """One fundamental cycle per strong component of a closed p-pseudomanifold.

    What is Being Computed?:
        Over Z: the coherently signed sum of the p-simplices of each strong component;
        together they are a Z-basis of ``H_p(K; Z) = Z^c``. Over Z2: the plain sum of
        each component's p-simplices, a basis of ``H_p(K; F_2) = F_2^c`` (no
        orientability needed).

    Algorithm:
        1. Check the hypotheses (``_top_structure``), refusing by name.
        2. Coherent orientation (Python BFS or the Julia kernel).
        3. Verify ``d z = 0`` for every component in exact arithmetic.

    Args:
        K: A simplicial complex.
        p: The dimension (default ``K.dimension``).
        coefficient_ring: ``"Z"`` or ``"Z2"``.
        backend: 'auto', 'julia' or 'python'.

    Returns:
        The fundamental cycles, one per strong component, all verified.

    Raises:
        NoFundamentalClassError: If a hypothesis fails, or (over Z) some component is
            non-orientable.
        ValueError: For an unsupported coefficient ring.
    """
    ring = str(coefficient_ring).upper().replace("/", "").replace("Z2Z", "Z2")
    if ring not in ("Z", "Z2"):
        raise ValueError(f"coefficient_ring must be 'Z' or 'Z2', got {coefficient_ring!r}")
    ori = coherent_orientation(K, p, backend=backend)
    if ring == "Z":
        bad = [c for c, ok in enumerate(ori.orientable) if not ok]
        if bad:
            raise NoFundamentalClassError(
                f"non-orientable: the coherent-sign propagation closes up with a "
                f"contradiction on {len(bad)} of {ori.n_components} component(s) {bad}, so "
                f"those have no integer fundamental cycle (over Z2 they still do: pass "
                f"coefficient_ring='Z2'; `top_homology_basis` gives H_{ori.p}(K; Z))."
            )
    out = []
    comp = np.asarray(ori.component)
    for c in range(ori.n_components):
        idx = np.flatnonzero(comp == c)
        simps = [ori.simplices[i] for i in idx]
        coeffs = [ori.signs[i] for i in idx] if ring == "Z" else [1] * len(idx)
        ok = _verify(simps, coeffs, modulus=0 if ring == "Z" else 2)
        if not ok:  # pragma: no cover - a failed verification is a bug, never an answer
            raise NoFundamentalClassError(
                "coherent orientation found but d z != 0 in exact arithmetic -- this "
                "should not be reachable; please report it"
            )
        out.append(FundamentalCycle(
            p=ori.p, simplices=simps, signs=coeffs, component=c,
            n_components=ori.n_components, coefficient_ring=ring, verified=ok,
        ))
    return out


def fundamental_cycle(
    K: "SimplicialComplex",
    p: int | None = None,
    coefficient_ring: str = "Z",
    backend: str = "auto",
) -> FundamentalCycle:
    """THE generator of ``H_p(K) = Z`` for a closed, orientable, strongly connected K.

    Args:
        K: A simplicial complex.
        p: The dimension (default ``K.dimension``).
        coefficient_ring: ``"Z"`` or ``"Z2"``.
        backend: 'auto', 'julia' or 'python'.

    Returns:
        The fundamental cycle, verified.

    Raises:
        NoFundamentalClassError: If K is not a closed (orientable, over Z)
            p-pseudomanifold, or has several strong components (then ``H_p`` has no
            single generator: use ``fundamental_cycles`` and choose explicitly).

    Example:
        >>> z = fundamental_cycle(torus, 2)
        >>> (torus.boundary_matrix(2) @ z.as_chain(torus) == 0).all()
        True
    """
    cycles = fundamental_cycles(K, p, coefficient_ring=coefficient_ring, backend=backend)
    if len(cycles) != 1:
        raise NoFundamentalClassError(
            f"{len(cycles)} strong components: H_{cycles[0].p} = Z^{len(cycles)} has no "
            f"single generator. Use `fundamental_cycles` and choose the components explicitly."
        )
    return cycles[0]


def top_homology_basis(
    K: "SimplicialComplex", p: int | None = None, backend: str = "auto"
) -> List[FundamentalCycle]:
    """A Z-basis of ``H_p(K; Z)`` for a closed p-pseudomanifold: its orientable components.

    What is Being Computed?:
        With nothing above dimension p, ``H_p(K; Z) = Z_p(K; Z)``, and an integer
        p-cycle is a multiple of the coherent sum on each orientable strong component
        and zero on each non-orientable one. So the orientable components' fundamental
        cycles form a basis (empty when every component is non-orientable).

    Args:
        K: A simplicial complex.
        p: The dimension (default ``K.dimension``).
        backend: 'auto', 'julia' or 'python'.

    Returns:
        The verified fundamental cycles of the orientable components.

    Raises:
        NoFundamentalClassError: If K is not a closed p-pseudomanifold.
    """
    ori = coherent_orientation(K, p, backend=backend)
    comp = np.asarray(ori.component)
    out = []
    for c, ok in enumerate(ori.orientable):
        if not ok:
            continue
        idx = np.flatnonzero(comp == c)
        simps = [ori.simplices[i] for i in idx]
        coeffs = [ori.signs[i] for i in idx]
        verified = _verify(simps, coeffs)
        out.append(FundamentalCycle(
            p=ori.p, simplices=simps, signs=coeffs, component=c,
            n_components=ori.n_components, coefficient_ring="Z", verified=verified,
        ))
    return out


def is_orientable_pseudomanifold(
    K: "SimplicialComplex", p: int | None = None, backend: str = "auto"
) -> bool:
    """Is the closed p-pseudomanifold K orientable (every strong component)?

    Decided purely combinatorially, with no coordinates and no frames: exact.

    Args:
        K: A simplicial complex.
        p: The dimension (default ``K.dimension``).
        backend: 'auto', 'julia' or 'python'.

    Returns:
        True iff every strong component admits a coherent orientation.

    Raises:
        NoFundamentalClassError: If K is not a closed p-pseudomanifold.
    """
    return coherent_orientation(K, p, backend=backend).is_orientable
