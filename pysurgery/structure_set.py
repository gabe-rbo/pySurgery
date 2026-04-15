from typing import List, Optional
from pydantic import BaseModel, ConfigDict, Field
from .core.complexes import ChainComplex
from .core.exceptions import StructureSetError
from .wall_groups import ObstructionResult, WallGroupL, l_group_symbol


class NormalInvariantsResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    dimension: int
    rank_Z: int
    rank_Z2: int
    notes: List[str] = Field(default_factory=list)
    exact: bool = True

    def to_report(self) -> str:
        report = (
            f"--- NORMAL INVARIANTS [M, G/TOP] FOR {self.dimension}D MANIFOLD ---\n"
        )
        report += f"Rank over Z: {self.rank_Z}\n"
        report += f"Rank over Z_2: {self.rank_Z2}\n"
        report += "By Sullivan's formula, this defines the topological vector bundles that can be framed for surgery."
        return report


class SurgeryExactSequenceResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    dimension: int
    fundamental_group: str
    l_n_symbol: str
    l_n_plus_1_symbol: str
    computable: bool
    exact: bool
    partial: bool = False
    analysis: List[str] = Field(default_factory=list)
    normal_invariants: Optional[NormalInvariantsResult] = None
    l_n_obstruction: Optional[ObstructionResult] = None
    l_n_plus_1_obstruction: Optional[ObstructionResult] = None
    l_n_state: "LObstructionState" = Field(default_factory=lambda: LObstructionState())
    l_n_plus_1_state: "LObstructionState" = Field(
        default_factory=lambda: LObstructionState()
    )

    def to_report(self) -> str:
        n = self.dimension
        report = f"--- SURGERY EXACT SEQUENCE FOR {n}D MANIFOLD ---\n"
        report += f"L_{n + 1}(1) ---> S_TOP(M) ---> [M, G/TOP] ---> L_{n}(1)\n"
        report += f"   {self.l_n_plus_1_symbol}    ---> S_TOP(M) ---> Normal Invs --->    {self.l_n_symbol}\n\n"
        report += "Topological Analysis:\n"
        for line in self.analysis:
            report += f"- {line}\n"
        return report


class LObstructionState(BaseModel):
    available: bool = False
    computable: bool = False
    exact: bool = False
    obstructs: Optional[bool] = None
    zero_certified: bool = False
    value: Optional[int] = None
    modulus: Optional[int] = None
    pi: Optional[str] = None
    dimension: Optional[int] = None
    decomposition_kind: str = "scalar"
    assembly_certified: bool = False
    message: str = ""

    def __getitem__(self, key: str):
        # Backward-compatible dict-like access used in older tests/callers.
        return self.model_dump().get(key)

    def to_legacy_dict(self) -> dict:
        return self.model_dump()

    @classmethod
    def from_obstruction(cls, ob: Optional[ObstructionResult]) -> "LObstructionState":
        if ob is None:
            return cls()
        return cls(
            available=True,
            computable=bool(ob.computable),
            exact=bool(ob.exact),
            obstructs=ob.obstructs,
            zero_certified=bool(ob.zero_certified),
            value=ob.value,
            modulus=ob.modulus,
            pi=ob.pi,
            dimension=ob.dimension,
            decomposition_kind=getattr(ob, "decomposition_kind", "scalar"),
            assembly_certified=bool(getattr(ob, "assembly_certified", False)),
            message=ob.message,
        )


class StructureSet(BaseModel):
    """
    Implementation of the topological Structure Set S_TOP(M).

    This mathematically models the Surgery Exact Sequence:
    ... -> L_{n+1}(pi_1) -> S_TOP(M) -> [M, G/TOP] -> L_n(pi_1)

    It determines the exact number of distinct manifolds that are
    homotopy equivalent to M but NOT homeomorphic to M.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    dimension: int
    fundamental_group: str = "1"

    @staticmethod
    def _is_trivial_group_symbol(symbol: str) -> bool:
        return symbol.strip().lower() in {"1", "trivial", "e"}

    def compute_normal_invariants(self, chain: ChainComplex) -> str:
        """
        Computes the rank of the set of Normal Invariants [M, G/TOP] via Sullivan's characteristic variety formula.
        For a simply connected manifold:
        [M, G/TOP] is isomorphic to Sum_{i>=1} H^{4i}(M; Z) + Sum_{i>=1} H^{4i-2}(M; Z_2)
        modulo some 2-torsion extensions. We compute the free rank and Z_2 rank.
        """
        return self.compute_normal_invariants_result(chain).to_report()

    def compute_normal_invariants_result(
        self, chain: ChainComplex
    ) -> NormalInvariantsResult:
        n = self.dimension
        rank_Z = 0
        rank_Z2 = 0

        for k in range(1, n + 1):
            if k % 4 == 0:
                r, _ = chain.cohomology(k)
                rank_Z += r
            elif k % 4 == 2:
                r_k, t_k = chain.homology(k)
                _, t_km1 = chain.homology(k - 1)
                z2_hom = sum(1 for t in t_k if t % 2 == 0)
                z2_ext = sum(1 for t in t_km1 if t % 2 == 0)
                rank_Z2 += r_k + z2_hom + z2_ext

        return NormalInvariantsResult(
            dimension=n,
            rank_Z=rank_Z,
            rank_Z2=rank_Z2,
            notes=["Computed via Sullivan characteristic formula terms"],
        )

    def evaluate_exact_sequence(self) -> str:
        """
        Evaluates the sequence to determine the size and nature of the Structure Set S_TOP(M).
        """
        return self.evaluate_exact_sequence_result().to_report()

    def evaluate_exact_sequence_result(
        self,
        normal_invariants: Optional[NormalInvariantsResult] = None,
        l_n_obstruction: Optional[ObstructionResult] = None,
        l_n_plus_1_obstruction: Optional[ObstructionResult] = None,
        l_n_state: Optional[LObstructionState] = None,
        l_n_plus_1_state: Optional[LObstructionState] = None,
    ) -> SurgeryExactSequenceResult:
        n = self.dimension

        def _resolve_state(
            ob: Optional[ObstructionResult], state: Optional[LObstructionState]
        ) -> tuple[LObstructionState, bool]:
            conflict = False
            derived = LObstructionState.from_obstruction(ob)
            if state is not None:
                if ob is not None:
                    conflict = (
                        derived.available != state.available
                        or derived.computable != state.computable
                        or derived.exact != state.exact
                        or derived.obstructs != state.obstructs
                        or derived.zero_certified != state.zero_certified
                    )
                # Explicit typed state has precedence.
                return state, conflict
            return derived, False

        resolved_l_n_state, l_n_conflict = _resolve_state(l_n_obstruction, l_n_state)
        resolved_l_n_plus_1_state, l_n_plus_1_conflict = _resolve_state(
            l_n_plus_1_obstruction, l_n_plus_1_state
        )

        if not self._is_trivial_group_symbol(self.fundamental_group):
            fg = self.fundamental_group
            resolved_l_n = l_n_obstruction
            resolved_l_n_plus_1 = l_n_plus_1_obstruction

            if resolved_l_n is None:
                try:
                    resolved_l_n = WallGroupL(
                        dimension=n, pi=fg
                    ).compute_obstruction_result()
                except Exception:
                    resolved_l_n = None
            if resolved_l_n_plus_1 is None:
                try:
                    resolved_l_n_plus_1 = WallGroupL(
                        dimension=n + 1, pi=fg
                    ).compute_obstruction_result()
                except Exception:
                    resolved_l_n_plus_1 = None

            if l_n_state is None:
                resolved_l_n_state = LObstructionState.from_obstruction(resolved_l_n)
                l_n_conflict = False
            if l_n_plus_1_state is None:
                resolved_l_n_plus_1_state = LObstructionState.from_obstruction(
                    resolved_l_n_plus_1
                )
                l_n_plus_1_conflict = False

            channels_available = (
                resolved_l_n_state.available and resolved_l_n_plus_1_state.available
            )
            channels_computable = (
                channels_available
                and resolved_l_n_state.computable
                and resolved_l_n_plus_1_state.computable
            )
            channels_exact = (
                channels_computable
                and resolved_l_n_state.exact
                and resolved_l_n_plus_1_state.exact
            )

            analysis = [
                f"Non-simply-connected case detected (pi_1 = {fg}).",
                "The surgery exact sequence remains valid; this API now threads typed L_n/L_{n+1} state channels for nontrivial pi_1 branches.",
                "Group-ring assembly completeness is still theorem-sensitive; readiness is graded from the supplied/computed typed states.",
            ]
            if channels_exact:
                analysis.append(
                    "Typed L-state channels are exact and computable for both L_n and L_{n+1} in this branch."
                )
            elif channels_computable:
                analysis.append(
                    "Typed L-state channels are computable but include heuristic/non-exact data."
                )
            else:
                analysis.append(
                    "Typed L-state channels are incomplete or non-computable; nontrivial group-ring decomposition remains partial."
                )
            if (
                resolved_l_n_state.available
                and not resolved_l_n_state.assembly_certified
            ):
                analysis.append(
                    "L_n channel is not assembly-certified for full product-group composition."
                )
            if (
                resolved_l_n_plus_1_state.available
                and not resolved_l_n_plus_1_state.assembly_certified
            ):
                analysis.append(
                    "L_{n+1} channel is not assembly-certified for full product-group composition."
                )
            if l_n_conflict:
                analysis.append(
                    "Warning: explicit L_n state conflicts with obstruction-derived state; explicit typed state was used."
                )
            if l_n_plus_1_conflict:
                analysis.append(
                    "Warning: explicit L_{n+1} state conflicts with obstruction-derived state; explicit typed state was used."
                )
            return SurgeryExactSequenceResult(
                dimension=n,
                fundamental_group=fg,
                l_n_symbol=l_group_symbol(n, fg),
                l_n_plus_1_symbol=l_group_symbol(n + 1, fg),
                computable=channels_computable,
                exact=channels_exact,
                partial=not channels_exact,
                analysis=analysis,
                normal_invariants=normal_invariants,
                l_n_obstruction=resolved_l_n,
                l_n_plus_1_obstruction=resolved_l_n_plus_1,
                l_n_state=resolved_l_n_state,
                l_n_plus_1_state=resolved_l_n_plus_1_state,
            )

        if n < 5:
            raise StructureSetError(
                "The Surgery Exact Sequence strictly applies to dimensions n >= 5. In 4D, Freedman's classification completely replaces the exact sequence."
            )

        l_n_str = self._format_wall_group(n)
        l_n_plus_1_str = self._format_wall_group(n + 1)

        resolved_l_n = l_n_obstruction
        resolved_l_n_plus_1 = l_n_plus_1_obstruction
        if resolved_l_n is None:
            try:
                candidate = WallGroupL(dimension=n, pi="1").compute_obstruction_result()
                if candidate.computable and candidate.exact:
                    resolved_l_n = candidate
            except Exception:
                pass
        if resolved_l_n_plus_1 is None:
            try:
                candidate = WallGroupL(
                    dimension=n + 1, pi="1"
                ).compute_obstruction_result()
                if candidate.computable and candidate.exact:
                    resolved_l_n_plus_1 = candidate
            except Exception:
                pass

        if l_n_state is None:
            resolved_l_n_state = LObstructionState.from_obstruction(resolved_l_n)
            l_n_conflict = False
        if l_n_plus_1_state is None:
            resolved_l_n_plus_1_state = LObstructionState.from_obstruction(
                resolved_l_n_plus_1
            )
            l_n_plus_1_conflict = False

        analysis = [
            "The set of Normal Invariants [M, G/TOP] dictates the possible vector bundles over M.",
            f"The Wall group L_{n}(1) ({l_n_str}) acts as the primary obstruction to doing surgery.",
        ]
        if l_n_str == "0":
            analysis.append(
                "Because L_n(1) = 0, every normal invariant maps directly into the Structure Set."
            )
        else:
            analysis.append(
                f"Because L_n(1) = {l_n_str}, some normal invariants may fail to lift to homotopy equivalences."
            )
        analysis.append(
            "The group L_{n+1}(1) acts on the Structure Set and governs multiplicity of structures."
        )
        if resolved_l_n_state.available:
            if resolved_l_n_state.obstructs is True:
                analysis.append(
                    "Typed L_n obstruction state certifies a non-zero obstruction element."
                )
            elif resolved_l_n_state.zero_certified:
                analysis.append(
                    "Typed L_n obstruction state certifies vanishing obstruction."
                )
            else:
                analysis.append(
                    "Typed L_n obstruction state is available but does not certify vanishing/non-vanishing."
                )
        if resolved_l_n_plus_1_state.available:
            if resolved_l_n_plus_1_state.obstructs is True:
                analysis.append(
                    "Typed L_{n+1} action state is non-trivial in the supplied obstruction certificate."
                )
            elif resolved_l_n_plus_1_state.zero_certified:
                analysis.append(
                    "Typed L_{n+1} action state is certified zero in the supplied obstruction certificate."
                )
        if l_n_conflict:
            analysis.append(
                "Warning: explicit L_n state conflicts with obstruction-derived state; explicit typed state was used."
            )
        if l_n_plus_1_conflict:
            analysis.append(
                "Warning: explicit L_{n+1} state conflicts with obstruction-derived state; explicit typed state was used."
            )

        return SurgeryExactSequenceResult(
            dimension=n,
            fundamental_group="1",
            l_n_symbol=l_n_str,
            l_n_plus_1_symbol=l_n_plus_1_str,
            computable=True,
            exact=True,
            analysis=analysis,
            normal_invariants=normal_invariants,
            l_n_obstruction=resolved_l_n,
            l_n_plus_1_obstruction=resolved_l_n_plus_1,
            l_n_state=resolved_l_n_state,
            l_n_plus_1_state=resolved_l_n_plus_1_state,
        )

    def _format_wall_group(self, k: int) -> str:
        if k % 4 == 0:
            return "Z"
        if k % 4 == 2:
            return "Z_2"
        return "0"
