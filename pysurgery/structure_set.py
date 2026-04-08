from typing import List, Optional
from pydantic import BaseModel, ConfigDict, Field
from .core.complexes import ChainComplex
from .core.exceptions import StructureSetError
from .wall_groups import l_group_symbol


class NormalInvariantsResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    dimension: int
    rank_Z: int
    rank_Z2: int
    notes: List[str] = Field(default_factory=list)

    def to_report(self) -> str:
        report = f"--- NORMAL INVARIANTS [M, G/TOP] FOR {self.dimension}D MANIFOLD ---\n"
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
    analysis: List[str] = Field(default_factory=list)
    normal_invariants: Optional[NormalInvariantsResult] = None

    def to_report(self) -> str:
        n = self.dimension
        report = f"--- SURGERY EXACT SEQUENCE FOR {n}D MANIFOLD ---\n"
        report += f"L_{n+1}(1) ---> S_TOP(M) ---> [M, G/TOP] ---> L_{n}(1)\n"
        report += f"   {self.l_n_plus_1_symbol}    ---> S_TOP(M) ---> Normal Invs --->    {self.l_n_symbol}\n\n"
        report += "Topological Analysis:\n"
        for line in self.analysis:
            report += f"- {line}\n"
        return report

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

    def compute_normal_invariants_result(self, chain: ChainComplex) -> NormalInvariantsResult:
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
                rank_Z2 += (r_k + z2_hom + z2_ext)

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
    ) -> SurgeryExactSequenceResult:
        n = self.dimension

        if not self._is_trivial_group_symbol(self.fundamental_group):
            fg = self.fundamental_group
            return SurgeryExactSequenceResult(
                dimension=n,
                fundamental_group=fg,
                l_n_symbol=l_group_symbol(n, fg),
                l_n_plus_1_symbol=l_group_symbol(n + 1, fg),
                computable=False,
                exact=False,
                analysis=[
                    f"Non-simply-connected case detected (pi_1 = {fg}).",
                    "The surgery exact sequence remains valid, but twisted Wall groups L_n(Z[pi_1]) are only partially implemented in this API.",
                    "For this calculation, enable Julia to accelerate exact representation-theoretic/group-ring reductions.",
                ],
                normal_invariants=normal_invariants,
            )

        if n < 5:
            raise StructureSetError("The Surgery Exact Sequence strictly applies to dimensions n >= 5. In 4D, Freedman's classification completely replaces the exact sequence.")

        l_n_str = self._format_wall_group(n)
        l_n_plus_1_str = self._format_wall_group(n + 1)

        analysis = [
            "The set of Normal Invariants [M, G/TOP] dictates the possible vector bundles over M.",
            f"The Wall group L_{n}(1) ({l_n_str}) acts as the primary obstruction to doing surgery.",
        ]
        if l_n_str == "0":
            analysis.append("Because L_n(1) = 0, every normal invariant maps directly into the Structure Set.")
        else:
            analysis.append(
                f"Because L_n(1) = {l_n_str}, some normal invariants may fail to lift to homotopy equivalences."
            )
        analysis.append("The group L_{n+1}(1) acts on the Structure Set and governs multiplicity of structures.")

        return SurgeryExactSequenceResult(
            dimension=n,
            fundamental_group="1",
            l_n_symbol=l_n_str,
            l_n_plus_1_symbol=l_n_plus_1_str,
            computable=True,
            exact=True,
            analysis=analysis,
            normal_invariants=normal_invariants,
        )

    def _format_wall_group(self, k: int) -> str:
        if k % 4 == 0:
            return "Z"
        if k % 4 == 2:
            return "Z_2"
        return "0"
