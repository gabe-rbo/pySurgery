from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field
from typing import List, Tuple, Optional, Dict


class Pi1GeneratorTrace(BaseModel):
    """Data-grounded trace for one presentation generator.

    Attributes:
        generator: The symbolic name of the generator (e.g., 'a', 'b^-1').
        vertex_path: Sequence of vertex IDs forming the loop.
        component_root: The root vertex ID of the connected component.
        edge_index: Optional edge index in the complex.
        directed_edge_path: Optional list of directed edges.
        undirected_edge_path: Optional list of undirected edges.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    generator: str
    vertex_path: List[int]
    component_root: int
    edge_index: Optional[int] = None
    directed_edge_path: List[Tuple[int, int]] = Field(default_factory=list)
    undirected_edge_path: List[Tuple[int, int]] = Field(default_factory=list)


class Pi1PresentationWithTraces(BaseModel):
    """Full pi_1 presentation with geometric traces for every generator.

    Attributes:
        generators: List of symbolic generator names.
        relations: List of relations as sequences of symbolic generator tokens.
        traces: List of Pi1GeneratorTrace objects.
        mode_used: The normalization mode used.
        generator_mode: The generator extraction mode.
        backend_used: The computation backend.
        raw_generator_count: Number of raw generators.
        optimized_generator_count: Number of generators after optimization.
        reduced_generator_count: Number of generators after reduction.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    generators: List[str]
    relations: List[List[str]]
    orientation_character: Dict[str, int] = Field(default_factory=dict)
    traces: List[Pi1GeneratorTrace]
    mode_used: str = ""
    generator_mode: str = ""
    backend_used: str = "python"
    raw_generator_count: int = 0
    optimized_generator_count: int = 0
    reduced_generator_count: int = 0


class HomologyGenerator(BaseModel):
    """A single homology generator (cycle) with geometric support.

    Attributes:
        dimension: Degree of the cycle.
        support_simplices: List of simplices in the cycle's support.
        support_edges: List of edges in the cycle's support.
        weight: Metric weight (e.g., total length or area).
        certified_cycle: Whether the cycle is verified to be a boundary-null chain.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    dimension: int
    support_simplices: List[Tuple[int, ...]]
    support_edges: List[Tuple[int, int]] = Field(default_factory=list)
    weight: float = 0.0
    certified_cycle: bool = True


class HomologyBasisResult(BaseModel):
    """Result of a homology basis computation.

    Attributes:
        dimension: The dimension of the homology group.
        rank: The rank of the homology group.
        generators: List of representative cycles.
        optimal: Whether the basis is metrically optimal.
        exact: Whether the result is topologically exact.
        message: Status message or error description.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    dimension: int
    rank: int
    generators: List[HomologyGenerator] = Field(default_factory=list)
    optimal: bool = False
    exact: bool = True
    message: str = ""
