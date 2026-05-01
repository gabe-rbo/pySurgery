from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field
from typing import List, Tuple, Optional, Dict


class Pi1GeneratorTrace(BaseModel):
    """Data-grounded trace for one presentation generator.

    Overview:
        A Pi1GeneratorTrace provides a geometric realization of a symbolic generator
        in the fundamental group presentation. It maps an abstract group generator 
        back to a concrete path in the original CW or simplicial complex.

    Key Concepts:
        - **Geometric Realization**: Linking symbolic algebra back to spatial topology.
        - **Vertex Path**: The sequence of vertices traversed to form the loop.
        - **Component Root**: The basepoint vertex for the fundamental group in this component.

    Common Workflows:
        1. **Extraction**: Generated during `extract_pi_1_with_traces()`.
        2. **Visualization**: Used to plot the generator loop on the complex geometry.
        3. **Mapping**: Used to pull back group elements to geometric cycles.

    Attributes:
        generator (str): The symbolic name of the generator (e.g., 'a', 'b^-1').
        vertex_path (List[int]): Sequence of vertex IDs forming the loop.
        component_root (int): The root vertex ID of the connected component.
        edge_index (Optional[int]): Optional edge index in the complex.
        directed_edge_path (List[Tuple[int, int]]): Optional list of directed edges.
        undirected_edge_path (List[Tuple[int, int]]): Optional list of undirected edges.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    generator: str
    vertex_path: List[int]
    component_root: int
    edge_index: Optional[int] = None
    directed_edge_path: List[Tuple[int, int]] = Field(default_factory=list)
    undirected_edge_path: List[Tuple[int, int]] = Field(default_factory=list)


class Pi1PresentationWithTraces(BaseModel):
    """Full π₁ presentation with geometric traces for every generator.

    Overview:
        Pi1PresentationWithTraces bundles a symbolic group presentation (generators and 
        relations) with the geometric path data (traces) required to reconstruct each 
        generator as a cycle in the underlying space.

    Key Concepts:
        - **Presentation (G, R)**: The abstract group defined by generators G and relations R.
        - **Geometric Tracing**: The correspondence between symbolic tokens and spatial loops.
        - **Orientation Character (w₁)**: Records the local orientation behavior for each generator.

    Common Workflows:
        1. **Complex Analysis** → `extract_pi_1_with_traces(cw)`
        2. **Presentation Reduction** → `simplify()` or Tietze transformations
        3. **Downstream Invariants** → Use for homology, K-theory, or surgery obstructions

    Attributes:
        generators (List[str]): List of symbolic generator names.
        relations (List[List[str]]): List of relations as sequences of symbolic generator tokens.
        orientation_character (Dict[str, int]): Map of generator → {-1, 1} recording orientation-reversing loops.
        traces (List[Pi1GeneratorTrace]): List of Pi1GeneratorTrace objects.
        mode_used (str): The normalization mode used during extraction.
        generator_mode (str): The generator extraction mode ('raw', 'optimized').
        backend_used (str): The computation backend ('python', 'julia').
        raw_generator_count (int): Number of generators before any simplification.
        optimized_generator_count (int): Number of generators after standard optimization.
        reduced_generator_count (int): Number of generators after full Tietze reduction.
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

    Overview:
        A HomologyGenerator represents a d-cycle in the chain complex that generates 
        a class in H_d. It includes the explicit geometric support (simplices) 
        and optional metric weights for optimization.

    Key Concepts:
        - **Cycle**: A chain c such that ∂c = 0.
        - **Support**: The set of simplices with non-zero coefficients in the chain.
        - **Metric Weight**: A value (length, area, volume) associated with the cycle's footprint.

    Common Workflows:
        1. **Homology Computation** → `homology_generators()`
        2. **Optimization** → Minimize `weight` via linear programming or local search.
        3. **Verification** → Check `certified_cycle` to ensure the boundary is indeed zero.

    Attributes:
        dimension (int): Degree of the cycle (d).
        support_simplices (List[Tuple[int, ...]]): List of simplices in the cycle's support.
        support_edges (List[Tuple[int, int]]): List of edges in the cycle's support.
        weight (float): Metric weight (e.g., total length or area).
        certified_cycle (bool): Whether the cycle is verified to be a boundary-null chain.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    dimension: int
    support_simplices: List[Tuple[int, ...]]
    support_edges: List[Tuple[int, int]] = Field(default_factory=list)
    weight: float = 0.0
    certified_cycle: bool = True


class HomologyBasisResult(BaseModel):
    """Result of a homology basis computation for a specific dimension.

    Overview:
        HomologyBasisResult encapsulates the full structure of H_d(X), including 
        its rank, representative cycles, and metadata regarding the optimality 
        and exactness of the basis found.

    Key Concepts:
        - **Basis**: A set of cycles that generates the free part of the homology group.
        - **Optimality**: Whether these representatives minimize some geometric functional.
        - **Exactness**: Whether the computation used exact arithmetic or heuristics.

    Common Workflows:
        1. **Direct Query** → `complex.homology(dimension=2)`
        2. **Basis Inspection** → Iterate over `generators` to analyze cycle geometry.
        3. **Metric Analysis** → Check `optimal` flag before using for geometric inference.

    Attributes:
        dimension (int): The dimension of the homology group (d).
        rank (int): The rank of the homology group (number of generators).
        generators (List[HomologyGenerator]): List of representative cycles.
        optimal (bool): Whether the basis is metrically optimal (e.g., shortest loops).
        exact (bool): Whether the result is topologically exact (using Smith Normal Form).
        message (str): Status message or error description.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    dimension: int
    rank: int
    generators: List[HomologyGenerator] = Field(default_factory=list)
    optimal: bool = False
    exact: bool = True
    message: str = ""
