from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class Pi1GeneratorTrace(BaseModel):
    """Data-grounded trace for one presentation generator."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    generator: str
    edge_index: int
    component_root: int
    vertex_path: list[int] = Field(default_factory=list)
    directed_edge_path: list[tuple[int, int]] = Field(default_factory=list)
    undirected_edge_path: list[tuple[int, int]] = Field(default_factory=list)


class Pi1PresentationWithTraces(BaseModel):
    """Fundamental-group presentation plus generator traces in the source complex."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    generators: list[str] = Field(default_factory=list)
    relations: list[list[str]] = Field(default_factory=list)
    traces: list[Pi1GeneratorTrace] = Field(default_factory=list)
    mode_used: str = "optimized"
    generator_mode: str = "optimized"
    backend_used: str = "python"
    raw_generator_count: int = 0
    optimized_generator_count: int = 0
    reduced_generator_count: int = 0


class HomologyGenerator(BaseModel):
    """One homology generator represented by data-native simplices/edges."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    dimension: int
    support_edges: list[tuple[int, int]] = Field(default_factory=list)
    support_simplices: list[tuple[int, ...]] = Field(default_factory=list)
    weight: float = 0.0
    certified_cycle: bool = True


class HomologyBasisResult(BaseModel):
    """Computed homology basis in data-native representative form."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    dimension: int
    rank: int
    generators: list[HomologyGenerator] = Field(default_factory=list)
    optimal: bool = False
    exact: bool = True
    message: str = ""

