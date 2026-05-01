"""Tests for Algebraic Poincaré Complexes and Poincaré Duality.

Overview:
    This suite verifies the implementation of Algebraic Poincaré Complexes (APC),
    including the cap product, dual complexes, and the verification of Betti 
    number symmetry for orientable manifolds.

Key Concepts:
    - **Algebraic Poincaré Complex**: A chain complex equipped with a fundamental 
      class and a chain homotopy equivalence (the cap product map).
    - **Cap Product (∩)**: A map Hᵏ(M) ⊗ Hₙ(M) → Hₙ₋ₖ(M) that induces duality.
    - **Fundamental Class ([M])**: A generator of Hₙ(M; ℤ) for an orientable manifold.
    - **Poincaré Duality**: The isomorphism Hᵏ(M) ≅ Hₙ₋ₖ(M) for closed orientable manifolds.
"""
import numpy as np
import scipy.sparse as sp
import pytest

from discrete_surface_data import get_surfaces, to_complex
from pysurgery.algebraic_poincare import AlgebraicPoincareComplex
from pysurgery.core.complexes import ChainComplex
from pysurgery.core.exceptions import DimensionError


def test_algebraic_poincare_dual():
    """Verify the construction of a dual complex from an APC.

    What is Being Computed?:
        The dual chain complex Cⁿ⁻* derived from C_*.

    Algorithm:
        1. Define a 1D chain complex (interval).
        2. Construct an AlgebraicPoincareComplex.
        3. Extract the dual complex and verify its dimensions.

    Preserved Invariants:
        - The dual complex should have dimensions consistent with the original.
    """
    d1 = sp.csr_matrix(np.array([[-1, 1]], dtype=np.int64))
    cc = ChainComplex(boundaries={1: d1}, dimensions=[0, 1], cells={0: 2, 1: 1})
    fund_class = np.array([1], dtype=np.int64)
    apc = AlgebraicPoincareComplex(
        chain_complex=cc, fundamental_class=fund_class, dimension=1
    )

    assert apc.dimension == 1
    dual = apc.dual_complex()
    assert dual.dimensions == [0, 1]


def test_cap_product():
    """Verify the cap product computation for a simple cycle (S¹).

    What is Being Computed?:
        The map ψₖ: Cᵏ → Cₙ₋ₖ defined by the cap product with the fundamental class.

    Algorithm:
        1. Define an S¹ chain complex and fundamental class.
        2. Provide a manual diagonal map ψ₀.
        3. Compute cap_product with a 0-cohomology class (constant 1).
        4. Verify the output shape.

    Preserved Invariants:
        - [M] ∩ 1 = [M] (Identity property of cap product).
    """
    # S1 case
    d1 = sp.csr_matrix(np.array([[-1, 1]], dtype=np.int64))
    cc = ChainComplex(boundaries={1: d1}, dimensions=[0, 1], cells={0: 2, 1: 1})
    fund_class = np.array([1], dtype=np.int64)
    # cohomology class at dim 0 (2 vertices)
    cohom_class = np.array([1, 1], dtype=np.int64)
    
    # Cap product of [S1] with 1 in H^0 is [S1] in H_1
    # n=1, k=0 -> n-k=1. psi_0: C^0 -> C_1. size(C^0)=2, size(C_1)=1. shape (1, 2)
    psi0 = np.array([[1, 0]], dtype=np.int64)
    apc = AlgebraicPoincareComplex(
        chain_complex=cc,
        fundamental_class=fund_class,
        dimension=1,
        psi={0: psi0}
    )
    # cap_product(cohomology_class, degree k)
    cap = apc.cap_product(cohom_class, 0)
    assert cap.shape == (1,)


def test_cap_product_requires_defined_psi():
    """Ensure cap_product fails if the diagonal map ψ is not provided.

    What is Being Computed?:
        Error handling for missing diagonal approximations.

    Algorithm:
        1. Construct APC with an empty psi dictionary.
        2. Attempt cap_product and catch DimensionError.
    """
    d1 = sp.csr_matrix(np.array([[-1, 1]], dtype=np.int64))
    cc = ChainComplex(boundaries={1: d1}, dimensions=[0, 1], cells={0: 2, 1: 1})
    apc = AlgebraicPoincareComplex(
        chain_complex=cc,
        fundamental_class=np.array([1]),
        dimension=1,
        psi={} # empty psi
    )
    with pytest.raises(DimensionError, match="Diagonal map psi_0 not defined"):
        apc.cap_product(np.array([1, 1]), 0)


def test_cap_product_dimension_mismatch_raises():
    """Ensure cap_product fails when the input cohomology class has wrong dimension.

    What is Being Computed?:
        Input validation for cohomology classes.

    Algorithm:
        1. Define a 1D complex (2 vertices).
        2. Provide a cohomology class of length 1 (wrong for C⁰).
        3. Verify that DimensionError is raised.
    """
    d1 = sp.csr_matrix(np.array([[-1, 1]], dtype=np.int64))
    cc = ChainComplex(boundaries={1: d1}, dimensions=[0, 1], cells={0: 2, 1: 1})
    # n=1, k=0 -> n-k=1. psi_0 shape (1, 2)
    apc = AlgebraicPoincareComplex(
        chain_complex=cc,
        fundamental_class=np.array([1]),
        dimension=1,
        psi={0: np.array([[1, 0]])}
    )
    with pytest.raises(DimensionError):
        apc.cap_product(np.array([1]), 0) # should be length 2


def test_psi_default_is_not_shared_between_instances():
    """Verify that the psi dictionary is uniquely instantiated for each APC.

    What is Being Computed?:
        Object isolation for diagonal maps.

    Algorithm:
        1. Create two APC instances without providing psi.
        2. Compare their psi attributes for identity.
    """
    d1 = sp.csr_matrix(np.array([[-1, 1]], dtype=np.int64))
    cc = ChainComplex(boundaries={1: d1}, dimensions=[0, 1], cells={0: 2, 1: 1})
    apc1 = AlgebraicPoincareComplex(
        chain_complex=cc, fundamental_class=np.array([1]), dimension=1
    )
    apc2 = AlgebraicPoincareComplex(
        chain_complex=cc, fundamental_class=np.array([1]), dimension=1
    )
    assert apc1.psi is not apc2.psi


def test_fundamental_class_length_mismatch_raises():
    """Ensure APC construction fails if fundamental class length ≠ cell count.

    What is Being Computed?:
        Structural validation of the fundamental class.
    """
    d1 = sp.csr_matrix(np.array([[-1, 1]], dtype=np.int64))
    cc = ChainComplex(boundaries={1: d1}, dimensions=[0, 1], cells={0: 2, 1: 1})
    with pytest.raises(DimensionError):
        AlgebraicPoincareComplex(
            chain_complex=cc, fundamental_class=np.array([1, 1]), dimension=1
        )


def test_fundamental_class_must_be_cycle_when_boundary_known():
    """Verify that the fundamental class must be a cycle (∂[M] = 0).

    What is Being Computed?:
        Cycle condition for fundamental classes.

    Algorithm:
        1. Construct a complex where a specific class has a non-zero boundary.
        2. Attempt to use it as a fundamental class and catch DimensionError.
    """
    # Chain complex with d1: [1, 1] mapping 1 -> 1
    d1 = sp.csr_matrix(np.array([[1, 1]], dtype=np.int64))
    # Dim 1 has 2 cells, dim 0 has 1 cell.
    cc = ChainComplex(boundaries={1: d1}, dimensions=[0, 1], cells={0: 1, 1: 2})
    # Fundamental class [1, 0] is NOT a cycle because d1([1, 0]) = 1 != 0
    with pytest.raises(DimensionError, match="must be a cycle"):
        AlgebraicPoincareComplex(
            chain_complex=cc, fundamental_class=np.array([1, 0]), dimension=1
        )


def test_psi_shape_checked_at_construction():
    """Ensure diagonal maps have correct matrix dimensions at construction time.

    What is Being Computed?:
        Static shape validation for diagonal maps.

    Algorithm:
        1. Provide a psi_0 matrix with transposed dimensions.
        2. Catch DimensionError during construction.
    """
    d1 = sp.csr_matrix(np.array([[-1, 1]], dtype=np.int64))
    cc = ChainComplex(boundaries={1: d1}, dimensions=[0, 1], cells={0: 2, 1: 1})
    # n=1, k=0 -> n-k=1. psi_0 shape (1, 2)
    with pytest.raises(DimensionError):
        AlgebraicPoincareComplex(
            chain_complex=cc,
            fundamental_class=np.array([1]),
            dimension=1,
            psi={0: np.array([[1], [0]], dtype=np.int64)}, # (2, 1) instead of (1, 2)
        )


@pytest.mark.parametrize(
    "name, builder, bettis, torsion, euler",
    get_surfaces(),
)
def test_poincare_duality_betti(name, builder, bettis, torsion, euler):
    """Verify Betti number symmetry for various surfaces and manifolds.

    What is Being Computed?:
        The symmetry βₖ = βₙ₋ₖ for orientable closed manifolds.

    Algorithm:
        1. Iterate through a suite of standard surfaces (S², Torus, S³, etc.).
        2. Check for orientability based on the manifold name.
        3. Verify rank equality between Hₖ and Hₙ₋ₖ.

    Preserved Invariants:
        - Betti number symmetry is a fundamental property of closed orientable manifolds.
    """
    st = builder()
    c = to_complex(st)

    dim_max = c.dimension
    # S2, Torus are orientable. RP2 is non-orientable.
    orientable = name in ["Tetrahedron", "Octahedron", "Icosahedron", "Torus", "S3", "Cube"] or "Torus" in name
    if orientable:
        for k in range(dim_max + 1):
            hk_rank, _ = c.homology(k)
            hnk_rank, _ = c.homology(dim_max - k)
            # modulo torsion, ranks should match
            assert hk_rank == hnk_rank
