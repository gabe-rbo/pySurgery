import numpy as np
import pytest

from pysurgery.core.bundles import SimplicialVectorBundle
from pysurgery.core.characteristic_classes import (
    extract_euler_class,
    extract_stiefel_whitney_tangent,
)
from pysurgery.core.complexes import SimplicialComplex
from pysurgery.core.fundamental_group import (
    extract_pi_1,
    infer_standard_group_descriptor,
)


# --- Canonical Space Builders ---


def build_s1():
    """1-sphere triangulation."""
    return SimplicialComplex.from_simplices([[0, 1], [1, 2], [2, 0]])


def build_s2():
    """2-sphere (tetrahedron boundary)."""
    return SimplicialComplex.from_simplices(
        [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]
    )


def build_s3():
    """3-sphere (boundary of 4-simplex)."""
    simplices = []
    for i in range(5):
        face = [j for j in range(5) if j != i]
        simplices.append(face)
    return SimplicialComplex.from_simplices(simplices)


def build_torus():
    """Minimal triangulation of T^2 (18 triangles, 9 vertices)."""
    faces = [
        [0, 1, 4],
        [0, 4, 3],
        [1, 2, 5],
        [1, 5, 4],
        [2, 0, 3],
        [2, 3, 5],
        [3, 4, 7],
        [3, 7, 6],
        [4, 5, 8],
        [4, 8, 7],
        [5, 3, 6],
        [5, 6, 8],
        [6, 7, 1],
        [6, 1, 0],
        [7, 8, 2],
        [7, 2, 1],
        [8, 6, 0],
        [8, 0, 2],
    ]
    return SimplicialComplex.from_simplices(faces)


def build_rp2():
    """Minimal triangulation of RP^2 (10 triangles, 6 vertices)."""
    faces = [
        [0, 1, 2],
        [0, 2, 3],
        [0, 3, 4],
        [0, 4, 5],
        [0, 5, 1],
        [1, 2, 4],
        [2, 3, 5],
        [3, 4, 1],
        [4, 5, 2],
        [5, 1, 3],
    ]
    return SimplicialComplex.from_simplices(faces)


def build_cp2():
    """Kühnel-Banchoff minimal triangulation of CP^2 (36 facets, 9 vertices)."""
    # 0-indexed vertices 0..8
    # Generator S = (0, 3, 6)(1, 4, 7)(2, 5, 8)
    mapping = {0: 3, 3: 6, 6: 0, 1: 4, 4: 7, 7: 1, 2: 5, 5: 8, 8: 2}

    def apply_S(s):
        return tuple(sorted([mapping[v] for v in s]))

    # 12 base 4-simplices representing the orbits
    bases = [
        [0, 1, 4, 7, 8],
        [0, 1, 2, 7, 8],
        [0, 2, 5, 7, 8],
        [1, 3, 4, 7, 8],
        [1, 2, 3, 7, 8],
        [2, 3, 5, 7, 8],
        [0, 1, 3, 4, 5],
        [0, 2, 3, 4, 5],
        [0, 1, 3, 4, 8],
        [0, 2, 3, 5, 7],
        [0, 1, 3, 5, 6],
        [0, 3, 5, 6, 7],
    ]

    all_simplices = set()
    for b in bases:
        curr = tuple(sorted(b))
        for _ in range(3):
            all_simplices.add(curr)
            curr = apply_S(curr)

    return SimplicialComplex.from_simplices(list(all_simplices))


def build_cylinder():
    """Cylinder triangulation."""
    faces = []
    for i in range(3):
        v00, v01 = i * 2, i * 2 + 1
        v10, v11 = ((i + 1) % 3) * 2, ((i + 1) % 3) * 2 + 1
        faces.append([v00, v01, v11])
        faces.append([v00, v10, v11])
    return SimplicialComplex.from_simplices(faces)


def build_mobius_band():
    """Möbius band triangulation."""
    faces = [[0, 1, 3], [0, 2, 3], [2, 3, 5], [2, 4, 5], [4, 5, 0], [4, 0, 1]]
    return SimplicialComplex.from_simplices(faces)


def build_punctured_torus():
    """Torus with one triangle removed."""
    torus = build_torus()
    simplices = list(torus.n_simplices(2))
    return SimplicialComplex.from_simplices(simplices[:-1])


# --- Fixtures ---


@pytest.fixture
def s1():
    return build_s1()


@pytest.fixture
def s2():
    return build_s2()


@pytest.fixture
def s3():
    return build_s3()


@pytest.fixture
def torus():
    return build_torus()


@pytest.fixture
def rp2():
    return build_rp2()


@pytest.fixture
def cp2():
    return build_cp2()


@pytest.fixture
def cylinder():
    return build_cylinder()


@pytest.fixture
def mobius():
    return build_mobius_band()


@pytest.fixture
def p_torus():
    return build_punctured_torus()


# --- Tests ---


def test_spheres_invariants(s1, s2, s3):
    # S1
    assert s1.betti_number(0) == 1
    assert s1.betti_number(1) == 1
    assert s1.euler_characteristic() == 0
    pi1_s1 = extract_pi_1(s1.to_cw_complex())
    assert infer_standard_group_descriptor(pi1_s1) == "Z"

    # S2
    assert s2.betti_number(0) == 1
    assert s2.betti_number(1) == 0
    assert s2.betti_number(2) == 1
    assert s2.euler_characteristic() == 2
    pi1_s2 = extract_pi_1(s2.to_cw_complex())
    assert infer_standard_group_descriptor(pi1_s2) == "1"

    # S3
    assert s3.betti_number(0) == 1
    assert s3.betti_number(1) == 0
    assert s3.betti_number(2) == 0
    assert s3.betti_number(3) == 1
    assert s3.euler_characteristic() == 0
    pi1_s3 = extract_pi_1(s3.to_cw_complex())
    assert infer_standard_group_descriptor(pi1_s3) == "1"


def test_torus_invariants(torus):
    assert torus.betti_number(0) == 1
    assert torus.betti_number(1) == 2
    assert torus.betti_number(2) == 1
    assert torus.euler_characteristic() == 0
    pi1_t2 = extract_pi_1(torus.to_cw_complex())
    assert infer_standard_group_descriptor(pi1_t2) == "Z x Z"


def test_rp2_invariants(rp2):
    # H_1(RP2) = Z_2
    h1_rank, h1_torsion = rp2.homology(1)
    assert h1_rank == 0
    assert h1_torsion == [2]

    assert rp2.betti_number(2) == 0
    assert rp2.euler_characteristic() == 1

    pi1_rp2 = extract_pi_1(rp2.to_cw_complex())
    assert infer_standard_group_descriptor(pi1_rp2) == "Z_2"


def test_cp2_invariants(cp2):
    assert cp2.betti_number(0) == 1
    assert cp2.betti_number(1) == 0
    assert cp2.betti_number(2) == 1
    assert cp2.betti_number(3) == 0
    assert cp2.betti_number(4) == 1
    assert cp2.euler_characteristic() == 3


def test_tangent_bundle_invariants(s2, torus, rp2, cp2):
    # Evaluated on the fundamental class (sum over all n-simplices mod 2)
    def evaluate_on_fundamental(sc, cochain):
        return int(np.sum(cochain) % 2)

    # w1 (Orientability)
    # S2, T2, CP2 are orientable (w1 = 0 cocycle)
    w1_s2 = extract_stiefel_whitney_tangent(s2, 1)

    assert np.all(w1_s2 % 2 == 0)

    # RP2 is non-orientable (w1 != 0)
    w1_rp2 = extract_stiefel_whitney_tangent(rp2, 1)
    assert np.any(w1_rp2 % 2 != 0)

    # Top Class w^n evaluates to chi(M) mod 2
    w2_s2 = extract_stiefel_whitney_tangent(s2, 2)
    assert evaluate_on_fundamental(s2, w2_s2) == s2.euler_characteristic() % 2

    w2_t2 = extract_stiefel_whitney_tangent(torus, 2)
    assert evaluate_on_fundamental(torus, w2_t2) == torus.euler_characteristic() % 2

    w2_rp2 = extract_stiefel_whitney_tangent(rp2, 2)
    assert evaluate_on_fundamental(rp2, w2_rp2) == rp2.euler_characteristic() % 2

    w4_cp2 = extract_stiefel_whitney_tangent(cp2, 4)
    assert evaluate_on_fundamental(cp2, w4_cp2) == cp2.euler_characteristic() % 2

    # Euler class integration (integral Chi)
    assert extract_euler_class(s2) == 2
    assert extract_euler_class(torus) == 0
    assert extract_euler_class(rp2) == 1
    assert extract_euler_class(cp2) == 3


def test_bundle_cocycle_and_w1(rp2):
    # Möbius Bundle over S1
    s1 = build_s1()
    # Edges: (0, 1), (1, 2), (2, 0)
    transitions = {
        (0, 1): np.array([[1.0]]),
        (1, 0): np.array([[1.0]]),
        (1, 2): np.array([[1.0]]),
        (2, 1): np.array([[1.0]]),
        (0, 2): np.array([[-1.0]]),
        (2, 0): np.array([[-1.0]]),
    }
    bundle = SimplicialVectorBundle(base_complex=s1, rank=1, transitions=transitions)
    # No triangles in S1, but we can verify it doesn't crash
    assert bundle.check_cocycle()
    w1 = bundle.stiefel_whitney_class(1)
    assert int(np.sum(w1) % 2) == 1

    # Real Tangent Bundle of RP2
    tb_rp2 = SimplicialVectorBundle.tangent_bundle(rp2)
    assert tb_rp2.check_cocycle()
    w1_rp2 = tb_rp2.stiefel_whitney_class(1)
    # w1 extraction for tangent bundle handles RP2 via manifold fast-path
    assert np.any(w1_rp2 == 1)


def test_homotopy_collapses(cylinder, mobius, p_torus):
    # Verify via simplicial collapses (Homotopy Equivalence preserves homology)

    # 1. Cylinder ~ S1
    cyl_collapsed = cylinder.collapse()
    assert cyl_collapsed.betti_number(1) == 1
    # Minimal core of S1 is 3 vertices, 3 edges
    assert cyl_collapsed.count_simplices(0) <= 6
    assert cyl_collapsed.count_simplices(2) == 0

    # 2. Möbius Band ~ S1
    mob_collapsed = mobius.collapse()
    assert mob_collapsed.betti_number(1) == 1
    assert mob_collapsed.count_simplices(2) == 0

    # 3. Punctured Torus ~ S1 v S1
    pt_collapsed = p_torus.collapse()
    assert pt_collapsed.betti_number(1) == 2
    assert pt_collapsed.count_simplices(2) == 0

    # 4. Standard Sphere S2 (Tetrahedron)
    # Removing one triangle from S2 makes it a Disk D2, which collapses to a point.
    s2 = build_s2()
    simplices = list(s2.n_simplices(2))
    disk = SimplicialComplex.from_simplices(simplices[:-1])
    disk_collapsed = disk.collapse()
    assert disk_collapsed.betti_number(0) == 1
    assert disk_collapsed.betti_number(1) == 0
    assert disk_collapsed.count_simplices(0) == 1  # Should collapse to a single point!


def test_morse_reduction(s2, torus, rp2):
    # Verify that Morse complex preserves homology
    for sc in [s2, torus, rp2]:
        mc = sc.morse_complex()
        for d in range(sc.dimension + 1):
            assert mc.betti_number(d) == sc.betti_number(d)
