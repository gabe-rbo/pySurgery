import numpy as np
from pysurgery.topology.complexes import SimplicialComplex

def test_long_exact_sequence_of_pair_circle_point():
    # X = S^1 (triangle: (0,1), (1,2), (0,2))
    # A = {0} (point)
    sc = SimplicialComplex()
    sc.add_simplex((0, 1))
    sc.add_simplex((1, 2))
    sc.add_simplex((0, 2))
    
    A = sc.subcomplex([(0,)])
    
    les = sc.long_exact_sequence_of_pair(A)
    
    # H_1(A) -> H_1(X) -> H_1(X, A) -> H_0(A) -> H_0(X) -> H_0(X, A)
    # Ranks: 0 -> 1 -> 1 -> 1 -> 1 -> 0
    
    # Check H_1(A) -> H_1(X) [morphisms[0]]
    # Im(m0) = 0, Ker(m1) should be 0.
    assert les.morphisms[0].matrix.shape == (1, 0)
    
    # Check H_1(X) -> H_1(X, A) [morphisms[1]]
    # Should be an isomorphism (1x1 matrix with +/- 1)
    assert abs(les.morphisms[1].matrix[0, 0]) == 1
    
    # Check exactness at H_1(X, A)
    # Im(m1) = H_1(X, A), so Ker(m2) should be everything.
    # d: H_1(X, A) -> H_0(A)
    # For S^1 relative to a point, the boundary of the loop is 0 in H_0(A)
    assert np.all(les.morphisms[2].matrix == 0)
    
    # Verify exactness at H_0(A)
    # Im(d) = 0, Ker(H_0(A) -> H_0(X)) should be 0.
    # i*: H_0(A) -> H_0(X) is inclusion of pt into S^1, which is isomorphism on H0
    assert abs(les.morphisms[3].matrix[0, 0]) == 1
    
    # Verify exactness globally
    for i in range(len(les.morphisms) - 1):
        assert les.verify_exactness(i)

def test_mayer_vietoris_circle():
    from pysurgery.homology.topological_sequences import compute_mayer_vietoris
    # X = S^1
    # U = Upper half (0,1,2) -> (0,1), (1,2)
    # V = Lower half (0,3,2) -> (0,3), (3,2)
    # U n V = {0, 2}
    X = SimplicialComplex()
    X.add_simplex((0, 1))
    X.add_simplex((1, 2))
    X.add_simplex((2, 3))
    X.add_simplex((3, 0))
    
    U = X.subcomplex([(0, 1), (1, 2)])
    V = X.subcomplex([(2, 3), (3, 0)])
    
    mv = compute_mayer_vietoris(X, U, V, k_max=1)
    
    # H_1(A) -> H_1(U)+H_1(V) -> H_1(X) -> H_0(A) -> H_0(U)+H_0(V) -> H_0(X)
    # A = {0, 2}, H_0(A) rank 2.
    # U, V are intervals, H_0 rank 1, H_1 rank 0.
    # H_1(A) rank 0.
    # Sequence: 0 -> 0 -> Z -> Z^2 -> Z^2 -> Z -> 0
    
    # Verify exactness at H_1(X)
    # Im(H_1(U)+H_1(V) -> H_1(X)) is 0. 
    # Ker(H_1(X) -> H_0(A)) should be 0.
    # Boundary map d: H_1(X) -> H_0(A) should be injective.
    # A loop in S^1 is (0->1->2) + (2->3->0).
    # d[loop] = [pt 2] - [pt 0] in H_0(A).
    assert mv.morphisms[2].matrix.shape == (2, 1)
    assert np.all(mv.morphisms[2].matrix != 0)
    
    # Verify exactness at H_0(A)
    # Im(d) has rank 1. 
    # Ker(H_0(A) -> H_0(U)+H_0(V)) should have rank 1.
    # i*: H_0(A) -> H_0(U)+H_0(V)
    # i*(a, b) = (a+b, a+b)  -- wait, i_U*(pt0)=pt0, i_U*(pt2)=pt2. In H_0(U), pt0 ~ pt2.
    # So i_U*(a, b) = a+b. Similarly for V.
    # i*(a, b) = (a+b, a+b).
    # Ker is {(a, -a)}, rank 1. Correct.
    
    for i in range(len(mv.morphisms) - 1):
        assert mv.verify_exactness(i)
