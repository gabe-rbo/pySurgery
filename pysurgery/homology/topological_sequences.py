"""
pysurgery/homology/topological_sequences.py

Construction of standard topological exact sequences.
"""

from __future__ import annotations
import numpy as np
from typing import TYPE_CHECKING, List, Tuple, Optional, Any
from pysurgery.algebra.exact_sequences import Morphism, ExactSequence
from pysurgery.homology.homology_generators import hk_generators_z

if TYPE_CHECKING:
    from pysurgery.topology.complexes import SimplicialComplex

def _induced_map_matrix(
    source_simplices: List[Tuple[int, ...]],
    target_simplices: List[Tuple[int, ...]],
    map_func: Any = None
) -> np.ndarray:
    """Compute the matrix of the induced chain map f_#: C_k(X) -> C_k(Y).
    
    If map_func is None, assumes inclusion.
    """
    m = len(target_simplices)
    n = len(source_simplices)
    matrix = np.zeros((m, n), dtype=object)
    
    target_idx = {s: i for i, s in enumerate(target_simplices)}
    
    for j, s in enumerate(source_simplices):
        mapped_s = tuple(sorted(map_func(s))) if map_func else s
        if mapped_s in target_idx:
            matrix[target_idx[mapped_s], j] = 1
            
    return matrix

def _induced_homology_morphism(
    source_complex: Any,
    target_complex: Any,
    k: int,
    chain_map_matrix: np.ndarray
) -> Morphism:
    """Compute the induced morphism f_*: H_k(X) -> H_k(Y)."""
    # We need basis for H_k(X) and H_k(Y)
    # Then express f_#(basis_X) in terms of basis_Y
    
    # 1. Get cycle basis for H_k(X)
    # (Using a simplified version of the logic in SimplicialComplex.chain_to_homology_class)
    # This is non-trivial because we need to handle torsion.
    # For now, let's assume free modules for the basic LES/MV implementations
    # and improve robustness if needed.
    
    # Actually, SimplicialComplex.homology() returns (rank, torsion).
    # hk_generators_z returns the generators themselves.
    
    gens_source = hk_generators_z(source_complex, k)
    gens_target = hk_generators_z(target_complex, k)
    
    rank_s = len(gens_source)
    rank_t = len(gens_target)
    
    if rank_s == 0:
        return Morphism(np.zeros((rank_t, 0), dtype=object), 0, rank_t)
    if rank_t == 0:
        return Morphism(np.zeros((0, rank_s), dtype=object), rank_s, 0)
        
    # Build the matrix: each column is f_*(gen_j) in basis of H_k(Y)
    # We use chain_to_homology_class on target_complex for each mapped generator
    
    # Wait, chain_to_homology_class needs the chain vector in target_complex's basis
    mat = np.zeros((rank_t, rank_s), dtype=object)
    
    target_simplices = target_complex.n_simplices(k)
    target_idx = {s: i for i, s in enumerate(target_simplices)}
    
    for j, gen in enumerate(gens_source):
        # f_#(gen)
        # gen is a HomologyGenerator with cycle_coefficients
        mapped_chain = np.zeros(len(target_simplices), dtype=np.int64)
        for simplex, coef in gen.cycle_coefficients.items():
            mapped_s = simplex # Assuming inclusion for now
            if mapped_s in target_idx:
                mapped_chain[target_idx[mapped_s]] += coef
        
        # Map to homology class
        h_class = target_complex.chain_to_homology_class(k, mapped_chain)
        # print(f"DEBUG induced mapping H_{k}: {j}-th gen maps to {h_class}")
        if len(h_class) > rank_t:
             # This can happen if rank_t was computed differently (e.g. including torsion)
             # But mat is (rank_t, rank_s).
             pass
        mat[:, j] = h_class
        
    return Morphism(mat, rank_s, rank_t)

def compute_long_exact_sequence_of_pair(
    X: SimplicialComplex, 
    A: SimplicialComplex, 
    max_dim: Optional[int] = None
) -> ExactSequence:
    """Construct the Long Exact Sequence of the pair (X, A).
    
    ... -> H_n(A) -> H_n(X) -> H_n(X, A) -> H_{n-1}(A) -> ...
    """
    if max_dim is None:
        max_dim = X.dimension
        
    modules = []
    morphisms = []
    
    # We build the sequence from high dimension to low
    for n in range(max_dim, -1, -1):
        # 1. H_n(A)
        hk_generators_z(A, n)
        modules.append(f"H_{n}(A)")
        
        # 2. H_n(X)
        hk_generators_z(X, n)
        modules.append(f"H_{n}(X)")
        
        # Induced map i_*: H_n(A) -> H_n(X)
        # Inclusion matrix at chain level
        A_simplices = A.n_simplices(n)
        X_simplices = X.n_simplices(n)
        i_sharp = _induced_map_matrix(A_simplices, X_simplices)
        i_star = _induced_homology_morphism(A, X, n, i_sharp)
        morphisms.append(i_star)
        
        # 3. H_n(X, A)
        rel_X_A = X.relative_chain_complex(A)
        Hn_rel_gens = hk_generators_z(rel_X_A, n)
        rank_Hn_rel = len(Hn_rel_gens)
        modules.append(f"H_{n}(X, A)")
        
        # Induced map j_*: H_n(X) -> H_n(X, A)
        # Projection matrix at chain level: X_n -> X_n / A_n
        # relative_chain_complex basis is [s for s in X_n if s not in A_n]
        rel_simplices = [s for s in X_simplices if s not in set(A_simplices)]
        j_sharp = _induced_map_matrix(X_simplices, rel_simplices)
        j_star = _induced_homology_morphism(X, rel_X_A, n, j_sharp)
        morphisms.append(j_star)
        
        # 4. Boundary map d: H_n(X, A) -> H_{n-1}(A)
        if n > 0:
            Anm1_simplices = A.n_simplices(n - 1)
            d_mat = np.zeros((len(Anm1_simplices), len(Hn_rel_gens)), dtype=object)
            Anm1_idx = {s: i for i, s in enumerate(Anm1_simplices)}
            
            for col_idx, gen in enumerate(Hn_rel_gens):
                # Lift relative cycle to X-chain
                chain_X = np.zeros(len(X_simplices), dtype=np.int64)
                for simplex_key, coef in gen.cycle_coefficients.items():
                    if simplex_key in X_simplices:
                        chain_X[X_simplices.index(simplex_key)] = coef
                    else:
                        # Handle potential mapping issues
                        pass
                # Take boundary in X
                boundary_X = X.boundary_matrix(n) @ chain_X
                
                # Result should be in A
                chain_A = np.zeros(len(Anm1_simplices), dtype=np.int64)
                X_nm1_simplices = X.n_simplices(n - 1)
                for i, coef in enumerate(boundary_X):
                    if coef == 0:
                        continue
                    s_nm1 = X_nm1_simplices[i]
                    if s_nm1 in Anm1_idx:
                        chain_A[Anm1_idx[s_nm1]] = coef
                
                # Map to H_{n-1}(A)
                h_class = A.chain_to_homology_class(n - 1, chain_A)
                d_mat[:, col_idx] = h_class
            
            Hnm1_A_rank = len(hk_generators_z(A, n - 1))
            d_morphism = Morphism(d_mat, rank_Hn_rel, Hnm1_A_rank)
            morphisms.append(d_morphism)
            
    return ExactSequence(modules, morphisms)

def compute_mayer_vietoris(
    X: SimplicialComplex, 
    U: SimplicialComplex, 
    V: SimplicialComplex, 
    k_max: Optional[int] = None
) -> ExactSequence:
    """Construct the Mayer-Vietoris sequence for X = U U V.
    
    ... -> H_k(U n V) -> H_k(U) + H_k(V) -> H_k(X) -> H_{k-1}(U n V) -> ...
    """
    if k_max is None:
        k_max = X.dimension
        
    # 1. Compute intersection A = U n V
    U_simplices = set(U.all_simplices())
    V_simplices = set(V.all_simplices())
    intersection_simplices = U_simplices.intersection(V_simplices)
    A = X.subcomplex(intersection_simplices)
    
    modules = []
    morphisms = []
    
    for k in range(k_max, -1, -1):
        # 1. H_k(A)
        Hk_A_rank = len(hk_generators_z(A, k))
        modules.append(f"H_{k}(A)")
        
        # 2. H_k(U) + H_k(V)
        Hk_U_rank = len(hk_generators_z(U, k))
        Hk_V_rank = len(hk_generators_z(V, k))
        modules.append(f"H_{k}(U) + H_{k}(V)")
        
        # Map phi: H_k(A) -> H_k(U) + H_k(V),  z |-> (i_U*(z), i_V*(z))
        A_simplices = A.n_simplices(k)
        U_simplices_k = U.n_simplices(k)
        V_simplices_k = V.n_simplices(k)
        
        i_U_star = _induced_homology_morphism(A, U, k, _induced_map_matrix(A_simplices, U_simplices_k))
        i_V_star = _induced_homology_morphism(A, V, k, _induced_map_matrix(A_simplices, V_simplices_k))
        
        phi_mat = np.vstack([i_U_star.matrix, i_V_star.matrix])
        phi_morphism = Morphism(phi_mat, Hk_A_rank, Hk_U_rank + Hk_V_rank)
        morphisms.append(phi_morphism)
        
        # 3. H_k(X)
        Hk_X_rank = len(hk_generators_z(X, k))
        modules.append(f"H_{k}(X)")
        
        # Map psi: H_k(U) + H_k(V) -> H_k(X),  (u, v) |-> j_U*(u) - j_V*(v)
        X_simplices_k = X.n_simplices(k)
        j_U_star = _induced_homology_morphism(U, X, k, _induced_map_matrix(U_simplices_k, X_simplices_k))
        j_V_star = _induced_homology_morphism(V, X, k, _induced_map_matrix(V_simplices_k, X_simplices_k))
        
        psi_mat = np.hstack([j_U_star.matrix, -j_V_star.matrix])
        psi_morphism = Morphism(psi_mat, Hk_U_rank + Hk_V_rank, Hk_X_rank)
        morphisms.append(psi_morphism)
        
        # 4. Boundary map d: H_k(X) -> H_{k-1}(A)
        if k > 0:
            Hk_X_gens = hk_generators_z(X, k)
            Hkm1_A_gens = hk_generators_z(A, k - 1)
            Anm1_simplices = A.n_simplices(k - 1)
            d_mat = np.zeros((len(Hkm1_A_gens), len(Hk_X_gens)), dtype=object)
            Anm1_idx = {s: i for i, s in enumerate(Anm1_simplices)}
            
            for col_idx, gen in enumerate(Hk_X_gens):
                # Barycentric/Simplicial MV boundary is more subtle.
                # Standard algorithm: write z = u + v where u in C_k(U), v in C_k(V).
                # Then d[z] = [d u] = - [d v].
                z_chain = np.zeros(len(X_simplices_k), dtype=np.int64)
                for s, c in gen.cycle_coefficients.items():
                    z_chain[X_simplices_k.index(s)] = c
                    
                # Split z into u and v
                u_chain = np.zeros(len(U_simplices_k), dtype=np.int64)
                v_chain = np.zeros(len(V_simplices_k), dtype=np.int64)
                U_idx = {s: i for i, s in enumerate(U_simplices_k)}
                V_idx = {s: i for i, s in enumerate(V_simplices_k)}
                
                # Heuristic: if simplex is in U, put it in u. Else in v.
                # If in both, put it in u (choice doesn't matter for d).
                for s, c in gen.cycle_coefficients.items():
                    if s in U_idx:
                        u_chain[U_idx[s]] = c
                    elif s in V_idx:
                        v_chain[V_idx[s]] = c
                    else:
                        raise ValueError(f"Simplex {s} in cycle not in U or V.")
                
                # Take boundary of u in U
                du = U.boundary_matrix(k) @ u_chain
                
                # Map to H_{k-1}(A)
                # du should have support in A
                chain_A = np.zeros(len(Anm1_simplices), dtype=np.int64)
                U_nm1_simplices = U.n_simplices(k - 1)
                for i, coef in enumerate(du):
                    if coef == 0:
                        continue
                    s_nm1 = U_nm1_simplices[i]
                    if s_nm1 in Anm1_idx:
                        chain_A[Anm1_idx[s_nm1]] = coef
                
                h_class = A.chain_to_homology_class(k - 1, chain_A)
                d_mat[:, col_idx] = h_class
                
            d_morphism = Morphism(d_mat, Hk_X_rank, len(Hkm1_A_gens))
            morphisms.append(d_morphism)
            
    return ExactSequence(modules, morphisms)

def compute_bockstein_sequence(X: SimplicialComplex, p: int, k: int) -> ExactSequence:
    """The Bockstein sequence associated with 0 -> Z --p--> Z -> Z/pZ -> 0.
    
    ... -> H_k(X; Z) --p--> H_k(X; Z) -> H_k(X; Z/pZ) -> H_{k-1}(X; Z) -> ...
    """
    # 1. H_k(X; Z)
    Hk_Z_rank = len(hk_generators_z(X, k))
    
    # 2. H_k(X; Z/pZ)
    X_p = X.model_copy()
    X_p.coefficient_ring = f"Z/{p}Z"
    Hk_Zp_rank, _ = X_p.homology(k) # Over field, only rank
    
    # 3. H_{k-1}(X; Z)
    Hkm1_Z_rank = len(hk_generators_z(X, k - 1))
    
    modules = [f"H_{k}(X; Z)", f"H_{k}(X; Z)", f"H_{k}(X; Z/{p}Z)", f"H_{k-1}(X; Z)"]
    
    # Map i*: H_k(X; Z) --p--> H_k(X; Z)
    # This is just p * Identity
    i_star_mat = p * np.eye(Hk_Z_rank, dtype=object)
    i_star = Morphism(i_star_mat, Hk_Z_rank, Hk_Z_rank)
    
    # Map rho*: H_k(X; Z) -> H_k(X; Z/pZ)
    # Reduction mod p.
    rho_mat = np.zeros((Hk_Zp_rank, Hk_Z_rank), dtype=object)
    # TODO: implementation of reduction mapping
    
    # Map beta: H_k(X; Z/pZ) -> H_{k-1}(X; Z)
    # The Bockstein homomorphism
    beta_mat = np.zeros((Hkm1_Z_rank, Hk_Zp_rank), dtype=object)
    # TODO: implementation of Bockstein
    
    return ExactSequence(modules, [i_star, Morphism(rho_mat, Hk_Z_rank, Hk_Zp_rank), Morphism(beta_mat, Hk_Zp_rank, Hkm1_Z_rank)])
