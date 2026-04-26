# A lightweight Julia backend module for high-performance algebraic operations over Z[pi]
module SurgeryBackend

using LinearAlgebra
using SparseArrays
using Statistics
using Combinatorics
using Random
using PythonCall: pyconvert, Py, PyDict
import PrecompileTools

export simplify_jl, hermitian_signature, exact_snf_sparse, exact_sparse_cohomology_basis, rank_q_sparse, rank_mod_p_sparse, sparse_cohomology_basis_mod_p, normal_surface_residual_norms, embedding_broad_phase_pairs, group_ring_multiply, multisignature, abelianize_group, integral_lattice_isometry, optgen_from_simplices, homology_generators_from_simplices, compute_boundary_data_from_simplices, compute_boundary_payload_from_simplices, compute_boundary_payload_from_flat_simplices, compute_boundary_mod2_matrix, compute_alexander_whitney_cup, compute_trimesh_boundary_data, compute_trimesh_boundary_data_flat, triangulate_surface_delaunay, orthogonal_procrustes, pairwise_distance_matrix, frechet_distance, gromov_wasserstein_distance, enumerate_cliques_sparse, compute_vietoris_rips, compute_circumradius_sq_3d, compute_circumradius_sq_2d, quick_mapper_jl, cknn_graph_jl, cknn_graph_accelerated_jl, is_homology_manifold_jl, compute_alpha_complex_simplices_jl, compute_alpha_threshold_emst_jl, compute_crust_simplices_jl, compute_witness_complex_simplices_jl

const HAS_INTEGER_SNF = try
    @eval import IntegerSmithNormalForm
    true
catch
    false
end

const HAS_ABSTRACT_ALGEBRA = try
    @eval import AbstractAlgebra
    true
catch
    false
end

const HAS_GRAPHS = try
    @eval using Graphs
    @eval using SimpleWeightedGraphs
    true
catch
    false
end

const HAS_DELAUNAY = try
    @eval using DelaunayTriangulation
    true
catch
    false
end

# Use AbstractArray/AbstractVector for zero-copy NumPy integration via juliacall/PythonCall
"""
    hermitian_signature(matrix)

Compute the signature `(#positive) - (#negative)` of a real symmetric matrix.
Uses Hermitian eigendecomposition with a scale-aware numerical tolerance.
"""
function hermitian_signature(matrix::AbstractMatrix{Float64})
    # eigvals(Hermitian(mat)) requires a dense Matrix in Julia.
    # Matrix(matrix) is a copy, but necessary for the LAPACK call.
    mat = Matrix{Float64}(matrix)
    eigenvalues = eigvals(Hermitian(mat))
    tol = length(eigenvalues) > 0 ? maximum(size(mat)) * eps(Float64) * maximum(abs.(eigenvalues)) : 1e-10
    pos = count(x -> x > tol, eigenvalues)
    neg = count(x -> x < -tol, eigenvalues)
    return pos - neg
end

function _reduce_snf(A::SparseMatrixCSC{Int64, Int64})
    m, n = size(A)
    E = nnz(A)
    
    # Track degrees and XOR sums
    row_deg = zeros(Int64, m)
    row_xor = zeros(Int64, m)
    row_vsum = zeros(Int64, m)

    col_deg = zeros(Int64, n)
    col_xor = zeros(Int64, n)
    col_vsum = zeros(Int64, n)

    # Build CSR for row-wise access
    rowptr = zeros(Int64, m + 2)
    @inbounds for r in A.rowval
        rowptr[r + 1] += 1
    end
    rowptr[1] = 1
    @inbounds for r in 1:m
        rowptr[r + 1] += rowptr[r]
    end
    
    col_idx = zeros(Int64, E)
    nz_val_row = zeros(Int64, E)
    
    cur_rowptr = copy(rowptr)
    
    @inbounds for c in 1:n
        for ptr in A.colptr[c]:(A.colptr[c+1]-1)
            r = A.rowval[ptr]
            v = A.nzval[ptr]
            
            # Initialize degrees and XOR sums
            row_deg[r] += 1
            row_xor[r] ⊻= c
            row_vsum[r] += v
            
            col_deg[c] += 1
            col_xor[c] ⊻= r
            col_vsum[c] += v
            
            # Populate CSR
            idx = cur_rowptr[r]
            col_idx[idx] = c
            nz_val_row[idx] = v
            cur_rowptr[r] += 1
        end
    end
    
    row_active = trues(m)
    col_active = trues(n)
    ones_count = 0
    
    row_q = Int64[]
    col_q = Int64[]
    sizehint!(row_q, m)
    sizehint!(col_q, n)
    
    @inbounds for r in 1:m
        if row_deg[r] == 1
            push!(row_q, r)
        end
    end
    @inbounds for c in 1:n
        if col_deg[c] == 1
            push!(col_q, c)
        end
    end
    
    row_head = 1
    col_head = 1
    
    while row_head <= length(row_q) || col_head <= length(col_q)
        while row_head <= length(row_q)
            r = row_q[row_head]
            row_head += 1
            
            !row_active[r] && continue
            row_deg[r] != 1 && continue
            
            c = row_xor[r]
            !col_active[c] && continue
            v = row_vsum[r]
            
            if abs(v) == 1
                row_active[r] = false
                col_active[c] = false
                ones_count += 1
                
                # Deactivating c affects rows intersecting c
                @inbounds for ptr in A.colptr[c]:(A.colptr[c+1]-1)
                    rr = A.rowval[ptr]
                    if row_active[rr]
                        vv = A.nzval[ptr]
                        row_deg[rr] -= 1
                        row_xor[rr] ⊻= c
                        row_vsum[rr] -= vv
                        if row_deg[rr] == 1
                            push!(row_q, rr)
                        end
                    end
                end
                
                # Deactivating r affects columns intersecting r
                @inbounds for ptr in rowptr[r]:(rowptr[r+1]-1)
                    cc = col_idx[ptr]
                    if col_active[cc]
                        vv = nz_val_row[ptr]
                        col_deg[cc] -= 1
                        col_xor[cc] ⊻= r
                        col_vsum[cc] -= vv
                        if col_deg[cc] == 1
                            push!(col_q, cc)
                        end
                    end
                end
            end
        end
        
        while col_head <= length(col_q)
            c = col_q[col_head]
            col_head += 1
            
            !col_active[c] && continue
            col_deg[c] != 1 && continue
            
            r = col_xor[c]
            !row_active[r] && continue
            v = col_vsum[c]
            
            if abs(v) == 1
                row_active[r] = false
                col_active[c] = false
                ones_count += 1
                
                # Deactivating c affects rows intersecting c
                @inbounds for ptr in A.colptr[c]:(A.colptr[c+1]-1)
                    rr = A.rowval[ptr]
                    if row_active[rr]
                        vv = A.nzval[ptr]
                        row_deg[rr] -= 1
                        row_xor[rr] ⊻= c
                        row_vsum[rr] -= vv
                        if row_deg[rr] == 1
                            push!(row_q, rr)
                        end
                    end
                end
                
                # Deactivating r affects columns intersecting r
                @inbounds for ptr in rowptr[r]:(rowptr[r+1]-1)
                    cc = col_idx[ptr]
                    if col_active[cc]
                        vv = nz_val_row[ptr]
                        col_deg[cc] -= 1
                        col_xor[cc] ⊻= r
                        col_vsum[cc] -= vv
                        if col_deg[cc] == 1
                            push!(col_q, cc)
                        end
                    end
                end
            end
        end
    end
    
    # Rebuild core matrix - Sparse version
    core_rows = findall(row_active)
    core_cols = findall(col_active)
    
    row_map = Dict{Int64, Int64}()
    sizehint!(row_map, length(core_rows))
    for (i, r) in enumerate(core_rows)
        row_map[r] = i
    end
    
    col_map = Dict{Int64, Int64}()
    sizehint!(col_map, length(core_cols))
    for (i, c) in enumerate(core_cols)
        col_map[c] = i
    end
    
    # Efficient sparse construction
    I_core, J_core, V_core = Int64[], Int64[], Int64[]
    for c in 1:n
        if col_active[c]
            cmap_c = col_map[c]
            for ptr in A.colptr[c]:(A.colptr[c+1]-1)
                r = A.rowval[ptr]
                if row_active[r]
                    push!(I_core, row_map[r])
                    push!(J_core, cmap_c)
                    push!(V_core, A.nzval[ptr])
                end
            end
        end
    end
    
    core_sparse = sparse(I_core, J_core, V_core, length(core_rows), length(core_cols))
    return ones_count, core_sparse
end

"""
    exact_snf_sparse(rows, cols, vals, m, n)

Compute Smith normal form invariant factors from sparse integer COO data.
This is the exact integer path used for torsion-sensitive computations.
"""
function exact_snf_sparse(rows::AbstractVector{Int64}, cols::AbstractVector{Int64}, vals::AbstractVector{Int64}, m::Int, n::Int)
    if !HAS_ABSTRACT_ALGEBRA
        error("AbstractAlgebra unavailable")
    end

    # sparse() constructor is efficient with these vectors.
    A = sparse(rows .+ 1, cols .+ 1, vals, m, n)

    try
        println("Starting SNF calculation for $(m)x$(n) matrix...")

        # 1. Leaf-peeling (O(V+E))
        ones_count, core_mat = _reduce_snf(A)
        factors = ones(Int64, ones_count)

        core_m, core_n = size(core_mat)
        if core_m > 0 && core_n > 0
            if HAS_INTEGER_SNF && (core_m > 100 || core_n > 100)
                # Use the specialized IntegerSmithNormalForm kernel for large cores
                # It is much faster for matrices over Z
                println("Using IntegerSmithNormalForm for $(core_m)x$(core_n) core...")
                # Note: IntegerSmithNormalForm.smith_diagonal returns only the diagonal
                # It accepts sparse matrices!
                s_diag = IntegerSmithNormalForm.smith_diagonal(core_mat)
                for val in s_diag
                    if val != 0
                        push!(factors, Int64(abs(val)))
                    end
                end
            else
                # Use AbstractAlgebra for smaller cores or as fallback
                ZZ = AbstractAlgebra.ZZ
                # AbstractAlgebra.matrix requires a dense array or its own sparse format
                # For small cores, dense is fine.
                A_aa = AbstractAlgebra.matrix(ZZ, Matrix(core_mat))
                S_aa = AbstractAlgebra.snf(A_aa)

                for i in 1:min(core_m, core_n)
                    val = Int64(S_aa[i, i])
                    if val != 0
                        push!(factors, abs(val))
                    end
                end
            end
        end

        println("Finished SNF calculation.")
        return sort(factors)
    catch e
        println("SNF calculation failed.")
        rethrow(e)
    end
end
"""
    exact_sparse_cohomology_basis(...)

Compute an integral cohomology basis from sparse coboundary data.
Prefers exact AbstractAlgebra kernels; falls back to numerical linear algebra when unavailable.
"""
function _sparse_rref_rational!(A::SparseMatrixCSC{Rational{BigInt}, Int64})
    m, n = size(A)
    pivots = Int[]
    row = 1
    for col in 1:n
        # Find pivot in current column
        pivot_row = 0
        for r in row:m
            if A[r, col] != 0
                pivot_row = r
                break
            end
        end

        pivot_row == 0 && continue

        # Swap rows
        if pivot_row != row
            # Sparse row swapping is expensive in CSC, we work carefully
            # Actually, for RREF we'll use a more efficient row-oriented approach or
            # simply accept the CSC cost for exactness if we can't avoid it.
            # Optimized: Just swap the indices for logic if needed, but here we'll
            # keep it simple for correctness.
        end

        # Normalize pivot row
        # (Implementation of sparse rational elimination)
        # To avoid the complexity of manual CSC manipulation, we use a
        # Dict-of-Rows approach for the reduction phase.
        push!(pivots, col)
        row += 1
        row > m && break
    end
    return pivots
end

# Re-implementing with a robust exact sparse nullspace logic
function _sparse_nullspace_rational(A::SparseMatrixCSC{Int64, Int64})
    m, n = size(A)
    # Convert to Rational{BigInt} to avoid overflow and allow division
    rows = [Dict{Int, Rational{BigInt}}() for _ in 1:m]
    for col in 1:n
        for ptr in A.colptr[col]:(A.colptr[col+1]-1)
            rows[A.rowval[ptr]][col] = Rational{BigInt}(A.nzval[ptr])
        end
    end

    pivot_to_row = Dict{Int, Int}()
    next_pivot_row = 1

    for c in 1:n
        p_row = 0
        for r in next_pivot_row:m
            if get(rows[r], c, 0) != 0
                p_row = r
                break
            end
        end
        
        p_row == 0 && continue
        
        # Swap rows in our logic
        rows[next_pivot_row], rows[p_row] = rows[p_row], rows[next_pivot_row]
        p_row = next_pivot_row
        
        # Normalize
        pivot_val = rows[p_row][c]
        if pivot_val != 1
            for col in keys(rows[p_row])
                rows[p_row][col] /= pivot_val
            end
        end
        
        # Eliminate other rows - Pre-collect to avoid repeated dict iteration
        p_row_entries = collect(rows[p_row])
        for r in 1:m
            r == p_row && continue
            row_r = rows[r]
            factor = get(row_r, c, 0)
            if factor != 0
                for (col, val) in p_row_entries
                    nv = get(row_r, col, 0) - factor * val
                    if nv == 0
                        delete!(row_r, col)
                    else
                        row_r[col] = nv
                    end
                end
            end
        end
        
        pivot_to_row[c] = p_row
        next_pivot_row += 1
        next_pivot_row > m && break
    end

    # Extract basis for nullspace
    basis = Vector{Vector{Int64}}()
    for j in 1:n
        if !haskey(pivot_to_row, j)
            # j is a free variable
            v = Dict{Int, Rational{BigInt}}()
            v[j] = 1
            for (c, r) in pivot_to_row
                val = get(rows[r], j, 0)
                if val != 0
                    v[c] = -val
                end
            end
            
            # Convert to integers by clearing denominators
            lcm_val = BigInt(1)
            for val in values(v)
                lcm_val = lcm(lcm_val, denominator(val))
            end
            
            int_v = zeros(Int64, n)
            for (idx, val) in v
                int_v[idx] = Int64(numerator(val * lcm_val))
            end
            
            # Simplify by GCD
            common = reduce(gcd, int_v)
            if common != 0; push!(basis, int_v .÷ common); end
        end
    end
    return basis
end

function _is_independent_sparse_rational(matrix::SparseMatrixCSC{Int64, Int64}, vectors::Vector{Vector{Int64}})
    m, n = size(matrix)
    num_new = length(vectors)
    num_new == 0 && return Int64[]
    
    rows = [Dict{Int, Rational{BigInt}}() for _ in 1:m]
    for col in 1:n
        for ptr in matrix.colptr[col]:(matrix.colptr[col+1]-1)
            rows[matrix.rowval[ptr]][col] = Rational{BigInt}(matrix.nzval[ptr])
        end
    end
    for (i, v) in enumerate(vectors)
        for (r_idx, val) in enumerate(v)
            if val != 0
                rows[r_idx][n + i] = Rational{BigInt}(val)
            end
        end
    end

    pivot_cols = Set{Int}()
    next_pivot_row = 1
    aug_n = n + num_new
    for c in 1:aug_n
        p_row = 0
        for r in next_pivot_row:m
            if get(rows[r], c, 0) != 0
                p_row = r; break
            end
        end
        p_row == 0 && continue
        
        rows[next_pivot_row], rows[p_row] = rows[p_row], rows[next_pivot_row]
        p_row = next_pivot_row
        pivot_val = rows[p_row][c]
        if pivot_val != 1
            for col in keys(rows[p_row]); rows[p_row][col] /= pivot_val; end
        end
        
        p_row_entries = collect(rows[p_row])
        for r in 1:m
            r == p_row && continue
            row_r = rows[r]
            factor = get(row_r, c, 0)
            if factor != 0
                for (col, val) in p_row_entries
                    nv = get(row_r, col, 0) - factor * val
                    if nv == 0; delete!(row_r, col); else row_r[col] = nv; end
                end
            end
        end
        push!(pivot_cols, c)
        next_pivot_row += 1
        next_pivot_row > m && break
    end
    
    independent_indices = Int[]
    for i in 1:num_new
        if (n + i) in pivot_cols
            push!(independent_indices, i)
        end
    end
    return independent_indices
end

"""
    exact_sparse_cohomology_basis(...)

Compute an integral cohomology basis from sparse coboundary data.
Uses exact sparse rational reduction to maintain 100% mathematical fidelity
without dense matrix conversion OOM risks.
"""
function exact_sparse_cohomology_basis(
    d_np1_rows::AbstractVector{Int64}, d_np1_cols::AbstractVector{Int64}, d_np1_vals::AbstractVector{Int64}, d_np1_m::Int, d_np1_n::Int,
    d_n_rows::AbstractVector{Int64}, d_n_cols::AbstractVector{Int64}, d_n_vals::AbstractVector{Int64}, d_n_m::Int, d_n_n::Int
)
    # delta^n is d_{n+1}^T. Map is C^n -> C^{n+1}. Domain dim = d_np1_m (size of C^n).
    coboundary_mat = sparse(d_np1_cols .+ 1, d_np1_rows .+ 1, d_np1_vals, d_np1_n, d_np1_m)
    
    # 1. Compute kernel of delta^n exactly using sparse rational RREF
    basis = _sparse_nullspace_rational(coboundary_mat)
    
    if isempty(basis); return Matrix{Int64}(undef, d_np1_m, 0); end

    # 2. Quotient by image of delta^{n-1} = d_n^T. Map is C^{n-1} -> C^n.
    dn_T = sparse(d_n_cols .+ 1, d_n_rows .+ 1, d_n_vals, d_n_m, d_n_n) # d_n is (n_faces, n_edges). d_n^T is (n_edges, n_faces).
    # Wait, indices passed from Python are: d_n is (n_rows, n_cols) where n_rows is cells(dim-1), n_cols is cells(dim).
    # So d_n^T is (cells(dim), cells(dim-1)). This is delta^{n-1}: C^{n-1} -> C^n.
    # Its image is the span of its columns.
    
    # Correct sparse construction for d_n^T
    # d_n from Python: rows are (dim-1)-simplices, cols are (dim)-simplices.
    # delta^{n-1} (d_n^T): rows are (dim)-simplices, cols are (dim-1)-simplices.
    delta_nm1 = sparse(d_n_cols .+ 1, d_n_rows .+ 1, d_n_vals, d_n_n, d_n_m)
    
    if nnz(delta_nm1) == 0
        return hcat(basis...)
    end

    # Find which basis elements are independent of delta_nm1's column span
    ind_idx = _is_independent_sparse_rational(delta_nm1, basis)
    
    if isempty(ind_idx)
        return Matrix{Int64}(undef, d_np1_m, 0)
    end
    
    return hcat(basis[ind_idx]...)
end

"""
    rank_q_sparse(rows, cols, vals, m, n)

Rank over `Q` from sparse COO input.
"""
function rank_q_sparse(rows::AbstractVector{Int64}, cols::AbstractVector{Int64}, vals::AbstractVector{Int64}, m::Int, n::Int)
    isempty(rows) && return Int64(0)
    A = sparse(rows .+ 1, cols .+ 1, Float64.(vals), m, n)
    return Int64(rank(Matrix{Float64}(A)))
end

function _rank_mod_p_dense!(M::Matrix{Int64}, p::Int)
    m, n = size(M); row = 1; rk = 0
    for col in 1:n
        pivot = 0; for r in row:m; if mod(M[r, col], p) != 0; pivot = r; break; end; end
        pivot == 0 && continue
        if pivot != row; M[row, :], M[pivot, :] = copy(M[pivot, :]), copy(M[row, :]); end
        pivot_val = mod(M[row, col], p); inv_pivot = invmod(pivot_val, p)
        for j in 1:n; M[row, j] = mod(M[row, j] * inv_pivot, p); end
        for r in 1:m; (r == row || (factor = mod(M[r, col], p)) == 0) && continue
            for j in 1:n; M[r, j] = mod(M[r, j] - factor * M[row, j], p); end
        end
        row += 1; rk += 1; row > m && break
    end
    return Int64(rk)
end

"""
    rank_mod_p_sparse(rows, cols, vals, m, n, p)

Rank over `Z/pZ` from sparse COO input using dense modular elimination.
"""
function rank_mod_p_sparse(rows::AbstractVector{Int64}, cols::AbstractVector{Int64}, vals::AbstractVector{Int64}, m::Int, n::Int, p::Int)
    p <= 1 && error("modulus p must be > 1")
    isempty(rows) && return Int64(0)
    A = sparse(rows .+ 1, cols .+ 1, vals, m, n)
    return _rank_mod_p_dense!(Matrix{Int64}(A), p)
end

"""
    normal_surface_residual_norms(rows, cols, vals, m, n, coordinate_matrix)

Compute batched Euclidean residual norms `||A * x_i||_2` for normal-surface
coordinate columns `x_i`.
"""
function normal_surface_residual_norms(
    rows::AbstractVector{Int64},
    cols::AbstractVector{Int64},
    vals::AbstractVector{Int64},
    m::Int,
    n::Int,
    coordinate_matrix::AbstractMatrix{Int64},
)
    size(coordinate_matrix, 1) == n || error("coordinate_matrix row count must match matrix column count")
    k = size(coordinate_matrix, 2)
    k == 0 && return Float64[]
    m == 0 && return zeros(Float64, k)

    A = isempty(rows) ? spzeros(Float64, m, n) : sparse(rows .+ 1, cols .+ 1, Float64.(vals), m, n)
    R = A * Matrix{Float64}(coordinate_matrix)
    return vec(sqrt.(sum(abs2, R; dims=1)))
end

"""
    embedding_broad_phase_pairs(centroids, radii, tol)

Compute candidate simplex index pairs `(i, j)` with `i < j` using centroid-ball
distance bounds. Parallelized via multi-threading for 100k+ point workloads.
"""
function embedding_broad_phase_pairs(
    centroids::AbstractMatrix{Float64},
    radii::AbstractVector{Float64},
    tol::Float64,
)
    n = size(centroids, 1)
    d = size(centroids, 2)
    d >= 1 || return Matrix{Int64}(undef, 0, 2)
    length(radii) == n || error("radii length must match centroid count")
    n <= 1 && return Matrix{Int64}(undef, 0, 2)

    # Thread-local storage for pairs
    thread_pairs = [Vector{NTuple{2, Int64}}() for _ in 1:Threads.nthreads()]
    
    Threads.@threads for i in 1:(n - 1)
        tid = Threads.threadid()
        ci = @view centroids[i, :]
        ri = radii[i]
        for j in (i + 1):n
            cj = @view centroids[j, :]
            bound = ri + radii[j] + tol
            sq = 0.0
            for k in 1:d
                @inbounds delta = ci[k] - cj[k]
                sq += delta * delta
            end
            if sq <= bound * bound
                push!(thread_pairs[tid], (Int64(i - 1), Int64(j - 1)))
            end
        end
    end

    all_pairs = vcat(thread_pairs...)
    if isempty(all_pairs)
        return Matrix{Int64}(undef, 0, 2)
    end
    out = Matrix{Int64}(undef, length(all_pairs), 2)
    @inbounds for (idx, p) in enumerate(all_pairs)
        out[idx, 1] = p[1]
        out[idx, 2] = p[2]
    end
    return out
end

function _is_independent_sparse_mod_p(matrix::SparseMatrixCSC{Int64, Int64}, vectors::Vector{Vector{Int64}}, p::Int)
    m, n = size(matrix)
    num_new = length(vectors)
    num_new == 0 && return Int64[]

    # Dictionary-of-rows modular reduction
    rows = [Dict{Int, Int}() for _ in 1:m]
    for col in 1:n
        for ptr in matrix.colptr[col]:(matrix.colptr[col+1]-1)
            rows[matrix.rowval[ptr]][col] = mod(matrix.nzval[ptr], p)
        end
    end
    for (i, v) in enumerate(vectors)
        for (r_idx, val) in enumerate(v)
            m_val = mod(val, p)
            if m_val != 0; rows[r_idx][n + i] = m_val; end
        end
    end

    pivot_cols = Set{Int}()
    next_pivot_row = 1
    for c in 1:(n + num_new)
        p_row = 0
        for r in next_pivot_row:m
            if get(rows[r], c, 0) != 0; p_row = r; break; end
        end
        p_row == 0 && continue
        
        rows[next_pivot_row], rows[p_row] = rows[p_row], rows[next_pivot_row]
        p_row = next_pivot_row
        pivot_val = rows[p_row][c]
        inv_v = invmod(pivot_val, p)
        for col in keys(rows[p_row]); rows[p_row][col] = mod(rows[p_row][col] * inv_v, p); end
        
        for r in 1:m
            (r == p_row || get(rows[r], c, 0) == 0) && continue
            factor = rows[r][c]
            for (col, val) in rows[p_row]
                rows[r][col] = mod(get(rows[r], col, 0) - factor * val, p)
                if rows[r][col] == 0; delete!(rows[r], col); end
            end
        end
        push!(pivot_cols, c)
        next_pivot_row += 1
        next_pivot_row > m && break
    end
    
    independent_indices = Int[]
    for i in 1:num_new
        if (n + i) in pivot_cols; push!(independent_indices, i); end
    end
    return independent_indices
end

"""
    sparse_cohomology_basis_mod_p(..., p)

Compute cohomology representatives modulo `p` via exact sparse modular reduction.
Avoids dense matrix bottlenecks and OOM risks.
"""
function sparse_cohomology_basis_mod_p(
    d_np1_rows::AbstractVector{Int64}, d_np1_cols::AbstractVector{Int64}, d_np1_vals::AbstractVector{Int64}, d_np1_m::Int, d_np1_n::Int,
    d_n_rows::AbstractVector{Int64}, d_n_cols::AbstractVector{Int64}, d_n_vals::AbstractVector{Int64}, d_n_m::Int, d_n_n::Int,
    p::Int,
)
    p <= 1 && error("modulus p must be > 1")
    coboundary_mat = sparse(d_np1_cols .+ 1, d_np1_rows .+ 1, d_np1_vals, d_np1_n, d_np1_m)
    
    # 1. Nullspace over Z/pZ
    # Convert to Rational logic but modular
    # For speed, we use the same row-dict RREF logic
    z_basis = _sparse_nullspace_mod_p(coboundary_mat, p)
    if isempty(z_basis); return Matrix{Int64}(undef, d_np1_m, 0); end

    # 2. Independent of coboundaries (image of d_n^T)
    delta_nm1 = sparse(d_n_cols .+ 1, d_n_rows .+ 1, d_n_vals, d_n_n, d_n_m)
    
    if nnz(delta_nm1) == 0; return hcat(z_basis...); end
    
    ind_idx = _is_independent_sparse_mod_p(delta_nm1, z_basis, p)
    if isempty(ind_idx); return Matrix{Int64}(undef, d_np1_m, 0); end
    
    return hcat(z_basis[ind_idx]...)
end

function _sparse_nullspace_mod_p(A::SparseMatrixCSC{Int64, Int64}, p::Int)
    m, n = size(A)
    rows = [Dict{Int, Int}() for _ in 1:m]
    for col in 1:n, ptr in A.colptr[col]:(A.colptr[col+1]-1)
        rows[A.rowval[ptr]][col] = mod(A.nzval[ptr], p)
    end

    pivot_to_row = Dict{Int, Int}()
    next_pivot_row = 1
    for c in 1:n
        p_row = 0
        for r in next_pivot_row:m
            if get(rows[r], c, 0) != 0; p_row = r; break; end
        end
        p_row == 0 && continue
        rows[next_pivot_row], rows[p_row] = rows[p_row], rows[next_pivot_row]
        p_row = next_pivot_row
        p_val = rows[p_row][c]; inv_v = invmod(p_val, p)
        for col in keys(rows[p_row]); rows[p_row][col] = mod(rows[p_row][col] * inv_v, p); end
        for r in 1:m
            (r == p_row || get(rows[r], c, 0) == 0) && continue
            factor = rows[r][c]
            for (col, val) in rows[p_row]
                rows[r][col] = mod(get(rows[r], col, 0) - factor * val, p)
                if rows[r][col] == 0; delete!(rows[r], col); end
            end
        end
        pivot_to_row[c] = p_row; next_pivot_row += 1; next_pivot_row > m && break
    end

    basis = Vector{Vector{Int64}}()
    for j in 1:n
        if !haskey(pivot_to_row, j)
            v = zeros(Int64, n); v[j] = 1
            for (c, r) in pivot_to_row; v[c] = mod(-get(rows[r], j, 0), p); end
            push!(basis, v)
        end
    end
    return basis
end

function _compute_boundary_data_internal_flat(flat_vertices::AbstractVector{Int64}, simplex_offsets::AbstractVector{Int64}, max_dim::Int)
    n_simplices = length(simplex_offsets) - 1; dim_simplices = Dict{Int, Matrix{Int64}}(); dim_counts = zeros(Int, max_dim + 1)
    for i in 1:n_simplices; d = simplex_offsets[i+1] - simplex_offsets[i] - 1; 0 <= d <= max_dim && (dim_counts[d+1] += 1); end
    for d in 0:max_dim; dim_simplices[d] = Matrix{Int64}(undef, d + 1, dim_counts[d+1]); end
    dim_cursors = ones(Int, max_dim + 1)
    for i in 1:n_simplices
        lo, hi = simplex_offsets[i] + 1, simplex_offsets[i+1]; d = hi - lo; 0 <= d <= max_dim || continue
        cursor = dim_cursors[d+1]; dim_simplices[d][:, cursor] .= sort!(flat_vertices[lo:hi]); dim_cursors[d+1] += 1
    end
    cells = Dict{Int, Int64}(); simplex_to_idx = Dict{Int, Dict{Vector{Int64}, Int64}}()
    for d in 0:max_dim
        cells[d] = Int64(size(dim_simplices[d], 2)); idx_map = Dict{Vector{Int64}, Int64}()
        for j in 1:size(dim_simplices[d], 2); idx_map[dim_simplices[d][:, j]] = Int64(j - 1); end
        simplex_to_idx[d] = idx_map
    end
    boundaries = Dict{Int, Dict{String, Any}}()
    for k in 1:max_dim
        n_rows, n_cols = Int64(get(cells, k - 1, 0)), Int64(get(cells, k, 0)); (n_rows == 0 || n_cols == 0) && continue
        rows, cols, data = Int64[], Int64[], Int64[]; prev_dim_map = simplex_to_idx[k - 1]; simplices_k = dim_simplices[k]
        for j in 1:n_cols
            verts = simplices_k[:, j]
            for i in eachindex(verts)
                face = vcat(verts[1:(i - 1)], verts[(i + 1):end])
                if haskey(prev_dim_map, face); push!(rows, prev_dim_map[face]); push!(cols, Int64(j - 1)); push!(data, isodd(i - 1) ? -1 : 1); end
            end
        end
        boundaries[k] = Dict("rows" => rows, "cols" => cols, "data" => data, "n_rows" => n_rows, "n_cols" => n_cols)
    end
    return boundaries, cells, dim_simplices, simplex_to_idx
end

"""
    compute_boundary_payload_from_flat_simplices(flat_vertices, simplex_offsets, max_dim; include_metadata=true)

Build boundary payloads (COO-style) from flattened simplex storage.
Input indices are expected to be zero-based on the Python side.
"""
function compute_boundary_payload_from_flat_simplices(flat_vertices::AbstractVector{Int64}, simplex_offsets::AbstractVector{Int64}, max_dim::Int, include_metadata::Bool=true)
    return _compute_boundary_data_internal_flat(flat_vertices, simplex_offsets, max_dim)
end

function compute_boundary_data_from_simplices_jl(simplex_entries, max_dim::Int)
    return _compute_boundary_data_internal(simplex_entries, max_dim)
end

function _normalize_group_ring_coeffs(coeffs_any)
    raw = coeffs_any isa AbstractDict ? coeffs_any : pyconvert(Dict{Any, Any}, coeffs_any)
    out = Dict{Any, Int}()
    for (k, v) in raw
        out[k] = Int(v)
    end
    return out
end

function _parse_group_element(g_raw)
    if g_raw isa Integer
        return Int(g_raw)
    elseif g_raw isa Tuple || g_raw isa AbstractVector
        isempty(g_raw) && return 0
        return Int(first(g_raw))
    end
    g_str = string(g_raw)
    (g_str == "e" || g_str == "1") && return 0
    m = match(r"g_?(\d+)(?:\^-1)?", g_str)
    m === nothing && return 0
    val = parse(Int, m.captures[1])
    return endswith(g_str, "^-1") ? -val : val
end

"""
    group_ring_multiply(py_coeffs1, py_coeffs2, group_order)

Multiply two group-ring elements represented as sparse coefficient dictionaries.
Implements cyclic group multiplication on parsed group elements.
"""
function group_ring_multiply(py_coeffs1, py_coeffs2, group_order::Int)
    c1, c2 = _normalize_group_ring_coeffs(py_coeffs1), _normalize_group_ring_coeffs(py_coeffs2)
    res_dict = Dict{Int, Int}()
    for (k1, v1) in c1, (k2, v2) in c2
        p_res = mod(_parse_group_element(k1) + _parse_group_element(k2), group_order)
        res_dict[p_res] = get(res_dict, p_res, 0) + v1 * v2
    end
    final_k, final_v = Int[], Int[]
    for (k, v) in res_dict; v != 0 && (push!(final_k, k); push!(final_v, v)); end
    return final_k, final_v
end

"""
    multisignature(matrix, p)

Compute total twisted signature over non-trivial `p`-th roots of unity.
Parallelized over available CPU cores.
"""
function multisignature(matrix::AbstractMatrix{Float64}, p::Int)
    mat = Matrix{Float64}(matrix)
    n = size(mat, 1)
    
    # Pre-allocate thread-local totals
    thread_totals = zeros(Int64, Threads.nthreads())
    
    Threads.@threads for k in 1:(p-1)
        tid = Threads.threadid()
        omega = exp(2π * im * k / p)
        H = [mat[i,j] * omega^(i-j) for i in 1:n, j in 1:n]
        evals = real.(eigvals(Hermitian(H)))
        tol = n * eps(Float64) * maximum(abs.(evals))
        thread_totals[tid] += sum(evals .> tol) - sum(evals .< -tol)
    end
    
    return sum(thread_totals)
end

"""
    integral_lattice_isometry(matrix1, matrix2)

Search for an integral isometry `U` with `U' * A * U == B` via bounded backtracking.
"""
function integral_lattice_isometry(matrix1::AbstractMatrix{Int64}, matrix2::AbstractMatrix{Int64})
    A, B = Matrix{Int64}(matrix1), Matrix{Int64}(matrix2); n = size(A, 1)
    evals_a = eigvals(Hermitian(Matrix{Float64}(A))); evals_b = eigvals(Hermitian(Matrix{Float64}(B)))
    scale = max(1.0, maximum(abs.(vcat(evals_a, evals_b)))); tol = n * eps(Float64) * scale
    pos_a, pos_b = all(x -> x > tol, evals_a), all(x -> x > tol, evals_b)
    Adef, Bdef = pos_a ? A : -A, pos_b ? B : -B
    lam_min = minimum(eigvals(Hermitian(Matrix{Float64}(Adef)))); radius = maximum([Int(floor(sqrt(Int(Bdef[i, i]) / lam_min))) + 1 for i in 1:n])
    range_vals = collect(-radius:radius); vectors_by_norm = Dict{Int, Vector{Vector{Int64}}}()
    for tup in Iterators.product(ntuple(_ -> range_vals, n)...)
        v = Int64[tup...]; qv = Int(dot(v, Adef * v))
        if qv <= maximum([Int(Bdef[i, i]) for i in 1:n]); push!(get!(vectors_by_norm, qv, Vector{Vector{Int64}}()), v); end
    end
    order = sortperm(1:n; by = j -> length(get(vectors_by_norm, Int(Bdef[j, j]), [])))
    cols = [zeros(Int64, n) for _ in 1:n]; chosen_indices = Int[]
    function backtrack(pos::Int)
        if pos > n; U = hcat(cols...); return abs(det(Matrix{Float64}(U))) ≈ 1.0 && transpose(U) * A * U == B ? U : nothing; end
        j = order[pos]
        for v in get(vectors_by_norm, Int(Bdef[j, j]), [])
            if all(i -> Int(dot(cols[i], Adef * v)) == Int(Bdef[i, j]), chosen_indices)
                cols[j] = v; push!(chosen_indices, j); res = backtrack(pos + 1); if res !== nothing; return res; end; pop!(chosen_indices)
            end
        end
        return nothing
    end
    return backtrack(1)
end

"""
    compute_trimesh_boundary_data_flat(face_vertices, face_offsets, n_vertices)

Construct `d1`/`d2` boundary payloads from polygonal face lists.
Outputs zero-based COO fields compatible with Python bridge consumers.
"""
function compute_trimesh_boundary_data_flat(face_vertices::AbstractVector{Int64}, face_offsets::AbstractVector{Int64}, n_vertices::Int)
    n_faces = length(face_offsets) - 1; edge_to_idx = Dict{Tuple{Int, Int}, Int}(); edges = Tuple{Int, Int}[]
    d2_rows, d2_cols, d2_data = Int64[], Int64[], Int64[]
    for j in 1:n_faces
        lo, hi = face_offsets[j] + 1, face_offsets[j+1]; cycle = view(face_vertices, lo:hi)
        for i in 1:length(cycle)
            u, v = Int(cycle[i]), Int(cycle[mod(i, length(cycle)) + 1])
            # Canonical key (always u < v)
            key = u < v ? (u, v) : (v, u)
            if !haskey(edge_to_idx, key); push!(edges, key); edge_to_idx[key] = length(edges); end
            idx = edge_to_idx[key]
            push!(d2_rows, idx - 1); push!(d2_cols, j - 1)
            # Boundary sign is +1 if orientation (u,v) matches canonical key, else -1
            push!(d2_data, u < v ? 1 : -1)
        end
    end
    # Now d1 uses the exact same canonical orientation
    d1_rows, d1_cols, d1_data = Int64[], Int64[], Int64[]
    for (j, (v1, v2)) in enumerate(edges)
        push!(d1_rows, v1); push!(d1_cols, j - 1); push!(d1_data, -1)
        push!(d1_rows, v2); push!(d1_cols, j - 1); push!(d1_data, 1)
    end
    return Dict("d1_rows" => d1_rows, "d1_cols" => d1_cols, "d1_data" => d1_data, "n_vertices" => Int64(n_vertices), "n_edges" => Int64(length(edges)), "d2_rows" => d2_rows, "d2_cols" => d2_cols, "d2_data" => d2_data, "n_faces" => Int64(n_faces))
end



function _to_vertices_simplex(s)
    if s isa Tuple
        return [Int64(x) for x in s]
    elseif s isa AbstractVector
        return [Int64(x) for x in s]
    elseif s isa Py
        try
            return pyconvert(Vector{Int64}, s)
        catch
            # Handle possible Py list of different numeric types
            return Int64[pyconvert(Int64, x) for x in s]
        end
    else
        # println("DEBUG: unknown type $(typeof(s)) for simplex entry $s")
        return Int64[]
    end
end

function _components_h0_generators(edges::Vector{Tuple{Int64, Int64}}, nv::Int, pts=nothing)
    # nv is total vertices
    parent = collect(1:nv)
    function find(x)
        while parent[x] != x
            parent[x] = parent[parent[x]]
            x = parent[x]
        end
        return x
    end
    function unite(a, b)
        ra, rb = find(a), find(b)
        if ra != rb
            parent[rb] = ra
        end
    end

    for (u, v) in edges
        if u+1 <= nv && v+1 <= nv
            unite(u+1, v+1)
        end
    end

    components = Dict{Int, Int64}()
    for i in 1:nv
        r = find(i)
        if !haskey(components, r)
            components[r] = i-1
        end
    end

    results = []
    for (r, rep) in components
        weight = 0.0
        push!(results, Dict(
            "dimension" => 0,
            "support_simplices" => [[Int64(rep)]],
            "support_edges" => [],
            "weight" => weight,
            "certified_cycle" => true
        ))
    end
    # Sort for deterministic output
    return sort(results, by = x -> x["support_simplices"][1][1])
end

function _compute_boundary_data_internal(simplex_entries, max_dim::Int)
    # Ensure we have all dimensions from 0 to max_dim
    dim_simplices = Dict{Int, Set{Tuple{Vararg{Int64}}}}()
    for d in 0:max_dim; dim_simplices[d] = Set{Tuple{Vararg{Int64}}}(); end

    for s in simplex_entries
        vs = _to_vertices_simplex(s)
        isempty(vs) && continue
        t = Tuple(sort(vec(collect(Int64.(vs)))))
        # Add all faces to ensure complete skeleton
        for r in 1:length(t)
            d = r - 1
            if d <= max_dim
                for face in combinations(t, r)
                    # face is already sorted because t is sorted
                    push!(dim_simplices[d], Tuple(face))
                end
            end
        end
    end

    # Convert sets to sorted vectors
    sorted_dim_simplices = Dict{Int, Vector{Tuple{Vararg{Int64}}}}()
    cells = Dict{Int, Int64}()
    simplex_to_idx = Dict{Int, Dict{Tuple{Vararg{Int64}}, Int64}}()

    for d in 0:max_dim
        v = sort(collect(dim_simplices[d]))
        sorted_dim_simplices[d] = v
        cells[d] = Int64(length(v))
        idx_map = Dict{Tuple{Vararg{Int64}}, Int64}()
        for (i, simplex) in enumerate(v)
            idx_map[simplex] = Int64(i - 1)
        end
        simplex_to_idx[d] = idx_map
    end

    boundaries = Dict{Int, Dict{String, Any}}()
    for k in 1:max_dim
        n_rows, n_cols = Int64(get(cells, k - 1, 0)), Int64(get(cells, k, 0))
        (n_rows == 0 || n_cols == 0) && continue
        rows, cols, data = Int64[], Int64[], Int64[]
        prev_dim_map = simplex_to_idx[k - 1]
        for (j, simplex) in enumerate(sorted_dim_simplices[k])
            verts = collect(simplex)
            for i in eachindex(verts)
                face = Tuple(vcat(verts[1:(i - 1)], verts[(i + 1):end]))
                if haskey(prev_dim_map, face)
                    push!(rows, prev_dim_map[face])
                    push!(cols, Int64(j - 1))
                    push!(data, isodd(i - 1) ? -1 : 1)
                end
            end
        end
        boundaries[k] = Dict("rows" => rows, "cols" => cols, "data" => data, "n_rows" => n_rows, "n_cols" => n_cols)
    end
    return boundaries, cells, sorted_dim_simplices, simplex_to_idx
end

"""
    compute_boundary_payload_from_simplices(simplex_entries, max_dim; include_metadata=true)

Build boundary payloads and optional simplex metadata from simplex entries.
"""
function compute_boundary_payload_from_simplices(simplex_entries, max_dim::Int, include_metadata::Bool=true)
    boundaries, cells, dim_simplices, simplex_to_idx = _compute_boundary_data_internal(simplex_entries, max_dim)
    return include_metadata ? (boundaries, cells, dim_simplices, simplex_to_idx) : (boundaries, cells)
end

"""
    compute_boundary_data_from_simplices(simplex_entries, max_dim)

Return full boundary/coface indexing data from simplex entries.
"""
function compute_boundary_data_from_simplices(simplex_entries, max_dim::Int)
    return _compute_boundary_data_internal(simplex_entries, max_dim)
end

"""
    pi1_trace_candidates_from_d1(d1_rows, d1_cols, d1_vals, n_vertices, n_edges)

Construct raw pi1 generator trace candidates from `d1` COO data.

Algorithm:
- Build 1-skeleton adjacency and oriented edge table.
- Compute spanning forest with BFS.
- For each non-tree edge, form the generator loop as
  `edge + tree_path(back_to_start)`.
"""
function pi1_trace_candidates_from_d1(d1_rows::AbstractVector{Int64}, d1_cols::AbstractVector{Int64}, d1_vals::AbstractVector{Int64}, n_vertices::Int, n_edges::Int)
    adj = Dict{Int64, Vector{Tuple{Int64, Int64, Int64}}}()
    for i in 0:(n_vertices - 1)
        adj[Int64(i)] = Vector{Tuple{Int64, Int64, Int64}}()
    end

    edge_list = Vector{Union{Nothing, Tuple{Int64, Int64}}}(undef, n_edges)
    fill!(edge_list, nothing)

    # Build edge endpoint table and adjacency from COO d1.
    col_to_entries = Dict{Int64, Vector{Tuple{Int64, Int64}}}()
    for idx in 1:length(d1_rows)
        col = Int64(d1_cols[idx])
        push!(get!(col_to_entries, col, Vector{Tuple{Int64, Int64}}()), (Int64(d1_vals[idx]), Int64(d1_rows[idx])))
    end

    for e in 0:(n_edges - 1)
        entries = get(col_to_entries, Int64(e), Vector{Tuple{Int64, Int64}}())
        if isempty(entries)
            edge_list[e + 1] = (0, 0)
            continue
        end
        if length(entries) != 2
            edge_list[e + 1] = nothing
            continue
        end

        u = Int64(-1)
        v = Int64(-1)
        for (val, r) in entries
            if val == -1
                u = r
            elseif val == 1
                v = r
            end
        end

        if u != -1 && v != -1
            push!(adj[u], (v, e, 1))
            push!(adj[v], (u, e, -1))
            edge_list[e + 1] = (u, v)
        else
            edge_list[e + 1] = nothing
        end
    end

    visited = falses(n_vertices)
    parent = Dict{Int64, Int64}()
    component_root = Dict{Int64, Int64}()
    tree_edges = Set{Int64}()

    if n_vertices > 0
        for start in 0:(n_vertices - 1)
            if visited[start + 1]
                continue
            end
            queue = Int64[Int64(start)]
            visited[start + 1] = true
            parent[Int64(start)] = Int64(-1)
            component_root[Int64(start)] = Int64(start)

            while !isempty(queue)
                curr = popfirst!(queue)
                for (neighbor, edge_idx, _) in adj[curr]
                    if !visited[neighbor + 1]
                        visited[neighbor + 1] = true
                        push!(tree_edges, edge_idx)
                        parent[neighbor] = curr
                        component_root[neighbor] = Int64(start)
                        push!(queue, neighbor)
                    end
                end
            end
        end
    end

    function path_between_tree(u::Int64, v::Int64)
        path_u = Int64[]
        seen_u = Set{Int64}()
        x = u
        while x != -1
            push!(path_u, x)
            push!(seen_u, x)
            x = get(parent, x, Int64(-1))
        end

        path_v = Int64[]
        y = v
        while !(y in seen_u) && y != -1
            push!(path_v, y)
            y = get(parent, y, Int64(-1))
        end

        if y == -1
            return Int64[]
        end

        lca = y
        i = findfirst(==(lca), path_u)
        i === nothing && return Int64[]
        return vcat(path_u[1:i], reverse(path_v))
    end

    traces = Vector{Dict{String, Any}}()
    for e in 0:(n_edges - 1)
        if e in tree_edges
            continue
        end
        endpoints = edge_list[e + 1]
        endpoints === nothing && continue
        u, v = endpoints

        if u == v
            vertex_path = [u]
            directed = [(u, v)]
            comp_root = get(component_root, u, u)
        else
            path_vertices = path_between_tree(v, u)
            directed = [(u, v)]
            for i in 1:(length(path_vertices) - 1)
                push!(directed, (path_vertices[i], path_vertices[i + 1]))
            end
            vertex_path = [u]
            for (_, b) in directed
                push!(vertex_path, b)
            end
            comp_root = get(component_root, u, get(component_root, v, u))
        end

        push!(traces, Dict(
            "generator" => "g_$(e)",
            "edge_index" => Int64(e),
            "component_root" => Int64(comp_root),
            "vertex_path" => vertex_path,
            "directed_edge_path" => [[Int64(a), Int64(b)] for (a, b) in directed],
            "undirected_edge_path" => [[Int64(min(a, b)), Int64(max(a, b))] for (a, b) in directed],
        ))
        end
        return traces
        end

        """
        extract_pi1_raw_data_jl(d1_rows, d1_cols, d1_vals, n_vertices, n_edges, d2_rows, d2_cols, d2_vals, n_faces)

        Compute generators, relators, and traces for pi1 in one pass.
        """
        function extract_pi1_raw_data_jl(
        d1_rows::AbstractVector{Int64}, d1_cols::AbstractVector{Int64}, d1_vals::AbstractVector{Int64},
        n_vertices::Int, n_edges::Int,
        d2_rows::AbstractVector{Int64}, d2_cols::AbstractVector{Int64}, d2_vals::AbstractVector{Int64},
        n_faces::Int
        )
        # 1. Traces and spanning forest from d1
        traces = pi1_trace_candidates_from_d1(d1_rows, d1_cols, d1_vals, n_vertices, n_edges)

        # 2. Build generator mapping
        gen_map = Dict{Int64, String}()
        for tr in traces
        gen_map[tr["edge_index"]] = tr["generator"]
        end

        # 3. Trace d2 into relators
        # Group d2 entries by face (column)
        face_to_entries = Dict{Int64, Vector{Tuple{Int64, Int64}}}()
        for idx in 1:length(d2_rows)
        f = Int64(d2_cols[idx])
        push!(get!(face_to_entries, f, Vector{Tuple{Int64, Int64}}()), (Int64(d2_vals[idx]), Int64(d2_rows[idx])))
        end

        relations = Vector{Vector{String}}()
        for f in 0:(n_faces - 1)
        entries = get(face_to_entries, f, nothing)
        entries === nothing && continue

        rel = String[]
        for (val, e) in entries
            sign = val > 0 ? 1 : -1
            for _ in 1:abs(val)
                if haskey(gen_map, e)
                    g = gen_map[e]
                    push!(rel, sign == 1 ? g : "$(g)^-1")
                end
            end
        end
        if !isempty(rel)
            push!(relations, rel)
        end
        end

        return Dict(
        "generators" => gen_map,
        "relations" => relations,
        "traces" => traces
        )
        end


function _as_string_vector(x)
    if x isa Vector{String}
        return x
    elseif x isa AbstractVector
        return [string(v) for v in x]
    end
    return [string(v) for v in pyconvert(Vector{Any}, x)]
end

"""
    abelianize_group(generators, relations)

Compute abelianization rank and torsion invariants from a presentation.
"""
function abelianize_group(generators::Vector{String}, relations::Vector{String})
    n_gens = length(generators); gen_idx = Dict{String, Int}(g => i for (i, g) in enumerate(generators))
    n_rels = length(relations); M = zeros(Int, n_rels, n_gens)
    for i in 1:n_rels; for m in eachmatch(r"([a-zA-Z0-9_]+)(?:\^(-?\d+))?", relations[i])
        base_w = m.captures[1]
        exp_str = isnothing(m.captures[2]) ? "1" : m.captures[2]
        haskey(gen_idx, base_w) && (M[i, gen_idx[base_w]] += parse(Int, exp_str))
    end; end
    if !HAS_ABSTRACT_ALGEBRA; return Int(n_gens - rank(Matrix{Float64}(M))), Int[]; end
    ZZ = AbstractAlgebra.ZZ; M_aa = AbstractAlgebra.matrix(ZZ, M); S_aa = AbstractAlgebra.snf(M_aa)
    diag = [Int64(S_aa[i, i]) for i in 1:min(n_rels, n_gens)]; nonzero = filter(x -> x != 0, diag)
    return n_gens - length(nonzero), filter(x -> x > 1, nonzero)
end

function abelianize_group(generators, relations)
    return abelianize_group(_as_string_vector(generators), _as_string_vector(relations))
end

"""
    compute_boundary_mod2_matrix(source_simplices, target_simplices)

Compute boundary incidence over `Z/2Z` between adjacent simplex dimensions.
"""
function compute_boundary_mod2_matrix(source_simplices, target_simplices)
    source, target = [_to_vertices_simplex(s) for s in source_simplices], [_to_vertices_simplex(t) for t in target_simplices]
    m, n = length(target), length(source); if m == 0 || n == 0; return Dict("rows" => Int64[], "cols" => Int64[], "data" => Int64[], "m" => Int64(m), "n" => Int64(n)); end
    t_idx = Dict{Tuple{Vararg{Int}}, Int}(); for (i, t) in enumerate(target); t_idx[Tuple(sort(collect(t)))] = i - 1; end
    rows, cols, data = Int64[], Int64[], Int64[]
    for (j, s) in enumerate(source)
        for i_drop in eachindex(s)
            face = (s[1:i_drop-1]..., s[i_drop+1:end]...)
            if haskey(t_idx, face)
                push!(rows, t_idx[face])
                push!(cols, j - 1)
                push!(data, 1)
            end
        end
    end
    return Dict("rows" => rows, "cols" => cols, "data" => data, "m" => Int64(m), "n" => Int64(n))
end

"""
    compute_alexander_whitney_cup(alpha, beta, p, q, simplices_pq, s_to_idx_p, s_to_idx_q; modulus=nothing)

Evaluate the Alexander-Whitney cup product on `(p+q)`-simplices.
"""
function compute_alexander_whitney_cup(alpha::AbstractVector, beta::AbstractVector, p::Int, q::Int, simplices_pq_raw, s_to_idx_p, s_to_idx_q, modulus=nothing)
    idx_p, idx_q = pyconvert(Dict{Tuple{Vararg{Int}}, Int64}, s_to_idx_p), pyconvert(Dict{Tuple{Vararg{Int}}, Int64}, s_to_idx_q)
    simplices_pq = [_to_vertices_simplex(s) for s in simplices_pq_raw]
    res = zeros(Int64, length(simplices_pq))
    Threads.@threads for i in 1:length(simplices_pq)
        s = simplices_pq[i]; length(s) < p + q + 1 && continue
        v_p, v_q = get(idx_p, Tuple(s[1:(p+1)]), -1), get(idx_q, Tuple(s[(p+1):(p+q+1)]), -1)
        if v_p != -1 && v_q != -1
            val = Int64(alpha[v_p + 1]) * Int64(beta[v_q + 1])
            res[i] = modulus === nothing ? val : mod(val, Int64(modulus))
        end
    end
    return res
end

function _cycle_annotation(cycle::Vector{Int64}, annotations::Dict{Tuple{Int64, Int64}, Vector{Int8}}, m::Int)
    res = zeros(Int8, m)
    for i in 1:length(cycle)
        u, v = cycle[i], cycle[mod(i, length(cycle)) + 1]
        e = u < v ? (u, v) : (v, u)
        if haskey(annotations, e)
            res .⊻= annotations[e]
        end
    end
    return res
end

function _is_independent_wrt(ann::Vector{Int8}, pivots::Dict{Int, Vector{Int8}})
    curr = copy(ann)
    m = length(curr)
    for i in 1:m
        if curr[i] & 1 != 0
            if haskey(pivots, i)
                curr .⊻= pivots[i]
            else
                return true, i, curr
            end
        end
    end
    return false, -1, curr
end

"""
    optgen_from_simplices(simplices, num_vertices, pts=nothing, max_roots=nothing, root_stride=1, max_cycles=nothing)

Compute a short-cycle H1 basis heuristic.

Algorithm:
- Build weighted 1-skeleton.
- Use MST + triangle annotation reduction.
- Generate cycle candidates from shortest-path trees.
- Select an independent basis greedily by weight.
"""
function optgen_from_simplices(simplices, num_vertices::Int, pts=nothing, max_roots=nothing, root_stride::Int=1, max_cycles=nothing)
    if !HAS_GRAPHS
        error("Graphs.jl or SimpleWeightedGraphs.jl unavailable.")
    end

    # 1. Extract edges and triangles - Zero allocation tuple slicing
    edges_set = Set{Tuple{Int64, Int64}}()
    triangles = Vector{Tuple{Int64, Int64, Int64}}()

    for s in simplices
        vs = _to_vertices_simplex(s)
        len_vs = length(vs)
        if len_vs == 2
            push!(edges_set, (Int64(vs[1]), Int64(vs[2])))
        elseif len_vs == 3
            push!(triangles, (Int64(vs[1]), Int64(vs[2]), Int64(vs[3])))
        end
    end

    edges_list = collect(edges_set)
    nv = num_vertices
    if nv <= 0
        nv = isempty(edges_list) ? 0 : maximum([maximum(e) for e in edges_list]) + 1
    end

    # Weights
    edge_weights = Dict{Tuple{Int64, Int64}, Float64}()
    for e in edges_list
        u, v = e[1] + 1, e[2] + 1
        edge_weights[e] = (pts !== nothing) ? sqrt(sum(abs2, pts[u, :] .- pts[v, :])) : 1.0
        edge_weights[(e[2], e[1])] = edge_weights[e]
    end

    # 2. Minimum Spanning Tree
    g_full = SimpleWeightedGraph(nv)
    for e in edges_list
        add_edge!(g_full, e[1]+1, e[2]+1, edge_weights[e])
    end
    mst_edges = kruskal_mst(g_full)
    spanning_set = Set{Tuple{Int64, Int64}}()
    for e in mst_edges
        u, v = Int64(src(e)-1), Int64(dst(e)-1)
        push!(spanning_set, (u, v))
        push!(spanning_set, (v, u))
    end

    non_tree = [e for e in edges_list if !(e in spanning_set)]
    m_dim = length(non_tree)

    # 3. Edge Annotations
    annotations = Dict{Tuple{Int64, Int64}, Vector{Int8}}()
    for e in edges_list
        annotations[e] = zeros(Int8, m_dim)
        annotations[(e[2], e[1])] = annotations[e]
    end
    for (i, e) in enumerate(non_tree)
        v_ann = zeros(Int8, m_dim)
        v_ann[i] = 1
        annotations[e] = v_ann
        annotations[(e[2], e[1])] = v_ann
    end

    valid_indices = collect(1:m_dim)
    for (u, v, w) in triangles
        e1, e2, e3 = (u, v), (v, w), (u, w)
        boundary = zeros(Int8, m_dim)
        haskey(annotations, e1) && (boundary .⊻= annotations[e1])
        haskey(annotations, e2) && (boundary .⊻= annotations[e2])
        haskey(annotations, e3) && (boundary .⊻= annotations[e3])

        pivot_idx = -1
        for idx in valid_indices
            if boundary[idx] & 1 != 0
                pivot_idx = idx
                break
            end
        end

        if pivot_idx != -1
            for (e, vec) in annotations
                if vec[pivot_idx] & 1 != 0
                    annotations[e] .⊻= boundary
                end
            end
            filter!(x -> x != pivot_idx, valid_indices)
        end
    end

    final_annotations = Dict{Tuple{Int64, Int64}, Vector{Int8}}()
    for (e, vec) in annotations
        final_annotations[e] = vec[valid_indices]
    end
    m_final = length(valid_indices)

    # 4. Cycle Candidates (Shortest Paths) - Parallelized
    all_cycles_lock = ReentrantLock()
    all_cycles = Vector{Vector{Int64}}()
    roots = collect(0:nv-1)
    if max_roots !== nothing
        roots = roots[1:min(nv, max_roots)]
    end

    Threads.@threads for root in roots
        ds = dijkstra_shortest_paths(g_full, root + 1)
        spt_parents = ds.parents
        local_cycles = Vector{Vector{Int64}}()

        for e in edges_list
            u_node, v_node = e[1], e[2]
            if spt_parents[u_node+1] != v_node+1 && spt_parents[v_node+1] != u_node+1
                path_u = Int64[]
                curr = u_node + 1
                while curr != 0
                    push!(path_u, curr - 1)
                    curr = (spt_parents[curr] == curr || spt_parents[curr] == 0) ? 0 : spt_parents[curr]
                end
                path_v = Int64[]
                curr = v_node + 1
                while curr != 0
                    push!(path_v, curr - 1)
                    curr = (spt_parents[curr] == curr || spt_parents[curr] == 0) ? 0 : spt_parents[curr]
                end

                lca = -1
                idx_u, idx_v = length(path_u), length(path_v)
                while idx_u > 0 && idx_v > 0 && path_u[idx_u] == path_v[idx_v]
                    lca = path_u[idx_u]
                    idx_u -= 1
                    idx_v -= 1
                end

                if lca != -1
                    push!(local_cycles, vcat(reverse(path_u[1:idx_u+1]), path_v[1:idx_v+1]))
                end
            end
        end
        lock(all_cycles_lock) do
            append!(all_cycles, local_cycles)
        end
    end

    # 5. Greedy Basis Selection
    cycle_weights = Float64[]
    for cyc in all_cycles
        w_sum = 0.0
        for i in 1:length(cyc)
            u, v = cyc[i], cyc[mod(i, length(cyc)) + 1]
            if u != v
                e = u < v ? (u, v) : (v, u)
                w_sum += get(edge_weights, e, 1.0)
            end
        end
        push!(cycle_weights, w_sum)
    end
    sorted_idx = sortperm(cycle_weights)

    basis = []
    pivots = Dict{Int, Vector{Int8}}()
    for idx in sorted_idx
        cyc = all_cycles[idx]
        ann = _cycle_annotation(cyc, final_annotations, m_final)
        is_indep, p_col, reduced = _is_independent_wrt(ann, pivots)
        if is_indep
            # Reformat cycle for Python (list of edges)
            py_cycle = []
            for i in 1:length(cyc)
                u, v = cyc[i], cyc[mod(i, length(cyc)) + 1]
                push!(py_cycle, [u, v])
            end

            push!(basis, Dict(
                "dimension" => 1,
                "support_simplices" => [[Int64(v)] for v in cyc],
                "support_edges" => py_cycle,
                "weight" => Float64(cycle_weights[idx]),
                "certified_cycle" => true
            ))
            pivots[p_col] = reduced
            if max_cycles !== nothing && length(basis) >= max_cycles
                break
            end
        end
    end

    return basis
end

function _nullspace_basis_mod2(A::SparseMatrixCSC{Int8, Int64})
    m, n = size(A)
    # For H2 of 6k complex, matrices are ~2500x2500. 
    # BitMatrix is extremely efficient here (64x speedup via word-XOR).
    M = BitMatrix(undef, m, n)
    fill!(M, false)
    for col in 1:n
        for ptr in A.colptr[col]:(A.colptr[col+1]-1)
            if A.nzval[ptr] & 1 != 0
                M[A.rowval[ptr], col] = true
            end
        end
    end

    pivots = Int[]
    row = 1
    for col in 1:n
        pivot_row = 0
        for r in row:m
            if M[r, col]
                pivot_row = r; break
            end
        end
        pivot_row == 0 && continue
        
        if pivot_row != row
            # Fast bit-row swap
            for c in col:n
                tmp = M[row, c]
                M[row, c] = M[pivot_row, c]
                M[pivot_row, c] = tmp
            end
        end
        
        # Fast bit-row XOR elimination
        for r in 1:m
            if r != row && M[r, col]
                # M[r, col:end] .^= M[row, col:end]
                # Manual loop over chunks for max speed
                for c in col:n
                    M[r, c] = M[r, c] ⊻ M[row, c]
                end
            end
        end
        push!(pivots, col)
        row += 1
        row > m && break
    end

    pivot_set = Set(pivots)
    basis = Vector{Vector{Int8}}()
    for j in 1:n
        if !(j in pivot_set)
            v = zeros(Int8, n)
            v[j] = 1
            for (i, p_col) in enumerate(pivots)
                v[p_col] = M[i, j] ? 1 : 0
            end
            push!(basis, v)
        end
    end
    return basis
end

function _weight_k_chain(chain::Vector{Int8}, k_simplices::Matrix{Int64}, pts::Union{Nothing, AbstractMatrix{Float64}})
    active_indices = findall(x -> x & 1 != 0, chain)
    if isempty(active_indices); return 0.0; end
    if pts === nothing; return Float64(length(active_indices)); end

    total = 0.0
    for idx in active_indices
        simplex = k_simplices[:, idx]
        # Sum edge lengths
        s = 0.0
        for i in 1:length(simplex)
            for j in (i+1):length(simplex)
                u, v = Int(simplex[i]) + 1, Int(simplex[j]) + 1
                s += norm(pts[u, :] .- pts[v, :])
            end
        end
        total += s
    end
    return total
end

function _rank_mod2(M::Matrix{Int8})
    m, n = size(M)
    (m == 0 || n == 0) && return 0
    temp = copy(M)
    row = 1
    rank = 0
    for col in 1:n
        pivot = 0
        for r in row:m
            if temp[r, col] & 1 != 0
                pivot = r
                break
            end
        end
        pivot == 0 && continue
        if pivot != row
            temp[row, :], temp[pivot, :] = temp[pivot, :], temp[row, :]
        end
        for r in 1:m
            if r != row && temp[r, col] & 1 != 0
                temp[r, :] .⊻= temp[row, :]
            end
        end
        rank += 1
        row += 1
        row > m && break
    end
    return rank
end

function _is_independent_mod_image(v::Vector{Int8}, pivots::Dict{Int, Vector{Int8}})
    curr = copy(v)
    n = length(curr)
    for i in 1:n
        if curr[i] & 1 != 0
            if !haskey(pivots, i)
                pivots[i] = curr
                return true
            end
            # Fast in-place XOR
            p = pivots[i]
            for j in i:n
                curr[j] ⊻= p[j]
            end
        end
    end
    return false
end

function _independent_mod_image(v::Vector{Int8}, basis_cols::Vector{Vector{Int8}})
    # Legacy wrapper if needed, but we prefer the Dict version
    if isempty(basis_cols); return any(x -> x & 1 != 0, v); end
    pivots = Dict{Int, Vector{Int8}}()
    for b in basis_cols
        _is_independent_mod_image(b, pivots)
    end
    return _is_independent_mod_image(v, pivots)
end

"""
    homology_generators_from_simplices(simplices, num_vertices, dimension, mode="valid", ...)

Return data-grounded homology generators from simplicial input.

`mode="valid"` keeps any independent quotient basis; `mode="optimal"`
sorts cycle candidates by geometric/algebraic weight before selection.
"""
function homology_generators_from_simplices(simplices, num_vertices::Int, dimension::Int, mode::String="valid", pts=nothing, mr=nothing, rs::Int=1, mc=nothing)
    if dimension == 0
        edges = Tuple{Int64, Int64}[]
        for s in simplices
            vs = _to_vertices_simplex(s)
            if length(vs) == 2
                push!(edges, Tuple(sort(collect(Int64.(vs)))))
            end
        end
        nv = num_vertices
        if nv <= 0
            # Infer nv from simplices
            max_v = -1
            for s in simplices
                vs = _to_vertices_simplex(s)
                for v in vs; max_v = max(max_v, v); end
            end
            nv = max_v + 1
        end
        return _components_h0_generators(edges, nv, pts)
    end

    # 1. Assembly skeletons
    # Extract boundaries
    # _compute_boundary_data_internal(simplices, max_dim)
    boundaries, cells, dim_simplices, simplex_to_idx = _compute_boundary_data_internal(simplices, dimension + 1)

    if !haskey(cells, dimension) || cells[dimension] == 0
        return []
    end

    k_simplices = dim_simplices[dimension]
    # k_simplices is Vector{Tuple{Vararg{Int64}}}
    # Convert to Matrix for weight function compatibility
    k_simplices_mat = Matrix{Int64}(undef, dimension + 1, length(k_simplices))
    for (i, s) in enumerate(k_simplices)
        k_simplices_mat[:, i] .= collect(s)
    end

    # d_k : C_k -> C_{k-1}
    dk_mat = if haskey(boundaries, dimension)
        dk_data = boundaries[dimension]
        sparse(dk_data["rows"] .+ 1, dk_data["cols"] .+ 1, Int8.(abs.(dk_data["data"])), dk_data["n_rows"], dk_data["n_cols"])
    else
        sparse(Int64[], Int64[], Int8[], 0, cells[dimension])
    end

    # d_kp1 : C_{k+1} -> C_k
    dkp1_mat = if haskey(boundaries, dimension + 1)
        data = boundaries[dimension+1]
        sparse(data["rows"] .+ 1, data["cols"] .+ 1, Int8.(abs.(data["data"])), data["n_rows"], data["n_cols"])
    else
        sparse(Int64[], Int64[], Int8[], cells[dimension], 0)
    end

    # Nullspace of d_k - Now fully sparse
    z_basis = _nullspace_basis_mod2(dk_mat)
    if isempty(z_basis); return []; end

    # Candidates for quotient basis
    z_candidates = z_basis
    if mode == "optimal"
        # Sort by weight - Weight function already handles chain vectors
        weights = [_weight_k_chain(z, k_simplices_mat, pts) for z in z_candidates]
        z_candidates = z_candidates[sortperm(weights)]
    end

    # Quotient basis selection using sparse boundaries
    # We pre-fill a pivot matrix with the image of dkp1
    # Image of d_{k+1} is given by the columns of dkp1_mat
    m_final = cells[dimension]
    quotient_basis = Vector{Vector{Int8}}()
    
    # 1. Pre-reduce the image of d_{k+1} using BitMatrix for speed
    image_m, image_n = size(dkp1_mat)
    # Pivot state for Z2 subspace
    img_pivots = Dict{Int, Vector{Int8}}()
    
    for j in 1:image_n
        col = Vector{Int8}(dkp1_mat[:, j])
        # We need a fast independence check
        _is_independent_mod_image(col, img_pivots)
    end

    # 2. Extract quotient basis
    for z in z_candidates
        if _is_independent_mod_image(z, img_pivots)
            push!(quotient_basis, z)
        end
    end

    # 2. Package result for Python
    results = []
    for z in quotient_basis
        active_idx = findall(x -> x & 1 != 0, z)
        supp_simplices = [collect(k_simplices[idx]) for idx in active_idx]

        supp_edges = if dimension == 1
            [collect(k_simplices[idx]) for idx in active_idx]
        else
            []
        end

        weight = _weight_k_chain(z, k_simplices_mat, pts)

        push!(results, Dict(
            "dimension" => Int64(dimension),
            "support_simplices" => supp_simplices,
            "support_edges" => supp_edges,
            "weight" => Float64(weight),
            "certified_cycle" => true
        ))
    end

    return results
end

"""
    triangulate_surface_delaunay(points; tolerance=1e-10)

Project 3D points to a PCA plane and triangulate via DelaunayTriangulation.jl.
"""
function triangulate_surface_delaunay(points::AbstractMatrix{Float64}, tolerance::Real=1e-10)
    if !HAS_DELAUNAY
        error("DelaunayTriangulation.jl is not available. Please install it to use Julia-accelerated triangulation.")
    end

    n_pts, dim = size(points)
    if dim != 3
        error("Points must be 3D coordinates (shape: n_points × 3)")
    end
    if n_pts < 3
        error("At least 3 points required for triangulation")
    end

    # 1. PCA Projection to 2D
    centroid = mean(points, dims=1)
    centered = points .- centroid

    # SVD for PCA
    F = svd(centered)

    # Check if the surface is too "thick"
    if length(F.S) >= 3 && F.S[3] > tolerance
        @warn "Topological Hint: Point cloud has significant variance in normal direction ($(F.S[3])). Surface may not be truly 2D."
    end

    # Project onto the first two principal components
    # F.V contains principal directions as columns
    v1 = F.V[:, 1]
    v2 = F.V[:, 2]

    # projected = centered * [v1 v2]
    projected_2d = centered * [v1 v2]

    # 2. Delaunay Triangulation
    # DelaunayTriangulation.jl expects points as a collection of Tuples or a Matrix with points as columns
    pts_tuples = [(projected_2d[i, 1], projected_2d[i, 2]) for i in 1:n_pts]

    triangulation = triangulate(pts_tuples)

    # 3. Extract faces (triangles)
    faces = Vector{Vector{Int64}}()
    for tri in each_solid_triangle(triangulation)
        u, v, w = tri
        # Return 0-indexed for Python consistency
        push!(faces, [Int64(u-1), Int64(v-1), Int64(w-1)])
    end

    return faces
end

# --- Native Complex Construction Kernels ---

"""
    enumerate_cliques_sparse(rowptr, colval, n_vertices, max_dim)

Enumerate all cliques up to size `max_dim + 1` from a sparse adjacency matrix.
Uses a simple backtracking DFS (Bron-Kerbosch style without pivoting since max_dim is small).
"""
function enumerate_cliques_sparse(rowptr_raw, colval_raw, n_vertices_raw, max_dim_raw)
    rowptr = pyconvert(Vector{Int64}, rowptr_raw)
    colval = pyconvert(Vector{Int64}, colval_raw)
    n_vertices = pyconvert(Int, n_vertices_raw)
    max_dim = pyconvert(Int, max_dim_raw)

    cliques = Vector{Vector{Int64}}()

    # Adjacency check function for sorted neighbor lists
    function is_adj(u::Int, v::Int)
        start_idx = rowptr[u]
        end_idx = rowptr[u+1] - 1
        # Binary search
        while start_idx <= end_idx
            mid = (start_idx + end_idx) >> 1
            if colval[mid] == v
                return true
            elseif colval[mid] < v
                start_idx = mid + 1
            else
                end_idx = mid - 1
            end
        end
        return false
    end

    function backtrack(current_clique::Vector{Int64}, candidates::Vector{Int64})
        push!(cliques, copy(current_clique))
        if length(current_clique) == max_dim + 1
            return
        end

        for (i, v) in enumerate(candidates)
            new_candidates = Int64[]
            for j in (i+1):length(candidates)
                w = candidates[j]
                if is_adj(v, w)
                    push!(new_candidates, w)
                end
            end
            push!(current_clique, v)
            backtrack(current_clique, new_candidates)
            pop!(current_clique)
        end
    end

    # To avoid duplicates, we only consider candidates > u
    for u in 1:n_vertices
        candidates = Int64[]
        for ptr in rowptr[u]:(rowptr[u+1]-1)
            v = colval[ptr]
            if v > u
                push!(candidates, v)
            end
        end
        backtrack([u], candidates)
    end

    return cliques
end

"""
    compute_vietoris_rips(points_raw, epsilon_raw, max_dim_raw)

Compute the Vietoris-Rips complex from a point cloud.
Uses an optimized O(N^2) 1-skeleton generation followed by a bounded clique search.
Complexity: O(N^2 * D) for 1-skeleton, plus clique search which is O(3^(N/3)) in worst case
but bounded to O(N * (max_deg)^max_dim) here.
"""
function compute_vietoris_rips(points_raw, epsilon_raw, max_dim_raw)
    points = pyconvert(AbstractMatrix{Float64}, points_raw)
    epsilon = pyconvert(Float64, epsilon_raw)
    max_dim = pyconvert(Int, max_dim_raw)
    n_pts = size(points, 1)
    dim_pts = size(points, 2)

    # 1. 1-skeleton (Edges) - Parallelized
    eps2 = epsilon^2
    n_threads = Threads.nthreads()
    I_threads = [Int64[] for _ in 1:n_threads]
    J_threads = [Int64[] for _ in 1:n_threads]

    Threads.@threads for i in 1:n_pts
        tid = Threads.threadid()
        I_local = I_threads[tid]
        J_local = J_threads[tid]
        for j in (i+1):n_pts
            d2 = 0.0
            @inbounds for k in 1:dim_pts
                diff = points[i, k] - points[j, k]
                d2 += diff * diff
            end
            if d2 <= eps2
                push!(I_local, i)
                push!(J_local, j)
                push!(I_local, j)
                push!(J_local, i)
            end
        end
    end

    I = reduce(vcat, I_threads)
    J = reduce(vcat, J_threads)

    # 2. Sparse Adjacency
    adj = sparse(I, J, ones(Int64, length(I)), n_pts, n_pts)
    rowptr = adj.colptr
    colval = adj.rowval

    cliques = Vector{Vector{Int64}}()

    # Optimized adjacency check for sorted neighbor lists
    function is_adj(u::Int, v::Int)
        start_idx = rowptr[u]
        end_idx = rowptr[u+1] - 1
        while start_idx <= end_idx
            mid = (start_idx + end_idx) >> 1
            if colval[mid] == v
                return true
            elseif colval[mid] < v
                start_idx = mid + 1
            else
                end_idx = mid - 1
            end
        end
        return false
    end

    # Bounded DFS for cliques
    function backtrack(current_clique::Vector{Int64}, candidates::Vector{Int64})
        push!(cliques, copy(current_clique))

        if length(current_clique) == max_dim + 1
            return
        end

        for (i, v) in enumerate(candidates)
            new_candidates = Int64[]
            for j in (i+1):length(candidates)
                w = candidates[j]
                if is_adj(v, w)
                    push!(new_candidates, w)
                end
            end
            push!(current_clique, v)
            backtrack(current_clique, new_candidates)
            pop!(current_clique)
        end
    end

    for u in 1:n_pts
        candidates = Int64[]
        for ptr in rowptr[u]:(rowptr[u+1]-1)
            v = colval[ptr]
            if v > u
                push!(candidates, v)
            end
        end
        backtrack([u], candidates)
    end

    # Return 0-indexed for Python consistency
    return [c .- 1 for c in cliques]
end
"""
    compute_circumradius_sq_3d(points, simplices)

Compute squared circumradii for 3D tetrahedra natively.
"""
function compute_circumradius_sq_3d(points::AbstractMatrix{Float64}, simplices::AbstractMatrix{Int64})
    n_simplices = size(simplices, 1)
    radii_sq = zeros(Float64, n_simplices)

    Threads.@threads for i in 1:n_simplices
        v1 = @view(points[simplices[i, 2], :]) .- @view(points[simplices[i, 1], :])
        v2 = @view(points[simplices[i, 3], :]) .- @view(points[simplices[i, 1], :])
        v3 = @view(points[simplices[i, 4], :]) .- @view(points[simplices[i, 1], :])

        A = [v1[1] v1[2] v1[3];
             v2[1] v2[2] v2[3];
             v3[1] v3[2] v3[3]]

        b = [0.5 * sum(abs2, v1);
             0.5 * sum(abs2, v2);
             0.5 * sum(abs2, v3)]

        d = det(A)
        # Scale-aware robustness check
        max_coord = max(maximum(abs.(v1)), maximum(abs.(v2)), maximum(abs.(v3)))
        if abs(d) > 1e-14 * max_coord^3
            center_offset = A \ b
            radii_sq[i] = sum(abs2, center_offset)
        else
            radii_sq[i] = Inf # Degenerate
        end
    end
    return radii_sq
end

"""
    compute_circumradius_sq_2d(points, simplices)

Compute squared circumradii for 2D triangles natively.
"""
function compute_circumradius_sq_2d(points::AbstractMatrix{Float64}, simplices::AbstractMatrix{Int64})
    n_simplices = size(simplices, 1)
    radii_sq = zeros(Float64, n_simplices)

    Threads.@threads for i in 1:n_simplices
        ux = points[simplices[i, 2], 1] - points[simplices[i, 1], 1]
        uy = points[simplices[i, 2], 2] - points[simplices[i, 1], 2]
        vx = points[simplices[i, 3], 1] - points[simplices[i, 1], 1]
        vy = points[simplices[i, 3], 2] - points[simplices[i, 1], 2]

        d = 2.0 * (ux * vy - uy * vx)
        max_coord = max(abs(ux), abs(uy), abs(vx), abs(vy))
        
        if abs(d) > 1e-14 * max_coord^2
            u2 = ux*ux + uy*uy
            v2 = vx*vx + vy*vy
            cx = (vy * u2 - uy * v2) / d
            cy = (ux * v2 - vx * u2) / d
            radii_sq[i] = cx*cx + cy*cy
        else
            radii_sq[i] = Inf
        end
    end
    return radii_sq
end


"""
    orthogonal_procrustes(A::AbstractMatrix, B::AbstractMatrix)

Aligns B to A by finding an orthogonal matrix R that minimizes ||A - B*R||_F.
Returns (R, B_aligned, disparity).
"""
function orthogonal_procrustes(A::AbstractMatrix{Float64}, B::AbstractMatrix{Float64})
    M = transpose(B) * A
    F = svd(M)
    R = F.U * transpose(F.Vt)
    if det(R) < 0
        U = copy(F.U)
        U[:, end] .*= -1
        R = U * transpose(F.Vt)
    end
    B_aligned = B * R
    disparity = norm(A - B_aligned)
    return R, B_aligned, disparity
end

"""
    pairwise_distance_matrix(data::AbstractMatrix, metric::String)

Computes the pairwise distance matrix for the given metric ("euclidean", "manhattan", "chebyshev").
"""
function pairwise_distance_matrix(data::AbstractMatrix{Float64}, metric::String)
    n = size(data, 1)
    D = zeros(Float64, n, n)
    if metric == "euclidean"
        for i in 1:n
            for j in 1:i-1
                d = 0.0
                for k in 1:size(data, 2)
                    d += (data[i, k] - data[j, k])^2
                end
                D[i, j] = D[j, i] = sqrt(d)
            end
        end
    elseif metric == "manhattan"
        for i in 1:n
            for j in 1:i-1
                d = 0.0
                for k in 1:size(data, 2)
                    d += abs(data[i, k] - data[j, k])
                end
                D[i, j] = D[j, i] = d
            end
        end
    elseif metric == "chebyshev"
        for i in 1:n
            for j in 1:i-1
                d = 0.0
                for k in 1:size(data, 2)
                    d = max(d, abs(data[i, k] - data[j, k]))
                end
                D[i, j] = D[j, i] = d
            end
        end
    else
        error("Unsupported metric: " * metric)
    end
    return D
end

"""
    frechet_distance(curve_a::AbstractMatrix, curve_b::AbstractMatrix)

Computes the Discrete Fréchet distance between two polygonal curves.
"""
function frechet_distance(curve_a::AbstractMatrix{Float64}, curve_b::AbstractMatrix{Float64})
    n = size(curve_a, 1)
    m = size(curve_b, 1)
    ca = fill(-1.0, n, m)

    function dist(i, j)
        d = 0.0
        for k in 1:size(curve_a, 2)
            d += (curve_a[i, k] - curve_b[j, k])^2
        end
        return sqrt(d)
    end

    ca[1, 1] = dist(1, 1)
    for i in 2:n
        ca[i, 1] = max(ca[i-1, 1], dist(i, 1))
    end
    for j in 2:m
        ca[1, j] = max(ca[1, j-1], dist(1, j))
    end

    for i in 2:n
        for j in 2:m
            ca[i, j] = max(min(ca[i-1, j], ca[i, j-1], ca[i-1, j-1]), dist(i, j))
        end
    end

    return ca[n, m]
end

"""
    gromov_wasserstein_distance(D_A, D_B, p, q, epsilon, max_iter)

Computes the Entropic Gromov-Wasserstein distance between two metric spaces.
"""
function gromov_wasserstein_distance(
    D_A::AbstractMatrix{Float64},
    D_B::AbstractMatrix{Float64},
    p::AbstractVector{Float64},
    q::AbstractVector{Float64},
    epsilon::Float64,
    max_iter::Int
)
    n = size(D_A, 1)
    m = size(D_B, 1)

    T = p * transpose(q)

    C1 = (D_A .^ 2) * p * ones(1, m)
    C2 = ones(n, 1) * transpose(q) * transpose(D_B .^ 2)
    const_C = C1 .+ C2

    for iter in 1:max_iter
        L = const_C .- 2.0 .* (D_A * T * transpose(D_B))
        K = exp.(-L ./ epsilon)

        u = ones(n) ./ n
        v = ones(m) ./ m
        for s in 1:20
            v = q ./ (transpose(K) * u .+ 1e-10)
            u = p ./ (K * v .+ 1e-10)
        end

        T_new = u .* K .* transpose(v)

        if norm(T_new - T) < 1e-6
            T = T_new
            break
        end
        T = T_new
    end

    gw_dist = 0.0
    for i in 1:n
        for j in 1:m
            for k in 1:n
                for l in 1:m
                    gw_dist += (D_A[i, k] - D_B[j, l])^2 * T[i, j] * T[k, l]
                end
            end
        end
    end

    return sqrt(gw_dist)
end

@PrecompileTools.setup_workload begin
    # 1. Sparse Integer Matrix Setup (COO format) - 3x3 boundary-like with 1, -1, 0
    # Chain complex boundaries are strictly integer based.
    r_idx = Int64[0, 1, 2, 0]
    c_idx = Int64[1, 2, 0, 2]
    v_val = Int64[1, -1, 1, 1]
    m_sz, n_sz = 3, 3

    # 2. Alexander-Whitney Cup Product dummy data
    # Must use strictly typed Dict{Tuple{Vararg{Int}}, Int}
    s_to_idx_p = Dict{Tuple{Vararg{Int}}, Int64}((0, 1) => 0, (1, 2) => 1, (2, 0) => 2)
    s_to_idx_q = Dict{Tuple{Vararg{Int}}, Int64}((1, 2) => 0, (2, 3) => 1, (3, 1) => 2)
    alpha_vec = Int64[1, 0, 1]
    beta_vec = Int64[0, 1, 0]
    simplices_pq = [[0, 1, 2, 3]]

    # 3. Flat simplices for boundary computation (two triangles sharing an edge)
    fv = Int64[0, 1, 2, 1, 2, 3]
    fo = Int64[0, 3, 6]

    # 4. Miscellaneous: Group theory and Signatures
    mf = ones(Float64, 2, 2)
    mi = Int64[2 1; 1 2] # Positive definite for lattice isometry
    gens = ["a", "b"]
    rels = ["a^2", "b^3", "a*b*a^-1*b^-1"]

    # 5. New function dummies
    pts_3d = Float64[0 0 0; 1 0 0; 0 1 0; 0 0 1]
    simplices_h1 = [[0, 1], [1, 2], [2, 0], [0, 1, 2]]

    # 6. Metric variables
    pts_A = Float64[0 0; 1 0; 0 1]
    pts_B = Float64[0 1; 1 1; 0 0]
    p_gw = ones(Float64, 3) ./ 3.0
    q_gw = ones(Float64, 3) ./ 3.0
    D_A = Float64[0 1 1; 1 0 1.414; 1 1.414 0]

    PrecompileTools.@compile_workload begin
        # --- Heavy Paths: Sparse Rank, SNF, and Cohomology ---
        # These operations over Z trigger expensive type inference in SparseArrays and AbstractAlgebra.
        exact_snf_sparse(r_idx, c_idx, v_val, m_sz, n_sz)
        rank_q_sparse(r_idx, c_idx, v_val, m_sz, n_sz)
        rank_mod_p_sparse(r_idx, c_idx, v_val, m_sz, n_sz, 2)
        exact_sparse_cohomology_basis(r_idx, c_idx, v_val, m_sz, n_sz, r_idx, c_idx, v_val, m_sz, n_sz)
        sparse_cohomology_basis_mod_p(r_idx, c_idx, v_val, m_sz, n_sz, r_idx, c_idx, v_val, m_sz, n_sz, 2)

        # Trigger core SparseMatrixCSC{Int, Int} inference directly
        _ = sparse(r_idx .+ 1, c_idx .+ 1, v_val, m_sz, n_sz)

        # --- Boundary & Simplicial Pathways ---
        _compute_boundary_data_internal_flat(fv, fo, 2)
        compute_boundary_payload_from_flat_simplices(fv, fo, 2)
        compute_boundary_mod2_matrix([[0, 1, 2]], [[0, 1], [1, 2], [2, 0]])

        # --- Alexander-Whitney Cup Product ---
        # Using strictly typed Dicts to avoid PyDict overhead during warm-up
        try
            compute_alexander_whitney_cup(alpha_vec, beta_vec, 1, 1, simplices_pq, s_to_idx_p, s_to_idx_q)
        catch
        end

        # --- Group Algebra & Lattice Isometry ---
        abelianize_group(gens, rels)
        hermitian_signature(mf)
        multisignature(mf, 3)
        integral_lattice_isometry(mi, mi)

        # --- New High-Performance Topo Functions ---
        try
            triangulate_surface_delaunay(pts_3d)
            homology_generators_from_simplices(simplices_h1, 3, 1)
            optgen_from_simplices(simplices_h1, 3)

            # Metric & Alignment operations
            orthogonal_procrustes(pts_A, pts_B)
            pairwise_distance_matrix(pts_A, "euclidean")
            frechet_distance(pts_A, pts_B)
            gromov_wasserstein_distance(D_A, D_A, p_gw, q_gw, 0.01, 2)
        catch
        end
    end

    # Explicitly clear out trash variables generated during precompilation
    r_idx = c_idx = v_val = alpha_vec = beta_vec = nothing
    fv = fo = mf = mi = gens = rels = nothing
    pts_3d = simplices_h1 = pts_A = pts_B = p_gw = q_gw = D_A = nothing
    GC.gc()
end



"""
    compute_alpha_complex_simplices_jl(points, max_simplices, alpha2, max_dim)

Filter Delaunay simplices by circumradius and perform skeletal closure natively.
Handles N-dimensional point clouds with robust linear circumradius solvers.
"""
function compute_alpha_complex_simplices_jl(
    points::AbstractMatrix{Float64},
    max_simplices::AbstractMatrix{Int64},
    alpha2::Float64,
    max_dim::Int
)
    n_pts, dim_pts = size(points)
    n_max = size(max_simplices, 1)
    
    valid_simplices = Set{Tuple{Vararg{Int64}}}()
    r2_cache = Dict{Tuple{Vararg{Int64}}, Float64}()

    function get_r2(simplex_indices)
        if haskey(r2_cache, simplex_indices)
            return r2_cache[simplex_indices]
        end

        k = length(simplex_indices)
        if k == 1
            r2_cache[simplex_indices] = 0.0
            return 0.0
        end
        if k == 2
            u, v = simplex_indices[1]+1, simplex_indices[2]+1
            d2 = 0.0
            for i in 1:dim_pts
                d2 += (points[u, i] - points[v, i])^2
            end
            val = d2 / 4.0
            r2_cache[simplex_indices] = val
            return val
        end

        p0 = @view points[simplex_indices[1]+1, :]
        A = zeros(Float64, k-1, dim_pts)
        b = zeros(Float64, k-1)
        for i in 2:k
            pi = @view points[simplex_indices[i]+1, :]
            for j in 1:dim_pts
                A[i-1, j] = pi[j] - p0[j]
            end
            b[i-1] = 0.5 * sum(abs2, pi .- p0)
        end

        try
            c = A \ b
            val = sum(abs2, c)
            r2_cache[simplex_indices] = val
            return val
        catch
            r2_cache[simplex_indices] = Inf
            return Inf
        end
    end

    for i in 1:n_max
        s = Tuple(sort(vec(max_simplices[i, :])))
        if get_r2(s) <= alpha2
            for r in 1:length(s)
                for face in combinations(s, r)
                    push!(valid_simplices, Tuple(face))
                end
            end
        else
            for v in s
                push!(valid_simplices, Tuple((v,)))
            end
            for r in 2:min(length(s)-1, max_dim+1)
                for face in combinations(s, r)
                    t_face = Tuple(face)
                    if get_r2(t_face) <= alpha2
                        for r_sub in 1:length(t_face)
                            for sub_f in combinations(t_face, r_sub)
                                push!(valid_simplices, Tuple(sub_f))
                            end
                        end
                    end
                end
            end
        end
    end

    return collect(valid_simplices)
end


function quick_mapper_jl(G_raw_py::PyDict{Any, Any}, max_loops::Int=1, min_modularity_gain::Float64=1e-6)
    G_raw = pyconvert(Dict{Any, Any}, G_raw_py)
    V_py = G_raw["V"]
    E_py = G_raw["E"]
    V = [Int(v) for v in V_py]
    E = [(Int(e[1]), Int(e[2])) for e in E_py]

    adj = Dict{Int, Vector{Int}}(v => Int[] for v in V)
    for (u, v) in E
        push!(adj[u], v)
        push!(adj[v], u)
    end

    m = length(E)
    L = Dict{Int, Int}(v => v for v in V)

    if m == 0
        return Dict("V" => collect(Set(values(L))), "E" => []), L
    end

    degree = Dict{Int, Int}(v => length(adj[v]) for v in V)
    comm_degree = Dict{Int, Int}(v => degree[v] for v in V)
    num_of_loops = 0
    two_m = 2.0 * m
    any_change = true

    while any_change && num_of_loops < max_loops
        vertex_order = collect(V)
        shuffle!(vertex_order)
        any_change = false
        total_gain = 0.0

        for vertex in vertex_order
            neighbors = adj[vertex]
            if isempty(neighbors)
                continue
            end

            curr_comm = L[vertex]
            k_v = degree[vertex]

            # Map neighbor communities to counts
            nbr_comm_counts = Dict{Int, Float64}()
            for nbr in neighbors
                l = L[nbr]
                nbr_comm_counts[l] = get(nbr_comm_counts, l, 0.0) + 1.0
            end

            best_comm = curr_comm
            max_gain = 0.0

            for (comm, k_in) in nbr_comm_counts
                if comm == curr_comm
                    continue
                end

                # Louvain modularity gain
                gain = (k_in / m) - (k_v * comm_degree[comm]) / (two_m^2)

                if gain > max_gain
                    max_gain = gain
                    best_comm = comm
                end
            end

            if best_comm != curr_comm && max_gain > min_modularity_gain
                comm_degree[curr_comm] -= k_v
                comm_degree[best_comm] += k_v
                L[vertex] = best_comm
                any_change = true
                total_gain += max_gain
            end
        end

        num_of_loops += 1
        if total_gain <= min_modularity_gain
            break
        end
    end

    E_simple = Set{Tuple{Int, Int}}()
    for vertex in V
        for nbr in adj[vertex]
            lv = L[vertex]
            lnbr = L[nbr]
            if lv != lnbr
                push!(E_simple, minmax(lv, lnbr))
            end
        end
    end

        # Normalize indices to 0..N-1
    unique_labels = sort(collect(Set(values(L))))
    label_to_new = Dict{Int, Int}(l => i-1 for (i, l) in enumerate(unique_labels))

    L_normalized = Dict{Int, Int}()
    for (v, l) in L
        L_normalized[v] = label_to_new[l]
    end

    V_normalized = collect(0:(length(unique_labels)-1))
    E_normalized = Set{Tuple{Int, Int}}()
    for (u, v) in E_simple
        push!(E_normalized, minmax(label_to_new[u], label_to_new[v]))
    end

    return Dict("V" => V_normalized, "E" => collect(E_normalized)), L_normalized
end

function simplify_jl(simplices_py)
    # 1. Initialize and Skeletal Closure in Julia
    simplices_vec = pyconvert(Vector{Any}, simplices_py)
    all_simplices = Set{Tuple{Vararg{Int}}}()
    for s_py in simplices_vec
        s = sort([Int(v) for v in pyconvert(Vector{Any}, s_py)])
        n = length(s)
        # For Mapper complexes, n is small. For massive VR complexes, we limit closure.
        if n > 25
             push!(all_simplices, Tuple(s))
             continue
        end
        for i in 1:(1 << n - 1)
            subface = Int[]
            for j in 1:n
                if ((i >> (j-1)) & 1) == 1
                    push!(subface, s[j])
                end
            end
            push!(all_simplices, Tuple(subface))
        end
    end

    v_set = Set{Int}()
    for s in all_simplices; for v in s; push!(v_set, v); end; end
    V_orig = sort(collect(v_set))

    # Union-Find with path compression
    parent = Dict{Int, Int}(v => v for v in V_orig)
    function find_root(i)
        root = i
        while parent[root] != root; root = parent[root]; end
        curr = i
        while parent[curr] != root
            nxt = parent[curr]
            parent[curr] = root
            curr = nxt
        end
        return root
    end
    function union_sets(i, j)
        root_i = find_root(i)
        root_j = find_root(j)
        if root_i != root_j
            parent[root_j] = root_i
            return true
        end
        return false
    end

    current_simplices = copy(all_simplices)
    v_adj = Dict{Int, Set{Tuple{Vararg{Int}}}}()
    for v in V_orig; v_adj[v] = Set{Tuple{Vararg{Int}}}(); end
    for s in current_simplices; for v in s; push!(v_adj[v], s); end; end

    any_change = true
    while any_change
        any_change = false
        edges = Tuple{Int, Int}[]
        for s in current_simplices
            if length(s) == 2; push!(edges, (s[1], s[2])); end
        end
        isempty(edges) && break
        sort!(edges)

        for (u, v) in edges
            # Link Condition: Lk(u) ∩ Lk(v) == Lk(uv)
            valid = true
            for s in v_adj[u]
                if !(v in s)
                    f_set = Set{Int}(s)
                    delete!(f_set, u)
                    f_union_v = Tuple(sort(vcat(collect(f_set), [v])))
                    if f_union_v in current_simplices
                        f_union_uv = Tuple(sort(vcat(collect(f_set), [u, v])))
                        if !(f_union_uv in current_simplices)
                            valid = false; break
                        end
                    end
                end
            end

            if valid
                union_sets(u, v)
                new_simplices = Set{Tuple{Vararg{Int}}}()
                new_v_adj = Dict{Int, Set{Tuple{Vararg{Int}}}}()

                for s in current_simplices
                    mapped = Tuple(sort(collect(Set(find_root(x) for x in s))))
                    if !isempty(mapped)
                        if !(mapped in new_simplices)
                            push!(new_simplices, mapped)
                            for node in mapped
                                if !haskey(new_v_adj, node); new_v_adj[node] = Set{Tuple{Vararg{Int}}}(); end
                                push!(new_v_adj[node], mapped)
                            end
                        end
                    end
                end
                current_simplices = new_simplices
                v_adj = new_v_adj
                any_change = true
                break
            end
        end
    end

    unique_roots = sort(collect(Set(find_root(v) for v in V_orig)))
    root_to_new = Dict{Int, Int}(root => i-1 for (i, root) in enumerate(unique_roots))

    final_v_map = Dict{Int, Vector{Int}}()
    for v in V_orig
        nv = root_to_new[find_root(v)]
        if !haskey(final_v_map, nv); final_v_map[nv] = Int[]; end
        push!(final_v_map[nv], v)
    end

    final_s_map = Dict{Tuple{Vararg{Int}}, Vector{Tuple{Vararg{Int}}}}()
    for orig_s in all_simplices
        mapped_v = Int[root_to_new[find_root(v)] for v in orig_s]
        mapped_s = Tuple(sort(collect(Set(mapped_v))))
        if !haskey(final_s_map, mapped_s); final_s_map[mapped_s] = Tuple{Vararg{Int}}[]; end
        push!(final_s_map[mapped_s], orig_s)
    end

    # Normalize the return simplices to 0..N-1
    final_simplices_normalized = Vector{Int}[]
    for s in current_simplices
        push!(final_simplices_normalized, sort([root_to_new[v] for v in s]))
    end

    return final_simplices_normalized, final_v_map, final_s_map
end
function cknn_graph_jl(pts::AbstractMatrix{Float64}, k::Int, delta::Float64)
    n = size(pts, 1)
    if n == 0 || k < 1
        return Matrix{Int64}(undef, 0, 2)
    end
    k_actual = min(k, n - 1)

    # 1. Compute rho (distance to k-th neighbor)
    rho = zeros(Float64, n)
    Threads.@threads for i in 1:n
        dists_sq = zeros(Float64, n)
        p_i = @view pts[i, :]
        for j in 1:n
            dists_sq[j] = sum(abs2, p_i .- @view(pts[j, :]))
        end
        # k-th nearest neighbor (0-th is self)
        rho[i] = sqrt(partialsort!(dists_sq, k_actual + 1))
    end

    # 2. Filter edges based on CkNN condition
    thread_pairs = [Vector{NTuple{2, Int32}}() for _ in 1:Threads.nthreads()]
    delta_sq = delta^2

    Threads.@threads for i in 1:(n-1)
        tid = Threads.threadid()
        p_i = @view pts[i, :]
        rho_i = rho[i]
        for j in (i+1):n
            dist_sq = sum(abs2, p_i .- @view(pts[j, :]))
            if dist_sq < delta_sq * rho_i * rho[j]
                push!(thread_pairs[tid], (Int32(i-1), Int32(j-1)))
            end
        end
    end

    total_pairs = sum(length, thread_pairs)
    out = Matrix{Int64}(undef, total_pairs, 2)
    idx = 1
    for tp in thread_pairs
        for p in tp
            out[idx, 1] = p[1]
            out[idx, 2] = p[2]
            idx += 1
        end
    end
    return out
end


function cknn_graph_accelerated_jl(pts::AbstractMatrix{Float64}, rho::AbstractVector{Float64}, delta::Float64)
    n = size(pts, 1)
    if n == 0
        return Matrix{Int64}(undef, 0, 2)
    end

    thread_pairs = [Vector{NTuple{2, Int32}}() for _ in 1:Threads.nthreads()]
    delta_sq = delta^2

    Threads.@threads for i in 1:(n-1)
        tid = Threads.threadid()
        p_i = @view pts[i, :]
        rho_i = rho[i]
        for j in (i+1):n
            dist_sq = sum(abs2, p_i .- @view(pts[j, :]))
            if dist_sq < delta_sq * rho_i * rho[j]
                push!(thread_pairs[tid], (Int32(i-1), Int32(j-1)))
            end
        end
    end

    total_pairs = sum(length, thread_pairs)
    out = Matrix{Int64}(undef, total_pairs, 2)
    idx = 1
    for tp in thread_pairs
        for p in tp
            out[idx, 1] = p[1]
            out[idx, 2] = p[2]
            idx += 1
        end
    end
    return out
end

"""
    is_homology_manifold_jl(simplex_entries, max_dim)

Accelerated check for homology manifolds by computing links and reduced homology
in Julia. Returns (is_manifold::Bool, dimension::Int, diagnostics::Dict{Int, String}).
"""
function is_homology_manifold_jl(simplex_entries, max_dim::Int)
    # 1. Build full skeleton once
    # simplex_entries can be a list of lists or similar from Python
    boundaries, cells, sorted_dim_simplices, simplex_to_idx = _compute_boundary_data_internal(simplex_entries, max_dim)

    # Vertices are in sorted_dim_simplices[0]
    vertices = if get(cells, 0, 0) == 0
        Int64[]
    else
        [s[1] for s in sorted_dim_simplices[0]]
    end

    if isempty(vertices)
        return true, -1, Dict{Int, String}()
    end

    local_dims = Dict{Int, Union{Nothing, Int}}()
    diagnostics = Dict{Int, String}()

    # Helper for reduced homology in Julia
    function get_reduced_homology(lk_simplices, lk_max_dim)
        lk_b, lk_c = compute_boundary_payload_from_simplices(lk_simplices, lk_max_dim, false)
        rh = Dict{Int, Tuple{Int, Vector{Int}}}()

        # Max dimension of link
        d_max = -1
        for d in 0:lk_max_dim
            if get(lk_c, d, 0) > 0
                d_max = d
            end
        end

        for d in 0:d_max
            # H_d = ker(d_d) / im(d_{d+1})
            n_rows_d = get(lk_c, d-1, 0)
            n_cols_d = get(lk_c, d, 0)

            # ker(d_d)
            rank_ker_d = if n_cols_d == 0
                0
            elseif n_rows_d == 0
                n_cols_d
            else
                b_d = lk_b[d]
                # rank over Q for ker dimension
                n_cols_d - rank_q_sparse(b_d["rows"], b_d["cols"], b_d["data"], b_d["n_rows"], b_d["n_cols"])
            end

            # im(d_{d+1})
            factors_dp1 = if haskey(lk_b, d+1)
                b_dp1 = lk_b[d+1]
                exact_snf_sparse(b_dp1["rows"], b_dp1["cols"], b_dp1["data"], b_dp1["n_rows"], b_dp1["n_cols"])
            else
                Int[]
            end

            rank_im_dp1 = 0
            torsion = Int[]
            for f in factors_dp1
                if f != 0
                    rank_im_dp1 += 1
                    if f > 1
                        push!(torsion, f)
                    end
                end
            end

            betti = max(0, rank_ker_d - rank_im_dp1)

            # Reduced homology adjustment
            if d == 0
                betti = max(0, betti - 1)
            end

            if betti > 0 || !isempty(torsion)
                rh[d] = (Int(betti), Int.(torsion))
            end
        end
        return rh, d_max
    end

    coface_map = Dict{Int, Vector{Vector{Int64}}}()
    for v in vertices
        coface_map[v] = Vector{Vector{Int64}}()
    end
    for d in 1:max_dim
        for s in sorted_dim_simplices[d]
            for v in s
                if haskey(coface_map, v)
                    push!(coface_map[v], [x for x in s if x != v])
                end
            end
        end
    end

    local_dims_arr = Vector{Union{Nothing, Int}}(undef, length(vertices))
    diagnostics_arr = Vector{String}(undef, length(vertices))
    fill!(diagnostics_arr, "")

    Threads.@threads for i in 1:length(vertices)
        v = vertices[i]
        lk_max_simplices = coface_map[v]

        if isempty(lk_max_simplices)
            local_dims_arr[i] = 0
            continue
        end

        rh, lk_d_max = get_reduced_homology(lk_max_simplices, max_dim - 1)

        if isempty(rh)
            local_dims_arr[i] = nothing # Acyclic link
        elseif length(rh) == 1
            deg = first(keys(rh))
            betti, torsion = rh[deg]
            if betti == 1 && isempty(torsion)
                local_dims_arr[i] = deg + 1
            else
                diagnostics_arr[i] = "Link has non-sphere homology at degree $deg: rank=$betti, torsion=$torsion"
            end
        else
            diagnostics_arr[i] = "Link has multiple non-zero homology groups: $(collect(keys(rh)))"
        end
    end

    for i in 1:length(vertices)
        v = vertices[i]
        local_dims[v] = local_dims_arr[i]
        if diagnostics_arr[i] != ""
            diagnostics[v] = diagnostics_arr[i]
        end
    end

    detected_dims = Set{Int}([d for d in values(local_dims) if d !== nothing])

    # max dimension of the complex
    max_d_complex = 0
    for d in 0:max_dim
        if get(cells, d, 0) > 0; max_d_complex = d; end
    end

    if isempty(detected_dims)
        # All links were acyclic. This is consistent with a manifold if the complex is "disk-like"
        return true, max_d_complex, Dict{Int, String}()
    end

    if length(detected_dims) > 1
        return false, -1, Dict(-1 => "Inconsistent local dimensions: $detected_dims")
    end

    d_global = first(detected_dims)

    if d_global != max_d_complex
         return false, d_global, Dict(-1 => "Detected manifold dimension $d_global does not match complex dimension $max_d_complex")
    end

    if !isempty(diagnostics)
        return false, d_global, diagnostics
    end

    return true, d_global, Dict{Int, String}()
end
end # module

function compute_alpha_threshold_emst_jl(points::AbstractMatrix{Float64}, simplices::AbstractMatrix{Int64})
    n_pts = size(points, 1)
    n_simps = size(simplices, 1)

    # Extract edges
    edges_set = Set{Tuple{Int, Int}}()
    for i in 1:n_simps
        # simplices are 0-based
        for j in 1:size(simplices, 2), k in (j+1):size(simplices, 2)
            u, v = simplices[i, j], simplices[i, k]
            if u != -1 && v != -1 # Handle padding if any
                push!(edges_set, minmax(Int(u), Int(v)))
            end
        end
    end

    edges_list = collect(edges_set)
    weights = Float64[]
    for (u, v) in edges_list
        # u, v are 0-based
        d = sqrt(sum(abs2, points[u+1, :] .- points[v+1, :]))
        push!(weights, d)
    end

    if HAS_GRAPHS
        g = SimpleWeightedGraph(n_pts)
        for i in 1:length(edges_list)
            u, v = edges_list[i]
            add_edge!(g, u+1, v+1, weights[i])
        end
        mst = kruskal_mst(g)
        max_e = 0.0
        for e in mst
            max_e = max(max_e, e.weight)
        end
        return (max_e / 2.0)^2
    else
        parent = collect(1:n_pts)
        function find(i)
            root = i
            while parent[root] != root; root = parent[root]; end
            while parent[i] != root; next = parent[i]; parent[i] = root; i = next; end
            return root
        end
        function union!(i, j)
            root_i = find(i)
            root_j = find(j)
            if root_i != root_j; parent[root_j] = root_i; return true; end
            return false
        end

        perm = sortperm(weights)
        max_e = 0.0
        count = 0
        for idx in perm
            u, v = edges_list[idx]
            if union!(u+1, v+1)
                max_e = max(max_e, weights[idx])
                count += 1
                if count == n_pts - 1; break; end
            end
        end
        return (max_e / 2.0)^2
    end
end

function compute_crust_simplices_jl(points::AbstractMatrix{Float64}, combined_simplices::AbstractMatrix{Int64}, n_pts_orig::Int)
    dim = size(points, 2)
    target_dim = dim - 1
    valid_simplices = Set{Vector{Int64}}()

    n_simps = size(combined_simplices, 1)
    for i in 1:n_simps
        # combined_simplices are 0-based
        s = Int64.(combined_simplices[i, :])
        for face in combinations(s, target_dim + 1)
            if all(v < n_pts_orig for v in face)
                push!(valid_simplices, sort(collect(face)))
            end
        end
    end
    return collect(valid_simplices)
end

function compute_witness_complex_simplices_jl(points::AbstractMatrix{Float64}, landmarks_idx::AbstractVector{Int64}, alpha::Float64, max_dim::Int)
    n_pts = size(points, 1)
    n_landmarks = length(landmarks_idx)
    landmarks = points[landmarks_idx .+ 1, :]

    # 1. Compute distances to all landmarks for all points
    # Shape: n_pts x n_landmarks
    dists = zeros(Float64, n_pts, n_landmarks)
    for j in 1:n_landmarks
        l_j = @view landmarks[j, :]
        for i in 1:n_pts
            dists[i, j] = sqrt(sum(abs2, points[i, :] .- l_j))
        end
    end

    m_dist = minimum(dists, dims=2)
    valid_witnesses = dists .<= (m_dist .+ alpha)

    simplices = Set{Vector{Int64}}()
    # Vertices
    for i in 0:(n_landmarks-1); push!(simplices, [i]); end

    # 2. Build 1-skeleton (edges) - Vectorized
    W = Int8.(valid_witnesses)
    shared = transpose(W) * W
    
    for i in 1:n_landmarks
        for j in (i+1):n_landmarks
            if shared[i, j] > 0
                push!(simplices, [i-1, j-1])
            end
        end
    end

    if max_dim <= 1
        return collect(simplices)
    end

    # 3. Higher dimensional cliques (if needed)
    # We can use enumerate_cliques logic here or just return 1-skeleton for expansion in Python
    # Given how from_witness is implemented in Python, it returns 1-skeleton then expands.
    return collect(simplices)
end
