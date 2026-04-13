# A lightweight Julia backend module for high-performance algebraic operations over Z[pi]
module SurgeryBackend

using LinearAlgebra
using SparseArrays
using Statistics
using PythonCall: pyconvert, Py
import PrecompileTools

export hermitian_signature, exact_snf_sparse, exact_sparse_cohomology_basis, rank_q_sparse, rank_mod_p_sparse, sparse_cohomology_basis_mod_p, group_ring_multiply, multisignature, abelianize_group, integral_lattice_isometry, optgen_from_simplices, homology_generators_from_simplices, compute_boundary_data_from_simplices, compute_boundary_payload_from_simplices, compute_boundary_payload_from_flat_simplices, compute_boundary_mod2_matrix, compute_alexander_whitney_cup, compute_trimesh_boundary_data, compute_trimesh_boundary_data_flat, triangulate_surface_delaunay

const HAS_ABSTRACT_ALGEBRA = try
    @eval import AbstractAlgebra
    true
catch
    false
end

# Use AbstractArray/AbstractVector for zero-copy NumPy integration via juliacall/PythonCall
function hermitian_signature(matrix::AbstractMatrix{Float64})
    # Convert to Julia Matrix (this may still copy if it's a PyArray, but Matrix(matrix) is often required for eigen)
    mat = Matrix{Float64}(matrix)
    eigenvalues = eigvals(Hermitian(mat))
    tol = length(eigenvalues) > 0 ? maximum(size(mat)) * eps(Float64) * maximum(abs.(eigenvalues)) : 1e-10
    pos = count(x -> x > tol, eigenvalues)
    neg = count(x -> x < -tol, eigenvalues)
    return pos - neg
end

function exact_snf_sparse(rows::AbstractVector{Int64}, cols::AbstractVector{Int64}, vals::AbstractVector{Int64}, m::Int, n::Int)
    if !HAS_ABSTRACT_ALGEBRA
        error("AbstractAlgebra unavailable")
    end

    # Python passes zero-based COO indices; lift to Julia's one-based indexing.
    # rows .+ 1 on a PyArray creates a new Julia Array{Int64}, which is fine for the sparse() constructor.
    A = sparse(rows .+ 1, cols .+ 1, vals, m, n)

    try
        ZZ = AbstractAlgebra.ZZ
        A_aa = AbstractAlgebra.matrix(ZZ, Matrix(A))
        S_aa = AbstractAlgebra.snf(A_aa)
        factors = Int64[]
        for i in 1:min(m, n)
            val = Int64(S_aa[i, i])
            if val != 0
                push!(factors, abs(val))
            end
        end
        return sort(factors)
    catch e
        rethrow(e)
    end
end

function exact_sparse_cohomology_basis(
    d_np1_rows::AbstractVector{Int64}, d_np1_cols::AbstractVector{Int64}, d_np1_vals::AbstractVector{Int64}, d_np1_m::Int, d_np1_n::Int,
    d_n_rows::AbstractVector{Int64}, d_n_cols::AbstractVector{Int64}, d_n_vals::AbstractVector{Int64}, d_n_m::Int, d_n_n::Int
)
    # Python passes zero-based COO indices; lift to Julia's one-based indexing.
    coboundary_mat = sparse(d_np1_cols .+ 1, d_np1_rows .+ 1, d_np1_vals, d_np1_n, d_np1_m)

    basis = Vector{Vector{Int64}}()
    try
        if !HAS_ABSTRACT_ALGEBRA
            error("AbstractAlgebra unavailable")
        end
        QQ = AbstractAlgebra.QQ
        M_qq = AbstractAlgebra.matrix(QQ, Matrix(coboundary_mat))
        nullity, nullspace_mat = AbstractAlgebra.nullspace(M_qq)
        
        for j in 1:nullity
            col = nullspace_mat[:, j]
            denoms = [AbstractAlgebra.denominator(x) for x in col]
            lcm_val = 1
            for d in denoms
                lcm_val = lcm(lcm_val, d)
            end
            
            int_vec = Int64[]
            for i in 1:d_np1_m
                val = AbstractAlgebra.numerator(col[i] * lcm_val)
                push!(int_vec, val)
            end
            push!(basis, int_vec)
        end
    catch e
        dense_M = Matrix{Float64}(coboundary_mat)
        F = svd(dense_M)
        smax = isempty(F.S) ? 0.0 : maximum(F.S)
        tol = maximum(size(dense_M)) * eps(Float64) * smax
        rank_est = count(x -> x > tol, F.S)
        nullity = size(dense_M, 2) - rank_est
        null_basis_float = nullity > 0 ? [F.V[:, i] for i in (size(F.V, 2) - nullity + 1):size(F.V, 2)] : Vector{Vector{Float64}}()
        for v in null_basis_float
            max_comp = maximum(abs.(v))
            if max_comp < 1e-12
                continue
            end
            scale = round(Int64, 1.0 / max_comp * 1000)
            int_v = round.(Int64, v .* scale)
            g = reduce(gcd, int_v)
            if g != 0
                push!(basis, int_v .÷ g)
            end
        end
    end
    
    dn_mat = sparse(d_n_cols .+ 1, d_n_rows .+ 1, d_n_vals, d_n_n, d_n_m)

    quotient_basis = Vector{Vector{Int64}}()
    if size(dn_mat, 2) > 0
        dense_dn = Matrix{Float64}(dn_mat)
        curr_rank = rank(dense_dn)
        
        for vec in basis
            test_mat = hcat(dense_dn, Float64.(vec))
            new_rank = rank(test_mat)
            
            is_indep = false
            if new_rank > curr_rank
                try
                    if !HAS_ABSTRACT_ALGEBRA
                        error("AbstractAlgebra unavailable")
                    end
                    ZZ = AbstractAlgebra.ZZ
                    int_test = AbstractAlgebra.matrix(ZZ, Matrix{Int}(hcat(Matrix(dn_mat), vec)))
                    snf_test = AbstractAlgebra.snf(int_test)
                    new_rank_int = count(x -> x != 0, [snf_test[i, i] for i in 1:min(size(int_test)...)])
                    
                    int_base = AbstractAlgebra.matrix(ZZ, Matrix{Int}(Matrix(dn_mat)))
                    snf_base = AbstractAlgebra.snf(int_base)
                    base_rank_int = count(x -> x != 0, [snf_base[i, i] for i in 1:min(size(int_base)...)])
                    
                    if new_rank_int > base_rank_int
                        is_indep = true
                    end
                catch
                    is_indep = true
                end
            end
            
            if is_indep
                push!(quotient_basis, vec)
                dense_dn = test_mat
                curr_rank = new_rank
            end
        end
    else
        quotient_basis = basis
    end
    
    # Return as a Matrix{Int64} to avoid Python tuple-unpacking overhead
    if isempty(quotient_basis)
        return Matrix{Int64}(undef, d_np1_m, 0)
    end
    return hcat(quotient_basis...)
end

function rank_q_sparse(rows::AbstractVector{Int64}, cols::AbstractVector{Int64}, vals::AbstractVector{Int64}, m::Int, n::Int)
    isempty(rows) && return Int64(0)
    A = sparse(rows .+ 1, cols .+ 1, Float64.(vals), m, n)
    return Int64(rank(Matrix{Float64}(A)))
end

function _rank_mod_p_dense!(M::Matrix{Int64}, p::Int)
    m, n = size(M)
    row = 1
    rk = 0
    for col in 1:n
        pivot = 0
        for r in row:m
            if mod(M[r, col], p) != 0
                pivot = r
                break
            end
        end
        pivot == 0 && continue
        if pivot != row
            M[row, :], M[pivot, :] = copy(M[pivot, :]), copy(M[row, :])
        end

        pivot_val = mod(M[row, col], p)
        inv_pivot = invmod(pivot_val, p)
        for j in 1:n
            M[row, j] = mod(M[row, j] * inv_pivot, p)
        end

        for r in 1:m
            if r == row
                continue
            end
            factor = mod(M[r, col], p)
            factor == 0 && continue
            for j in 1:n
                M[r, j] = mod(M[r, j] - factor * M[row, j], p)
            end
        end

        row += 1
        rk += 1
        row > m && break
    end
    return Int64(rk)
end

function rank_mod_p_sparse(rows::AbstractVector{Int64}, cols::AbstractVector{Int64}, vals::AbstractVector{Int64}, m::Int, n::Int, p::Int)
    p <= 1 && error("modulus p must be > 1")
    isempty(rows) && return Int64(0)
    A = sparse(rows .+ 1, cols .+ 1, vals, m, n)
    dense = Matrix{Int64}(A)
    return _rank_mod_p_dense!(dense, p)
end

function _rref_mod_p_dense(M::Matrix{Int64}, p::Int)
    m, n = size(M)
    row = 1
    pivots = Int[]
    for col in 1:n
        pivot = 0
        for r in row:m
            if mod(M[r, col], p) != 0
                pivot = r
                break
            end
        end
        pivot == 0 && continue
        if pivot != row
            M[row, :], M[pivot, :] = copy(M[pivot, :]), copy(M[row, :])
        end

        pivot_val = mod(M[row, col], p)
        inv_pivot = invmod(pivot_val, p)
        for j in 1:n
            M[row, j] = mod(M[row, j] * inv_pivot, p)
        end

        for r in 1:m
            if r == row
                continue
            end
            factor = mod(M[r, col], p)
            factor == 0 && continue
            for j in 1:n
                M[r, j] = mod(M[r, j] - factor * M[row, j], p)
            end
        end

        push!(pivots, col)
        row += 1
        row > m && break
    end
    return M, pivots
end

function _nullspace_basis_mod_p_dense(A::Matrix{Int64}, p::Int)
    _, n = size(A)
    rref, pivots = _rref_mod_p_dense(copy(A), p)
    pivot_set = Set(pivots)
    basis = Vector{Vector{Int64}}()
    for free in 1:n
        in(free, pivot_set) && continue
        v = zeros(Int64, n)
        v[free] = 1
        for (i, col) in enumerate(pivots)
            v[col] = mod(-rref[i, free], p)
        end
        push!(basis, v)
    end
    return basis
end

function _independent_mod_p(v::Vector{Int64}, cols::Vector{Vector{Int64}}, p::Int)
    vv = mod.(v, p)
    if isempty(cols)
        return any(x -> x != 0, vv)
    end
    M_prev = hcat(cols...)
    M_new = hcat(M_prev, vv)
    return _rank_mod_p_dense!(copy(M_new), p) > _rank_mod_p_dense!(copy(M_prev), p)
end

function sparse_cohomology_basis_mod_p(
    d_np1_rows::AbstractVector{Int64}, d_np1_cols::AbstractVector{Int64}, d_np1_vals::AbstractVector{Int64}, d_np1_m::Int, d_np1_n::Int,
    d_n_rows::AbstractVector{Int64}, d_n_cols::AbstractVector{Int64}, d_n_vals::AbstractVector{Int64}, d_n_m::Int, d_n_n::Int,
    p::Int,
)
    p <= 1 && error("modulus p must be > 1")

    coboundary_mat = sparse(d_np1_cols .+ 1, d_np1_rows .+ 1, d_np1_vals, d_np1_n, d_np1_m)
    z_basis = _nullspace_basis_mod_p_dense(Matrix{Int64}(coboundary_mat), p)

    dn_mat = sparse(d_n_cols .+ 1, d_n_rows .+ 1, d_n_vals, d_n_n, d_n_m)
    dn_dense = Matrix{Int64}(dn_mat)
    b_cols = [mod.(dn_dense[:, j], p) for j in 1:size(dn_dense, 2)]

    reps = Vector{Vector{Int64}}()
    span_cols = copy(b_cols)
    for z in z_basis
        if _independent_mod_p(z, span_cols, p)
            zz = mod.(z, p)
            push!(reps, zz)
            push!(span_cols, zz)
        end
    end
    
    if isempty(reps)
        return Matrix{Int64}(undef, d_np1_m, 0)
    end
    return hcat(reps...)
end

function _compute_boundary_data_internal_flat(flat_vertices::AbstractVector{Int64}, simplex_offsets::AbstractVector{Int64}, max_dim::Int)
    # Optimized internal version working on flat buffers to avoid Tuple allocations
    n_simplices = length(simplex_offsets) - 1
    dim_simplices = Dict{Int, Matrix{Int64}}()
    dim_counts = zeros(Int, max_dim + 1)
    
    # First pass: count by dimension
    for i in 1:n_simplices
        slen = simplex_offsets[i+1] - simplex_offsets[i]
        d = slen - 1
        if 0 <= d <= max_dim
            dim_counts[d+1] += 1
        end
    end
    
    # Pre-allocate matrices
    for d in 0:max_dim
        dim_simplices[d] = Matrix{Int64}(undef, d + 1, dim_counts[d+1])
    end
    
    # Second pass: fill
    dim_cursors = ones(Int, max_dim + 1)
    for i in 1:n_simplices
        lo, hi = simplex_offsets[i] + 1, simplex_offsets[i+1]
        slen = hi - lo + 1
        d = slen - 1
        if 0 <= d <= max_dim
            cursor = dim_cursors[d+1]
            # Copy sorted vertices into the matrix column
            vs = sort(flat_vertices[lo:hi])
            dim_simplices[d][:, cursor] .= vs
            dim_cursors[d+1] += 1
        end
    end

    cells = Dict{Int, Int64}()
    simplex_to_idx = Dict{Int, Dict{Vector{Int64}, Int64}}()
    for d in 0:max_dim
        cells[d] = Int64(size(dim_simplices[d], 2))
        idx_map = Dict{Vector{Int64}, Int64}()
        for j in 1:size(dim_simplices[d], 2)
            idx_map[dim_simplices[d][:, j]] = Int64(j - 1)
        end
        simplex_to_idx[d] = idx_map
    end

    boundaries = Dict{Int, Dict{String, Any}}()
    for k in 1:max_dim
        n_rows = Int64(get(cells, k - 1, 0))
        n_cols = Int64(get(cells, k, 0))
        if n_rows == 0 || n_cols == 0
            continue
        end

        rows = Int64[]
        cols = Int64[]
        data = Int64[]
        prev_dim_map = simplex_to_idx[k - 1]
        
        simplices_k = dim_simplices[k]
        for j in 1:n_cols
            verts = simplices_k[:, j]
            for i in eachindex(verts)
                # Create face by dropping one vertex
                face = vcat(verts[1:(i - 1)], verts[(i + 1):end])
                if haskey(prev_dim_map, face)
                    push!(rows, prev_dim_map[face])
                    push!(cols, Int64(j - 1))
                    push!(data, isodd(i - 1) ? -1 : 1)
                end
            end
        end

        boundaries[k] = Dict(
            "rows" => rows,
            "cols" => cols,
            "data" => data,
            "n_rows" => n_rows,
            "n_cols" => n_cols,
        )
    end

    return boundaries, cells, dim_simplices, simplex_to_idx
end

function compute_boundary_payload_from_simplices(simplex_entries, max_dim::Int, include_metadata::Bool=true)
    # Legacy wrapper for list-of-tuples
    dim_simplices_list = Dict{Int, Vector{Tuple{Vararg{Int64}}}}()
    for d in 0:max_dim; dim_simplices_list[d] = Tuple{Vararg{Int64}}[]; end
    for s in simplex_entries
        vs = _to_vertices_simplex(s); isempty(vs) && continue; sort!(vs)
        d = length(vs) - 1
        if 0 <= d <= max_dim; push!(dim_simplices_list[d], Tuple(Int64.(vs))); end
    end
    
    # For simplicity, we just use the original internal logic if input is not flat
    boundaries, cells, dim_simplices, simplex_to_idx = _compute_boundary_data_internal(simplex_entries, max_dim)
    if include_metadata
        return boundaries, cells, dim_simplices, simplex_to_idx
    end
    return boundaries, cells
end

function compute_boundary_payload_from_flat_simplices(flat_vertices::AbstractVector{Int64}, simplex_offsets::AbstractVector{Int64}, max_dim::Int, include_metadata::Bool=true)
    boundaries, cells, dim_simplices, simplex_to_idx = _compute_boundary_data_internal_flat(flat_vertices, simplex_offsets, max_dim)
    if include_metadata
        return boundaries, cells, dim_simplices, simplex_to_idx
    end
    return boundaries, cells
end

function compute_boundary_data_from_simplices(simplex_entries, max_dim::Int)
    return _compute_boundary_data_internal(simplex_entries, max_dim)
end

function group_ring_multiply(py_coeffs1::Py, py_coeffs2::Py, group_order::Int)
    # Use pyconvert to escape slow PyDict wrapper immediately
    coeffs1 = pyconvert(Dict{String, Int}, py_coeffs1)
    coeffs2 = pyconvert(Dict{String, Int}, py_coeffs2)
    
    res_dict = Dict{String, Int}()
    for (k1, v1) in coeffs1
        for (k2, v2) in coeffs2
            function parse_gen(g_str)
                if g_str == "e" || g_str == "1"
                    return 0
                end
                m = match(r"g_?(\d+)(?:\^-1)?", g_str)
                if m === nothing
                    throw(ArgumentError("Invalid generator format: " * g_str))
                end
                val = parse(Int, m.captures[1])
                inv = endswith(g_str, "^-1")
                return inv ? -val : val
            end
            
            p1 = parse_gen(k1)
            p2 = parse_gen(k2)
            p_res = (p1 + p2) % group_order
            if p_res < 0; p_res += group_order; end
            g_str = p_res == 0 ? "1" : "g_$(p_res)"
            
            res_dict[g_str] = get(res_dict, g_str, 0) + v1 * v2
        end
    end
    
    # Filter zeros and return parallel vectors
    final_k = String[]
    final_v = Int[]
    for (k, v) in res_dict
        if v != 0
            push!(final_k, k)
            push!(final_v, v)
        end
    end
    return final_k, final_v
end

# Fallback for explicit vectors
function group_ring_multiply(k1::AbstractVector, v1::AbstractVector, k2::AbstractVector, v2::AbstractVector, group_order::Int)
    k1s = String[string(x) for x in k1]
    v1i = Int[Int(x) for x in v1]
    k2s = String[string(x) for x in k2]
    v2i = Int[Int(x) for x in v2]
    
    res_dict = Dict{String, Int}()
    for i in 1:length(k1s)
        for j in 1:length(k2s)
            function parse_gen(g_str)
                if g_str == "e" || g_str == "1"
                    return 0
                end
                m = match(r"g_?(\d+)(?:\^-1)?", g_str)
                if m === nothing; return 0; end
                val = parse(Int, m.captures[1])
                return endswith(g_str, "^-1") ? -val : val
            end
            p1, p2 = parse_gen(k1s[i]), parse_gen(k2s[j])
            p_res = mod(p1 + p2, group_order)
            g_str = p_res == 0 ? "1" : "g_$(p_res)"
            res_dict[g_str] = get(res_dict, g_str, 0) + v1i[i] * v2i[j]
        end
    end
    final_k = String[]; final_v = Int[]
    for (k, v) in res_dict
        if v != 0; push!(final_k, k); push!(final_v, v); end
    end
    return final_k, final_v
end

function multisignature(matrix::AbstractMatrix{Float64}, p::Int)
    mat = Matrix{Float64}(matrix)
    n = size(mat, 1)
    total = 0
    for k in 1:(p-1)
        omega = exp(2π * im * k / p)
        H = [mat[i,j] * omega^(i-j) for i in 1:n, j in 1:n]
        evals = real.(eigvals(Hermitian(H)))
        tol = n * eps(Float64) * maximum(abs.(evals))
        total += sum(evals .> tol) - sum(evals .< -tol)
    end
    return total
end

function integral_lattice_isometry(matrix1::AbstractMatrix{Int64}, matrix2::AbstractMatrix{Int64})
    A = Matrix{Int64}(matrix1)
    B = Matrix{Int64}(matrix2)
    n = size(A, 1)
    if size(A, 2) != n || size(B, 1) != n || size(B, 2) != n; return nothing; end

    evals_a = eigvals(Hermitian(Matrix{Float64}(A)))
    evals_b = eigvals(Hermitian(Matrix{Float64}(B)))
    scale = max(1.0, maximum(abs.(vcat(evals_a, evals_b))))
    tol = n * eps(Float64) * scale

    pos_a = all(x -> x > tol, evals_a); neg_a = all(x -> x < -tol, evals_a)
    pos_b = all(x -> x > tol, evals_b); neg_b = all(x -> x < -tol, evals_b)
    if !((pos_a && pos_b) || (neg_a && neg_b)); return nothing; end

    Adef = pos_a ? A : -A; Bdef = pos_b ? B : -B
    lam_min = minimum(eigvals(Hermitian(Matrix{Float64}(Adef))))
    if lam_min <= 0; return nothing; end

    targets = [Int(Bdef[i, i]) for i in 1:n]
    if any(t -> t <= 0, targets); return nothing; end

    radius = maximum([Int(floor(sqrt(t / lam_min))) + 1 for t in targets])
    range_vals = collect(-radius:radius)
    vectors_by_norm = Dict{Int, Vector{Vector{Int64}}}()
    for t in unique(targets); vectors_by_norm[t] = Vector{Vector{Int64}}(); end

    for tup in Iterators.product(ntuple(_ -> range_vals, n)...)
        v = Int64[tup...]
        qv = Int(dot(v, Adef * v))
        if haskey(vectors_by_norm, qv)
            push!(vectors_by_norm[qv], v)
            if length(vectors_by_norm[qv]) > 20000; return nothing; end
        end
    end

    if any(t -> isempty(vectors_by_norm[t]), targets); return nothing; end
    order = sortperm(1:n; by = j -> length(vectors_by_norm[targets[j]]))
    cols = [zeros(Int64, n) for _ in 1:n]; chosen_original_indices = Int[]

    function backtrack(pos::Int)
        if pos > n
            U = hcat(cols...)
            d = round(Int, det(Matrix{Float64}(U)))
            if abs(d) != 1; return nothing; end
            return transpose(U) * A * U == B ? U : nothing
        end
        j = order[pos]
        for v in vectors_by_norm[targets[j]]
            ok = true
            for i in chosen_original_indices
                if Int(dot(cols[i], Adef * v)) != Int(Bdef[i, j]); ok = false; break; end
            end
            if !ok; continue; end
            cols[j] = v; push!(chosen_original_indices, j)
            out = backtrack(pos + 1)
            if out !== nothing; return out; end
            pop!(chosen_original_indices)
        end
        return nothing
    end
    return backtrack(1)
end

const HEdge = NTuple{2,Int}
const HTriangle = NTuple{3,Int}
const HCycle = Vector{HEdge}

@inline _order_hedge(u::Int, v::Int) = u <= v ? (u, v) : (v, u)

function _to_vertices_simplex(s)
    if s isa Tuple; return [Int(x) for x in s]
    elseif s isa AbstractVector; return [Int(x) for x in s]
    else; return Int[]; end
end

function _normalize_simplex_to_idx_dict(raw)
    try; return pyconvert(Dict{Tuple{Vararg{Int}}, Int64}, raw)
    catch; end
    out = Dict{Tuple{Vararg{Int}}, Int}()
    for (k, v) in raw
        vs = _to_vertices_simplex(k); isempty(vs) && continue
        out[Tuple(vs)] = Int(v)
    end
    return out
end

function _edge_weight_h1(u::Int, v::Int, points)
    if points === nothing; return 1.0; end
    s = 0.0
    @inbounds @simd for j in axes(points, 2)
        d = points[u + 1, j] - points[v + 1, j]
        s += d * d
    end
    return sqrt(s)
end

function _normalize_edges_triangles(simplices)
    edges = HEdge[]; triangles = HTriangle[]; vertex_ids = Set{Int}()
    for s in simplices
        vs = _to_vertices_simplex(s)
        if length(vs) == 2
            e = _order_hedge(vs[1], vs[2]); push!(edges, e)
            push!(vertex_ids, e[1]); push!(vertex_ids, e[2])
        elseif length(vs) == 3
            sort!(vs); t = (vs[1], vs[2], vs[3]); push!(triangles, t)
            push!(vertex_ids, t[1]); push!(vertex_ids, t[2]); push!(vertex_ids, t[3])
        end
    end
    unique!(edges); unique!(triangles)
    return edges, triangles, vertex_ids
end

mutable struct HDSU
    parent::Vector{Int}
    rank::Vector{UInt8}
end
HDSU(n::Int) = HDSU(collect(1:n), fill(UInt8(0), n))

@inline function _find_h!(d::HDSU, x::Int)
    px = d.parent[x]; px == x && return x
    r = _find_h!(d, px); d.parent[x] = r; return r
end

@inline function _unite_h!(d::HDSU, a::Int, b::Int)
    ra = _find_h!(d, a); rb = _find_h!(d, b); ra == rb && return false
    if d.rank[ra] < d.rank[rb]; ra, rb = rb, ra; end
    d.parent[rb] = ra; d.rank[ra] == d.rank[rb] && (d.rank[ra] += UInt8(1))
    return true
end

function _minimum_spanning_edges_h(edges::Vector{HEdge}, weights::Dict{HEdge,Float64}, num_vertices::Int)
    order = sortperm(edges; by = e -> get(weights, e, 1.0))
    dsu = HDSU(max(num_vertices, 1)); spanning = Set{HEdge}()
    for idx in order
        u, v = edges[idx]
        if _unite_h!(dsu, u + 1, v + 1); push!(spanning, (u, v)); end
    end
    return spanning
end

function _annot_edge_h(simplices, num_vertices::Int, edge_weights::Dict{HEdge,Float64})
    edges, triangles, _ = _normalize_edges_triangles(simplices)
    weights = Dict{HEdge,Float64}()
    for e in edges; weights[e] = get(edge_weights, e, 1.0); end
    spanning = _minimum_spanning_edges_h(edges, weights, num_vertices)
    non_tree = [e for e in edges if !in(e, spanning)]; m = length(non_tree)
    annotations = Dict{HEdge,BitVector}()
    if m == 0; (for e in edges; annotations[e] = BitVector(); end; return annotations, 0); end
    for e in spanning; annotations[e] = falses(m); end
    for (i, e) in enumerate(non_tree); v = falses(m); v[i] = true; annotations[e] = v; end
    active = trues(m); boundary = falses(m)
    for t in triangles
        u, v, w = t; e1, e2, e3 = _order_hedge(u, v), _order_hedge(v, w), _order_hedge(u, w)
        fill!(boundary, false)
        haskey(annotations, e1) && (boundary .⊻= annotations[e1])
        haskey(annotations, e2) && (boundary .⊻= annotations[e2])
        haskey(annotations, e3) && (boundary .⊻= annotations[e3])
        pivot = 0
        @inbounds for i in 1:m; (boundary[i] && active[i]) || continue; pivot = i; break; end
        pivot == 0 && continue
        for (e, vec) in annotations; vec[pivot] && (vec .⊻= boundary); end
        active[pivot] = false
    end
    final = Dict{HEdge,BitVector}()
    for (e, vec) in annotations; final[e] = vec[active]; end
    return final, count(active)
end

function _shortest_path_tree_h(root::Int, adjacency::Dict{Int,Vector{Tuple{Int,Float64}}})
    dist = Dict{Int,Float64}(root => 0.0); parent = Dict{Int,Int}(root => -1); heap = [(0.0, root)]
    while !isempty(heap)
        sort!(heap, by = x -> x[1]); d, u = popfirst!(heap)
        d != dist[u] && continue
        for (v, w) in get(adjacency, u, Tuple{Int,Float64}[])
            nd = d + w
            if nd < get(dist, v, Inf); dist[v] = nd; parent[v] = u; push!(heap, (nd, v)); end
        end
    end
    tree_edges = Set{HEdge}()
    for (child, par) in parent; par == -1 && continue; push!(tree_edges, _order_hedge(child, par)); end
    return parent, tree_edges
end

function _path_between_h_vertices(u::Int, v::Int, parent::Dict{Int,Int})
    path_u = Int[]; seen_u = Set{Int}(); x = u
    while x != -1; push!(path_u, x); push!(seen_u, x); x = get(parent, x, -1); end
    path_v = Int[]; y = v
    while !in(y, seen_u) && y != -1; push!(path_v, y); y = get(parent, y, -1); end
    y == -1 && return Int[]; i = findfirst(==(y), path_u)
    return vcat(path_u[1:i], reverse(path_v))
end

function _path_edges_h(path_vertices::Vector{Int})
    edges = HEdge[]
    for i in 1:(length(path_vertices) - 1); push!(edges, _order_hedge(path_vertices[i], path_vertices[i + 1])); end
    return edges
end

function _generator_cycles_h(simplices, num_vertices::Int, points; max_roots=nothing, root_stride::Int=1, max_cycles=nothing)
    edges, _, vertex_ids = _normalize_edges_triangles(simplices); isempty(edges) && return HCycle[]
    adjacency = Dict{Int,Vector{Tuple{Int,Float64}}}()
    for (u, v) in edges
        w = _edge_weight_h1(u, v, points)
        push!(get!(adjacency, u, Tuple{Int,Float64}[]), (v, w)); push!(get!(adjacency, v, Tuple{Int,Float64}[]), (u, w))
    end
    vertices = isempty(vertex_ids) ? collect(0:max(num_vertices - 1, 0)) : sort(collect(vertex_ids))
    selected_roots = vertices[begin:max(root_stride, 1):end]
    if max_roots !== nothing; selected_roots = selected_roots[1:min(Int(max_roots), length(selected_roots))]; end
    cycles = HCycle[]
    for root in selected_roots
        parent, tree_edges = _shortest_path_tree_h(root, adjacency)
        for (u, v) in edges
            e = _order_hedge(u, v); if in(e, tree_edges) || !haskey(parent, u) || !haskey(parent, v); continue; end
            path_vertices = _path_between_h_vertices(u, v, parent); length(path_vertices) < 2 && continue
            cyc = HEdge[e]; append!(cyc, _path_edges_h(path_vertices)); push!(cycles, cyc)
            if max_cycles !== nothing && length(cycles) >= Int(max_cycles); return cycles; end
        end
    end
    return cycles
end

function _cycle_annotation_h(cycle::HCycle, simplex_annotations::Dict{HEdge,BitVector}, len::Int)
    ann = falses(len)
    for e in cycle; se = _order_hedge(e[1], e[2]); haskey(simplex_annotations, se) && (ann .⊻= simplex_annotations[se]); end
    return ann
end

function _is_independent_h!(cv::BitVector, pivots::Dict{Int,BitVector})
    for i in eachindex(cv)
        cv[i] || continue
        if haskey(pivots, i); cv .⊻= pivots[i]
        else; pivots[i] = copy(cv); return true; end
    end
    return false
end

function _greedy_basis_h(cycles::Vector{HCycle}, simplices, num_vertices::Int, points)
    edges, triangles, vertex_ids = _normalize_edges_triangles(simplices)
    if isempty(edges)
        for cyc in cycles; append!(edges, cyc); for (u, v) in cyc; push!(vertex_ids, u); push!(vertex_ids, v); end; end
    end
    if num_vertices <= 0 && !isempty(vertex_ids); num_vertices = maximum(vertex_ids) + 1; end
    ew = Dict{HEdge,Float64}()
    for e in edges; ew[e] = _edge_weight_h1(e[1], e[2], points); end
    simplices_all = Any[]; append!(simplices_all, edges); append!(simplices_all, triangles)
    simplex_annotations, vec_dim = _annot_edge_h(simplices_all, num_vertices, ew)
    cycles_sorted = sort(cycles; by = cyc -> sum(_edge_weight_h1(u, v, points) for (u, v) in cyc))
    basis = HCycle[]; pivots = Dict{Int,BitVector}()
    for cyc in cycles_sorted
        ann = _cycle_annotation_h(cyc, simplex_annotations, vec_dim)
        _is_independent_h!(ann, pivots) && push!(basis, cyc)
    end
    return basis
end

function optgen_from_simplices(simplices, num_vertices::Int, point_cloud=nothing, max_roots=nothing, root_stride::Int=1, max_cycles=nothing)
    points = point_cloud === nothing ? nothing : Matrix{Float64}(point_cloud)
    cycles = _generator_cycles_h(simplices, num_vertices, points; max_roots=max_roots, root_stride=root_stride, max_cycles=max_cycles)
    return _greedy_basis_h(cycles, simplices, num_vertices, points)
end

function _simplices_by_dim(simplices)
    out = Dict{Int, Vector{Vector{Int}}}()
    for s in simplices
        vs = _to_vertices_simplex(s); isempty(vs) && continue; sort!(vs)
        for r in 1:length(vs)
            v = get!(out, r - 1, Vector{Vector{Int}}())
            for face_vec in _k_subsets(vs, r); if !any(x -> x == face_vec, v); push!(v, face_vec); end; end
        end
    end
    return out
end

function _k_subsets(v::Vector{Int}, k::Int)
    out = Vector{Vector{Int}}(); n = length(v)
    if k <= 0 || k > n; return out; end
    idx = collect(1:k)
    while true
        push!(out, [v[i] for i in idx]); i = k
        while i >= 1 && idx[i] == i + n - k; i -= 1; end
        i == 0 && break; idx[i] += 1
        for j in (i + 1):k; idx[j] = idx[j - 1] + 1; end
    end
    return out
end

function _boundary_mod2(source::Vector{Vector{Int}}, target::Vector{Vector{Int}})
    m, n = length(target), length(source); M = zeros(Int, m, n); (m == 0 || n == 0) && return M
    t_idx = Dict{Tuple{Vararg{Int}}, Int}()
    for (i, t) in enumerate(target); t_idx[Tuple(t)] = i; end
    for (j, s) in enumerate(source)
        for drop in eachindex(s)
            face = [s[i] for i in eachindex(s) if i != drop]
            row = get(t_idx, Tuple(face), 0); row == 0 && continue; M[row, j] ⊻= 1
        end
    end
    return M
end

function _rref_mod2(A::Matrix{Int})
    M = A .% 2; m, n = size(M); row = 1; pivots = Int[]
    for col in 1:n
        pivot = 0; for r in row:m; if M[r, col] == 1; pivot = r; break; end; end
        pivot == 0 && continue
        if pivot != row; M[row, :], M[pivot, :] = copy(M[pivot, :]), copy(M[row, :]); end
        for r in 1:m; if r != row && M[r, col] == 1; M[r, :] .⊻= M[row, :] ; end ; end
        push!(pivots, col); row += 1; row > m && break
    end
    return M, pivots
end

function _nullspace_basis_mod2(A::Matrix{Int})
    _, n = size(A); rref, pivots = _rref_mod2(A); pivot_set = Set(pivots); basis = Vector{Vector{Int}}()
    for free in 1:n
        in(free, pivot_set) && continue; v = zeros(Int, n); v[free] = 1
        for (i, col) in enumerate(pivots); v[col] = rref[i, free] % 2; end
        push!(basis, v)
    end
    return basis
end

_rank_mod2(A::Matrix{Int}) = length(last(_rref_mod2(A)))

function _independent_mod2(v::Vector{Int}, cols::Vector{Vector{Int}})
    vv = v .% 2; if isempty(cols); return any(x -> x != 0, vv); end
    M_prev = hcat(cols...); M_new = hcat(M_prev, vv); return _rank_mod2(M_new) > _rank_mod2(M_prev)
end

function _chain_weight(chain::Vector{Int}, simplices_k::Vector{Vector{Int}}, points)
    active = [simplices_k[i] for i in eachindex(chain) if chain[i] % 2 == 1]; isempty(active) && return 0.0
    points === nothing && return float(length(active))
    total = 0.0
    for s in active
        if length(s) <= 1; total += 1.0; continue; end
        for i in 1:length(s); for j in (i + 1):length(s); u, v = s[i] + 1, s[j] + 1; total += norm(points[u, :] .- points[v, :]); end; end
    end
    return total
end

function _h0_generators(edges::Vector{HEdge}, num_vertices::Int)
    num_vertices <= 0 && return Vector{Dict{String, Any}}()
    dsu = HDSU(num_vertices); for (u, v) in edges; if 0 <= u < num_vertices && 0 <= v < num_vertices; _unite_h!(dsu, u + 1, v + 1); end; end
    comps = Dict{Int, Vector{Int}}()
    for v in 0:(num_vertices - 1); r = _find_h!(dsu, v + 1); push!(get!(comps, r, Int[]), v); end
    out = Vector{Dict{String, Any}}()
    for verts in values(comps)
        rep = minimum(verts)
        push!(out, Dict("dimension" => 0, "support_simplices" => [Any[rep]], "support_edges" => Any[], "weight" => 0.0, "certified_cycle" => true))
    end
    return out
end

function homology_generators_from_simplices(simplices, num_vertices::Int, dimension::Int, mode::String="valid", point_cloud=nothing, max_roots=nothing, root_stride::Int=1, max_cycles=nothing)
    points = point_cloud === nothing ? nothing : Matrix{Float64}(point_cloud)
    dimension < 0 && error("dimension must be >= 0"); (mode != "valid" && mode != "optimal") && error("mode must be 'valid' or 'optimal'")
    if dimension == 1 && mode == "optimal"
        basis = optgen_from_simplices(simplices, num_vertices, points, max_roots, root_stride, max_cycles)
        out = Vector{Dict{String, Any}}()
        for cyc in basis
            support = Any[[e[1], e[2]] for e in cyc]
            push!(out, Dict("dimension" => 1, "support_simplices" => support, "support_edges" => support, "weight" => sum(_edge_weight_h1(e[1], e[2], points) for e in cyc), "certified_cycle" => true))
        end
        return out
    end
    by_dim = _simplices_by_dim(simplices)
    if dimension == 0
        edges = HEdge[]
        for e in get(by_dim, 1, Vector{Vector{Int}}()); length(e) == 2 && push!(edges, _order_hedge(e[1], e[2])); end
        nv = num_vertices; if nv <= 0; mx = -1; for s in simplices; for v in _to_vertices_simplex(s); mx = max(mx, v); end; end; nv = mx + 1; end
        return _h0_generators(edges, nv)
    end
    simplices_k = get(by_dim, dimension, Vector{Vector{Int}}()); simplices_km1 = get(by_dim, dimension - 1, Vector{Vector{Int}}()); simplices_kp1 = get(by_dim, dimension + 1, Vector{Vector{Int}}())
    isempty(simplices_k) && return Vector{Dict{String, Any}}()
    d_k = _boundary_mod2(simplices_k, simplices_km1); d_kp1 = _boundary_mod2(simplices_kp1, simplices_k)
    z_basis = _nullspace_basis_mod2(d_k); isempty(z_basis) && return Vector{Dict{String, Any}}()
    b_cols = [d_kp1[:, j] .% 2 for j in 1:size(d_kp1, 2)]; z_candidates = copy(z_basis)
    if mode == "optimal"; sort!(z_candidates, by = z -> _chain_weight(z, simplices_k, points)); end
    reps = Vector{Vector{Int}}(); span_cols = copy(b_cols)
    for z in z_candidates; if _independent_mod2(z, span_cols); push!(reps, z); push!(span_cols, z .% 2); end; end
    out = Vector{Dict{String, Any}}()
    for z in reps
        support = Any[]; for i in eachindex(z); z[i] % 2 == 1 && push!(support, Any[simplices_k[i]...]); end
        support_edges = dimension == 1 ? support : Any[]
        push!(out, Dict("dimension" => dimension, "support_simplices" => support, "support_edges" => support_edges, "weight" => _chain_weight(z, simplices_k, points), "certified_cycle" => true))
    end
    return out
end

function abelianize_group(generators::Vector{String}, relations::Vector{String})
    n_gens = length(generators); gen_idx = Dict{String, Int}(g => i for (i, g) in enumerate(generators))
    n_rels = length(relations); M = zeros(Int, n_rels, n_gens)
    for i in 1:n_rels; for m in eachmatch(r"([a-zA-Z0-9_]+)(?:\^(-?\d+))?", relations[i])
        base_w = m.captures[1]; if haskey(gen_idx, base_w); pow_str = m.captures[2]; pow = pow_str === nothing ? 1 : parse(Int, pow_str); M[i, gen_idx[base_w]] += pow; end
    end; end
    if !HAS_ABSTRACT_ALGEBRA; rq = rank(Matrix{Float64}(M)); return Int(n_gens - rq), Int[]; end
    ZZ = AbstractAlgebra.ZZ; M_aa = AbstractAlgebra.matrix(ZZ, M); S_aa = AbstractAlgebra.snf(M_aa)
    diag = [Int64(S_aa[i, i]) for i in 1:min(n_rels, n_gens)]; nonzero = filter(x -> x != 0, diag); torsion = filter(x -> x > 1, nonzero)
    return n_gens - length(nonzero), torsion
end

function compute_boundary_mod2_matrix(source_simplices, target_simplices)
    source = [_to_vertices_simplex(s) for s in source_simplices]; target = [_to_vertices_simplex(t) for t in target_simplices]
    m, n = length(target), length(source); if m == 0 || n == 0; return Dict("rows" => Int64[], "cols" => Int64[], "data" => Int64[], "m" => Int64(m), "n" => Int64(n)); end
    t_idx = Dict{Tuple{Vararg{Int}}, Int}(); for (i, t) in enumerate(target); t_idx[Tuple(sort(t))] = i - 1; end
    rows, cols, data = Int64[], Int64[], Int64[]
    for (j, s) in enumerate(source); for i_drop in eachindex(s); face_vec = vcat(s[1:(i_drop-1)], s[(i_drop+1):end]); face = Tuple(sort(Int.(face_vec)))
        if haskey(t_idx, face); push!(rows, t_idx[face]); push!(cols, j - 1); push!(data, 1); end
    end; end
    return Dict("rows" => rows, "cols" => cols, "data" => data, "m" => Int64(m), "n" => Int64(n))
end

function compute_alexander_whitney_cup(alpha::AbstractVector, beta::AbstractVector, p::Int, q::Int, simplices_p_plus_q, simplex_to_idx_p, simplex_to_idx_q, modulus=nothing)
    idx_p_map, idx_q_map = _normalize_simplex_to_idx_dict(simplex_to_idx_p), _normalize_simplex_to_idx_dict(simplex_to_idx_q)
    n_simplices = length(simplices_p_plus_q); result = zeros(Int64, n_simplices)
    for i in 1:n_simplices
        simplex = _to_vertices_simplex(simplices_p_plus_q[i]); length(simplex) < p + q + 1 && continue
        front_face, back_face = Tuple(simplex[1:(p+1)]), Tuple(simplex[(p+1):(p+q+1)])
        idx_p, idx_q = get(idx_p_map, front_face, -1), get(idx_q_map, back_face, -1)
        if idx_p != -1 && idx_q != -1; val = Int64(alpha[idx_p + 1]) * Int64(beta[idx_q + 1]); if modulus !== nothing; val = val % Int64(modulus); end; result[i] = val; end
    end
    return result
end

function compute_trimesh_boundary_data(faces, n_vertices::Int)
    n_faces = length(faces); edge_to_idx = Dict{Tuple{Int, Int}, Int}(); edges = Tuple{Int, Int}[]; face_boundary_edges_list = Vector{Vector{Tuple{Int, Int, Int}}}()
    for face_vec in faces
        face_arr = _to_vertices_simplex(face_vec); cycle = Int.(face_arr); cyc_edges = Tuple{Int, Int, Int}[]
        for i in 1:length(cycle); u, v = cycle[i], cycle[mod(i, length(cycle)) + 1]; key = (u, v) <= (v, u) ? (u, v) : (v, u)
            if !haskey(edge_to_idx, key); edge_to_idx[key] = length(edges) + 1; push!(edges, key); end
            push!(cyc_edges, (u, v, edge_to_idx[key])); end
        push!(face_boundary_edges_list, cyc_edges)
    end
    n_edges = length(edges); d1_rows, d1_cols, d1_data = Int64[], Int64[], Int64[]
    for (j, (v1, v2)) in enumerate(edges); push!(d1_rows, v1 - 1); push!(d1_cols, j - 1); push!(d1_data, -1); push!(d1_rows, v2 - 1); push!(d1_cols, j - 1); push!(d1_data, 1); end
    d2_rows, d2_cols, d2_data = Int64[], Int64[], Int64[]
    for (j, edge_cycle) in enumerate(face_boundary_edges_list); for (u, v, idx) in edge_cycle; sign = (u, v) == edges[idx] ? 1 : -1; push!(d2_rows, idx - 1); push!(d2_cols, j - 1); push!(d2_data, sign); end; end
    return Dict("d1_rows" => d1_rows, "d1_cols" => d1_cols, "d1_data" => d1_data, "n_vertices" => Int64(n_vertices), "n_edges" => Int64(n_edges), "d2_rows" => d2_rows, "d2_cols" => d2_cols, "d2_data" => d2_data, "n_faces" => Int64(n_faces))
end

function compute_trimesh_boundary_data_flat(face_vertices::AbstractVector{Int64}, face_offsets::AbstractVector{Int64}, n_vertices::Int)
    # Optimized internal to avoid Tuple list creation
    n_faces = length(face_offsets) - 1
    edge_to_idx = Dict{Tuple{Int, Int}, Int}()
    edges = Tuple{Int, Int}[]
    
    d2_rows, d2_cols, d2_data = Int64[], Int64[], Int64[]
    for j in 1:n_faces
        lo, hi = face_offsets[j] + 1, face_offsets[j+1]
        cycle = face_vertices[lo:hi]
        for i in 1:length(cycle)
            u = Int(cycle[i])
            v = Int(cycle[mod(i, length(cycle)) + 1])
            key = u <= v ? (u, v) : (v, u)
            if !haskey(edge_to_idx, key)
                push!(edges, key)
                edge_to_idx[key] = length(edges)
            end
            idx = edge_to_idx[key]
            sign = (u, v) == edges[idx] ? 1 : -1
            push!(d2_rows, idx - 1)
            push!(d2_cols, j - 1)
            push!(d2_data, sign)
        end
    end
    
    n_edges = length(edges)
    d1_rows, d1_cols, d1_data = Int64[], Int64[], Int64[]
    for (j, (v1, v2)) in enumerate(edges)
        push!(d1_rows, v1); push!(d1_cols, j - 1); push!(d1_data, -1)
        push!(d1_rows, v2); push!(d1_cols, j - 1); push!(d1_data, 1)
    end
    
    return Dict("d1_rows" => d1_rows, "d1_cols" => d1_cols, "d1_data" => d1_data, "n_vertices" => Int64(n_vertices), "n_edges" => Int64(n_edges), "d2_rows" => d2_rows, "d2_cols" => d2_cols, "d2_data" => d2_data, "n_faces" => Int64(n_faces))
end

function triangulate_surface_delaunay(points::AbstractMatrix{Float64}, tolerance::Real=1e-10)
    centroid = mean(points, dims=1); centered = points .- centroid; U, S, Vt = svd(centered)
    v1, v2 = Vt[1, :], Vt[2, :]; projected_2d = hcat(centered * v1, centered * v2)
    try
        n_points = size(points, 1); centroid_2d = vec(mean(projected_2d, dims=1))
        angles = [atan(projected_2d[i, 2] - centroid_2d[2], projected_2d[i, 1] - centroid_2d[1]) for i in 1:n_points]
        sorted_idx = sortperm(angles); triangles = []
        for i in 2:(n_points - 1); push!(triangles, sort([sorted_idx[1], sorted_idx[i], sorted_idx[i + 1]])); end
        return unique!(triangles)
    catch e; error("Triangulation failed: $(e)"); end
end

# Enhanced Precompilation Workload
PrecompileTools.@setup_workload begin
    rows = Int64[0, 1]; cols = Int64[0, 1]; vals = Int64[1, 1]
    simplices = [(0,), (1,), (2,), (0, 1), (1, 2), (0, 2), (0, 1, 2)]
    flat_vertices = Int64[0, 1, 2, 0, 1, 1, 2, 0, 2, 0, 1, 2]
    simplex_offsets = Int64[0, 1, 2, 3, 5, 7, 9, 12]
    simplex_to_idx = Dict((0, 1) => 0, (1, 2) => 1, (0, 2) => 2)
    alpha = Int64[1, 1, 1]; beta = Int64[1, 1, 1]
    mat_f = ones(2, 2); mat_i = ones(Int64, 2, 2)

    PrecompileTools.@compile_workload begin
        exact_snf_sparse(rows, cols, vals, 2, 2)
        rank_q_sparse(rows, cols, vals, 2, 2)
        rank_mod_p_sparse(rows, cols, vals, 2, 2, 2)
        compute_boundary_payload_from_flat_simplices(flat_vertices, simplex_offsets, 2, false)
        compute_alexander_whitney_cup(alpha, beta, 1, 1, [(0, 1, 2)], simplex_to_idx, simplex_to_idx, nothing)
        exact_sparse_cohomology_basis(rows, cols, vals, 2, 2, rows, cols, vals, 2, 2)
        sparse_cohomology_basis_mod_p(rows, cols, vals, 2, 2, rows, cols, vals, 2, 2, 2)
        multisignature(mat_f, 2)
        integral_lattice_isometry(mat_i, mat_i)
    end
end

function _compute_boundary_data_internal(simplex_entries, max_dim::Int)
    dim_simplices = Dict{Int, Vector{Tuple{Vararg{Int64}}}}()
    for d in 0:max_dim; dim_simplices[d] = Tuple{Vararg{Int64}}[]; end
    for s in simplex_entries
        vs = _to_vertices_simplex(s); isempty(vs) && continue; sort!(vs); d = length(vs) - 1
        if 0 <= d <= max_dim; push!(dim_simplices[d], Tuple(Int64.(vs))); end
    end
    cells = Dict{Int, Int64}(); simplex_to_idx = Dict{Int, Dict{Tuple{Vararg{Int64}}, Int64}}()
    for d in 0:max_dim
        simplices_d = dim_simplices[d]; sort!(simplices_d); cells[d] = Int64(length(simplices_d))
        idx_map = Dict{Tuple{Vararg{Int64}}, Int64}()
        for (i, simplex) in enumerate(simplices_d); idx_map[simplex] = Int64(i - 1); end
        simplex_to_idx[d] = idx_map
    end
    boundaries = Dict{Int, Dict{String, Any}}()
    for k in 1:max_dim
        n_rows, n_cols = Int64(get(cells, k - 1, 0)), Int64(get(cells, k, 0))
        if n_rows == 0 || n_cols == 0; continue; end
        rows, cols, data = Int64[], Int64[], Int64[]; prev_dim_map = simplex_to_idx[k - 1]
        for (j, simplex) in enumerate(dim_simplices[k])
            verts = collect(simplex)
            for i in eachindex(verts)
                face = Tuple(vcat(verts[1:(i - 1)], verts[(i + 1):end]))
                if haskey(prev_dim_map, face); push!(rows, prev_dim_map[face]); push!(cols, Int64(j - 1)); push!(data, isodd(i - 1) ? -1 : 1); end
            end
        end
        boundaries[k] = Dict("rows" => rows, "cols" => cols, "data" => data, "n_rows" => n_rows, "n_cols" => n_cols)
    end
    return boundaries, cells, dim_simplices, simplex_to_idx
end

end # module SurgeryBackend
