# A lightweight Julia backend module for high-performance algebraic operations over Z[pi]
module SurgeryBackend

using LinearAlgebra
using SparseArrays
using Statistics
using Combinatorics
using PythonCall: pyconvert, Py
import PrecompileTools

export hermitian_signature, exact_snf_sparse, exact_sparse_cohomology_basis, rank_q_sparse, rank_mod_p_sparse, sparse_cohomology_basis_mod_p, group_ring_multiply, multisignature, abelianize_group, integral_lattice_isometry, optgen_from_simplices, homology_generators_from_simplices, compute_boundary_data_from_simplices, compute_boundary_payload_from_simplices, compute_boundary_payload_from_flat_simplices, compute_boundary_mod2_matrix, compute_alexander_whitney_cup, compute_trimesh_boundary_data, compute_trimesh_boundary_data_flat, triangulate_surface_delaunay

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

function exact_snf_sparse(rows::AbstractVector{Int64}, cols::AbstractVector{Int64}, vals::AbstractVector{Int64}, m::Int, n::Int)
    if !HAS_ABSTRACT_ALGEBRA
        error("AbstractAlgebra unavailable")
    end
    # sparse() constructor is efficient with these vectors.
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
    coboundary_mat = sparse(d_np1_cols .+ 1, d_np1_rows .+ 1, d_np1_vals, d_np1_n, d_np1_m)
    basis = Vector{Vector{Int64}}()
    try
        if !HAS_ABSTRACT_ALGEBRA; error("AbstractAlgebra unavailable"); end
        QQ = AbstractAlgebra.QQ
        M_qq = AbstractAlgebra.matrix(QQ, Matrix(coboundary_mat))
        nullity, nullspace_mat = AbstractAlgebra.nullspace(M_qq)
        for j in 1:nullity
            col = nullspace_mat[:, j]
            denoms = [AbstractAlgebra.denominator(x) for x in col]
            lcm_val = 1; for d in denoms; lcm_val = lcm(lcm_val, d); end
            int_vec = Int64[]; for i in 1:d_np1_m; push!(int_vec, AbstractAlgebra.numerator(col[i] * lcm_val)); end
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
            if max_comp < 1e-12; continue; end
            scale = round(Int64, 1.0 / max_comp * 1000)
            int_v = round.(Int64, v .* scale)
            g = reduce(gcd, int_v); if g != 0; push!(basis, int_v .÷ g); end
        end
    end
    dn_mat = sparse(d_n_cols .+ 1, d_n_rows .+ 1, d_n_vals, d_n_n, d_n_m)
    quotient_basis = Vector{Vector{Int64}}()
    if size(dn_mat, 2) > 0
        dense_dn = Matrix{Float64}(dn_mat); curr_rank = rank(dense_dn)
        for vec in basis
            test_mat = hcat(dense_dn, Float64.(vec)); new_rank = rank(test_mat); is_indep = false
            if new_rank > curr_rank
                try
                    if !HAS_ABSTRACT_ALGEBRA; error("AbstractAlgebra unavailable"); end
                    ZZ = AbstractAlgebra.ZZ
                    int_test = AbstractAlgebra.matrix(ZZ, Matrix{Int}(hcat(Matrix(dn_mat), vec)))
                    snf_test = AbstractAlgebra.snf(int_test)
                    new_rank_int = count(x -> x != 0, [snf_test[i, i] for i in 1:min(size(int_test)...)])
                    int_base = AbstractAlgebra.matrix(ZZ, Matrix{Int}(Matrix(dn_mat)))
                    snf_base = AbstractAlgebra.snf(int_base)
                    base_rank_int = count(x -> x != 0, [snf_base[i, i] for i in 1:min(size(int_base)...)])
                    if new_rank_int > base_rank_int; is_indep = true; end
                catch; is_indep = true; end
            end
            if is_indep; push!(quotient_basis, vec); dense_dn = test_mat; curr_rank = new_rank; end
        end
    else; quotient_basis = basis; end
    if isempty(quotient_basis); return Matrix{Int64}(undef, d_np1_m, 0); end
    return hcat(quotient_basis...)
end

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

function rank_mod_p_sparse(rows::AbstractVector{Int64}, cols::AbstractVector{Int64}, vals::AbstractVector{Int64}, m::Int, n::Int, p::Int)
    p <= 1 && error("modulus p must be > 1")
    isempty(rows) && return Int64(0)
    A = sparse(rows .+ 1, cols .+ 1, vals, m, n)
    return _rank_mod_p_dense!(Matrix{Int64}(A), p)
end

function _rref_mod_p_dense(M::Matrix{Int64}, p::Int)
    m, n = size(M); row = 1; pivots = Int[]
    for col in 1:n
        pivot = 0; for r in row:m; if mod(M[r, col], p) != 0; pivot = r; break; end; end
        pivot == 0 && continue
        if pivot != row; M[row, :], M[pivot, :] = copy(M[pivot, :]), copy(M[row, :]); end
        pivot_val = mod(M[row, col], p); inv_pivot = invmod(pivot_val, p)
        for j in 1:n; M[row, j] = mod(M[row, j] * inv_pivot, p); end
        for r in 1:m; (r == row || (factor = mod(M[r, col], p)) == 0) && continue
            for j in 1:n; M[r, j] = mod(M[r, j] - factor * M[row, j], p); end
        end
        push!(pivots, col); row += 1; row > m && break
    end
    return M, pivots
end

function _nullspace_basis_mod_p_dense(A::Matrix{Int64}, p::Int)
    _, n = size(A); rref, pivots = _rref_mod_p_dense(copy(A), p); pivot_set = Set(pivots); basis = Vector{Vector{Int64}}()
    for free in 1:n
        in(free, pivot_set) && continue
        v = zeros(Int64, n); v[free] = 1
        for (i, col) in enumerate(pivots); v[col] = mod(-rref[i, free], p); end
        push!(basis, v)
    end
    return basis
end

function _independent_mod_p(v::Vector{Int64}, cols::Vector{Vector{Int64}}, p::Int)
    vv = mod.(v, p); if isempty(cols); return any(x -> x != 0, vv); end
    M_prev = hcat(cols...); M_new = hcat(M_prev, vv); return _rank_mod_p_dense!(copy(M_new), p) > _rank_mod_p_dense!(copy(M_prev), p)
end

function sparse_cohomology_basis_mod_p(
    d_np1_rows::AbstractVector{Int64}, d_np1_cols::AbstractVector{Int64}, d_np1_vals::AbstractVector{Int64}, d_np1_m::Int, d_np1_n::Int,
    d_n_rows::AbstractVector{Int64}, d_n_cols::AbstractVector{Int64}, d_n_vals::AbstractVector{Int64}, d_n_m::Int, d_n_n::Int,
    p::Int,
)
    p <= 1 && error("modulus p must be > 1")
    coboundary_mat = sparse(d_np1_cols .+ 1, d_np1_rows .+ 1, d_np1_vals, d_np1_n, d_np1_m)
    z_basis = _nullspace_basis_mod_p_dense(Matrix{Int64}(coboundary_mat), p)
    dn_mat = sparse(d_n_cols .+ 1, d_n_rows .+ 1, d_n_vals, d_n_n, d_n_m); dn_dense = Matrix{Int64}(dn_mat)
    b_cols = [mod.(dn_dense[:, j], p) for j in 1:size(dn_dense, 2)]
    reps = Vector{Vector{Int64}}(); span_cols = copy(b_cols)
    for z in z_basis; if _independent_mod_p(z, span_cols, p); zz = mod.(z, p); push!(reps, zz); push!(span_cols, zz); end; end
    if isempty(reps); return Matrix{Int64}(undef, d_np1_m, 0); end
    return hcat(reps...)
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

function compute_boundary_payload_from_flat_simplices(flat_vertices::AbstractVector{Int64}, simplex_offsets::AbstractVector{Int64}, max_dim::Int, include_metadata::Bool=true)
    return _compute_boundary_data_internal_flat(flat_vertices, simplex_offsets, max_dim)
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

function group_ring_multiply(py_coeffs1, py_coeffs2, group_order::Int)
    c1, c2 = _normalize_group_ring_coeffs(py_coeffs1), _normalize_group_ring_coeffs(py_coeffs2)
    res_dict = Dict{String, Int}()
    for (k1, v1) in c1, (k2, v2) in c2
        p_res = mod(_parse_group_element(k1) + _parse_group_element(k2), group_order)
        g_str = p_res == 0 ? "1" : "g_$(p_res)"
        res_dict[g_str] = get(res_dict, g_str, 0) + v1 * v2
    end
    final_k, final_v = String[], Int[]
    for (k, v) in res_dict; v != 0 && (push!(final_k, k); push!(final_v, v)); end
    return final_k, final_v
end

function multisignature(matrix::AbstractMatrix{Float64}, p::Int)
    mat = Matrix{Float64}(matrix); n = size(mat, 1); total = 0
    for k in 1:(p-1)
        omega = exp(2π * im * k / p); H = [mat[i,j] * omega^(i-j) for i in 1:n, j in 1:n]
        evals = real.(eigvals(Hermitian(H))); tol = n * eps(Float64) * maximum(abs.(evals))
        total += sum(evals .> tol) - sum(evals .< -tol)
    end
    return total
end

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

function compute_trimesh_boundary_data_flat(face_vertices::AbstractVector{Int64}, face_offsets::AbstractVector{Int64}, n_vertices::Int)
    n_faces = length(face_offsets) - 1; edge_to_idx = Dict{Tuple{Int, Int}, Int}(); edges = Tuple{Int, Int}[]
    d2_rows, d2_cols, d2_data = Int64[], Int64[], Int64[]
    for j in 1:n_faces
        lo, hi = face_offsets[j] + 1, face_offsets[j+1]; cycle = view(face_vertices, lo:hi)
        for i in 1:length(cycle)
            u, v = Int(cycle[i]), Int(cycle[mod(i, length(cycle)) + 1]); key = u <= v ? (u, v) : (v, u)
            if !haskey(edge_to_idx, key); push!(edges, key); edge_to_idx[key] = length(edges); end
            idx = edge_to_idx[key]; push!(d2_rows, idx - 1); push!(d2_cols, j - 1); push!(d2_data, (u, v) == edges[idx] ? 1 : -1)
        end
    end
    d1_rows, d1_cols, d1_data = Int64[], Int64[], Int64[]
    for (j, (v1, v2)) in enumerate(edges); push!(d1_rows, v1); push!(d1_cols, j - 1); push!(d1_data, -1); push!(d1_rows, v2); push!(d1_cols, j - 1); push!(d1_data, 1); end
    return Dict("d1_rows" => d1_rows, "d1_cols" => d1_cols, "d1_data" => d1_data, "n_vertices" => Int64(n_vertices), "n_edges" => Int64(length(edges)), "d2_rows" => d2_rows, "d2_cols" => d2_cols, "d2_data" => d2_data, "n_faces" => Int64(n_faces))
end



function _to_vertices_simplex(s)
    if s isa Tuple
        return [Int64(x) for x in s]
    elseif s isa AbstractVector
        return [Int64(x) for x in s]
    elseif s isa Py
        # Convert Py list to Julia Vector
        return pyconvert(Vector{Int64}, s)
    else
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
                    push!(dim_simplices[d], Tuple(sort(collect(Int64.(face)))))
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

function compute_boundary_payload_from_simplices(simplex_entries, max_dim::Int, include_metadata::Bool=true)
    boundaries, cells, dim_simplices, simplex_to_idx = _compute_boundary_data_internal(simplex_entries, max_dim)
    return include_metadata ? (boundaries, cells, dim_simplices, simplex_to_idx) : (boundaries, cells)
end

function compute_boundary_data_from_simplices(simplex_entries, max_dim::Int)
    return _compute_boundary_data_internal(simplex_entries, max_dim)
end

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

function _as_string_vector(x)
    if x isa Vector{String}
        return x
    elseif x isa AbstractVector
        return [string(v) for v in x]
    end
    return [string(v) for v in pyconvert(Vector{Any}, x)]
end

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

function compute_boundary_mod2_matrix(source_simplices, target_simplices)
    source, target = [_to_vertices_simplex(s) for s in source_simplices], [_to_vertices_simplex(t) for t in target_simplices]
    m, n = length(target), length(source); if m == 0 || n == 0; return Dict("rows" => Int64[], "cols" => Int64[], "data" => Int64[], "m" => Int64(m), "n" => Int64(n)); end
    t_idx = Dict{Tuple{Vararg{Int}}, Int}(); for (i, t) in enumerate(target); t_idx[Tuple(sort(collect(t)))] = i - 1; end
    rows, cols, data = Int64[], Int64[], Int64[]
    for (j, s) in enumerate(source); for i_drop in eachindex(s)
        face = Tuple(sort(collect(Int.(vcat(s[1:(i_drop-1)], s[(i_drop+1):end]))))); haskey(t_idx, face) && (push!(rows, t_idx[face]); push!(cols, j - 1); push!(data, 1))
    end; end
    return Dict("rows" => rows, "cols" => cols, "data" => data, "m" => Int64(m), "n" => Int64(n))
end

function compute_alexander_whitney_cup(alpha::AbstractVector, beta::AbstractVector, p::Int, q::Int, simplices_pq, s_to_idx_p, s_to_idx_q, modulus=nothing)
    idx_p, idx_q = pyconvert(Dict{Tuple{Vararg{Int}}, Int64}, s_to_idx_p), pyconvert(Dict{Tuple{Vararg{Int}}, Int64}, s_to_idx_q)
    res = zeros(Int64, length(simplices_pq))
    for i in 1:length(simplices_pq)
        s = _to_vertices_simplex(simplices_pq[i]); length(s) < p + q + 1 && continue
        v_p, v_q = get(idx_p, Tuple(s[1:(p+1)]), -1), get(idx_q, Tuple(s[(p+1):(p+q+1)]), -1)
        if v_p != -1 && v_q != -1; val = Int64(alpha[v_p + 1]) * Int64(beta[v_q + 1]); res[i] = modulus === nothing ? val : mod(val, Int64(modulus)); end
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

function optgen_from_simplices(simplices, num_vertices::Int, pts=nothing, max_roots=nothing, root_stride::Int=1, max_cycles=nothing)
    if !HAS_GRAPHS
        error("Graphs.jl or SimpleWeightedGraphs.jl unavailable.")
    end
    
    # 1. Extract edges and triangles
    edges_set = Set{Tuple{Int64, Int64}}()
    triangles = Vector{Tuple{Int64, Int64, Int64}}()
    
    for s in simplices
        vs = _to_vertices_simplex(s)
        if length(vs) == 2
            push!(edges_set, Tuple(sort(vec(Int64.(collect(vs))))))
        elseif length(vs) == 3
            push!(triangles, Tuple(sort(vec(Int64.(collect(vs))))))
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
        edge_weights[e] = (pts !== nothing) ? norm(pts[u, :] .- pts[v, :]) : 1.0
    end
    
    # 2. Minimum Spanning Tree (Kruskal)
    g_full = SimpleWeightedGraph(nv)
    for e in edges_list
        add_edge!(g_full, e[1]+1, e[2]+1, edge_weights[e])
    end
    mst_edges = kruskal_mst(g_full)
    spanning_set = Set{Tuple{Int64, Int64}}()
    for e in mst_edges
        push!(spanning_set, Tuple(sort(collect((Int64(src(e)-1), Int64(dst(e)-1))))))
    end
    
    non_tree = [e for e in edges_list if !(e in spanning_set)]
    m_dim = length(non_tree)
    
    # 3. Edge Annotations
    annotations = Dict{Tuple{Int64, Int64}, Vector{Int8}}()
    for e in spanning_set; annotations[e] = zeros(Int8, m_dim); end
    for (i, e) in enumerate(non_tree)
        v = zeros(Int8, m_dim)
        v[i] = 1
        annotations[e] = v
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
    
    # 4. Cycle Candidates (Shortest Paths)
    # Adjacency for Dijkstra
    adj = Dict{Int64, Vector{Tuple{Int64, Float64}}}()
    for e in edges_list
        u, v = e
        push!(get!(adj, u, []), (v, edge_weights[e]))
        push!(get!(adj, v, []), (u, edge_weights[e]))
    end
    
    all_cycles = Vector{Vector{Int64}}()
    roots = collect(0:nv-1)
    if max_roots !== nothing
        # Simple root selection strategy
        roots = roots[1:min(nv, max_roots)]
    end
    
    for root in roots
        ds = dijkstra_shortest_paths(g_full, root + 1)
        # For each edge (u, v) not in SPT, form cycle
        # SPT edges are those where parent[v] == u
        spt_edges = Set{Tuple{Int64, Int64}}()
        for v in 1:nv
            p = ds.parents[v]
            if p != 0 && p != v
                push!(spt_edges, Tuple(sort(collect((Int64(v-1), Int64(p-1))))))
            end
        end
        
        for e in edges_list
            if !(e in spt_edges)
                u_node, v_node = e[1], e[2]
                # Reconstruct path root->u and root->v
                path_u = Int64[]
                curr = u_node + 1
                while curr != 0
                    push!(path_u, curr - 1)
                    curr = ds.parents[curr] == curr ? 0 : ds.parents[curr]
                end
                path_v = Int64[]
                curr = v_node + 1
                while curr != 0
                    push!(path_v, curr - 1)
                    curr = ds.parents[curr] == curr ? 0 : ds.parents[curr]
                end
                
                # Find LCA
                lca = -1
                idx_u, idx_v = length(path_u), length(path_v)
                while idx_u > 0 && idx_v > 0 && path_u[idx_u] == path_v[idx_v]
                    lca = path_u[idx_u]
                    idx_u -= 1
                    idx_v -= 1
                end
                
                if lca != -1
                    cycle = vcat(reverse(path_u[1:idx_u+1]), path_v[1:idx_v+1])
                    push!(all_cycles, cycle)
                end
            end
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

function _rref_mod2!(M::Matrix{Int8})
    m, n = size(M)
    row = 1
    pivots = Int[]
    for col in 1:n
        pivot = 0
        for r in row:m
            if M[r, col] & 1 != 0
                pivot = r
                break
            end
        end
        pivot == 0 && continue
        if pivot != row
            M[row, :], M[pivot, :] = M[pivot, :], M[row, :]
        end
        for r in 1:m
            if r != row && M[r, col] & 1 != 0
                M[r, :] .⊻= M[row, :]
            end
        end
        push!(pivots, col)
        row += 1
        row > m && break
    end
    return M, pivots
end

function _nullspace_basis_mod2(A::Matrix{Int8})
    m, n = size(A)
    # A is boundary matrix. We want nullspace of d_k.
    M_rref, pivots = _rref_mod2!(copy(A))
    pivot_set = Set(pivots)
    basis = Vector{Vector{Int8}}()
    for free_col in 1:n
        if !(free_col in pivot_set)
            v = zeros(Int8, n)
            v[free_col] = 1
            for (i, p_col) in enumerate(pivots)
                v[p_col] = M_rref[i, free_col] & 1
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

function _independent_mod_image(v::Vector{Int8}, basis_cols::Vector{Vector{Int8}})
    if isempty(basis_cols); return any(x -> x & 1 != 0, v); end
    # Combine basis and candidate into a matrix
    M_new = hcat(basis_cols..., v)
    M_prev = hcat(basis_cols...)
    return _rank_mod2(M_new) > _rank_mod2(M_prev)
end

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
    
    # Nullspace of d_k
    z_basis = _nullspace_basis_mod2(Matrix{Int8}(dk_mat))
    if isempty(z_basis); return []; end
    
    # Image of d_{k+1}
    # dkp1_mat is sparse, convert to dense for rank checks
    b_cols = [Vector{Int8}(dkp1_mat[:, j]) for j in 1:size(dkp1_mat, 2)]
    
    # Candidates for quotient basis
    z_candidates = z_basis
    if mode == "optimal"
        # Sort by weight
        weights = [_weight_k_chain(z, k_simplices_mat, pts) for z in z_candidates]
        z_candidates = z_candidates[sortperm(weights)]
    end
    
    quotient_basis = Vector{Vector{Int8}}()
    span_cols = copy(b_cols)
    for z in z_candidates
        if _independent_mod_image(z, span_cols)
            push!(quotient_basis, z)
            push!(span_cols, z)
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
        u, v, w = indices(tri)
        # Return 0-indexed for Python consistency
        push!(faces, [Int64(u-1), Int64(v-1), Int64(w-1)])
    end
    
    return faces
end

PrecompileTools.@setup_workload begin
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
        catch
        end
    end
end

end # module SurgeryBackend
