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

function group_ring_multiply(py_coeffs1::Py, py_coeffs2::Py, group_order::Int)
    c1, c2 = pyconvert(Dict{String, Int}, py_coeffs1), pyconvert(Dict{String, Int}, py_coeffs2)
    res_dict = Dict{String, Int}()
    function parse_gen(g_str)
        (g_str == "e" || g_str == "1") && return 0
        m = match(r"g_?(\d+)(?:\^-1)?", g_str); m === nothing && return 0
        val = parse(Int, m.captures[1]); return endswith(g_str, "^-1") ? -val : val
    end
    for (k1, v1) in c1, (k2, v2) in c2
        p_res = mod(parse_gen(k1) + parse_gen(k2), group_order)
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

PrecompileTools.@setup_workload begin
    r, c, v = Int64[0, 1], Int64[0, 1], Int64[1, 1]; fv, fo = Int64[0, 1, 2], Int64[0, 3]; mf = ones(2, 2); mi = ones(Int64, 2, 2)
    PrecompileTools.@compile_workload begin
        exact_snf_sparse(r, c, v, 2, 2); rank_q_sparse(r, c, v, 2, 2); rank_mod_p_sparse(r, c, v, 2, 2, 2)
        _compute_boundary_data_internal_flat(fv, fo, 2); exact_sparse_cohomology_basis(r, c, v, 2, 2, r, c, v, 2, 2)
        multisignature(mf, 2); integral_lattice_isometry(mi, mi)
    end
end

function _to_vertices_simplex(s)
    if s isa Tuple; return [Int(x) for x in s]
    elseif s isa AbstractVector; return [Int(x) for x in s]
    else; return Int[]; end
end

function _compute_boundary_data_internal(simplex_entries, max_dim::Int)
    dim_simplices = Dict{Int, Vector{Tuple{Vararg{Int64}}}}(); for d in 0:max_dim; dim_simplices[d] = Tuple{Vararg{Int64}}[]; end
    for s in simplex_entries; vs = _to_vertices_simplex(s); isempty(vs) && continue; sort!(vs); d = length(vs) - 1; 0 <= d <= max_dim && push!(dim_simplices[d], Tuple(Int64.(vs))); end
    cells = Dict{Int, Int64}(); simplex_to_idx = Dict{Int, Dict{Tuple{Vararg{Int64}}, Int64}}()
    for d in 0:max_dim
        simplices_d = dim_simplices[d]; sort!(simplices_d); cells[d] = Int64(length(simplices_d))
        idx_map = Dict{Tuple{Vararg{Int64}}, Int64}(); for (i, simplex) in enumerate(simplices_d); idx_map[simplex] = Int64(i - 1); end; simplex_to_idx[d] = idx_map
    end
    boundaries = Dict{Int, Dict{String, Any}}()
    for k in 1:max_dim
        n_rows, n_cols = Int64(get(cells, k - 1, 0)), Int64(get(cells, k, 0)); (n_rows == 0 || n_cols == 0) && continue
        rows, cols, data = Int64[], Int64[], Int64[]; prev_dim_map = simplex_to_idx[k - 1]
        for (j, simplex) in enumerate(dim_simplices[k])
            verts = collect(simplex); for i in eachindex(verts)
                face = Tuple(vcat(verts[1:(i - 1)], verts[(i + 1):end])); haskey(prev_dim_map, face) && (push!(rows, prev_dim_map[face]); push!(cols, Int64(j - 1)); push!(data, isodd(i - 1) ? -1 : 1))
            end
        end
        boundaries[k] = Dict("rows" => rows, "cols" => cols, "data" => data, "n_rows" => n_rows, "n_cols" => n_cols)
    end
    return boundaries, cells, dim_simplices, simplex_to_idx
end

function compute_boundary_payload_from_simplices(simplex_entries, max_dim::Int, include_metadata::Bool=true)
    boundaries, cells, dim_simplices, simplex_to_idx = _compute_boundary_data_internal(simplex_entries, max_dim)
    return include_metadata ? (boundaries, cells, dim_simplices, simplex_to_idx) : (boundaries, cells)
end

function compute_boundary_data_from_simplices(simplex_entries, max_dim::Int)
    return _compute_boundary_data_internal(simplex_entries, max_dim)
end

function abelianize_group(generators::Vector{String}, relations::Vector{String})
    n_gens = length(generators); gen_idx = Dict{String, Int}(g => i for (i, g) in enumerate(generators))
    n_rels = length(relations); M = zeros(Int, n_rels, n_gens)
    for i in 1:n_rels; for m in eachmatch(r"([a-zA-Z0-9_]+)(?:\^(-?\d+))?", relations[i])
        base_w = m.captures[1]; haskey(gen_idx, base_w) && (M[i, gen_idx[base_w]] += parse(Int, m.captures[2] || "1"))
    end; end
    if !HAS_ABSTRACT_ALGEBRA; return Int(n_gens - rank(Matrix{Float64}(M))), Int[]; end
    ZZ = AbstractAlgebra.ZZ; M_aa = AbstractAlgebra.matrix(ZZ, M); S_aa = AbstractAlgebra.snf(M_aa)
    diag = [Int64(S_aa[i, i]) for i in 1:min(n_rels, n_gens)]; nonzero = filter(x -> x != 0, diag)
    return n_gens - length(nonzero), filter(x -> x > 1, nonzero)
end

function compute_boundary_mod2_matrix(source_simplices, target_simplices)
    source, target = [_to_vertices_simplex(s) for s in source_simplices], [_to_vertices_simplex(t) for t in target_simplices]
    m, n = length(target), length(source); if m == 0 || n == 0; return Dict("rows" => Int64[], "cols" => Int64[], "data" => Int64[], "m" => Int64(m), "n" => Int64(n)); end
    t_idx = Dict{Tuple{Vararg{Int}}, Int}(); for (i, t) in enumerate(target); t_idx[Tuple(sort(t))] = i - 1; end
    rows, cols, data = Int64[], Int64[], Int64[]
    for (j, s) in enumerate(source); for i_drop in eachindex(s)
        face = Tuple(sort(Int.(vcat(s[1:(i_drop-1)], s[(i_drop+1):end])))); haskey(t_idx, face) && (push!(rows, t_idx[face]); push!(cols, j - 1); push!(data, 1))
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

function optgen_from_simplices(simplices, num_vertices::Int, pts=nothing, max_roots=nothing, root_stride::Int=1, max_cycles=nothing)
    # Placeholder for the complex H1 logic, preserving core structure
    return Vector{HCycle}()
end

function homology_generators_from_simplices(simplices, num_vertices::Int, dimension::Int, mode::String="valid", pts=nothing, mr=nothing, rs::Int=1, mc=nothing)
    # Placeholder for homology logic, preserving core structure
    return Vector{Dict{String, Any}}()
end

function triangulate_surface_delaunay(points::AbstractMatrix{Float64}, tolerance::Real=1e-10)
    # Placeholder for triangulation logic
    return Vector{Vector{Int64}}()
end

end # module SurgeryBackend
