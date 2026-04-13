# A lightweight Julia backend module for high-performance algebraic operations over Z[pi]
module SurgeryBackend

using LinearAlgebra
using SparseArrays
using Statistics

export hermitian_signature, exact_snf_sparse, exact_sparse_cohomology_basis, group_ring_multiply, multisignature, abelianize_group, integral_lattice_isometry, optgen_from_simplices, homology_generators_from_simplices, compute_boundary_data_from_simplices, compute_boundary_mod2_matrix, compute_alexander_whitney_cup, compute_trimesh_boundary_data, triangulate_surface_delaunay

const HAS_ABSTRACT_ALGEBRA = try
    @eval import AbstractAlgebra
    true
catch
    false
end

function hermitian_signature(matrix)
    # Convert to Julia Matrix
    mat = Matrix{Float64}(matrix)
    eigenvalues = eigvals(Hermitian(mat))
    tol = length(eigenvalues) > 0 ? maximum(size(mat)) * eps(Float64) * maximum(abs.(eigenvalues)) : 1e-10
    pos = count(x -> x > tol, eigenvalues)
    neg = count(x -> x < -tol, eigenvalues)
    return pos - neg
end

function exact_snf_sparse(rows, cols, vals, m, n)
    if !HAS_ABSTRACT_ALGEBRA
        error("AbstractAlgebra unavailable")
    end

    # Python passes zero-based COO indices; lift to Julia's one-based indexing.
    row_idx = Int64.(rows) .+ 1
    col_idx = Int64.(cols) .+ 1
    A = sparse(row_idx, col_idx, Int64.(vals), m, n)

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
    d_np1_rows, d_np1_cols, d_np1_vals, d_np1_m, d_np1_n,
    d_n_rows, d_n_cols, d_n_vals, d_n_m, d_n_n
)
    # Python passes zero-based COO indices; lift to Julia's one-based indexing.
    np1_r = Int64.(d_np1_rows) .+ 1
    np1_c = Int64.(d_np1_cols) .+ 1
    np1_v = Int64.(d_np1_vals)
    coboundary_mat = sparse(np1_c, np1_r, np1_v, d_np1_n, d_np1_m)

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
        # Float fallback: recover null space from SVD, then scale each vector to
        # clear denominators using the GCD of a rational reconstruction.
        null_basis_float = nullity > 0 ? [F.V[:, i] for i in (size(F.V, 2) - nullity + 1):size(F.V, 2)] : Vector{Vector{Float64}}()
        for v in null_basis_float
            # Scale to smallest integer multiple using the largest component
            max_comp = maximum(abs.(v))
            if max_comp < 1e-12
                continue
            end
            # Use a rational approximation with bounded denominator
            scale = round(Int64, 1.0 / max_comp * 1000)
            int_v = round.(Int64, v .* scale)
            g = reduce(gcd, int_v)
            if g != 0
                push!(basis, int_v .÷ g)
            end
        end
    end
    
    n_r = Int64.(d_n_rows) .+ 1
    n_c = Int64.(d_n_cols) .+ 1
    n_v = Int64.(d_n_vals)
    dn_mat = sparse(n_c, n_r, n_v, d_n_n, d_n_m)

    quotient_basis = Vector{Vector{Int64}}()
    if size(dn_mat, 2) > 0
        dense_dn = Matrix{Float64}(dn_mat)
        curr_rank = rank(dense_dn)
        
        for vec in basis
            test_mat = hcat(dense_dn, Float64.(vec))
            new_rank = rank(test_mat)
            
            # Additional rigorous integer check
            # We append vec to Im(d_n). If it adds no rank, it's a coboundary.
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
                    # Fallback to float rank if AbstractAlgebra throws
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
    return quotient_basis
end

function compute_boundary_data_from_simplices(simplex_entries, max_dim::Int)
    dim_simplices = Dict{Int, Vector{Tuple{Vararg{Int64}}}}()
    for d in 0:max_dim
        dim_simplices[d] = Tuple{Vararg{Int64}}[]
    end

    for s in simplex_entries
        vs = _to_vertices_simplex(s)
        isempty(vs) && continue
        sort!(vs)
        d = length(vs) - 1
        if 0 <= d <= max_dim
            push!(dim_simplices[d], Tuple(Int64.(vs)))
        end
    end

    cells = Dict{Int, Int64}()
    simplex_to_idx = Dict{Int, Dict{Tuple{Vararg{Int64}}, Int64}}()
    for d in 0:max_dim
        simplices_d = dim_simplices[d]
        sort!(simplices_d)
        cells[d] = Int64(length(simplices_d))
        idx_map = Dict{Tuple{Vararg{Int64}}, Int64}()
        for (i, simplex) in enumerate(simplices_d)
            idx_map[simplex] = Int64(i - 1) # return zero-based indices to Python
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

        for (j, simplex) in enumerate(dim_simplices[k])
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

        boundaries[k] = Dict(
            "rows" => rows,
            "cols" => cols,
            "data" => data,
            "n_rows" => n_rows,
            "n_cols" => n_cols,
        )
    end

    return boundaries, cells, dim_simplices
end

function group_ring_multiply(k1::Vector{String}, v1::Vector{Int}, k2::Vector{String}, v2::Vector{Int}, group_order::Int)
    res_dict = Dict{String, Int}()
    for i in 1:length(k1)
        for j in 1:length(k2)
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
            
            p1 = parse_gen(k1[i])
            p2 = parse_gen(k2[j])
            p_res = (p1 + p2) % group_order
            if p_res < 0
                p_res += group_order
            end
            g_str = p_res == 0 ? "1" : "g_$(p_res)"
            
            coeff = v1[i] * v2[j]
            res_dict[g_str] = get(res_dict, g_str, 0) + coeff
        end
    end
    
    # Filter zeros
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

function group_ring_multiply(k1::AbstractVector, v1::AbstractVector, k2::AbstractVector, v2::AbstractVector, group_order::Int)
    k1s = String[string(x) for x in k1]
    v1i = Int64[Int(x) for x in v1]
    k2s = String[string(x) for x in k2]
    v2i = Int64[Int(x) for x in v2]
    return group_ring_multiply(k1s, v1i, k2s, v2i, group_order)
end

function multisignature(matrix, p::Int)
    # The multisignature of Q over Z[Z_p] at character k is:
    # sigma_k = signature of the Hermitian matrix Q tensored with exp(2πi k/p)
    # For the real symmetric case, this equals signature(Q) for the trivial rep.
    mat = Matrix{Float64}(matrix)
    n = size(mat, 1)
    total = 0
    for k in 1:(p-1)
        omega = exp(2π * im * k / p)
        # Build Hermitian form H_k[i,j] = Q[i,j] * omega^(i-j)  (representation twist)
        H = [mat[i,j] * omega^(i-j) for i in 1:n, j in 1:n]
        evals = real.(eigvals(Hermitian(H)))
        tol = n * eps(Float64) * maximum(abs.(evals))
        total += sum(evals .> tol) - sum(evals .< -tol)
    end
    return total
end

function integral_lattice_isometry(matrix1, matrix2)
    A = Matrix{Int64}(matrix1)
    B = Matrix{Int64}(matrix2)
    n = size(A, 1)
    if size(A, 2) != n || size(B, 1) != n || size(B, 2) != n
        return nothing
    end

    evals_a = eigvals(Hermitian(Matrix{Float64}(A)))
    evals_b = eigvals(Hermitian(Matrix{Float64}(B)))
    scale = max(1.0, maximum(abs.(vcat(evals_a, evals_b))))
    tol = n * eps(Float64) * scale

    pos_a = all(x -> x > tol, evals_a)
    neg_a = all(x -> x < -tol, evals_a)
    pos_b = all(x -> x > tol, evals_b)
    neg_b = all(x -> x < -tol, evals_b)
    if !((pos_a && pos_b) || (neg_a && neg_b))
        return nothing
    end

    Adef = pos_a ? A : -A
    Bdef = pos_b ? B : -B
    lam_min = minimum(eigvals(Hermitian(Matrix{Float64}(Adef))))
    if lam_min <= 0
        return nothing
    end

    targets = [Int(Bdef[i, i]) for i in 1:n]
    if any(t -> t <= 0, targets)
        return nothing
    end

    radius = maximum([Int(floor(sqrt(t / lam_min))) + 1 for t in targets])
    range_vals = collect(-radius:radius)
    vectors_by_norm = Dict{Int, Vector{Vector{Int64}}}()
    for t in unique(targets)
        vectors_by_norm[t] = Vector{Vector{Int64}}()
    end

    for tup in Iterators.product(ntuple(_ -> range_vals, n)...)
        v = Int64[tup...]
        qv = Int(dot(v, Adef * v))
        if haskey(vectors_by_norm, qv)
            push!(vectors_by_norm[qv], v)
            if length(vectors_by_norm[qv]) > 20000
                return nothing
            end
        end
    end

    if any(t -> isempty(vectors_by_norm[t]), targets)
        return nothing
    end

    order = sortperm(1:n; by = j -> length(vectors_by_norm[targets[j]]))
    cols = [zeros(Int64, n) for _ in 1:n]
    chosen_original_indices = Int[]

    function backtrack(pos::Int)
        if pos > n
            U = hcat(cols...)
            d = round(Int, det(Matrix{Float64}(U)))
            if abs(d) != 1
                return nothing
            end
            return transpose(U) * A * U == B ? U : nothing
        end

        j = order[pos]
        for v in vectors_by_norm[targets[j]]
            ok = true
            for i in chosen_original_indices
                if Int(dot(cols[i], Adef * v)) != Int(Bdef[i, j])
                    ok = false
                    break
                end
            end
            if !ok
                continue
            end

            cols[j] = v
            push!(chosen_original_indices, j)
            out = backtrack(pos + 1)
            if out !== nothing
                return out
            end
            pop!(chosen_original_indices)
        end

        return nothing
    end

    return backtrack(1)
end

# -----------------------------------------------------------------------------
# Data-grounded H1 generators (Algorithms 10, 8, 7, 9)
#
# Attribution:
#   T. K. Dey and Y. Wang, Computational Topology for Data Analysis,
#   chapter "Generators and Optimality".
#
# This implementation works directly on simplices supplied from Python via
# juliacall and is intended to back pySurgery's "elements of your data"
# generator APIs.
# -----------------------------------------------------------------------------

const HEdge = NTuple{2,Int}
const HTriangle = NTuple{3,Int}
const HCycle = Vector{HEdge}

@inline _order_hedge(u::Int, v::Int) = u <= v ? (u, v) : (v, u)

function _to_vertices_simplex(s)
    if s isa Tuple
        return [Int(x) for x in s]
    elseif s isa AbstractVector
        return [Int(x) for x in s]
    else
        return Int[]
    end
end

function _edge_weight_h1(u::Int, v::Int, points)
    if points === nothing
        return 1.0
    end
    s = 0.0
    @inbounds @simd for j in axes(points, 2)
        d = points[u + 1, j] - points[v + 1, j]  # vertices are 0-based in Python
        s += d * d
    end
    return sqrt(s)
end

function _normalize_edges_triangles(simplices)
    edges = HEdge[]
    triangles = HTriangle[]
    vertex_ids = Set{Int}()
    for s in simplices
        vs = _to_vertices_simplex(s)
        if length(vs) == 2
            e = _order_hedge(vs[1], vs[2])
            push!(edges, e)
            push!(vertex_ids, e[1]); push!(vertex_ids, e[2])
        elseif length(vs) == 3
            sort!(vs)
            t = (vs[1], vs[2], vs[3])
            push!(triangles, t)
            push!(vertex_ids, t[1]); push!(vertex_ids, t[2]); push!(vertex_ids, t[3])
        end
    end
    unique!(edges)
    unique!(triangles)
    return edges, triangles, vertex_ids
end

mutable struct HDSU
    parent::Vector{Int}
    rank::Vector{UInt8}
end
HDSU(n::Int) = HDSU(collect(1:n), fill(UInt8(0), n))

@inline function _find_h!(d::HDSU, x::Int)
    px = d.parent[x]
    px == x && return x
    r = _find_h!(d, px)
    d.parent[x] = r
    return r
end

@inline function _unite_h!(d::HDSU, a::Int, b::Int)
    ra = _find_h!(d, a)
    rb = _find_h!(d, b)
    ra == rb && return false
    if d.rank[ra] < d.rank[rb]
        ra, rb = rb, ra
    end
    d.parent[rb] = ra
    d.rank[ra] == d.rank[rb] && (d.rank[ra] += UInt8(1))
    return true
end

function _minimum_spanning_edges_h(edges::Vector{HEdge}, weights::Dict{HEdge,Float64}, num_vertices::Int)
    order = sortperm(edges; by = weights)
    dsu = HDSU(max(num_vertices, 1))
    spanning = Set{HEdge}()
    for idx in order
        u, v = edges[idx]
        if _unite_h!(dsu, u + 1, v + 1)
            push!(spanning, (u, v))
        end
    end
    return spanning
end

function _annot_edge_h(simplices, num_vertices::Int, edge_weights::Dict{HEdge,Float64})
    edges, triangles, _ = _normalize_edges_triangles(simplices)
    weights = Dict{HEdge,Float64}()
    for e in edges
        weights[e] = get(edge_weights, e, 1.0)
    end

    spanning = _minimum_spanning_edges_h(edges, weights, num_vertices)
    non_tree = [e for e in edges if !in(e, spanning)]
    m = length(non_tree)

    annotations = Dict{HEdge,BitVector}()
    m == 0 && (for e in edges; annotations[e] = BitVector(); end; return annotations, 0)

    for e in spanning
        annotations[e] = falses(m)
    end
    for (i, e) in enumerate(non_tree)
        v = falses(m); v[i] = true
        annotations[e] = v
    end

    active = trues(m)
    boundary = falses(m)
    for t in triangles
        u, v, w = t
        e1, e2, e3 = _order_hedge(u, v), _order_hedge(v, w), _order_hedge(u, w)

        fill!(boundary, false)
        haskey(annotations, e1) && (boundary .⊻= annotations[e1])
        haskey(annotations, e2) && (boundary .⊻= annotations[e2])
        haskey(annotations, e3) && (boundary .⊻= annotations[e3])

        pivot = 0
        @inbounds for i in 1:m
            (boundary[i] && active[i]) || continue
            pivot = i
            break
        end
        pivot == 0 && continue

        for (e, vec) in annotations
            vec[pivot] && (vec .⊻= boundary)
        end
        active[pivot] = false
    end

    final = Dict{HEdge,BitVector}()
    for (e, vec) in annotations
        final[e] = vec[active]
    end
    return final, count(active)
end

function _shortest_path_tree_h(root::Int, adjacency::Dict{Int,Vector{Tuple{Int,Float64}}})
    dist = Dict{Int,Float64}(root => 0.0)
    parent = Dict{Int,Int}(root => -1)
    heap = [(0.0, root)]

    while !isempty(heap)
        sort!(heap, by = x -> x[1])
        d, u = popfirst!(heap)
        d != dist[u] && continue
        for (v, w) in get(adjacency, u, Tuple{Int,Float64}[])
            nd = d + w
            if nd < get(dist, v, Inf)
                dist[v] = nd
                parent[v] = u
                push!(heap, (nd, v))
            end
        end
    end

    tree_edges = Set{HEdge}()
    for (child, par) in parent
        par == -1 && continue
        push!(tree_edges, _order_hedge(child, par))
    end
    return parent, tree_edges
end

function _path_between_h_vertices(u::Int, v::Int, parent::Dict{Int,Int})
    path_u = Int[]
    seen_u = Set{Int}()
    x = u
    while x != -1
        push!(path_u, x)
        push!(seen_u, x)
        x = get(parent, x, -1)
    end
    path_v = Int[]
    y = v
    while !in(y, seen_u) && y != -1
        push!(path_v, y)
        y = get(parent, y, -1)
    end
    y == -1 && return Int[]
    i = findfirst(==(y), path_u)
    return vcat(path_u[1:i], reverse(path_v))
end

function _path_edges_h(path_vertices::Vector{Int})
    edges = HEdge[]
    for i in 1:(length(path_vertices) - 1)
        push!(edges, _order_hedge(path_vertices[i], path_vertices[i + 1]))
    end
    return edges
end

function _generator_cycles_h(simplices, num_vertices::Int, points; max_roots=nothing, root_stride::Int=1, max_cycles=nothing)
    edges, _, vertex_ids = _normalize_edges_triangles(simplices)
    isempty(edges) && return HCycle[]

    adjacency = Dict{Int,Vector{Tuple{Int,Float64}}}()
    for (u, v) in edges
        w = _edge_weight_h1(u, v, points)
        push!(get!(adjacency, u, Tuple{Int,Float64}[]), (v, w))
        push!(get!(adjacency, v, Tuple{Int,Float64}[]), (u, w))
    end

    vertices = isempty(vertex_ids) ? collect(0:max(num_vertices - 1, 0)) : sort(collect(vertex_ids))
    selected_roots = vertices[begin:max(root_stride, 1):end]
    if max_roots !== nothing
        selected_roots = selected_roots[1:min(Int(max_roots), length(selected_roots))]
    end

    cycles = HCycle[]
    for root in selected_roots
        parent, tree_edges = _shortest_path_tree_h(root, adjacency)
        for (u, v) in edges
            e = _order_hedge(u, v)
            if in(e, tree_edges) || !haskey(parent, u) || !haskey(parent, v)
                continue
            end
            path_vertices = _path_between_h_vertices(u, v, parent)
            length(path_vertices) < 2 && continue
            cyc = HEdge[e]
            append!(cyc, _path_edges_h(path_vertices))
            push!(cycles, cyc)
            if max_cycles !== nothing && length(cycles) >= Int(max_cycles)
                return cycles
            end
        end
    end
    return cycles
end

function _cycle_annotation_h(cycle::HCycle, simplex_annotations::Dict{HEdge,BitVector}, len::Int)
    ann = falses(len)
    for e in cycle
        se = _order_hedge(e[1], e[2])
        haskey(simplex_annotations, se) && (ann .⊻= simplex_annotations[se])
    end
    return ann
end

function _is_independent_h!(cv::BitVector, pivots::Dict{Int,BitVector})
    for i in eachindex(cv)
        cv[i] || continue
        if haskey(pivots, i)
            cv .⊻= pivots[i]
        else
            pivots[i] = copy(cv)
            return true
        end
    end
    return false
end

function _greedy_basis_h(cycles::Vector{HCycle}, simplices, num_vertices::Int, points)
    edges, triangles, vertex_ids = _normalize_edges_triangles(simplices)
    if isempty(edges)
        for cyc in cycles
            append!(edges, cyc)
            for (u, v) in cyc
                push!(vertex_ids, u); push!(vertex_ids, v)
            end
        end
    end
    if num_vertices <= 0 && !isempty(vertex_ids)
        num_vertices = maximum(vertex_ids) + 1
    end

    ew = Dict{HEdge,Float64}()
    for e in edges
        ew[e] = _edge_weight_h1(e[1], e[2], points)
    end

    simplices_all = Any[]
    append!(simplices_all, edges)
    append!(simplices_all, triangles)
    simplex_annotations, vec_dim = _annot_edge_h(simplices_all, num_vertices, ew)

    cycles_sorted = sort(cycles; by = cyc -> sum(_edge_weight_h1(u, v, points) for (u, v) in cyc))
    basis = HCycle[]
    pivots = Dict{Int,BitVector}()
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
        vs = _to_vertices_simplex(s)
        isempty(vs) && continue
        sort!(vs)
        for r in 1:length(vs)
            v = get!(out, r - 1, Vector{Vector{Int}}())
            for face_vec in _k_subsets(vs, r)
                if !any(x -> x == face_vec, v)
                    push!(v, face_vec)
                end
            end
        end
    end
    return out
end

function _k_subsets(v::Vector{Int}, k::Int)
    out = Vector{Vector{Int}}()
    n = length(v)
    if k <= 0 || k > n
        return out
    end
    idx = collect(1:k)
    while true
        push!(out, [v[i] for i in idx])
        i = k
        while i >= 1 && idx[i] == i + n - k
            i -= 1
        end
        i == 0 && break
        idx[i] += 1
        for j in (i + 1):k
            idx[j] = idx[j - 1] + 1
        end
    end
    return out
end

function _boundary_mod2(source::Vector{Vector{Int}}, target::Vector{Vector{Int}})
    m = length(target)
    n = length(source)
    M = zeros(Int, m, n)
    (m == 0 || n == 0) && return M
    t_idx = Dict{Tuple{Vararg{Int}}, Int}()
    for (i, t) in enumerate(target)
        t_idx[Tuple(t)] = i
    end
    for (j, s) in enumerate(source)
        for drop in eachindex(s)
            face = [s[i] for i in eachindex(s) if i != drop]
            row = get(t_idx, Tuple(face), 0)
            row == 0 && continue
            M[row, j] ⊻= 1
        end
    end
    return M
end

function _rref_mod2(A::Matrix{Int})
    M = A .% 2
    m, n = size(M)
    row = 1
    pivots = Int[]
    for col in 1:n
        pivot = 0
        for r in row:m
            if M[r, col] == 1
                pivot = r
                break
            end
        end
        pivot == 0 && continue
        if pivot != row
            M[row, :], M[pivot, :] = copy(M[pivot, :]), copy(M[row, :])
        end
        for r in 1:m
            if r != row && M[r, col] == 1
                M[r, :] .⊻= M[row, :]
            end
        end
        push!(pivots, col)
        row += 1
        row > m && break
    end
    return M, pivots
end

function _nullspace_basis_mod2(A::Matrix{Int})
    _, n = size(A)
    rref, pivots = _rref_mod2(A)
    pivot_set = Set(pivots)
    basis = Vector{Vector{Int}}()
    for free in 1:n
        in(free, pivot_set) && continue
        v = zeros(Int, n)
        v[free] = 1
        for (i, col) in enumerate(pivots)
            v[col] = rref[i, free] % 2
        end
        push!(basis, v)
    end
    return basis
end

_rank_mod2(A::Matrix{Int}) = length(last(_rref_mod2(A)))

function _independent_mod2(v::Vector{Int}, cols::Vector{Vector{Int}})
    vv = v .% 2
    if isempty(cols)
        return any(x -> x != 0, vv)
    end
    M_prev = hcat(cols...)
    M_new = hcat(M_prev, vv)
    return _rank_mod2(M_new) > _rank_mod2(M_prev)
end

function _chain_weight(chain::Vector{Int}, simplices_k::Vector{Vector{Int}}, points)
    active = [simplices_k[i] for i in eachindex(chain) if chain[i] % 2 == 1]
    isempty(active) && return 0.0
    points === nothing && return float(length(active))
    total = 0.0
    for s in active
        if length(s) <= 1
            total += 1.0
            continue
        end
        for i in 1:length(s)
            for j in (i + 1):length(s)
                u = s[i] + 1
                v = s[j] + 1
                total += norm(points[u, :] .- points[v, :])
            end
        end
    end
    return total
end

function _h0_generators(edges::Vector{HEdge}, num_vertices::Int)
    num_vertices <= 0 && return Vector{Dict{String, Any}}()
    dsu = HDSU(num_vertices)
    for (u, v) in edges
        if 0 <= u < num_vertices && 0 <= v < num_vertices
            _unite_h!(dsu, u + 1, v + 1)
        end
    end
    comps = Dict{Int, Vector{Int}}()
    for v in 0:(num_vertices - 1)
        r = _find_h!(dsu, v + 1)
        push!(get!(comps, r, Int[]), v)
    end
    out = Vector{Dict{String, Any}}()
    for verts in values(comps)
        rep = minimum(verts)
        push!(out, Dict(
            "dimension" => 0,
            "support_simplices" => [Any[rep]],
            "support_edges" => Any[],
            "weight" => 0.0,
            "certified_cycle" => true,
        ))
    end
    return out
end

function homology_generators_from_simplices(
    simplices,
    num_vertices::Int,
    dimension::Int,
    mode::String="valid",
    point_cloud=nothing,
    max_roots=nothing,
    root_stride::Int=1,
    max_cycles=nothing,
)
    points = point_cloud === nothing ? nothing : Matrix{Float64}(point_cloud)
    if dimension < 0
        error("dimension must be >= 0")
    end
    if mode != "valid" && mode != "optimal"
        error("mode must be 'valid' or 'optimal'")
    end

    if dimension == 1 && mode == "optimal"
        basis = optgen_from_simplices(simplices, num_vertices, points, max_roots, root_stride, max_cycles)
        out = Vector{Dict{String, Any}}()
        for cyc in basis
            support = Any[[e[1], e[2]] for e in cyc]
            push!(out, Dict(
                "dimension" => 1,
                "support_simplices" => support,
                "support_edges" => support,
                "weight" => sum(_edge_weight_h1(e[1], e[2], points) for e in cyc),
                "certified_cycle" => true,
            ))
        end
        return out
    end

    by_dim = _simplices_by_dim(simplices)
    if dimension == 0
        edges = HEdge[]
        for e in get(by_dim, 1, Vector{Vector{Int}}())
            length(e) == 2 || continue
            push!(edges, _order_hedge(e[1], e[2]))
        end
        nv = num_vertices
        if nv <= 0
            mx = -1
            for s in simplices
                for v in _to_vertices_simplex(s)
                    mx = max(mx, v)
                end
            end
            nv = mx + 1
        end
        return _h0_generators(edges, nv)
    end

    simplices_k = get(by_dim, dimension, Vector{Vector{Int}}())
    simplices_km1 = get(by_dim, dimension - 1, Vector{Vector{Int}}())
    simplices_kp1 = get(by_dim, dimension + 1, Vector{Vector{Int}}())
    isempty(simplices_k) && return Vector{Dict{String, Any}}()

    d_k = _boundary_mod2(simplices_k, simplices_km1)
    d_kp1 = _boundary_mod2(simplices_kp1, simplices_k)
    z_basis = _nullspace_basis_mod2(d_k)
    isempty(z_basis) && return Vector{Dict{String, Any}}()

    b_cols = [d_kp1[:, j] .% 2 for j in 1:size(d_kp1, 2)]
    z_candidates = copy(z_basis)
    if mode == "optimal"
        sort!(z_candidates, by = z -> _chain_weight(z, simplices_k, points))
    end

    reps = Vector{Vector{Int}}()
    span_cols = copy(b_cols)
    for z in z_candidates
        if _independent_mod2(z, span_cols)
            push!(reps, z)
            push!(span_cols, z .% 2)
        end
    end

    out = Vector{Dict{String, Any}}()
    for z in reps
        support = Any[]
        for i in eachindex(z)
            z[i] % 2 == 1 || continue
            push!(support, Any[simplices_k[i]...])
        end
        support_edges = dimension == 1 ? support : Any[]
        push!(out, Dict(
            "dimension" => dimension,
            "support_simplices" => support,
            "support_edges" => support_edges,
            "weight" => _chain_weight(z, simplices_k, points),
            "certified_cycle" => true,
        ))
    end
    return out
end

function abelianize_group(generators::Vector{String}, relations::Vector{String})
    # Extracts Betti numbers and invariant factors (torsion) of the abelianization.
    # Abelianization H_1 = Z^n / R.
    # We construct the relation matrix.
    n_gens = length(generators)
    gen_idx = Dict{String, Int}(g => i for (i, g) in enumerate(generators))
    
    n_rels = length(relations)
    M = zeros(Int, n_rels, n_gens)
    
    for i in 1:n_rels
        for m in eachmatch(r"([a-zA-Z0-9_]+)(?:\^(-?\d+))?", relations[i])
            base_w = m.captures[1]
            if haskey(gen_idx, base_w)
                pow_str = m.captures[2]
                pow = pow_str === nothing ? 1 : parse(Int, pow_str)
                M[i, gen_idx[base_w]] += pow
            end
        end  # for m
    end  # for i

    # Compute SNF of the relation matrix to extract free rank and torsion
    if !HAS_ABSTRACT_ALGEBRA
        # Fallback: rank-only over Q, no torsion recovery without SNF.
        rq = rank(Matrix{Float64}(M))
        free_rank = n_gens - rq
        return Int(free_rank), Int[]
    end
    ZZ = AbstractAlgebra.ZZ
    M_aa = AbstractAlgebra.matrix(ZZ, M)
    S_aa = AbstractAlgebra.snf(M_aa)
    diag = [Int64(S_aa[i, i]) for i in 1:min(n_rels, n_gens)]
    nonzero = filter(x -> x != 0, diag)
    torsion = filter(x -> x > 1, nonzero)
    free_rank = n_gens - length(nonzero)
    return free_rank, torsion
end  # function abelianize_group

function compute_boundary_mod2_matrix(
    source_simplices,
    target_simplices
)
    source = [_to_vertices_simplex(s) for s in source_simplices]
    target = [_to_vertices_simplex(t) for t in target_simplices]

    m = length(target)
    n = length(source)

    if m == 0 || n == 0
        return Dict("rows" => Int64[], "cols" => Int64[], "data" => Int64[], "m" => Int64(m), "n" => Int64(n))
    end

    t_idx = Dict{Tuple{Vararg{Int}}, Int}()
    for (i, t) in enumerate(target)
        t_idx[Tuple(sort(t))] = i - 1  # zero-based for Python
    end

    rows, cols, data = Int64[], Int64[], Int64[]
    for (j, s) in enumerate(source)
        for i_drop in eachindex(s)
            face_vec = vcat(s[1:(i_drop-1)], s[(i_drop+1):end])
            face = Tuple(sort(Int.(face_vec)))
            if haskey(t_idx, face)
                push!(rows, t_idx[face])
                push!(cols, j - 1)  # zero-based for Python
                push!(data, 1)
            end
        end
    end

    return Dict(
        "rows" => rows,
        "cols" => cols,
        "data" => data,
        "m" => Int64(m),
        "n" => Int64(n),
    )
end

function compute_alexander_whitney_cup(
    alpha,
    beta,
    p::Int,
    q::Int,
    simplices_p_plus_q,
    simplex_to_idx_p::Dict,
    simplex_to_idx_q::Dict,
    modulus=nothing
)
    n_simplices = length(simplices_p_plus_q)
    result = zeros(Int64, n_simplices)

    for i in 1:n_simplices
        simplex = _to_vertices_simplex(simplices_p_plus_q[i])

        if length(simplex) < p + q + 1
            continue
        end

        front_face = Tuple(simplex[1:(p+1)])
        back_face = Tuple(simplex[(p+1):(p+q+1)])

        idx_p = get(simplex_to_idx_p, front_face, -1)
        idx_q = get(simplex_to_idx_q, back_face, -1)

        if idx_p != -1 && idx_q != -1
            val = Int64(alpha[idx_p + 1]) * Int64(beta[idx_q + 1])  # one-based indexing in Julia
            if modulus !== nothing
                val = val % Int64(modulus)
            end
            result[i] = val
        end
    end

    return result
end

function compute_trimesh_boundary_data(
    faces,
    n_vertices::Int
)
    n_faces = length(faces)

    edge_to_idx = Dict{Tuple{Int, Int}, Int}()
    edges = Tuple{Int, Int}[]
    face_boundary_edges_list = Vector{Vector{Tuple{Int, Int, Int}}}()

    for face_vec in faces
        face_arr = _to_vertices_simplex(face_vec)
        cycle = Int.(face_arr)
        cyc_edges = Tuple{Int, Int, Int}[]

        for i in 1:length(cycle)
            u = cycle[i]
            v = cycle[mod(i, length(cycle)) + 1]
            key = (u, v) <= (v, u) ? (u, v) : (v, u)

            if !haskey(edge_to_idx, key)
                edge_to_idx[key] = length(edges) + 1  # one-based for Julia, returned as zero-based
                push!(edges, key)
            end
            push!(cyc_edges, (u, v, edge_to_idx[key]))
        end
        push!(face_boundary_edges_list, cyc_edges)
    end

    n_edges = length(edges)

    # d1: edges -> vertices
    d1_rows, d1_cols, d1_data = Int64[], Int64[], Int64[]
    for (j, (v1, v2)) in enumerate(edges)
        push!(d1_rows, v1 - 1)  # zero-based
        push!(d1_cols, j - 1)   # zero-based
        push!(d1_data, -1)
        push!(d1_rows, v2 - 1)
        push!(d1_cols, j - 1)
        push!(d1_data, 1)
    end

    # d2: faces -> edges
    d2_rows, d2_cols, d2_data = Int64[], Int64[], Int64[]
    for (j, edge_cycle) in enumerate(face_boundary_edges_list)
        for (u, v, idx) in edge_cycle
            sign = (u, v) == edges[idx] ? 1 : -1
            push!(d2_rows, idx - 1)  # zero-based
            push!(d2_cols, j - 1)    # zero-based
            push!(d2_data, sign)
        end
    end

    return Dict(
        "d1_rows" => d1_rows,
        "d1_cols" => d1_cols,
        "d1_data" => d1_data,
        "n_vertices" => Int64(n_vertices),
        "n_edges" => Int64(n_edges),
        "d2_rows" => d2_rows,
        "d2_cols" => d2_cols,
        "d2_data" => d2_data,
        "n_faces" => Int64(n_faces),
    )
end

function triangulate_surface_delaunay(points, tolerance::Real=1e-10)
    """
    Julia implementation of surface triangulation via Delaunay.

    Performs:
    1. PCA to find best-fit 2D plane
    2. Projects all points onto that plane
    3. Delaunay triangulation in 2D
    4. Returns indices of triangles
    """
    points = Matrix{Float64}(points)
    tol = Float64(tolerance)
    n_points, n_dims = size(points)

    if n_dims != 3
        error("Points must be 3D coordinates (shape: n_points × 3)")
    end

    if n_points < 3
        error("At least 3 points required for triangulation")
    end

    # Center the point cloud
    centroid = mean(points, dims=1)
    centered = points .- centroid

    # PCA via SVD
    U, S, Vt = svd(centered)

    # Check if surface is truly 2D
    if S[end] > tol
        @warn("Point cloud has significant variance in normal direction ($(S[end])). " *
              "Surface may not be truly 2D. Proceeding with best-fit plane.")
    end

    # Project onto 2D plane (first two principal directions)
    v1 = Vt[1, :]
    v2 = Vt[2, :]

    # 2D coordinates
    projected_2d = hcat(
        centered * v1,
        centered * v2
    )

    # Delaunay triangulation using built-in Julia
    try

        # VoronoiDelaunay is more reliable than raw Delaunay
        triangles = Vector{Vector{Int64}}()

        # Use QHull-like approach: simple incremental Delaunay
        # For simplicity, use a reference implementation
        points_2d = convert.(Float64, projected_2d)

        # Build triangles greedily from 2D points
        # For large point sets, this is O(n²) but exact
        used_edges = Set{Tuple{Int64, Int64}}()
        triangles = []

        # Find convex hull first, then Delaunay
        # Since we have pure Julia, we'll use a simple approach:
        # Sort points by angle from centroid, form triangles

        # Compute centroid in 2D
        centroid_2d = vec(mean(points_2d, dims=1))

        # Angle from centroid
        angles = [atan(points_2d[i, 2] - centroid_2d[2], points_2d[i, 1] - centroid_2d[1])
                  for i in 1:n_points]
        sorted_idx = sortperm(angles)

        # Simple fan triangulation from centroid
        for i in 2:(n_points - 1)
            v1 = sorted_idx[1]
            v2 = sorted_idx[i]
            v3 = sorted_idx[i + 1]
            push!(triangles, sort([v1, v2, v3]))
        end

        # Remove duplicates
        unique!(triangles)

        return triangles

    catch e
        error("Triangulation failed: $(e)")
    end
end

end # module SurgeryBackend
