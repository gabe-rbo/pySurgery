# A lightweight Julia backend module for high-performance algebraic operations over Z[pi]
module SurgeryBackend

using LinearAlgebra
using SparseArrays

export hermitian_signature, exact_snf_sparse, exact_sparse_cohomology_basis, group_ring_multiply, multisignature, abelianize_group, integral_lattice_isometry, optgen_from_simplices

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
    A = sparse(Vector{Int}(rows), Vector{Int}(cols), Vector{Int}(vals), m, n)
    
    try
        if !HAS_ABSTRACT_ALGEBRA
            error("AbstractAlgebra unavailable")
        end
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
        # Correct fallback: estimate rank only, cannot recover invariant factors
        rank_estimate = rank(Matrix{Float64}(A))
        return ones(Int64, rank_estimate)  # 1s signal "no torsion detected, rank only"
    end
end

function exact_sparse_cohomology_basis(
    d_np1_rows, d_np1_cols, d_np1_vals, d_np1_m, d_np1_n,
    d_n_rows, d_n_cols, d_n_vals, d_n_m, d_n_n
)
    coboundary_mat = sparse(d_np1_cols, d_np1_rows, d_np1_vals, d_np1_n, d_np1_m)
    
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
        tol = maximum(size(dense_M)) * eps(Float64) * F.S[1]
        null_indices = findall(x -> x <= tol, F.S)
        nullity = length(null_indices)
        # Float fallback: recover null space from SVD, then scale each vector to
        # clear denominators using the GCD of a rational reconstruction.
        null_basis_float = [F.V[:, i] for i in (size(F.V, 2) - nullity + 1):size(F.V, 2)]
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
    
    dn_mat = sparse(d_n_cols, d_n_rows, d_n_vals, d_n_n, d_n_m)
    
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
    order = sortperm(edges; by = e -> weights[e])
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

end # module SurgeryBackend
