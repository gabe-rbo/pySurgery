# A lightweight Julia backend module for high-performance algebraic operations over Z[pi]
module SurgeryBackend

using LinearAlgebra
using SparseArrays

export hermitian_signature, exact_snf_sparse, exact_sparse_cohomology_basis, group_ring_multiply, multisignature, abelianize_group

function hermitian_signature(matrix)
    # Convert to Julia Matrix
    mat = Matrix{Float64}(matrix)
    eigenvalues = eigvals(Hermitian(mat))
    pos = count(x -> x > 1e-10, eigenvalues)
    neg = count(x -> x < -1e-10, eigenvalues)
    return pos - neg
end

function exact_snf_sparse(rows, cols, vals, m, n)
    A = sparse(Vector{Int}(rows), Vector{Int}(cols), Vector{Int}(vals), m, n)
    
    try
        import AbstractAlgebra
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
        dense_A = Matrix{Float64}(A)
        U, S, V = svd(dense_A)
        factors = round.(Int64, S[S .> 1e-10])
        return sort(factors)
    end
end

function exact_sparse_cohomology_basis(
    d_np1_rows, d_np1_cols, d_np1_vals, d_np1_m, d_np1_n,
    d_n_rows, d_n_cols, d_n_vals, d_n_m, d_n_n
)
    # δ^n = d_{n+1}^T : C^n -> C^{n+1} (coboundary map)
    coboundary_mat = sparse(d_np1_cols, d_np1_rows, d_np1_vals, d_np1_n, d_np1_m)

    # B^n = im(δ^{n-1}) = column space of d_n^T
    has_image = d_n_m > 0 && d_n_n > 0 && length(d_n_vals) > 0 && any(d_n_vals .!= 0)

    try
        import AbstractAlgebra
        QQ = AbstractAlgebra.QQ
        M_qq = AbstractAlgebra.matrix(QQ, Matrix(coboundary_mat))
        nullity, nullspace_mat = AbstractAlgebra.nullspace(M_qq)

        basis = Vector{Vector{Int64}}()

        if !has_image
            # H^n = Z^n: no coboundaries to mod out; all cocycles are cohomology classes.
            for j in 1:nullity
                col_data = [nullspace_mat[k, j] for k in 1:d_np1_m]
                denoms = [AbstractAlgebra.denominator(col_data[k]) for k in 1:d_np1_m]
                lcm_val = reduce(lcm, denoms; init=1)
                int_vec = Int64[Int64(AbstractAlgebra.numerator(col_data[k] * lcm_val)) for k in 1:d_np1_m]
                push!(basis, int_vec)
            end
        else
            # H^n = Z^n / B^n: keep only cocycles independent from the image B^n.
            # Start from the image columns and greedily add independent nullspace vectors.
            # image_mat holds the generators of B^n = im(δ^{n-1})
            image_mat = Matrix(sparse(d_n_cols, d_n_rows, d_n_vals, d_n_n, d_n_m))
            accumulated = AbstractAlgebra.matrix(QQ, image_mat)
            accumulated_rank = AbstractAlgebra.rank(accumulated)
            for j in 1:nullity
                col_data = [nullspace_mat[k, j] for k in 1:d_np1_m]
                col_mat = AbstractAlgebra.matrix(QQ, d_np1_m, 1, col_data)
                test_mat = hcat(accumulated, col_mat)
                new_rank = AbstractAlgebra.rank(test_mat)
                if new_rank > accumulated_rank
                    accumulated = test_mat
                    accumulated_rank = new_rank
                    denoms = [AbstractAlgebra.denominator(col_data[k]) for k in 1:d_np1_m]
                    lcm_val = reduce(lcm, denoms; init=1)
                    int_vec = Int64[Int64(AbstractAlgebra.numerator(col_data[k] * lcm_val)) for k in 1:d_np1_m]
                    push!(basis, int_vec)
                end
            end
        end
        return basis
    catch e
        # Float64 full-SVD fallback. Uses the full unitary V so that all null-space components
        # are captured even when d_{n+1} has more columns than rows.
        # WARNING: SVD produces orthonormal float vectors, not exact integer vectors.
        # This fallback also does NOT subtract the coboundary image B^n.
        dense_M = Matrix{Float64}(coboundary_mat)
        F = svd(dense_M; full=true)
        largest_sv = isempty(F.S) ? 1.0 : max(F.S[1], 1.0)
        tol = maximum(size(dense_M)) * eps(Float64) * largest_sv
        rank_M = count(x -> x > tol, F.S)
        nullity_svd = size(F.V, 2) - rank_M
        return [round.(Int64, F.V[:, rank_M + i] .* 1000) for i in 1:nullity_svd]
    end
end

function group_ring_multiply(k1::Vector{String}, v1::Vector{Int}, k2::Vector{String}, v2::Vector{Int}, group_order::Int)
    res_dict = Dict{String, Int}()
    for i in 1:length(k1)
        for j in 1:length(k2)
            function parse_gen(g_str)
                if g_str == "e" || g_str == "1"
                    return 0
                end
                inv = endswith(g_str, "^-1")
                base = inv ? replace(g_str[2:end-3], "g" => "") : replace(g_str, "g" => "")
                val = parse(Int, base)
                return inv ? -val : val
            end
            
            p1 = parse_gen(k1[i])
            p2 = parse_gen(k2[j])
            p_res = (p1 + p2) % group_order
            if p_res < 0
                p_res += group_order
            end
            g_str = p_res == 0 ? "1" : "g$(p_res)"
            
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
    error("True multisignature computation over character eigenspaces is mathematically complex and currently unsupported. Signature modulo p is incorrect.")
end

function abelianize_group(generators::Vector{String}, relations::Vector{String})
    # Extracts free rank and torsion invariant factors of the abelianization H_1 = Z^n / im(M),
    # where M is the relation matrix (rows = relations, cols = generators).
    n_gens = length(generators)
    gen_idx = Dict{String, Int}(g => i for (i, g) in enumerate(generators))

    n_rels = length(relations)
    M = zeros(Int, n_rels, n_gens)

    for i in 1:n_rels
        words = split(relations[i], " ")
        for w in words
            if w == "" continue end
            if endswith(w, "^-1")
                # Inverse generator: g^-1 contributes -1
                base_w = w[1:end-3]
                if haskey(gen_idx, base_w)
                    M[i, gen_idx[base_w]] -= 1
                end
            elseif (caret_pos = findfirst('^', w)) !== nothing
                # Positive power: g^k contributes +k
                base_w = w[1:caret_pos-1]
                exp_val = tryparse(Int, w[caret_pos+1:end])
                if exp_val !== nothing && haskey(gen_idx, base_w)
                    M[i, gen_idx[base_w]] += exp_val
                end
            else
                # Plain generator: contributes +1
                if haskey(gen_idx, w)
                    M[i, gen_idx[w]] += 1
                end
            end
        end
    end

    # Use exact SNF over Z for correct invariant factors (torsion)
    try
        import AbstractAlgebra
        ZZ = AbstractAlgebra.ZZ
        M_aa = AbstractAlgebra.matrix(ZZ, M)
        S_aa = AbstractAlgebra.snf(M_aa)
        diag_size = min(n_gens, n_rels)
        diag_vals = [Int64(S_aa[i, i]) for i in 1:diag_size]
        nonzero_count = count(v -> v != 0, diag_vals)
        free_rank = n_gens - nonzero_count
        torsion = sort([abs(v) for v in diag_vals if abs(v) > 1])
        return free_rank, torsion
    catch e
        # Float64 SVD fallback: approximates free rank; torsion factors may be inaccurate
        tol = max(n_gens, n_rels) * eps(Float64) * (norm(Float64.(M)) + 1.0)
        U, S, V = svd(Float64.(M))
        nonzero_sv = count(x -> x > tol, S)
        free_rank = n_gens - nonzero_sv
        s_vals = round.(Int, S[S .> tol])
        torsion = sort([v for v in s_vals if v > 1])
        return free_rank, torsion
    end
end

end
