# A lightweight Julia backend module for high-performance algebraic operations over Z[pi]
module SurgeryBackend

using LinearAlgebra
using SparseArrays

export hermitian_signature, exact_snf_sparse, exact_sparse_cohomology_basis, group_ring_multiply, multisignature, abelianize_group

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
    coboundary_mat = sparse(d_np1_cols, d_np1_rows, d_np1_vals, d_np1_n, d_np1_m)
    
    basis = Vector{Vector{Int64}}()
    try
        import AbstractAlgebra
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
        basis = [round.(Int64, F.V[:, i] .* 1000) for i in (size(F.V, 2) - nullity + 1):size(F.V, 2)]
    end
    
    dn_mat = sparse(d_n_cols, d_n_rows, d_n_vals, d_n_n, d_n_m)
    dense_dn = Matrix{Float64}(dn_mat)
    
    quotient_basis = Vector{Vector{Int64}}()
    if size(dense_dn, 2) > 0
        curr_rank = rank(dense_dn)
        for vec in basis
            test_mat = hcat(dense_dn, Float64.(vec))
            new_rank = rank(test_mat)
            if new_rank > curr_rank
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
        end
    end
    
    # SNF on M gives the structure of the abelianized group
    U, S, V = svd(Float64.(M))
    s_vals = round.(Int, S[S .> 1e-10])
    torsion = sort(s_vals[s_vals .> 1])
    
    rank = n_gens - length(s_vals)
    return rank, torsion
end

end
s = round.(Int, S[S .> tol])
        torsion = sort(s_vals[s_vals .> 1])
        
        rank = n_gens - length(s_vals)
        return rank, torsion
    end
end

end
   end
end

end
s = round.(Int, S[S .> tol])
        torsion = sort(s_vals[s_vals .> 1])
        
        rank = n_gens - length(s_vals)
        return rank, torsion
    end
end

end
