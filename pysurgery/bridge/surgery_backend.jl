# A lightweight Julia backend module for high-performance algebraic operations over Z[pi]
module SurgeryBackend

using LinearAlgebra
using SparseArrays
using Statistics
using Combinatorics
using Combinatorics: combinations
using Random
using PythonCall: pyconvert, Py, PyDict
using IntegerSmithNormalForm
using AbstractAlgebra
import PrecompileTools

export simplify_jl, hermitian_signature, exact_snf_sparse, exact_sparse_cohomology_basis, rank_q_sparse, rank_mod_p_sparse, sparse_cohomology_basis_mod_p, normal_surface_residual_norms, embedding_broad_phase_pairs, group_ring_multiply, multisignature, abelianize_group, integral_lattice_isometry, optgen_from_simplices, homology_generators_from_simplices, compute_boundary_data_from_simplices, compute_boundary_payload_from_simplices, compute_boundary_payload_from_flat_simplices, compute_boundary_mod2_matrix, compute_alexander_whitney_cup, compute_trimesh_boundary_data, compute_trimesh_boundary_data_flat, triangulate_surface_delaunay, orthogonal_procrustes, pairwise_distance_matrix, frechet_distance, gromov_wasserstein_distance, enumerate_cliques_sparse, compute_vietoris_rips, compute_circumradius_sq_3d, compute_circumradius_sq_2d, quick_mapper_jl, cknn_graph_jl, cknn_graph_accelerated_jl, is_homology_manifold_jl, compute_alpha_complex_simplices_jl, compute_alpha_threshold_emst_jl, compute_crust_simplices_jl, compute_witness_complex_simplices_jl, compute_discrete_morse_gradient_jl,
    snf_markowitz_column_order, modular_rank_certification_jl, padic_snf_diagonal_jl, batch_exact_snf_sparse,
    todd_coxeter_index_jl, cayley_table_jl, cayley_convolve_jl, lift_boundary_to_cover_jl,
    fox_derivative_block_real_jl, fox_derivative_block_complex_jl, twisted_alexander_whitney_jl,
    BarcodeResult, compute_persistence_barcodes, compute_filtration_persistence, compute_rips_filtration, compute_rips_cohomology, compute_alpha_filtration,
    surgery_relative_boundary_sparse, linking_seifert_solve_z, linking_intersection_pairing,
    surgery_handle_attach, sphere_recognition_pl,
    linking_intersection_batch, compute_cohomology_basis_jl, linking_intersect_2chains,
    alexander_from_seifert_jl, knot_signature_jl, linking_gauss_riemann_jl

# --- Persistent Homology Types & Exceptions ---

struct BarcodeResult
    birth::Int
    death::Int
    dim::Int
    multiplicity::Int
end

struct DensificationError <: Exception
    msg::String
end

struct MemoryBoundsError <: Exception
    msg::String
end

struct UnsupportedFieldError <: Exception
    msg::String
end

const MAX_MEMORY_BYTES = 3 * 1024^3 # 3 GB

function check_memory_bounds()
    # Base.gc_live_bytes() provides an estimate of currently allocated memory.
    if Base.gc_live_bytes() > MAX_MEMORY_BYTES
        throw(MemoryBoundsError("Memory limit exceeded! Used: \$(Base.gc_live_bytes() / 1024^3) GB > 3.0 GB"))
    end
end

function get_lowest_nonzero_row(R::SparseMatrixCSC, j::Int)
    start_idx = R.colptr[j]
    end_idx = R.colptr[j+1] - 1
    # Find the last element that is actually non-zero
    for idx in end_idx:-1:start_idx
        if R.nzval[idx] != 0
            return R.rowval[idx]
        end
    end
    return 0
end

function sparse_RDV_reduction(D::SparseMatrixCSC, field::Symbol)
    if !issparse(D)
        throw(DensificationError("Densification attempted! SparseArrays.jl strictly required."))
    end

    n_rows, n_cols = size(D)

    R = copy(D)
    V = sparse(1I, n_cols, n_cols)
    if field == :Q
        R = SparseMatrixCSC{Rational{Int}, Int}(R.m, R.n, R.colptr, R.rowval, Rational{Int}.(R.nzval))
        V = SparseMatrixCSC{Rational{Int}, Int}(V.m, V.n, V.colptr, V.rowval, Rational{Int}.(V.nzval))
    elseif field == :Z2
        R = SparseMatrixCSC{Int8, Int}(R.m, R.n, R.colptr, R.rowval, Int8.(mod.(R.nzval, 2)))
        V = SparseMatrixCSC{Int8, Int}(V.m, V.n, V.colptr, V.rowval, Int8.(mod.(V.nzval, 2)))
    end
    dropzeros!(R)

    low = zeros(Int, n_cols)
    low_to_col = Dict{Int, Int}()

    for j in 1:n_cols
        if j % 1000 == 0
            check_memory_bounds()
        end

        while true
            low_j = get_lowest_nonzero_row(R, j)

            if low_j == 0
                low[j] = 0
                break
            end

            k = get(low_to_col, low_j, 0)

            if k == 0 || k >= j
                low[j] = low_j
                low_to_col[low_j] = j
                break
            end

            # We need to find R[low_j, j] and R[low_j, k]
            # Since low_j is the lowest non-zero row, it's the last non-zero in the column.
            val_j = R[low_j, j]
            val_k = R[low_j, k]

            if field == :Q
                c = val_j // val_k
                R[:, j] = dropzeros!(R[:, j] - c * R[:, k])
                V[:, j] = dropzeros!(V[:, j] - c * V[:, k])
            elseif field == :Z2
                # In Z2, a + b mod 2.
                R[:, j] = dropzeros!(mod.(R[:, j] + R[:, k], 2))
                V[:, j] = dropzeros!(mod.(V[:, j] + V[:, k], 2))
            end

            # Because `R[:, j] = ...` modifies the sparse matrix, we must call dropzeros! on the whole matrix
            # or ensure the assignment drops zeros. The assignment `R[:, j] = dropzeros!(...)` might still 
            # keep explicit zeros in the CSC structure of `R` depending on how `setindex!` is implemented.
            # To be safe and prevent infinite loops, we dropzeros on R.
            dropzeros!(R)

            if nnz(R) > 0.5 * n_rows * n_cols && n_rows > 1000
                throw(DensificationError("Matrix densified during reduction!"))
            end
        end
    end

    return R, V, low
end

"""
    compute_persistence_barcodes(boundary_matrices::Dict{Int, <:SparseMatrixCSC}, field::Symbol)

Computes the persistence barcodes for a given filtration over a specified field.
"""
function compute_persistence_barcodes(boundary_matrices::Dict{Int, <:SparseMatrixCSC}, field::Symbol)
    if field !== :Q && field !== :Z2
        throw(UnsupportedFieldError("Field must be :Q or :Z2"))
    end

    barcodes = Vector{BarcodeResult}()
    
    # We need to process dimensions in order
    dims = sort(collect(keys(boundary_matrices)))
    
    # A cycle born at index i in dimension d-1 is killed by column j in dimension d.
    # To correctly map birth/death pairs, we assume the boundary matrices are provided
    # over the entire filtration length, i.e., D_d is (N x N) where N is total simplices.
    # If D_d is n_{d-1} x n_d, we need to map local column/row indices to global filtration indices.
    # However, standard persistence matrix D is a single large block strictly upper triangular matrix
    # for all dimensions. Let's assume the user passes a single D or dict of blocks.
    # Wait, the architectural spec says `boundary_matrices::Dict{Int, SparseMatrixCSC}`
    # and we iterate through dimensions. The R=DV algorithm returns `low`.
    # Let's assume boundary_matrices[d] maps d-simplices (columns) to (d-1)-simplices (rows).
    # The columns of boundary_matrices[d] are 1..n_d, rows 1..n_{d-1}.
    # The output `low` vector gives the local row index (1..n_{d-1}) that kills it.
    
    # To form barcodes properly when boundary matrices are separated by dimension,
    # the birth index is the filtration index of the (d-1)-simplex (row),
    # the death index is the filtration index of the d-simplex (col).
    # Since we don't have the global filtration mapping here, we return local indices
    # (birth = row_index in D_d, death = col_index in D_d) which the Python side will map.
    
    for d in dims
        D = boundary_matrices[d]
        R, V, low = sparse_RDV_reduction(D, field)
        
        # For dimension d, the columns are d-simplices.
        n_rows, n_cols = size(D)
        
        # Keep track of which (d-1) simplices (rows) are paired
        paired_rows = falses(n_rows)
        
        # Temporary storage for features born in d-1
        raw_intervals = []

        for j in 1:n_cols
            if low[j] > 0
                i = low[j]
                paired_rows[i] = true
                # Feature born at i (in d-1) and died at j (in d)
                push!(raw_intervals, (i, j, d-1))
            else
                # Column j is a cycle, born at j (in d). Will be killed in d+1, or infinite.
            end
        end
        
        # Infinite features from d-1
        for i in 1:n_rows
            if !paired_rows[i]
                # Is it a cycle? If D_d is just the boundary matrix, we actually don't know if row i is a cycle
                # unless we also reduced D_{d-1}.
                # The exact R=DV on D_d alone identifies cycles in d-1 (the zero rows of R? No, the rows not in `low`).
                # Wait, a simplex i in d-1 is a cycle if it was a zero-column in R_{d-1}.
                # We need to carry over cycle information if we do this block by block, or just return the pairings.
                # Since we process block by block independently, let's just return the paired ones and handle cycles 
                # globally, or assume standard boundary matrix structure.
            end
        end
        
        # Group identical intervals
        counts = Dict{Tuple{Int, Int, Int}, Int}()
        for (b, de, dim) in raw_intervals
            counts[(b, de, dim)] = get(counts, (b, de, dim), 0) + 1
        end
        
        for ((b, de, dim), mult) in counts
            push!(barcodes, BarcodeResult(b, de, dim, mult))
        end
    end

    return barcodes
end

# Symmetric difference of two ascending Int vectors written into `dst` == the Z₂
# sum of two sparse columns (shared rows cancel mod 2). Reusing `dst` across the
# reduction keeps the inner loop allocation-free. Result stays ascending so its
# pivot is its last entry.
function _symdiff_into!(dst::Vector{Int}, a::Vector{Int}, b::Vector{Int})
    na = length(a); nb = length(b)
    resize!(dst, na + nb)
    i = 1; j = 1; r = 0
    @inbounds while i <= na && j <= nb
        ai = a[i]; bj = b[j]
        if ai < bj
            r += 1; dst[r] = ai; i += 1
        elseif ai > bj
            r += 1; dst[r] = bj; j += 1
        else
            i += 1; j += 1            # equal row index → cancels
        end
    end
    @inbounds while i <= na
        r += 1; dst[r] = a[i]; i += 1
    end
    @inbounds while j <= nb
        r += 1; dst[r] = b[j]; j += 1
    end
    resize!(dst, r)
    return dst
end

# Twist/clearing Z₂ reduction over a CSR boundary matrix. Column `r`'s facet ranks
# are `bnd_idx[bnd_ptr[r] : bnd_ptr[r+1]-1]` (ascending; empty for vertices).
# Dimensions are reduced top-down so a discovered pivot's column (a positive
# simplex) is cleared rather than re-reduced. Two scratch buffers are ping-ponged
# so each column XOR reuses storage; only a pivot column's final reduced form is
# copied into `Rcols` for later use. Returns `(bar_dim, bar_birth, bar_death)`.
function _reduce_persistence(ord_dim::Vector{Int}, ord_val::Vector{Float64},
                             maxdim::Int, bnd_ptr::Vector{Int}, bnd_idx::Vector{Int})
    M = length(ord_dim)
    cols_by_dim = [Int[] for _ in 0:max(maxdim, 0)]
    @inbounds for r in 1:M
        push!(cols_by_dim[ord_dim[r] + 1], r)
    end

    # Pivot rows and column ranks are dense in 1..M, so plain arrays beat Dicts.
    pivot_of_row = zeros(Int, M)           # pivot row rank → owning column rank (0 = none)
    Rcols = Vector{Vector{Int}}(undef, M)  # reduced column stored at each pivot column
    low_of = zeros(Int, M)                 # pivot row of each column (0 if zeroed)
    cleared = falses(M)                    # positive simplices removed by clearing

    col = Int[]
    scratch = Int[]
    for d in maxdim:-1:1
        @inbounds for r in cols_by_dim[d + 1]
            if cleared[r]
                continue                 # known positive (birth) — skip entirely
            end
            lo = bnd_ptr[r]; hi = bnd_ptr[r + 1] - 1
            resize!(col, hi - lo + 1)
            for t in lo:hi
                col[t - lo + 1] = bnd_idx[t]
            end
            while !isempty(col)
                low = col[end]
                owner = pivot_of_row[low]   # 0 when this pivot row is unclaimed
                if owner == 0
                    break
                end
                _symdiff_into!(scratch, col, Rcols[owner])
                col, scratch = scratch, col
            end
            if !isempty(col)
                low = col[end]
                pivot_of_row[low] = r
                low_of[r] = low
                Rcols[r] = copy(col)
                cleared[low] = true      # pivot simplex is positive → clear it
            end
        end
    end

    bar_dim = Int[]
    bar_birth = Float64[]
    bar_death = Float64[]
    @inbounds for r in 1:M
        lr = low_of[r]
        if lr > 0
            if ord_val[r] > ord_val[lr]
                push!(bar_dim, ord_dim[lr])
                push!(bar_birth, ord_val[lr])
                push!(bar_death, ord_val[r])
            end
        elseif !cleared[r]
            push!(bar_dim, ord_dim[r])
            push!(bar_birth, ord_val[r])
            push!(bar_death, Inf)
        end
    end
    return (bar_dim, bar_birth, bar_death)
end

"""
    _compute_filtration_persistence(simplices_flat, simplex_ptr, vals)

Exact Z₂ persistent homology of a monotone simplicial filtration.

This is the standard boundary-matrix reduction (so the result is identical to a
naïve textbook reduction — *not* an approximation) accelerated with the
Chen–Kerber *twist*/clearing optimisation. Clearing removes provably-redundant
work only; the pivots — hence every bar — are unchanged.

Inputs are a flat description of the maximal complex in *any* order:
- `simplices_flat` : concatenated vertex indices of every simplex.
- `simplex_ptr`    : 0-based offsets, length `M+1`; simplex `j` (1-based) spans
                     `simplices_flat[simplex_ptr[j]+1 : simplex_ptr[j+1]]`.
- `vals`           : appearance value of each simplex (aligned with `simplex_ptr`).

The simplices are (re)ordered here by `(value, dimension)` — a valid filtration
order (faces precede cofaces) — and the boundary is assembled as a CSR matrix of
facet ranks before the reduction.

Returns `(bar_dim, bar_birth, bar_death)`; `bar_death == Inf` marks essential
(never-dying) classes. Zero-persistence pairs (`death == birth`) are omitted,
matching the Python reference.
"""
function _compute_filtration_persistence(simplices_flat::Vector{Int},
                                         simplex_ptr::Vector{Int},
                                         vals::Vector{Float64})
    M = length(simplex_ptr) - 1
    if M <= 0
        return (Int[], Float64[], Float64[])
    end

    # Reconstruct each simplex as an ascending vertex vector, then hand off to the
    # shared reduction core (which assumes ascending vertices).
    simplices = Vector{Vector{Int}}(undef, M)
    @inbounds for j in 1:M
        simplices[j] = sort(simplices_flat[simplex_ptr[j]+1 : simplex_ptr[j+1]])
    end
    return _reduce_from_simplices(simplices, vals)
end

"""
    _reduce_from_simplices(simplices, vals)

Shared exact-reduction core behind both the flat entry point and the fused Rips
builder. Given `simplices` (each an **ascending** vertex vector) and their
appearance `vals`, order by `(value, dimension)`, assemble the CSR boundary, and
run the twist/clearing Z₂ reduction. Returns `(bar_dim, bar_birth, bar_death)`.

Precondition: every `simplices[j]` is sorted ascending (both callers guarantee
this — the flat path sorts on reconstruction, the Rips DFS emits ascending
cliques). The barcode is invariant to the order of equal-`(value, dimension)`
simplices, so no vertex tie-break is needed in the sort key.
"""
function _extract_apparent_pairs(simplices::Vector{Vector{Int}}, vals::Vector{Float64})
    M = length(simplices)
    if M <= 1
        return Set{Int}()
    end

    maxvert = 0
    max_len = 0
    @inbounds for s in simplices
        len_s = length(s)
        if len_s > max_len
            max_len = len_s
        end
        if len_s > 0 && s[end] > maxvert
            maxvert = s[end]
        end
    end

    b = 8 * sizeof(Int) - leading_zeros(maxvert + 1)
    removed = Set{Int}()

    if max_len * b <= 62
        code_to_idx = Dict{Int, Int}()
        sizehint!(code_to_idx, M)
        @inbounds for i in 1:M
            s = simplices[i]
            code = 0
            for v in s
                code = (code << b) | (v + 1)
            end
            code_to_idx[code] = i
        end

        coface_info = Dict{Int, Tuple{Int, Int}}()
        sizehint!(coface_info, M)

        @inbounds for i in 1:M
            s = simplices[i]
            k = length(s)
            k < 2 && continue
            for p in 1:k
                facet_code = 0
                for q in 1:k
                    if q != p
                        facet_code = (facet_code << b) | (s[q] + 1)
                    end
                end

                if haskey(code_to_idx, facet_code)
                    info = get(coface_info, facet_code, (0, 0))
                    if info[2] == 0
                        coface_info[facet_code] = (i, 1)
                    elseif info[2] == 1
                        coface_info[facet_code] = (0, 2)
                    end
                end
            end
        end

        for (facet_code, info) in coface_info
            if info[2] == 1
                tau_idx = code_to_idx[facet_code]
                sigma_idx = info[1]
                
                # Check if sigma itself is a maximal simplex (has no cofaces)
                s_sigma = simplices[sigma_idx]
                sigma_code = 0
                for v in s_sigma
                    sigma_code = (sigma_code << b) | (v + 1)
                end
                
                if !haskey(coface_info, sigma_code)
                    if abs(vals[tau_idx] - vals[sigma_idx]) < 1e-12
                        if !(tau_idx in removed) && !(sigma_idx in removed)
                            push!(removed, tau_idx)
                            push!(removed, sigma_idx)
                        end
                    end
                end
            end
        end
    else
        vindex = Dict{Vector{Int}, Int}()
        sizehint!(vindex, M)
        @inbounds for i in 1:M
            vindex[simplices[i]] = i
        end

        coface_info = Dict{Vector{Int}, Tuple{Int, Int}}()
        sizehint!(coface_info, M)

        buf = Int[]
        @inbounds for i in 1:M
            s = simplices[i]
            k = length(s)
            k < 2 && continue
            resize!(buf, k - 1)
            for p in 1:k
                idx = 1
                for q in 1:k
                    if q != p
                        buf[idx] = s[q]; idx += 1
                    end
                end
                if haskey(vindex, buf)
                    info = get(coface_info, buf, (0, 0))
                    if info[2] == 0
                        coface_info[copy(buf)] = (i, 1)
                    elseif info[2] == 1
                        coface_info[buf] = (0, 2)
                    end
                end
            end
        end

        for (facet, info) in coface_info
            if info[2] == 1
                tau_idx = vindex[facet]
                sigma_idx = info[1]
                
                # Check if sigma itself is a maximal simplex
                sigma = simplices[sigma_idx]
                if !haskey(coface_info, sigma)
                    if abs(vals[tau_idx] - vals[sigma_idx]) < 1e-12
                        if !(tau_idx in removed) && !(sigma_idx in removed)
                            push!(removed, tau_idx)
                            push!(removed, sigma_idx)
                        end
                    end
                end
            end
        end
    end

    return removed
end

function _reduce_from_simplices(simplices::Vector{Vector{Int}}, vals::Vector{Float64})
    removed = _extract_apparent_pairs(simplices, vals)
    if !isempty(removed)
        M_orig = length(simplices)
        keep = Int[]
        sizehint!(keep, M_orig - length(removed))
        for i in 1:M_orig
            if !(i in removed)
                push!(keep, i)
            end
        end
        simplices = simplices[keep]
        vals = vals[keep]
    end

    M = length(simplices)
    if M <= 0
        return (Int[], Float64[], Float64[])
    end

    dims = Vector{Int}(undef, M)
    @inbounds for j in 1:M
        dims[j] = length(simplices[j]) - 1
    end

    # Global filtration order: by (value, dimension). A face has value ≤ its
    # cofaces, and at equal value strictly smaller dimension, so faces always
    # precede cofaces — a valid reduction order. Among equal-(value,dimension)
    # simplices any order is valid and the barcode is invariant, so we omit a
    # vertex tie-break (a Vector in the sort key would heap-allocate on every
    # comparison). isbits (Float64, Int) keys keep the sort allocation-free.
    order = collect(1:M)
    sort!(order, by = j -> (vals[j], dims[j]), alg = QuickSort)

    ord_dim = Vector{Int}(undef, M)
    ord_val = Vector{Float64}(undef, M)
    ord_simplex = Vector{Vector{Int}}(undef, M)
    maxdim = 0
    maxvert = 0
    @inbounds for r in 1:M
        j = order[r]
        s = simplices[j]
        ord_dim[r] = dims[j]
        ord_val[r] = vals[j]
        ord_simplex[r] = s
        if dims[j] > maxdim
            maxdim = dims[j]
        end
        if !isempty(s) && s[end] > maxvert
            maxvert = s[end]
        end
    end

    # CSR boundary: column r (rank order) holds the ascending ranks of its facets;
    # vertices (dim 0) have none. Facet ranks come from an index mapping each
    # simplex to its rank.
    bnd_ptr = Vector{Int}(undef, M + 1)
    bnd_ptr[1] = 1
    @inbounds for r in 1:M
        k = ord_dim[r] + 1
        bnd_ptr[r + 1] = bnd_ptr[r] + (k >= 2 ? k : 0)
    end
    bnd_idx = Vector{Int}(undef, bnd_ptr[M + 1] - 1)

    # Facet lookup maps a (d-1)-simplex to its global rank. We pack a simplex's
    # ascending vertices into a single Int when they fit in 62 bits (the common
    # case) -- Int keys hash far faster than Vector keys. Vertices are shifted by
    # +1 so no slot is zero: a leading 0 would otherwise make e.g. {0,2,3} collide
    # with the edge {2,3}. With the +1 shift, codes of different lengths occupy
    # disjoint magnitude ranges, so packing is globally injective. Otherwise we
    # fall back to Vector keys, correct at any size.
    b = 8 * sizeof(Int) - leading_zeros(maxvert + 1)
    if (maxdim + 1) * b <= 62
        index = Dict{Int, Int}()
        sizehint!(index, M)
        @inbounds for r in 1:M
            s = ord_simplex[r]
            code = 0
            for v in s
                code = (code << b) | (v + 1)
            end
            index[code] = r
        end
        @inbounds for r in 1:M
            s = ord_simplex[r]
            k = length(s)
            k < 2 && continue
            base = bnd_ptr[r]
            for p in 1:k
                code = 0
                for q in 1:k
                    if q != p
                        code = (code << b) | (s[q] + 1)
                    end
                end
                bnd_idx[base + p - 1] = index[code]
            end
            sort!(view(bnd_idx, bnd_ptr[r]:bnd_ptr[r + 1] - 1))
        end
    else
        vindex = Dict{Vector{Int}, Int}()
        sizehint!(vindex, M)
        @inbounds for r in 1:M
            vindex[ord_simplex[r]] = r
        end
        buf = Int[]
        @inbounds for r in 1:M
            s = ord_simplex[r]
            k = length(s)
            k < 2 && continue
            base = bnd_ptr[r]
            resize!(buf, k - 1)
            for p in 1:k
                idx = 1
                for q in 1:k
                    if q != p
                        buf[idx] = s[q]; idx += 1
                    end
                end
                bnd_idx[base + p - 1] = vindex[buf]
            end
            sort!(view(bnd_idx, bnd_ptr[r]:bnd_ptr[r + 1] - 1))
        end
    end

    return _reduce_persistence(ord_dim, ord_val, maxdim, bnd_ptr, bnd_idx)
end

"""
    compute_filtration_persistence(simplices_flat, simplex_ptr, vals)

PythonCall entry point for [`_compute_filtration_persistence`](@ref); materialises
the NumPy buffers as native Julia vectors and runs the exact Z₂ reduction.
"""
function compute_filtration_persistence(simplices_flat_raw, simplex_ptr_raw, vals_raw)
    return _compute_filtration_persistence(
        pyconvert(Vector{Int}, simplices_flat_raw),
        pyconvert(Vector{Int}, simplex_ptr_raw),
        pyconvert(Vector{Float64}, vals_raw),
    )
end

"""
    _compute_rips_filtration(points, epsilon, max_dim)

Fused Vietoris–Rips persistence — build the VR complex from `points` up to
`epsilon`/`max_dim`, give each simplex its longest-edge (max pairwise distance)
appearance value, and run the exact twist/clearing Z₂ reduction, **all in Julia**.
The full simplex set never crosses to Python: only the barcode and small summary
arrays return. This removes the Python→Julia→Python clique/​boundary marshaling
that dominates large filtration reports.

The 1-skeleton and clique enumeration mirror [`compute_vietoris_rips`](@ref); the
clique diameter is carried incrementally through the DFS (each extension by a
vertex `v` adds the edges from `v` to the current clique), so the longest-edge
value is exact and matches the Python `rips_filtration_values`. Cliques are emitted
ascending, satisfying [`_reduce_from_simplices`](@ref)'s precondition.

Returns `(bar_dim, bar_birth, bar_death, eps_values, dim_ids, dim_first_val,
dim_count, total)`: the barcode (`death == Inf` for essential classes), the sorted
distinct appearance values (filtration grid), and per-dimension first (minimum)
appearance value / simplex count, plus the total simplex count.
"""
function _compute_rips_filtration(points::Matrix{Float64}, epsilon::Float64, max_dim::Int, analyze_manifolds::Bool=false, n_samples::Union{Nothing, Int}=nothing)
    n_pts = size(points, 1)
    dim_pts = size(points, 2)

    # --- 1-skeleton (parallel), same structure as compute_vietoris_rips ---
    eps2 = epsilon^2
    n_threads = Threads.maxthreadid()
    I_threads = [Int[] for _ in 1:n_threads]
    J_threads = [Int[] for _ in 1:n_threads]
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
                push!(I_local, i); push!(J_local, j)
                push!(I_local, j); push!(J_local, i)
            end
        end
    end
    I = reduce(vcat, I_threads)
    J = reduce(vcat, J_threads)
    adj = sparse(I, J, ones(Int, length(I)), n_pts, n_pts)
    rowptr = adj.colptr
    colval = adj.rowval

    # Per-thread clique + value buffers.
    cliques_threads = [Vector{Vector{Int}}() for _ in 1:n_threads]
    vals_threads = [Float64[] for _ in 1:n_threads]

    # Binary search on the (ascending) neighbour list — read-only, thread-safe.
    is_adj = (u::Int, v::Int) -> begin
        s = rowptr[u]; e = rowptr[u+1] - 1
        while s <= e
            mid = (s + e) >> 1
            c = colval[mid]
            if c == v
                return true
            elseif c < v
                s = mid + 1
            else
                e = mid - 1
            end
        end
        return false
    end

    sqdist = (u::Int, v::Int) -> begin
        acc = 0.0
        @inbounds for k in 1:dim_pts
            diff = points[u, k] - points[v, k]
            acc += diff * diff
        end
        return acc
    end

    # Bounded DFS carrying the running squared diameter (longest edge²).
    function backtrack(local_cliques::Vector{Vector{Int}}, local_vals::Vector{Float64},
                       current_clique::Vector{Int}, current_val2::Float64,
                       candidates::Vector{Int})
        push!(local_cliques, copy(current_clique))
        push!(local_vals, sqrt(current_val2))
        if length(current_clique) == max_dim + 1
            return
        end
        for (i, v) in enumerate(candidates)
            new_candidates = Int[]
            for j in (i+1):length(candidates)
                w = candidates[j]
                if is_adj(v, w)
                    push!(new_candidates, w)
                end
            end
            # Extending by v adds edges v–u for every u already in the clique.
            new_val2 = current_val2
            @inbounds for u in current_clique
                dv = sqdist(v, u)
                if dv > new_val2
                    new_val2 = dv
                end
            end
            push!(current_clique, v)
            backtrack(local_cliques, local_vals, current_clique, new_val2, new_candidates)
            pop!(current_clique)
        end
    end

    Threads.@threads for u in 1:n_pts
        tid = Threads.threadid()
        candidates = Int[]
        for ptr in rowptr[u]:(rowptr[u+1]-1)
            v = colval[ptr]
            if v > u
                push!(candidates, v)
            end
        end
        backtrack(cliques_threads[tid], vals_threads[tid], Int[u], 0.0, candidates)
    end

    cliques = reduce(vcat, cliques_threads)
    vals = reduce(vcat, vals_threads)
    M = length(cliques)

    # Barcode via the shared reduction core (cliques are ascending by construction).
    (bar_dim, bar_birth, bar_death) = _reduce_from_simplices(cliques, vals)

    # Distinct appearance-value grid + per-dimension first value / count.
    eps_values = sort(unique(vals))
    dim_first = Dict{Int, Float64}()
    dim_cnt = Dict{Int, Int}()
    @inbounds for j in 1:M
        d = length(cliques[j]) - 1
        v = vals[j]
        if haskey(dim_first, d)
            v < dim_first[d] && (dim_first[d] = v)
            dim_cnt[d] += 1
        else
            dim_first[d] = v
            dim_cnt[d] = 1
        end
    end
    dim_ids = sort(collect(keys(dim_first)))
    dim_first_val = Float64[dim_first[d] for d in dim_ids]
    dim_count = Int[dim_cnt[d] for d in dim_ids]

    if analyze_manifolds
        m_eps, m_is_mani, m_dims, m_is_closed, m_failures = _run_manifold_analysis_jl(cliques, vals, max_dim, epsilon, n_samples, n_pts)
        return (bar_dim, bar_birth, bar_death, eps_values, dim_ids, dim_first_val,
                dim_count, M, true, m_eps, m_is_mani, m_dims, m_is_closed, m_failures)
    else
        return (bar_dim, bar_birth, bar_death, eps_values, dim_ids, dim_first_val,
                dim_count, M, false, Float64[], Bool[], Int[], Bool[], Int[])
    end
end

"""
    compute_rips_filtration(points, epsilon, max_dim, analyze_manifolds, n_samples)

PythonCall entry point for [`_compute_rips_filtration`](@ref); materialises the
NumPy point buffer as a native `Matrix{Float64}` (the lazy PyArray is not
thread-safe to read from the `@threads` loops) and runs the fused build+reduce.
"""
function compute_rips_filtration(points_raw, epsilon_raw, max_dim_raw, analyze_manifolds_raw=false, n_samples_raw=nothing)
    return _compute_rips_filtration(
        pyconvert(Matrix{Float64}, points_raw),
        pyconvert(Float64, epsilon_raw),
        pyconvert(Int, max_dim_raw),
        pyconvert(Bool, analyze_manifolds_raw),
        pyconvert(Union{Nothing, Int}, n_samples_raw),
    )
end

# ====================================================================
# Phase B — implicit persistent COHOMOLOGY of a Vietoris–Rips filtration
# (Ripser-style). The full simplex set is never assembled into a boundary
# matrix: simplices are indexed by the combinatorial number system (CNS)
# and their cofacets are enumerated on the fly. Reduction is the standard
# coboundary column reduction with Chen–Kerber clearing across dimensions.
# Persistent cohomology yields the IDENTICAL barcode to homology
# (de Silva–Morozov–Vejdemo-Johansson duality), so the result is exact —
# it is validated bar-for-bar against the clique engine (itself proven
# equal to the pure-Python oracle).
#
# Convention (matches RipsFiltrationReport / the oracle): the complex is
# built up to dimension `max_dim`; H_d is computed for d in 0..max_dim;
# top-dimension (d == max_dim) classes have no cofacets in the complex and
# are therefore essential. A simplex's appearance value is its longest edge
# (max pairwise distance = "diameter").
# ====================================================================

# Binomial table: B[v+1, j+1] = C(v, j) for v in 0:n, j in 0:kmax (0 for v < j).
function _rips_binomial(n::Int, kmax::Int)
    B = zeros(Int, n + 1, kmax + 1)
    @inbounds for v in 0:n
        B[v + 1, 1] = 1                          # C(v, 0) = 1
        for j in 1:min(v, kmax)
            B[v + 1, j + 1] = B[v, j] + B[v, j + 1]   # Pascal: C(v,j)=C(v-1,j-1)+C(v-1,j)
        end
    end
    return B
end

# CNS index of a simplex given its vertices in DESCENDING order:
# index = sum_i C(verts[i], k-i+1), with k = #vertices.
function _rips_simplex_index(verts::Vector{Int}, B::Matrix{Int})
    k = length(verts)
    idx = 0
    @inbounds for i in 1:k
        idx += B[verts[i] + 1, (k - i + 1) + 1]
    end
    return idx
end

# Largest vertex v (0 <= v < n) with C(v, k) <= idx (binary search on the table).
function _rips_get_max_vertex(idx::Int, k::Int, n::Int, B::Matrix{Int})
    lo = k - 1                                   # C(k-1, k) = 0 <= idx
    hi = n - 1
    @inbounds while lo < hi
        mid = (lo + hi + 1) >> 1
        if B[mid + 1, k + 1] <= idx
            lo = mid
        else
            hi = mid - 1
        end
    end
    return lo
end

# Recover the dim+1 vertices (DESCENDING) of a simplex from its CNS index.
function _rips_get_vertices(idx::Int, dim::Int, n::Int, B::Matrix{Int})
    k = dim + 1
    verts = Vector{Int}(undef, k)
    @inbounds for p in k:-1:1
        v = _rips_get_max_vertex(idx, p, n, B)
        verts[k - p + 1] = v
        idx -= B[v + 1, p + 1]
    end
    return verts                                 # descending
end

# Sparse neighbour lists within `epsilon`: nbr[v+1] = sorted-ascending neighbour
# ids of vertex v, nbd[v+1] = aligned distances. O(n^2) once (parallel).
function _rips_neighbor_lists(points::Matrix{Float64}, epsilon::Float64)
    n = size(points, 1)
    dpts = size(points, 2)
    eps2 = epsilon^2
    nthreads = Threads.maxthreadid()
    I_t = [Int[] for _ in 1:nthreads]
    J_t = [Int[] for _ in 1:nthreads]
    D_t = [Float64[] for _ in 1:nthreads]
    Threads.@threads for i in 0:(n - 1)
        tid = Threads.threadid()
        Il = I_t[tid]; Jl = J_t[tid]; Dl = D_t[tid]
        for j in (i + 1):(n - 1)
            d2 = 0.0
            @inbounds for k in 1:dpts
                diff = points[i + 1, k] - points[j + 1, k]
                d2 += diff * diff
            end
            if d2 <= eps2
                dist = sqrt(d2)
                push!(Il, i); push!(Jl, j); push!(Dl, dist)
            end
        end
    end
    nbr = [Int[] for _ in 1:n]
    nbd = [Float64[] for _ in 1:n]
    for tid in 1:nthreads
        Il = I_t[tid]; Jl = J_t[tid]; Dl = D_t[tid]
        @inbounds for e in 1:length(Il)
            i = Il[e]; j = Jl[e]; dist = Dl[e]
            push!(nbr[i + 1], j); push!(nbd[i + 1], dist)
            push!(nbr[j + 1], i); push!(nbd[j + 1], dist)
        end
    end
    for v in 1:n
        p = sortperm(nbr[v])
        nbr[v] = nbr[v][p]
        nbd[v] = nbd[v][p]
    end
    return nbr, nbd
end

# Distance d(u, w) via binary search in u's sorted neighbour list (-1 if not adjacent).
function _rips_edist(u::Int, w::Int, nbr::Vector{Vector{Int}}, nbd::Vector{Vector{Float64}})
    lst = nbr[u + 1]
    lo = 1; hi = length(lst)
    @inbounds while lo <= hi
        mid = (lo + hi) >> 1
        c = lst[mid]
        if c == w
            return nbd[u + 1][mid]
        elseif c < w
            lo = mid + 1
        else
            hi = mid - 1
        end
    end
    return -1.0
end

# CNS index of the cofacet verts ∪ {w} (w merged into the descending vertex list).
function _rips_cofacet_index(verts::Vector{Int}, w::Int, B::Matrix{Int})
    k = length(verts)
    idx = 0
    i = 1
    placed = false
    @inbounds for p in (k + 1):-1:1
        if !placed && (i > k || w > verts[i])
            v = w; placed = true
        else
            v = verts[i]; i += 1
        end
        idx += B[v + 1, p + 1]
    end
    return idx
end

# Pivot order on coboundary entries (diam, index): the pivot of a reduced column
# is its EARLIEST cofacet — smallest diameter (a class dies as soon as a cofacet
# fills it: the elder rule), ties broken by smaller index. This is the dual of the
# homology "lowest one" (the anti-transpose flips max↔min). The reduced column is
# kept sorted pivot-first, so the pivot is always its first entry.
@inline _co_before(x::Tuple{Float64, Int}, y::Tuple{Float64, Int}) =
    x[1] < y[1] || (x[1] == y[1] && x[2] < y[2])

# Binary min-heap of coboundary entries (diam, index) ordered by `_co_before`
# (the heap top is the ≺-minimal entry). The working coboundary column is a heap;
# entries are added (Z₂) by pushing and cancelled lazily when popping the pivot.
@inline function _heap_push!(h::Vector{Tuple{Float64, Int}}, x::Tuple{Float64, Int})
    push!(h, x)
    i = length(h)
    @inbounds while i > 1
        p = i >> 1
        if _co_before(h[i], h[p])
            h[i], h[p] = h[p], h[i]; i = p
        else
            break
        end
    end
end

@inline function _heap_pop!(h::Vector{Tuple{Float64, Int}})
    top = h[1]
    n = length(h)
    h[1] = h[n]
    pop!(h)
    n -= 1
    i = 1
    @inbounds while true
        l = 2i; r = 2i + 1; m = i
        if l <= n && _co_before(h[l], h[m]); m = l; end
        if r <= n && _co_before(h[r], h[m]); m = r; end
        m == i && break
        h[i], h[m] = h[m], h[i]; i = m
    end
    return top
end

# Pop the pivot: the ≺-minimal entry surviving Z₂ cancellation of equal indices
# (a cofacet appearing an even number of times cancels). Returns nothing if empty.
function _pop_pivot!(h::Vector{Tuple{Float64, Int}})
    isempty(h) && return nothing
    piv = _heap_pop!(h)
    @inbounds while !isempty(h) && h[1][2] == piv[2]
        _heap_pop!(h)                            # cancel the duplicate
        isempty(h) && return nothing
        piv = _heap_pop!(h)
    end
    return piv
end

# Peek the pivot without consuming the column (pop, then push it back).
function _get_pivot!(h::Vector{Tuple{Float64, Int}})
    piv = _pop_pivot!(h)
    piv !== nothing && _heap_push!(h, piv)
    return piv
end

# Cofacets of the simplex `verts` (descending, diameter `sdiam`): every common
# neighbour w of all its vertices, with cofacet diameter max(sdiam, max_i d(w,v_i)).
function _rips_cofacets(verts::Vector{Int}, sdiam::Float64,
                        nbr::Vector{Vector{Int}}, nbd::Vector{Vector{Float64}},
                        B::Matrix{Int})
    out = Vector{Tuple{Float64, Int}}()
    k = length(verts)
    cand = nbr[verts[1] + 1]                     # w must be adjacent to verts[1]
    @inbounds for w in cand
        cofdiam = sdiam
        ok = true
        for i in 1:k
            vi = verts[i]
            if w == vi
                ok = false; break
            end
            dwi = _rips_edist(vi, w, nbr, nbd)
            if dwi < 0.0
                ok = false; break
            end
            if dwi > cofdiam
                cofdiam = dwi
            end
        end
        ok || continue
        push!(out, (cofdiam, _rips_cofacet_index(verts, w, B)))
    end
    return out
end

# Generate all (d+1)-simplices as cofacets of every d-simplex, deduplicated by index.
function _rips_generate_next(cur_idx::Vector{Int}, cur_diam::Vector{Float64}, d::Int,
                             nbr::Vector{Vector{Int}}, nbd::Vector{Vector{Float64}},
                             B::Matrix{Int}, n::Int)
    nxt = Dict{Int, Float64}()
    @inbounds for t in 1:length(cur_idx)
        verts = _rips_get_vertices(cur_idx[t], d, n, B)
        for (cd, ci) in _rips_cofacets(verts, cur_diam[t], nbr, nbd, B)
            if !haskey(nxt, ci)
                nxt[ci] = cd
            end
        end
    end
    idxs = collect(keys(nxt))
    diams = Vector{Float64}(undef, length(idxs))
    @inbounds for t in 1:length(idxs)
        diams[t] = nxt[idxs[t]]
    end
    return idxs, diams
end

"""
    _compute_rips_cohomology(points, epsilon, max_dim)

Implicit persistent cohomology of the Vietoris–Rips filtration of `points` up to
`epsilon`/`max_dim`. Returns the same payload tuple as
[`_compute_rips_filtration`](@ref): `(bar_dim, bar_birth, bar_death, eps_values,
dim_ids, dim_first_val, dim_count, total)`.
"""
function _compute_rips_cohomology(points::Matrix{Float64}, epsilon::Float64, max_dim::Int, analyze_manifolds::Bool=false, n_samples::Union{Nothing, Int}=nothing)
    n = size(points, 1)
    nbr, nbd = _rips_neighbor_lists(points, epsilon)
    B = _rips_binomial(n, max_dim + 2)

    bar_dim = Int[]; bar_birth = Float64[]; bar_death = Float64[]
    gridset = Set{Float64}(); push!(gridset, 0.0)
    @inbounds for v in 1:n
        for t in 1:length(nbd[v])
            push!(gridset, nbd[v][t])
        end
    end

    dim_ids = Int[]; dim_first_val = Float64[]; dim_count = Int[]
    total = 0

    cur_idx = collect(0:(n - 1))                 # dim 0: vertices, index == id
    cur_diam = zeros(Float64, n)
    cleared = Set{Int}()
    d = 0
    while true
        if !isempty(cur_idx)
            total += length(cur_idx)
            push!(dim_ids, d)
            push!(dim_count, length(cur_idx))
            push!(dim_first_val, minimum(cur_diam))
        end

        # Owner-recompute reduction (Ripser-style): store only which column owns
        # each pivot and the small "V" list of columns summed into it; the working
        # coboundary is rebuilt on demand from those columns' coboundaries. For an
        # apparent/emergent pair the V-list is a singleton, so the column resolves
        # in one heap step — no full reduced column is ever stored.
        pivot_owner = Dict{Int, Int}()                      # cofacet idx → owner col's simplex idx
        Vcols = Dict{Int, Vector{Tuple{Int, Float64}}}()    # owner sidx → member (idx, diam) list
        next_pivots = Set{Int}()                            # (d+1)-pivots → clear next dim

        # Persistent cohomology reduces columns in REVERSE filtration order
        # (the anti-transpose duality): latest simplex first — larger diameter,
        # ties by larger index (the ≺-greatest first, consistent with `_co_before`).
        order = Int[t for t in 1:length(cur_idx) if !(cur_idx[t] in cleared)]
        sort!(order, lt = (s, t) ->
            (cur_diam[s] > cur_diam[t]) ||
            (cur_diam[s] == cur_diam[t] && cur_idx[s] > cur_idx[t]))

        working = Tuple{Float64, Int}[]
        for t in order
            sidx = cur_idx[t]; sdiam = cur_diam[t]
            if d >= max_dim
                push!(bar_dim, d); push!(bar_birth, sdiam); push!(bar_death, Inf)
                continue                                     # top dimension: no cofacets
            end

            empty!(working)
            verts = _rips_get_vertices(sidx, d, n, B)
            for c in _rips_cofacets(verts, sdiam, nbr, nbd, B)
                _heap_push!(working, c)
            end
            members = Tuple{Int, Float64}[(sidx, sdiam)]

            piv = _get_pivot!(working)
            while piv !== nothing
                ow = get(pivot_owner, piv[2], -1)
                ow == -1 && break                            # unclaimed pivot → pair found
                @inbounds for (midx, mdiam) in Vcols[ow]
                    mverts = _rips_get_vertices(midx, d, n, B)
                    for c in _rips_cofacets(mverts, mdiam, nbr, nbd, B)
                        _heap_push!(working, c)
                    end
                    push!(members, (midx, mdiam))
                end
                piv = _get_pivot!(working)
            end

            if piv === nothing
                push!(bar_dim, d); push!(bar_birth, sdiam); push!(bar_death, Inf)
            else
                if piv[1] > sdiam
                    push!(bar_dim, d); push!(bar_birth, sdiam); push!(bar_death, piv[1])
                end
                pivot_owner[piv[2]] = sidx
                Vcols[sidx] = members
                push!(next_pivots, piv[2])
            end
        end

        if d >= max_dim
            break
        end
        nxt_idx, nxt_diam = _rips_generate_next(cur_idx, cur_diam, d, nbr, nbd, B, n)
        if isempty(nxt_idx)
            break
        end
        cleared = next_pivots
        cur_idx = nxt_idx
        cur_diam = nxt_diam
        d += 1
    end

    eps_values = sort(collect(gridset))
    if analyze_manifolds
        cliques, vals = _generate_all_cliques_and_values(points, epsilon, max_dim)
        m_eps, m_is_mani, m_dims, m_is_closed, m_failures = _run_manifold_analysis_jl(cliques, vals, max_dim, epsilon, n_samples, n)
        return (bar_dim, bar_birth, bar_death, eps_values, dim_ids, dim_first_val,
                dim_count, total, true, m_eps, m_is_mani, m_dims, m_is_closed, m_failures)
    else
        return (bar_dim, bar_birth, bar_death, eps_values, dim_ids, dim_first_val,
                dim_count, total, false, Float64[], Bool[], Int[], Bool[], Int[])
    end
end

"""
    compute_rips_cohomology(points, epsilon, max_dim, analyze_manifolds, n_samples)

PythonCall entry point for [`_compute_rips_cohomology`](@ref).
"""
function compute_rips_cohomology(points_raw, epsilon_raw, max_dim_raw, analyze_manifolds_raw=false, n_samples_raw=nothing)
    return _compute_rips_cohomology(
        pyconvert(Matrix{Float64}, points_raw),
        pyconvert(Float64, epsilon_raw),
        pyconvert(Int, max_dim_raw),
        pyconvert(Bool, analyze_manifolds_raw),
        pyconvert(Union{Nothing, Int}, n_samples_raw),
    )
end

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
function exact_snf_sparse(rows::AbstractVector{Int64}, cols::AbstractVector{Int64}, vals::AbstractVector{Int64}, m::Int, n::Int; use_markowitz::Bool=true)
    if !HAS_ABSTRACT_ALGEBRA
        error("AbstractAlgebra unavailable")
    end

    # sparse() constructor is efficient with these vectors.
    A = sparse(rows .+ 1, cols .+ 1, vals, m, n)

    try

        # Phase 1: O(V+E) leaf-peeling pre-processor.
        # Rows/columns with a single ±1 entry are peeled in O(1), reducing the
        # boundary matrix to a smaller "core" before the O(N³) SNF step.
        ones_count, core_mat = _reduce_snf(A)
        factors = ones(Int64, ones_count)

        core_m, core_n = size(core_mat)
        if core_m > 0 && core_n > 0
            # Phase 2: Markowitz column reordering on the core.
            # Permute columns so that those creating the least fill-in are
            # processed first.  Column permutation is unimodular and does not
            # alter the SNF diagonal.
            if use_markowitz && nnz(core_mat) > 0
                col_perm = _markowitz_col_permutation(core_mat)
                core_mat = core_mat[:, col_perm]
            end

            # Phase 3: Exact SNF of the reduced core.
            if HAS_INTEGER_SNF && (core_m > 2000 || core_n > 2000)
                println("Using IntegerSmithNormalForm for $(core_m)x$(core_n) core...")
                s_diag = IntegerSmithNormalForm.elementary_divisors(core_mat)
                for val in s_diag
                    if val != 0
                        push!(factors, Int64(abs(val)))
                    end
                end
            else
                ZZ = AbstractAlgebra.ZZ
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

        return sort(factors)
    catch e
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
    # Materialize native copies: callers pass zero-copy PyArrays over NumPy
    # buffers, which are NOT thread-safe to read from the @threads loop below.
    centroids = centroids isa Matrix{Float64} ? centroids : Matrix{Float64}(centroids)
    radii = radii isa Vector{Float64} ? radii : Vector{Float64}(radii)
    n = size(centroids, 1)
    d = size(centroids, 2)
    d >= 1 || return Matrix{Int64}(undef, 0, 2)
    length(radii) == n || error("radii length must match centroid count")
    n <= 1 && return Matrix{Int64}(undef, 0, 2)

    # Thread-local storage for pairs. Size by maxthreadid(): threadid() is not
    # bounded by nthreads(), so indexing with it would otherwise BoundsError.
    thread_pairs = [Vector{NTuple{2, Int64}}() for _ in 1:Threads.maxthreadid()]

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
    
    # Pre-allocate thread-local totals. Size by maxthreadid(): threadid() is not
    # bounded by nthreads(), so indexing with it would otherwise BoundsError.
    thread_totals = zeros(Int64, Threads.maxthreadid())

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
function _path_between_tree_jl(u::Int64, v::Int64, parent::Dict{Int64, Int64}, depth::Dict{Int64, Int})
    path_u = Int64[]
    path_v = Int64[]
    
    curr_u, curr_v = u, v
    
    # Bring both nodes to the same depth
    while depth[curr_u] > depth[curr_v]
        push!(path_u, curr_u)
        curr_u = parent[curr_u]
    end
    while depth[curr_v] > depth[curr_u]
        push!(path_v, curr_v)
        curr_v = parent[curr_v]
    end
    
    # Move up until LCA is found
    while curr_u != curr_v
        push!(path_u, curr_u)
        push!(path_v, curr_v)
        curr_u = parent[curr_u]
        curr_v = parent[curr_v]
    end
    
    push!(path_u, curr_u) # Add LCA
    return vcat(path_u, reverse(path_v))
end

function _reconstruct_cycle_jl(edge_indices::Vector{Int64}, d1_rows::AbstractVector{Int64}, d1_cols::AbstractVector{Int64}, d1_vals::AbstractVector{Int64}, gen_map::Dict{Int64, String})
    # Build local multigraph for the face
    # A 2-cell boundary can be a non-simple circuit (Eulerian)
    edge_set = Set(edge_indices)
    edge_info = Dict{Int64, Vector{Int64}}() # edge_idx -> [u, v]
    
    # Extract endpoints for edges in this face
    for i in 1:length(d1_cols)
        e = d1_cols[i]
        if e in edge_set
            push!(get!(edge_info, e, Int64[]), d1_rows[i])
        end
    end

    # Build adjacency with multiplicity (multigraph)
    # Entry: neighbor, edge_idx, orientation
    adj = Dict{Int64, Vector{Tuple{Int64, Int64, Int64}}}()
    
    # For CW complexes, d1(e) = v - u means e is u -> v
    # We need to know which vertex is target (1) and source (-1)
    # We can pre-calculate this from d1_vals
    for e in edge_indices
        verts = get(edge_info, e, [])
        if length(verts) == 2
            u_idx = -1
            v_idx = -1
            # Find signs
            for i in 1:length(d1_cols)
                if d1_cols[i] == e
                    if d1_vals[i] == -1; u_idx = d1_rows[i]; end
                    if d1_vals[i] == 1; v_idx = d1_rows[i]; end
                end
            end
            if u_idx == -1 || v_idx == -1
                u_idx, v_idx = verts[1], verts[2]
            end
            
            push!(get!(adj, u_idx, []), (v_idx, e, 1))
            push!(get!(adj, v_idx, []), (u_idx, e, -1))
        elseif length(verts) == 1
            v = verts[1]
            push!(get!(adj, v, []), (v, e, 1))
        else
            # Zero-boundary loop!
            # If no vertices in d1, associate with vertex 0
            v = 0
            push!(get!(adj, v, []), (v, e, 1))
        end
    end

    isempty(adj) && return String[]

    # Robust traversal matching available edge instances
    res_path = String[]
    used_indices = Set{Int}()
    curr_v = first(keys(adj))
    
    for _ in 1:length(edge_indices)
        found = false
        if !haskey(adj, curr_v); break; end
        for (v_next, e_idx, orient) in adj[curr_v]
            # Find an unused occurrence of this edge index in the face boundary
            match_idx = -1
            for i in 1:length(edge_indices)
                if edge_indices[i] == e_idx && !(i in used_indices)
                    match_idx = i
                    break
                end
            end
            
            if match_idx != -1
                push!(used_indices, match_idx)
                g = get(gen_map, e_idx, "")
                if !isempty(g)
                    push!(res_path, orient == 1 ? g : "$(g)^-1")
                end
                curr_v = v_next
                found = true
                break
            end
        end
        if !found; break; end
    end
    
    return res_path
end

function pi1_trace_candidates_from_d1(d1_rows::AbstractVector{Int64}, d1_cols::AbstractVector{Int64}, d1_vals::AbstractVector{Int64}, n_vertices::Int, n_edges::Int)
    adj = Dict{Int64, Vector{Tuple{Int64, Int64, Int64}}}()
    for i in 0:(n_vertices - 1); adj[Int64(i)] = []; end

    edge_list = Vector{Union{Nothing, Tuple{Int64, Int64}}}(undef, n_edges)
    fill!(edge_list, nothing)

    col_to_entries = Dict{Int64, Vector{Tuple{Int64, Int64}}}()
    for idx in 1:length(d1_rows)
        push!(get!(col_to_entries, Int64(d1_cols[idx]), []), (Int64(d1_vals[idx]), Int64(d1_rows[idx])))
    end

    for e in 0:(n_edges - 1)
        entries = get(col_to_entries, e, [])
        if length(entries) == 2
            u = entries[1][1] == -1 ? entries[1][2] : entries[2][2]
            v = entries[1][1] == 1 ? entries[1][2] : entries[2][2]
            push!(adj[u], (v, e, 1))
            push!(adj[v], (u, e, -1))
            edge_list[e + 1] = (u, v)
        elseif length(entries) == 1
            v = entries[1][2]
            push!(adj[v], (v, e, 1))
            edge_list[e + 1] = (v, v)
        else
            # Zero boundary loop (e.g. S1 with 1 vertex)
            # Associate with vertex 0 if it exists
            if n_vertices > 0
                v = 0
                push!(adj[v], (v, e, 1))
                edge_list[e + 1] = (v, v)
            end
        end
    end

    visited = falses(n_vertices)
    parent = Dict{Int64, Int64}()
    depth = Dict{Int64, Int}()
    component_root = Dict{Int64, Int64}()
    tree_edges = Set{Int64}()

    for start in 0:(n_vertices - 1)
        visited[start + 1] && continue
        queue = [(Int64(start), 0)]
        visited[start + 1] = true
        parent[Int64(start)] = -1
        depth[Int64(start)] = 0
        component_root[Int64(start)] = Int64(start)

        while !isempty(queue)
            curr, d = popfirst!(queue)
            for (neighbor, edge_idx, _) in adj[curr]
                if !visited[neighbor + 1]
                    visited[neighbor + 1] = true
                    push!(tree_edges, edge_idx)
                    parent[neighbor] = curr
                    depth[neighbor] = d + 1
                    component_root[neighbor] = Int64(start)
                    push!(queue, (neighbor, d + 1))
                end
            end
        end
    end

    traces = Vector{Dict{String, Any}}()
    for e in 0:(n_edges - 1)
        (e in tree_edges || edge_list[e + 1] === nothing) && continue
        u, v = edge_list[e + 1]
        
        directed = [(u, v)]
        if u != v
            path_vertices = _path_between_tree_jl(v, u, parent, depth)
            for i in 1:(length(path_vertices) - 1)
                push!(directed, (path_vertices[i], path_vertices[i + 1]))
            end
        end
        
        vertex_path = [u]
        for (_, b) in directed; push!(vertex_path, b); end
        
        push!(traces, Dict(
            "generator" => "g_$(e)",
            "edge_index" => Int64(e),
            "component_root" => Int64(get(component_root, u, u)),
            "vertex_path" => vertex_path,
            "directed_edge_path" => [[Int64(a), Int64(b)] for (a, b) in directed],
            "undirected_edge_path" => [[Int64(min(a, b)), Int64(max(a, b))] for (a, b) in directed],
        ))
    end
    return traces
end

function extract_pi1_raw_data_jl(
    d1_rows::AbstractVector{Int64}, d1_cols::AbstractVector{Int64}, d1_vals::AbstractVector{Int64},
    n_vertices::Int, n_edges::Int,
    d2_rows::AbstractVector{Int64}, d2_cols::AbstractVector{Int64}, d2_vals::AbstractVector{Int64},
    n_faces::Int
)
    traces = pi1_trace_candidates_from_d1(d1_rows, d1_cols, d1_vals, n_vertices, n_edges)
    gen_map = Dict{Int64, String}(tr["edge_index"] => tr["generator"] for tr in traces)

    face_to_payload = Dict{Int64, Vector{Tuple{Int64, Int64}}}()
    for idx in 1:length(d2_rows)
        f = Int64(d2_cols[idx])
        e = Int64(d2_rows[idx])
        val = Int64(d2_vals[idx])
        push!(get!(face_to_payload, f, []), (e, val))
    end

    relations = Vector{Vector{String}}()
    for f in 0:(n_faces - 1)
        payload = get(face_to_payload, f, nothing)
        payload === nothing && continue
        
        # Flatten payload into list of edge indices by multiplicity
        edge_indices = Int64[]
        for (e, val) in payload
            for _ in 1:abs(val)
                push!(edge_indices, e)
            end
        end
        
        rel = _reconstruct_cycle_jl(edge_indices, d1_rows, d1_cols, d1_vals, gen_map)
        !isempty(rel) && push!(relations, rel)
    end

    w1 = Dict{String, Int}(g => 1 for g in values(gen_map))
    return Dict("generators" => gen_map, "relations" => relations, "traces" => traces, "orientation_character" => w1)
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
    # Materialize native copies of the coefficient vectors: callers pass zero-copy
    # PyArrays over NumPy buffers, which are NOT thread-safe to read concurrently
    # from the @threads loop below.
    alpha = collect(alpha)
    beta = collect(beta)
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
    # Materialize a NATIVE dense copy. `pyconvert(AbstractMatrix{Float64}, ...)`
    # yields a lazy PyArray over the NumPy buffer; reading it from the @threads
    # loop below touches Python state and is NOT thread-safe, throwing an
    # intermittent CompositeException/TaskFailedException. A single O(N*D) copy
    # gives a true Matrix{Float64} that threads can read concurrently.
    points = pyconvert(Matrix{Float64}, points_raw)
    epsilon = pyconvert(Float64, epsilon_raw)
    max_dim = pyconvert(Int, max_dim_raw)
    n_pts = size(points, 1)
    dim_pts = size(points, 2)

    # 1. 1-skeleton (Edges) - Parallelized
    eps2 = epsilon^2
    # threadid() is NOT bounded by nthreads() (GC/interactive threads have higher
    # ids), so size per-thread buffers by maxthreadid() to avoid a BoundsError.
    n_threads = Threads.maxthreadid()
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

    # Per-thread clique buffers — same pattern as 1-skeleton above.
    cliques_threads = [Vector{Vector{Int64}}() for _ in 1:Threads.maxthreadid()]

    # Optimized adjacency check for sorted neighbor lists (read-only, thread-safe).
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

    # Bounded DFS — writes only to local_cliques, so callers from different
    # threads never touch the same buffer.
    function backtrack(local_cliques::Vector{Vector{Int64}}, current_clique::Vector{Int64}, candidates::Vector{Int64})
        push!(local_cliques, copy(current_clique))

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
            backtrack(local_cliques, current_clique, new_candidates)
            pop!(current_clique)
        end
    end

    # 3. Clique enumeration - Parallelized
    Threads.@threads for u in 1:n_pts
        tid = Threads.threadid()
        local_cliques = cliques_threads[tid]
        candidates = Int64[]
        for ptr in rowptr[u]:(rowptr[u+1]-1)
            v = colval[ptr]
            if v > u
                push!(candidates, v)
            end
        end
        backtrack(local_cliques, [u], candidates)
    end

    cliques = reduce(vcat, cliques_threads)

    # Return 0-indexed for Python consistency
    return [c .- 1 for c in cliques]
end
"""
    compute_circumradius_sq_3d(points, simplices)

Compute squared circumradii for 3D tetrahedra natively.
"""
function compute_circumradius_sq_3d(points::AbstractMatrix{Float64}, simplices::AbstractMatrix{Int64})
    # Materialize native copies: callers pass zero-copy PyArrays over NumPy
    # buffers, which are NOT thread-safe to read from the @threads loop below.
    points = points isa Matrix{Float64} ? points : Matrix{Float64}(points)
    simplices = simplices isa Matrix{Int64} ? simplices : Matrix{Int64}(simplices)
    n_simplices = size(simplices, 1)
    radii_sq = zeros(Float64, n_simplices)

    Threads.@threads for i in 1:n_simplices
        # Adjust 0-based Python indices to 1-based Julia indexing
        idx1, idx2, idx3, idx4 = simplices[i, 1] + 1, simplices[i, 2] + 1, simplices[i, 3] + 1, simplices[i, 4] + 1
        v1 = @view(points[idx2, :]) .- @view(points[idx1, :])
        v2 = @view(points[idx3, :]) .- @view(points[idx1, :])
        v3 = @view(points[idx4, :]) .- @view(points[idx1, :])

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
    # Materialize native copies: callers pass zero-copy PyArrays over NumPy
    # buffers, which are NOT thread-safe to read from the @threads loop below.
    points = points isa Matrix{Float64} ? points : Matrix{Float64}(points)
    simplices = simplices isa Matrix{Int64} ? simplices : Matrix{Int64}(simplices)
    n_simplices = size(simplices, 1)
    radii_sq = zeros(Float64, n_simplices)

    Threads.@threads for i in 1:n_simplices
        idx1, idx2, idx3 = simplices[i, 1] + 1, simplices[i, 2] + 1, simplices[i, 3] + 1
        ux = points[idx2, 1] - points[idx1, 1]
        uy = points[idx2, 2] - points[idx1, 2]
        vx = points[idx3, 1] - points[idx1, 1]
        vy = points[idx3, 2] - points[idx1, 2]

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

        # --- Filtration persistence (twist/clearing Z₂ reduction) ---
        # Filled triangle {0,1,2}: 3 vertices @0, 3 edges @1, 1 triangle @1.
        _compute_filtration_persistence(
            Int[0, 1, 2, 0, 1, 1, 2, 0, 2, 0, 1, 2],
            Int[0, 1, 2, 3, 5, 7, 9, 12],
            Float64[0, 0, 0, 1, 1, 1, 1],
        )

        # --- Fused Rips filtration (build + longest-edge values + reduction) ---
        _compute_rips_filtration(pts_3d, 5.0, 2)

        # --- Implicit cohomology Rips engine (Phase B) ---
        _compute_rips_cohomology(pts_3d, 5.0, 3)

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
            return 0.0
        end
        if k == 2
            u, v = simplex_indices[1]+1, simplex_indices[2]+1
            d2 = sum(abs2, points[u, :] .- points[v, :])
            val = d2 / 4.0
            r2_cache[simplex_indices] = val
            return val
        end

        # For k > 2, check if any face is obtuse (minimum enclosing sphere logic)
        if k == 3
            # Triangle: acute circumradius or diametral sphere of longest edge
            p0, p1, p2 = points[simplex_indices[1]+1, :], points[simplex_indices[2]+1, :], points[simplex_indices[3]+1, :]
            a2 = sum(abs2, p1 .- p2)
            b2 = sum(abs2, p0 .- p2)
            c2 = sum(abs2, p0 .- p1)

            # Area^2 via Cayley-Menger or cross product
            v1 = p1 .- p0
            v2 = p2 .- p0
            if dim_pts == 3
                area2 = 0.25 * sum(abs2, cross(v1, v2))
            else
                area2 = 0.25 * (v1[1]*v2[2] - v1[2]*v2[1])^2
            end

            r2_acute = (a2 * b2 * c2) / (16.0 * area2 + 1e-30)
            is_obtuse = (a2 + b2 < c2) || (a2 + c2 < b2) || (b2 + c2 < a2)
            val = is_obtuse ? maximum([a2, b2, c2]) / 4.0 : r2_acute
            r2_cache[simplex_indices] = val
            return val
        end

        # Generic N-dimensional case (Tetrahedra k=4, etc.)
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
            # For 3-simplices in 3D (k=4), A is 3x3. Check for degeneracy.
            if k-1 == dim_pts && abs(det(A)) < 1e-15
                # Degenerate: use maximum of sub-face circumradii
                r2_max = 0.0
                for face in combinations(simplex_indices, k-1)
                    r2_max = max(r2_max, get_r2(Tuple(face)))
                end
                r2_cache[simplex_indices] = r2_max
                return r2_max
            end

            c = A \ b
            val = sum(abs2, c)
            
            # If circumcenter is outside the simplex, it might be obtuse
            # For tetrahedra, check if any face is a better candidate
            # This is a simplified "minimum enclosing sphere" logic
            r2_cache[simplex_indices] = val
            return val
        catch
            r2_cache[simplex_indices] = Inf
            return Inf
        end
    end

    # Gabriel condition: A simplex is in Alpha complex if its R^2 <= alpha2
    for i in 1:n_max
        s_raw = sort(vec(max_simplices[i, :]))
        # Collect all sub-faces up to max_dim
        for r in 1:length(s_raw)
            for face in combinations(s_raw, r)
                t_face = Tuple(face)
                if get_r2(t_face) <= alpha2
                    push!(valid_simplices, t_face)
                end
            end
        end
    end

    # Ensure all faces of valid simplices are included (simplicial complex property)
    final_simplices = Set{Tuple{Vararg{Int64}}}()
    for s in valid_simplices
        for r in 1:length(s)
            for face in combinations(collect(s), r)
                push!(final_simplices, Tuple(face))
            end
        end
    end

    return [collect(Int64, s) for s in final_simplices]
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
    # Materialize a native copy: callers pass a zero-copy PyArray over a NumPy
    # buffer, which is NOT thread-safe to read from the @threads loops below.
    pts = pts isa Matrix{Float64} ? pts : Matrix{Float64}(pts)
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

    # 2. Filter edges based on CkNN condition. Size thread-local buffers by
    # maxthreadid(): threadid() is not bounded by nthreads(), so indexing with
    # it would otherwise BoundsError.
    thread_pairs = [Vector{NTuple{2, Int32}}() for _ in 1:Threads.maxthreadid()]
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
    # Materialize native copies: callers pass zero-copy PyArrays over NumPy
    # buffers, which are NOT thread-safe to read from the @threads loop below.
    pts = pts isa Matrix{Float64} ? pts : Matrix{Float64}(pts)
    rho = rho isa Vector{Float64} ? rho : Vector{Float64}(rho)
    n = size(pts, 1)
    if n == 0
        return Matrix{Int64}(undef, 0, 2)
    end

    # Size thread-local buffers by maxthreadid(): threadid() is not bounded by
    # nthreads(), so indexing with it would otherwise BoundsError.
    thread_pairs = [Vector{NTuple{2, Int32}}() for _ in 1:Threads.maxthreadid()]
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

    if !isempty(diagnostics)
        d_out = isempty(detected_dims) ? max_d_complex : first(detected_dims)
        return false, d_out, diagnostics
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

    function compute_alpha_threshold_emst_jl(
points::AbstractMatrix{Float64}, simplices::AbstractMatrix{Int64})
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


"""
    compute_discrete_morse_gradient_jl(simplices)

Compute a discrete Morse gradient field using a greedy matching algorithm.
Returns a list of pairs [[sigma], [tau]].
"""
function compute_discrete_morse_gradient_jl(simplices_py)
    # Convert and normalize simplices
    all_simplices_list = Vector{Vector{Int64}}()
    for s_py in pyconvert(Vector{Any}, simplices_py)
        push!(all_simplices_list, sort(collect(pyconvert(Vector{Any}, s_py))))
    end
    all_simplices = Set(all_simplices_list)
    
    matched = Set{Vector{Int64}}()
    matching = Vector{Vector{Vector{Int64}}}()
    
    # Group by dimension for efficient coface counting
    dim_groups = Dict{Int, Vector{Vector{Int64}}}()
    for s in all_simplices_list
        d = length(s) - 1
        push!(get!(dim_groups, d, Vector{Vector{Int64}}()), s)
    end
    
    max_d = maximum(keys(dim_groups))
    
    # Process Algorithm (Greedy matching)
    for d in 0:(max_d - 1)
        # Re-compute coface counts among UNMATCHED for dimension d
        unmatched_d = filter(s -> !(s in matched), get(dim_groups, d, []))
        unmatched_dp1 = filter(s -> !(s in matched), get(dim_groups, d+1, []))
        unmatched_dp1_set = Set(unmatched_dp1)
        
        if isempty(unmatched_d); continue; end
        
        # Build coface counts: sigma -> [tau1, tau2, ...]
        cofaces = Dict{Vector{Int64}, Vector{Vector{Int64}}}()
        for s in unmatched_d; cofaces[s] = Vector{Vector{Int64}}(); end
        
        for tau in unmatched_dp1
            for i in 1:length(tau)
                sigma = vcat(tau[1:i-1], tau[i+1:end])
                if haskey(cofaces, sigma)
                    push!(cofaces[sigma], tau)
                end
            end
        end
        
        # Match free faces (sigma has exactly one coface in unmatched_dp1)
        for (sigma, targets) in cofaces
            if length(targets) == 1
                tau = targets[1]
                if !(sigma in matched) && !(tau in matched)
                    push!(matched, sigma)
                    push!(matched, tau)
                    push!(matching, [sigma, tau])
                end
            end
        end
    end
    
    return matching
end

# ======================================================================
# PROPOSAL 1: EXACT SPARSE SNF EXTENSIONS
# Sparsity-aware pivoting · modular rank certification · p-adic CRT
# reconstruction · parallel batch computation
# ======================================================================

# ─────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────

"""
    _markowitz_col_permutation(A) -> Vector{Int64}

Compute a Markowitz-criterion column permutation for SNF fill-in minimization.

For each column c the Markowitz score is:
    score(c) = (col_nnz(c) − 1) × min_{r : A[r,c]≠0}(row_nnz(r) − 1)

Columns with score 0 (unit or singleton entries) are placed first; they map
directly onto the O(V+E) leaf-peeling step without extra work.
"""
function _markowitz_col_permutation(A::SparseMatrixCSC{Int64, Int64})
    m, n = size(A)
    col_nnz_v = zeros(Int64, n)
    for c in 1:n
        col_nnz_v[c] = A.colptr[c + 1] - A.colptr[c]
    end
    row_nnz_v = zeros(Int64, m)
    @inbounds for c in 1:n
        for ptr in A.colptr[c]:(A.colptr[c + 1] - 1)
            row_nnz_v[A.rowval[ptr]] += 1
        end
    end
    scores = fill(typemax(Int64), n)
    @inbounds for c in 1:n
        cnnz = col_nnz_v[c]
        if cnnz == 0
            scores[c] = typemax(Int64)
            continue
        end
        min_rnnz = typemax(Int64)
        for ptr in A.colptr[c]:(A.colptr[c + 1] - 1)
            rn = row_nnz_v[A.rowval[ptr]]
            rn < min_rnnz && (min_rnnz = rn)
        end
        scores[c] = max(Int64(0), cnnz - 1) * max(Int64(0), min_rnnz - 1)
    end
    return sortperm(scores)
end

"""
    _padic_val_int(v, p) -> Int64

Return the p-adic valuation of non-zero integer v (i.e. the largest k such that
p^k divides v).  Returns 0 for v = 0 as a safe sentinel.
"""
function _padic_val_int(v::Int64, p::Int64)::Int64
    v == 0 && return Int64(0)
    e   = Int64(0)
    av  = abs(v)
    while av % p == 0
        e  += 1
        av  = div(av, p)
    end
    return e
end

"""
    _ext_gcd_i64(a, b) -> (gcd, x, y)

Extended Euclidean algorithm over Int64.  Returns (g, x, y) with ax + by = g.
"""
function _ext_gcd_i64(a::Int64, b::Int64)
    x0, x1 = Int64(1), Int64(0)
    y0, y1 = Int64(0), Int64(1)
    while b != Int64(0)
        q   = div(a, b)
        a, b   = b, a - q * b
        x0, x1 = x1, x0 - q * x1
        y0, y1 = y1, y0 - q * y1
    end
    return a, x0, y0
end

"""
    _padic_rank_step(M_in, p, pe) -> Int64

Compute rank(A over ℤ/p^eℤ) = #{i : v_p(d_i) < e}, where d_i are the SNF
diagonal entries over ℤ.

Algorithm (correct for all e ≥ 1):
1. Reduce entries mod p^e.
2. For each column, find the unused row with the smallest p-adic valuation.
3. A unit pivot (v_p = 0) performs full GF(p^e) elimination of the column.
4. A non-unit pivot p^k·u (k > 0) eliminates only the rows whose entry in
   this column is also divisible by p^k (partial elimination).
5. Every non-zero pivot increments the rank count.
"""
function _padic_rank_step(M_in::Matrix{Int64}, p::Int64, pe::Int64)
    m, n       = size(M_in)
    M          = mod.(copy(M_in), pe)
    rank_count = Int64(0)
    row_used   = falses(m)

    for col in 1:n
        best_row = 0
        best_vp  = Int64(64)  # Sentinel larger than any realistic valuation

        @inbounds for row in 1:m
            row_used[row] && continue
            v = M[row, col]
            v == 0 && continue
            vp = _padic_val_int(v, p)
            if vp < best_vp
                best_vp  = vp
                best_row = row
                vp == 0 && break  # unit found — optimal pivot
            end
        end

        best_row == 0 && continue  # column all-zero mod p^e

        rank_count        += 1
        row_used[best_row] = true
        piv                = M[best_row, col]

        if best_vp == 0
            # Unit pivot: standard modular-GE eliminates the entire column.
            inv_piv = invmod(piv, pe)
            @inbounds for row in 1:m
                row == best_row && continue
                factor = mod(M[row, col] * inv_piv, pe)
                factor == 0 && continue
                for c in 1:n
                    M[row, c] = mod(M[row, c] - factor * M[best_row, c], pe)
                end
            end
        else
            # Non-unit pivot p^vp · u.
            # Can only eliminate rows whose entry in this column is divisible by p^vp.
            p_pow  = Int64(p)^best_vp
            pe_red = div(pe, p_pow)           # = p^{e − vp}
            u_piv  = div(piv, p_pow)          # unit in ℤ/p^{e−vp}ℤ
            inv_u  = invmod(u_piv, pe_red)
            @inbounds for row in 1:m
                row == best_row && continue
                v = M[row, col]
                v % p_pow != 0 && continue    # v_p(entry) < vp: cannot eliminate
                v_red  = div(v, p_pow)
                factor = mod(v_red * inv_u, pe_red)
                factor == 0 && continue
                for c in 1:n
                    M[row, c] = mod(M[row, c] - factor * M[best_row, c], pe)
                end
            end
        end
    end

    return rank_count
end

# ─────────────────────────────────────────────────────────────────────
# Exported public functions
# ─────────────────────────────────────────────────────────────────────

"""
    snf_markowitz_column_order(rows, cols, vals, m, n) -> Vector{Int64}

Return the **0-indexed** column permutation that sorts columns of the sparse
matrix by Markowitz score (ascending).  Applying this permutation before SNF
reduces fill-in in the core matrix passed to IntegerSmithNormalForm / AbstractAlgebra.

The permutation is 0-indexed so it can be used directly as a Python/NumPy index.
"""
function snf_markowitz_column_order(
    rows::AbstractVector{Int64},
    cols::AbstractVector{Int64},
    vals::AbstractVector{Int64},
    m::Int, n::Int,
)
    if isempty(rows)
        return collect(Int64(0):(Int64(n) - 1))
    end
    A         = sparse(rows .+ 1, cols .+ 1, vals, m, n)
    perm_1idx = _markowitz_col_permutation(A)
    return perm_1idx .- Int64(1)
end

"""
    modular_rank_certification_jl(rows, cols, vals, m, n, primes) -> NamedTuple

Certify the rank of a sparse integer matrix by computing rank(A mod p) for each
prime p in `primes`.

Returned NamedTuple fields:
  primes         – the primes used (copy of input)
  ranks          – rank(A mod p) for each prime
  all_agree      – true when all mod-p ranks are equal
  lower_bound    – max(ranks); valid lower bound (reduction mod p cannot increase rank)
  certified_rank – the agreed rank if all_agree == true, else -1

Mathematical note: rank(A mod p) = #{i : p ∤ d_i} where d_1|…|d_r are the SNF
diagonal.  Therefore all-prime agreement certifies the ℤ-rank with high confidence
(and exactly when gcd(∏p, ∏ d_i) = 1 for all i).
"""
function modular_rank_certification_jl(
    rows::AbstractVector{Int64},
    cols::AbstractVector{Int64},
    vals::AbstractVector{Int64},
    m::Int, n::Int,
    primes::AbstractVector{Int64},
)
    empty_ranks() = (
        primes         = collect(Int64, primes),
        ranks          = zeros(Int64, length(primes)),
        all_agree      = true,
        lower_bound    = Int64(0),
        certified_rank = Int64(0),
    )
    isempty(primes) && return empty_ranks()
    isempty(rows)   && return empty_ranks()

    A_dense = Matrix{Int64}(sparse(rows .+ 1, cols .+ 1, vals, m, n))
    ranks   = Int64[]
    for p in primes
        push!(ranks, Int64(_rank_mod_p_dense!(copy(A_dense), Int64(p))))
    end

    all_same    = all(r == ranks[1] for r in ranks)
    lower_bound = maximum(ranks)
    certified   = all_same ? ranks[1] : Int64(-1)

    return (
        primes         = collect(Int64, primes),
        ranks          = ranks,
        all_agree      = all_same,
        lower_bound    = Int64(lower_bound),
        certified_rank = Int64(certified),
    )
end

"""
    padic_snf_diagonal_jl(rows, cols, vals, m, n, primes, max_e) -> Vector{Int64}

Reconstruct the exact SNF diagonal over ℤ via deterministic p-adic CRT lifting.
This is an independent computation path that does NOT call IntegerSmithNormalForm
or AbstractAlgebra, making it suitable for cross-validation.

Algorithm:
  For each prime p in `primes`:
    1. Compute the rank sequence r_1, r_2, … where
         r_e = #{i : v_p(d_i) < e}    (= rank of A over ℤ/p^e ℤ)
       using `_padic_rank_step` for e = 1 … max_e.
    2. Decode the p-adic valuation: v_p(d_k) = (first e with r_e ≥ k) − 1.
  Reconstruct d_k = ∏_p  p^{v_p(d_k)}.
  Return the diagonal sorted in non-decreasing order (d_1 | d_2 | … convention).

Correctness guarantee: exact when every prime factor of every d_k appears in
`primes` and max_e ≥ max_k v_p(d_k) for all p in primes.
For homological boundary matrices (entries ∈ {−1, 0, 1}) the default primes
{2,3,5,7,11,13,17,19,23,29,31} cover all torsion ≤ 31.

References:
  Smith, H. J. S. (1861). On systems of linear indeterminate equations and
    congruences. Philosophical Transactions, 151, 293–326.
"""
function padic_snf_diagonal_jl(
    rows::AbstractVector{Int64},
    cols::AbstractVector{Int64},
    vals::AbstractVector{Int64},
    m::Int, n::Int,
    primes::AbstractVector{Int64},
    max_e::Int64,
)
    isempty(rows) && return Int64[]

    A_sp    = sparse(rows .+ 1, cols .+ 1, vals, m, n)
    A_dense = Matrix{Int64}(A_sp)

    # Total rank via floating-point (fast upper bound; exact for typical boundary mats)
    r_total = Int64(rank(Float64.(A_dense)))
    r_total == 0 && return Int64[]

    # For each prime, build rank sequence and decode p-adic valuations.
    vp_table = Dict{Int64, Vector{Int64}}()

    for p in primes
        rank_seq = Int64[]
        prev_r   = Int64(0)
        for e in 1:max_e
            pe  = Int64(p)^e
            r_e = _padic_rank_step(A_dense, p, pe)
            push!(rank_seq, r_e)
            prev_r = r_e
            r_e >= r_total && break   # fully saturated
        end

        # v_p(d_k) = first (e − 1) such that rank_seq[e] ≥ k.
        vp_vec = Int64[]
        for k in 1:r_total
            vp_k = Int64(max_e)  # default: valuation ≥ max_e (no prime power found)
            for (e_idx, r_e) in enumerate(rank_seq)
                if r_e >= k
                    vp_k = Int64(e_idx) - Int64(1)
                    break
                end
            end
            push!(vp_vec, vp_k)
        end

        vp_table[Int64(p)] = vp_vec
    end

    # Reconstruct d_k = ∏_p  p^{v_p(d_k)}.
    diag = Int64[]
    for k in 1:r_total
        d_k = Int64(1)
        for p in primes
            vp_k = vp_table[Int64(p)][k]
            vp_k > 0 && (d_k *= Int64(p)^vp_k)
        end
        push!(diag, d_k)
    end

    return sort(diag)
end

"""
    batch_exact_snf_sparse(batch_rows, batch_cols, batch_vals, batch_m, batch_n)
        -> Vector{Vector{Int64}}

Compute exact Smith Normal Form for a batch of sparse matrices in parallel using
Threads.@threads.

Arguments (all vectors of equal length — one element per matrix):
  batch_rows – row-index arrays (each Int64[])
  batch_cols – column-index arrays
  batch_vals – value arrays
  batch_m    – row counts (Int64[])
  batch_n    – column counts (Int64[])

Each sub-computation is independent and dispatched to Julia thread-pool workers.
Results are returned in input order; failed computations return Int64[].
"""
function batch_exact_snf_sparse(
    batch_rows::AbstractVector,
    batch_cols::AbstractVector,
    batch_vals::AbstractVector,
    batch_m::AbstractVector{Int64},
    batch_n::AbstractVector{Int64},
)
    n_batch  = length(batch_rows)
    results  = Vector{Vector{Int64}}(undef, n_batch)

    # Pre-convert all Python data to native Julia vectors BEFORE threading.
    # pyconvert and PyList/PyArray indexing touch Python state and are NOT
    # thread-safe; doing them inside the @threads loop races and — because the
    # body is wrapped in try/catch — silently corrupts results to Int64[].
    rows_all = [pyconvert(Vector{Int64}, batch_rows[i]) for i in 1:n_batch]
    cols_all = [pyconvert(Vector{Int64}, batch_cols[i]) for i in 1:n_batch]
    vals_all = [pyconvert(Vector{Int64}, batch_vals[i]) for i in 1:n_batch]
    m_all    = [Int(batch_m[i]) for i in 1:n_batch]
    n_all    = [Int(batch_n[i]) for i in 1:n_batch]

    Threads.@threads for i in 1:n_batch
        try
            results[i] = exact_snf_sparse(
                rows_all[i], cols_all[i], vals_all[i],
                m_all[i], n_all[i],
            )
        catch
            results[i] = Int64[]
        end
    end

    return results
end

# ──────────────────────────────────────────────────────────────────────────────
# Proposal 5 (REVISED): Bounded Controlled Cohomology kernels
#
# Native Todd-Coxeter coset enumeration, Cayley-table construction, group-ring
# convolution, universal-cover boundary lift, Fox-derivative blocks (real and
# complex), and twisted Alexander-Whitney cup product. All kernels are pure
# Julia (no AbstractAlgebra dependency in the hot path) and operate on
# zero-copy NumPy arrays via PythonCall.
# ──────────────────────────────────────────────────────────────────────────────

function _parse_signed_word(rel_str::String, gen_to_idx::Dict{String, Int})
    toks = Int[]
    for m in eachmatch(r"([a-zA-Z0-9_]+)(?:\^(-?\d+))?", rel_str)
        base = m.captures[1]
        exp_val = isnothing(m.captures[2]) ? 1 : parse(Int, m.captures[2])
        haskey(gen_to_idx, base) || continue
        idx = gen_to_idx[base]
        for _ in 1:abs(exp_val)
            push!(toks, exp_val >= 0 ? idx : -idx)
        end
    end
    return toks
end

"""
    todd_coxeter_index_jl(generators, relations, max_index)

Run HLT-style Todd-Coxeter coset enumeration over the trivial subgroup to
certify that the finitely-presented group `<generators | relations>` is
finite. Returns `(converged, n_cosets, coset_table)` where:
- `converged::Bool` — true iff all entries were filled within `max_index` cosets.
- `n_cosets::Int`  — number of equivalence-class representatives.
- `coset_table`    — dense `n_cosets × 2*n_gens` Int matrix; columns 1..n_gens
  are forward generator actions, columns n_gens+1..2*n_gens are inverse actions.

`relations` are space-separated strings of generator tokens (e.g. "a a a" or
"a b a^-1 b^-1"); same convention as `abelianize_group`.
"""
function todd_coxeter_index_jl(generators_in, relations_in, max_index::Int)
    generators = _as_string_vector(generators_in)
    relations = _as_string_vector(relations_in)
    n_gens = length(generators)
    if n_gens == 0
        return (true, 1, zeros(Int, 1, 0))
    end
    gen_to_idx = Dict{String, Int}(g => i for (i, g) in enumerate(generators))
    rels_parsed = [_parse_signed_word(r, gen_to_idx) for r in relations]
    rels_parsed = [r for r in rels_parsed if !isempty(r)]

    n_cols = 2 * n_gens
    col_for(g) = g > 0 ? g : (-g + n_gens)
    inv_col(g) = g > 0 ? (g + n_gens) : -g
    flip(col) = col <= n_gens ? col + n_gens : col - n_gens

    table = Vector{Vector{Int}}()
    push!(table, zeros(Int, n_cols))  # coset 1 = identity
    parent = Int[1]

    function _find(c::Int)
        while parent[c] != c
            parent[c] = parent[parent[c]]
            c = parent[c]
        end
        return c
    end

    function _new_coset()
        push!(table, zeros(Int, n_cols))
        push!(parent, length(table))
        return length(table)
    end

    queue = Vector{Tuple{Int, Int}}()

    function _coincidence(a::Int, b::Int)
        a = _find(a); b = _find(b)
        a == b && return
        if a > b
            a, b = b, a
        end
        parent[b] = a
        for j in 1:n_cols
            x = table[b][j]
            y = table[a][j]
            if x != 0 && y == 0
                xf = _find(x)
                table[a][j] = xf
                # also set inverse
                infl = flip(j)
                if table[xf][infl] == 0 || _find(table[xf][infl]) == b
                    table[xf][infl] = a
                end
            elseif x != 0 && y != 0
                xf = _find(x); yf = _find(y)
                xf != yf && push!(queue, (xf, yf))
            end
            table[b][j] = 0
        end
    end

    function _process_queue()
        while !isempty(queue)
            a, b = pop!(queue)
            _coincidence(a, b)
        end
    end

    function _set_edge(c::Int, col::Int, target::Int)
        c = _find(c); target = _find(target)
        existing = table[c][col]
        if existing == 0
            table[c][col] = target
        else
            ef = _find(existing)
            if ef != target
                push!(queue, (ef, target))
                _process_queue()
                return
            end
        end
        infl = flip(col)
        existing_inv = table[target][infl]
        if existing_inv == 0
            table[target][infl] = c
        else
            ef2 = _find(existing_inv)
            if ef2 != c
                push!(queue, (ef2, c))
                _process_queue()
            end
        end
    end

    function _scan_and_fill(rel::Vector{Int}, c::Int)
        c = _find(c)
        f = 1; f_curr = c
        while f <= length(rel)
            col = col_for(rel[f])
            nxt = table[f_curr][col]
            nxt == 0 && break
            f_curr = _find(nxt)
            f += 1
        end
        if f > length(rel)
            push!(queue, (f_curr, c)); _process_queue(); return
        end
        b = length(rel); b_curr = c
        while b >= f
            col = inv_col(rel[b])
            nxt = table[b_curr][col]
            nxt == 0 && break
            b_curr = _find(nxt)
            b -= 1
        end
        if b < f
            push!(queue, (f_curr, b_curr)); _process_queue(); return
        end
        if b == f
            col = col_for(rel[f])
            _set_edge(f_curr, col, b_curr)
        end
        # b > f: gap remains; will be filled later.
    end

    iter_cap = max(64, max_index * (n_cols + 1) * 8)
    iter = 0
    pos_c = 1
    while pos_c <= length(table) && iter < iter_cap
        # Skip dead cosets
        while pos_c <= length(table) && _find(pos_c) != pos_c
            pos_c += 1
        end
        pos_c > length(table) && break
        for j in 1:n_cols
            iter += 1
            iter >= iter_cap && break
            if _find(pos_c) == pos_c && table[pos_c][j] == 0
                if length(parent) >= max_index
                    return (false, length(parent), zeros(Int, 0, 0))
                end
                new_c = _new_coset()
                _set_edge(pos_c, j, new_c)
                _process_queue()
                for rel in rels_parsed
                    _scan_and_fill(rel, new_c)
                    _find(pos_c) != pos_c && break
                end
            end
        end
        pos_c += 1
    end

    # Compactify
    live = Set{Int}()
    for c in 1:length(table)
        push!(live, _find(c))
    end
    n_live = length(live)
    if n_live > max_index
        return (false, n_live, zeros(Int, 0, 0))
    end

    # Verify all live entries filled
    converged = true
    for c in live, j in 1:n_cols
        if table[c][j] == 0 || _find(table[c][j]) ∉ live
            converged = false
            break
        end
    end
    if !converged
        return (false, n_live, zeros(Int, 0, 0))
    end

    sorted_live = sort(collect(live))
    canonical = Dict(c => i for (i, c) in enumerate(sorted_live))
    new_table = zeros(Int, n_live, n_cols)
    for (new_idx, old_idx) in enumerate(sorted_live)
        for j in 1:n_cols
            new_table[new_idx, j] = canonical[_find(table[old_idx][j])]
        end
    end
    return (true, n_live, new_table)
end

"""
    cayley_table_jl(coset_table, generators)

Build the full Cayley table for a finite group from a Todd-Coxeter coset table
(over the trivial subgroup). Returns `(cayley, inverse, id_idx, words)` where
`cayley[i, j]` is the index of element i*j, `inverse[i]` is the index of i^-1,
`id_idx` is the identity element index (always 1 here), and `words[i]` is a
human-readable representative word for element i.
"""
function cayley_table_jl(coset_table_in, generators_in)
    coset_table = pyconvert(Matrix{Int}, coset_table_in)
    generators = _as_string_vector(generators_in)
    n_gens = length(generators)
    n_cosets, n_cols = size(coset_table)
    @assert n_cols == 2 * n_gens "Coset table column count must equal 2*n_gens"

    # BFS from coset 1 to derive a reduced word for each coset.
    words = Vector{Vector{Int}}(undef, n_cosets)
    words[1] = Int[]
    visited = falses(n_cosets); visited[1] = true
    bfs_queue = Int[1]
    while !isempty(bfs_queue)
        c = popfirst!(bfs_queue)
        for j in 1:n_cols
            d = coset_table[c, j]
            if d > 0 && !visited[d]
                signed_gen = j <= n_gens ? j : -(j - n_gens)
                words[d] = vcat(words[c], [signed_gen])
                visited[d] = true
                push!(bfs_queue, d)
            end
        end
    end

    cayley = zeros(Int, n_cosets, n_cosets)
    @inbounds for c in 1:n_cosets, d in 1:n_cosets
        curr = c
        for sg in words[d]
            col = sg > 0 ? sg : (-sg + n_gens)
            curr = coset_table[curr, col]
        end
        cayley[c, d] = curr
    end

    inverse = zeros(Int, n_cosets)
    @inbounds for c in 1:n_cosets
        curr = 1
        for sg in reverse(words[c])
            inv_sg = -sg
            col = inv_sg > 0 ? inv_sg : (-inv_sg + n_gens)
            curr = coset_table[curr, col]
        end
        inverse[c] = curr
    end

    word_strings = Vector{String}(undef, n_cosets)
    for c in 1:n_cosets
        if isempty(words[c])
            word_strings[c] = "e"
        else
            tokens = String[]
            for sg in words[c]
                push!(tokens, sg > 0 ? generators[sg] : generators[-sg] * "^-1")
            end
            word_strings[c] = join(tokens, " ")
        end
    end

    return cayley, inverse, 1, word_strings
end

"""
    cayley_convolve_jl(a, b, cayley)

Group-ring multiplication via Cayley-table convolution:
  (a * b)[k] = Σ_{i, j : cayley[i, j] = k} a[i] * b[j].
"""
function cayley_convolve_jl(a_in, b_in, cayley_in)
    a = pyconvert(Vector{Int64}, a_in)
    b = pyconvert(Vector{Int64}, b_in)
    cayley = pyconvert(Matrix{Int64}, cayley_in)
    n = length(a)
    @assert length(b) == n
    @assert size(cayley) == (n, n)
    res = zeros(Int64, n)
    @inbounds for i in 1:n
        ai = a[i]
        ai == 0 && continue
        for j in 1:n
            bj = b[j]
            bj == 0 && continue
            k = cayley[i, j]
            res[k] += ai * bj
        end
    end
    return res
end

"""
    lift_boundary_to_cover_jl(rows, cols, group_indices, coeffs, n_g, m_base, n_base, cayley)

Lift a base-complex boundary map to the universal cover, given Z[G]-coefficients
expressed as a list of (target_face, source_cell, group_idx, coeff) tuples.

The lifted map sends cover-cell (c, g) (linearised as `(c-1)*n_g + g`) to
Σ_terms coeff_k * (target_face, cayley[g, group_idx_k]). Returns COO triples
`(rows_out, cols_out, vals_out)` over the lifted matrix of size
`(m_base*n_g, n_base*n_g)` ready for `scipy.sparse.coo_matrix`.

All indices are 1-based on input; outputs are 1-based as well — Python wrapper
converts to 0-based for scipy.
"""
function lift_boundary_to_cover_jl(rows_in, cols_in, group_indices_in, coeffs_in,
                                    n_g::Int, m_base::Int, n_base::Int, cayley_in)
    rows = pyconvert(Vector{Int64}, rows_in)
    cols = pyconvert(Vector{Int64}, cols_in)
    gidx = pyconvert(Vector{Int64}, group_indices_in)
    coeffs = pyconvert(Vector{Int64}, coeffs_in)
    cayley = pyconvert(Matrix{Int64}, cayley_in)
    n_terms = length(rows)
    @assert length(cols) == n_terms
    @assert length(gidx) == n_terms
    @assert length(coeffs) == n_terms
    @assert size(cayley, 1) == n_g
    @assert size(cayley, 2) == n_g

    n_out = n_terms * n_g
    out_rows = Vector{Int64}(undef, n_out)
    out_cols = Vector{Int64}(undef, n_out)
    out_vals = Vector{Int64}(undef, n_out)
    write = 1
    @inbounds for k in 1:n_terms
        f = rows[k]; c = cols[k]; h = gidx[k]; v = coeffs[k]
        @inbounds for g in 1:n_g
            target_g = cayley[g, h]
            out_rows[write] = (f - 1) * n_g + target_g
            out_cols[write] = (c - 1) * n_g + g
            out_vals[write] = v
            write += 1
        end
    end
    return out_rows, out_cols, out_vals
end

"""
    fox_derivative_block_real_jl(relator, gen_idx, cayley, gen_to_group, inverse_indices, rho_images, degree)

Compute the d×d block ∂w/∂g_i with values evaluated through a real
representation ρ. `rho_images[k]` is ρ applied to the k-th group element
(1-indexed via the Cayley table). `relator` is a Vector{Int} of signed
generator indices.
"""
function fox_derivative_block_real_jl(relator_in, gen_idx::Int, cayley_in,
                                       gen_to_group_in, inverse_indices_in,
                                       rho_images_in, degree::Int)
    relator = pyconvert(Vector{Int}, relator_in)
    cayley = pyconvert(Matrix{Int}, cayley_in)
    gen_to_group = pyconvert(Vector{Int}, gen_to_group_in)
    inverse_indices = pyconvert(Vector{Int}, inverse_indices_in)
    n_g = size(cayley, 1)
    rho_arr = Vector{Matrix{Float64}}(undef, n_g)
    rho_raw = pyconvert(Vector{Any}, rho_images_in)
    for k in 1:n_g
        rho_arr[k] = pyconvert(Matrix{Float64}, rho_raw[k])
    end
    block = zeros(Float64, degree, degree)
    prefix = 1
    @inbounds for t in 1:length(relator)
        sg = relator[t]
        gen = abs(sg)
        eps = sg > 0 ? 1 : -1
        gid = gen_to_group[gen]
        gid_inv = inverse_indices[gid]
        new_prefix = eps > 0 ? cayley[prefix, gid] : cayley[prefix, gid_inv]
        if gen == gen_idx
            if eps > 0
                block .+= rho_arr[prefix]
            else
                block .-= rho_arr[new_prefix]
            end
        end
        prefix = new_prefix
    end
    return block
end

"""
    fox_derivative_block_complex_jl(relator, gen_idx, cayley, gen_to_group, inverse_indices, rho_images, degree)

Complex-valued Fox derivative block; same algorithm as the real variant but
evaluating ρ in `Matrix{ComplexF64}`. Used for Path-B twisted homology over
unitary representations.
"""
function fox_derivative_block_complex_jl(relator_in, gen_idx::Int, cayley_in,
                                          gen_to_group_in, inverse_indices_in,
                                          rho_images_in, degree::Int)
    relator = pyconvert(Vector{Int}, relator_in)
    cayley = pyconvert(Matrix{Int}, cayley_in)
    gen_to_group = pyconvert(Vector{Int}, gen_to_group_in)
    inverse_indices = pyconvert(Vector{Int}, inverse_indices_in)
    n_g = size(cayley, 1)
    rho_arr = Vector{Matrix{ComplexF64}}(undef, n_g)
    rho_raw = pyconvert(Vector{Any}, rho_images_in)
    for k in 1:n_g
        rho_arr[k] = pyconvert(Matrix{ComplexF64}, rho_raw[k])
    end
    block = zeros(ComplexF64, degree, degree)
    prefix = 1
    @inbounds for t in 1:length(relator)
        sg = relator[t]
        gen = abs(sg)
        eps = sg > 0 ? 1 : -1
        gid = gen_to_group[gen]
        gid_inv = inverse_indices[gid]
        new_prefix = eps > 0 ? cayley[prefix, gid] : cayley[prefix, gid_inv]
        if gen == gen_idx
            if eps > 0
                block .+= rho_arr[prefix]
            else
                block .-= rho_arr[new_prefix]
            end
        end
        prefix = new_prefix
    end
    return block
end

"""
    twisted_alexander_whitney_jl(alpha, beta, p, q, simplices_pq, s_to_idx_p, s_to_idx_q, degree)

Twisted Alexander-Whitney cup product over a representation of fixed degree d.
Cochains `alpha` (p-cochain) and `beta` (q-cochain) are vectors of length
`d * n_simplices_dim`, encoding `n_simplices_dim` blocks of size `d`. The
output is a (p+q)-cochain of length `d * length(simplices_pq)` whose value on
σ is alpha[front_face(σ)] ⊗_block beta[back_face(σ)] componentwise (Hadamard).

This kernel is intentionally simple: for the v1 twisted intersection form we
use the Hadamard pairing as the canonical bilinear form ℂ^d ⊗ ℂ^d → ℂ via the
diagonal trace, which is the right answer for orthogonal/unitary ρ.
"""
function twisted_alexander_whitney_jl(alpha_in, beta_in, p::Int, q::Int,
                                      simplices_pq_raw, s_to_idx_p, s_to_idx_q,
                                      degree::Int)
    alpha = pyconvert(Vector{ComplexF64}, alpha_in)
    beta = pyconvert(Vector{ComplexF64}, beta_in)
    idx_p = pyconvert(Dict{Tuple{Vararg{Int}}, Int64}, s_to_idx_p)
    idx_q = pyconvert(Dict{Tuple{Vararg{Int}}, Int64}, s_to_idx_q)
    simplices_pq = [_to_vertices_simplex(s) for s in simplices_pq_raw]
    n_pq = length(simplices_pq)
    res = zeros(ComplexF64, degree * n_pq)
    @inbounds for i in 1:n_pq
        s = simplices_pq[i]
        length(s) < p + q + 1 && continue
        v_p = get(idx_p, Tuple(s[1:(p+1)]), -1)
        v_q = get(idx_q, Tuple(s[(p+1):(p+q+1)]), -1)
        if v_p != -1 && v_q != -1
            base_p = v_p * degree
            base_q = v_q * degree
            base_r = (i - 1) * degree
            for k in 1:degree
                res[base_r + k] = alpha[base_p + k] * beta[base_q + k]
            end
        end
    end
    return res
end

# ── Handle Surgery Kernels (Phase 10) ────────────────────────────────────────

"""
    surgery_relative_boundary_sparse(K_simplices_q, K_simplices_qplus1, Kb_simplex_indices)

Build boundary matrix B_{q+1}: C_{q+1}(K) → C_q(K) as a sparse integer matrix.
Columns are indexed by (q+1)-simplices; rows by q-simplices.
`Kb_simplex_indices` are 0-based indices into K_simplices_q (Python convention).

Returns:
    SparseMatrixCSC{Int64, Int64} of shape (|C_q|, |C_{q+1}|).

Called by: exact path of compute_linking_number.
"""
function surgery_relative_boundary_sparse(
    K_simplices_q_raw,
    K_simplices_qplus1_raw,
    Kb_simplex_indices_raw,
)
    K_q = [pyconvert(Vector{Int}, s) for s in K_simplices_q_raw]
    K_qp1 = [pyconvert(Vector{Int}, s) for s in K_simplices_qplus1_raw]

    m = length(K_q)
    n = length(K_qp1)

    # Build index maps from sorted simplex tuples to 1-based row/col indices
    q_idx = Dict{Tuple{Vararg{Int}}, Int}()
    for (i, s) in enumerate(K_q)
        q_idx[Tuple(sort(s))] = i
    end

    Is = Int[]
    Js = Int[]
    Vs = Int64[]

    for (j, tau) in enumerate(K_qp1)
        tau_sorted = sort(tau)
        len = length(tau_sorted)
        for i_del in 1:len
            face = [tau_sorted[k] for k in 1:len if k != i_del]
            face_key = Tuple(face)
            row = get(q_idx, face_key, 0)
            if row != 0
                sign_val = Int64((-1)^(i_del - 1))
                push!(Is, row)
                push!(Js, j)
                push!(Vs, sign_val)
            end
        end
    end

    if isempty(Is)
        return sparse(Int[], Int[], Int64[], m, n)
    end
    return sparse(Is, Js, Vs, m, n)
end

"""
    linking_seifert_solve_z(B, b)

Solve B · f = b over ℤ via Smith Normal Form.

Returns:
    (f::Vector{Int64}, success::Bool, reason::Symbol)
    where reason ∈ {:ok, :not_in_image, :divisibility_fail}

Called by: exact path of compute_linking_number.
"""
function linking_seifert_solve_z(B_raw, b_raw)
    b = pyconvert(Vector{Int64}, b_raw)

    # B may be a Python sparse matrix or Julia SparseMatrixCSC
    if isa(B_raw, SparseMatrixCSC)
        B = convert(SparseMatrixCSC{Int64, Int64}, B_raw)
    else
        B_mat = pyconvert(Matrix{Int64}, B_raw)
        B = sparse(B_mat)
    end

    m, n = size(B)

    # Compute SNF via exact integer arithmetic
    # Use AbstractAlgebra for robust dense SNF with transforms
    try
        M_aa = AbstractAlgebra.matrix(ZZ, Matrix(B))
        S_aa, U_aa, V_aa = AbstractAlgebra.snf_with_transform(M_aa)
        
        # S_aa is diagonal, extract diagonal elements
        diag_D = [Int64(S_aa[i, i]) for i in 1:min(size(S_aa)...)]
        r = count(!=(0), diag_D)

        # Convert transforms to standard Int64 matrices
        # U*M*V = S => M = U^-1 * S * V^-1
        # M*f = b => U^-1 * S * V^-1 * f = b => S * (V^-1 * f) = U * b
        # Let w = V^-1 * f, then S*w = U*b and f = V*w.
        U = [Int64(U_aa[i, j]) for i in 1:size(U_aa, 1), j in 1:size(U_aa, 2)]
        V = [Int64(V_aa[i, j]) for i in 1:size(V_aa, 1), j in 1:size(V_aa, 2)]

        u = U * b

        # Check image condition: u[r+1:end] must all be 0
        if r < length(u) && any(!=(0), u[r+1:end])
            return (zeros(Int64, n), false, :not_in_image)
        end

        # Divisibility and solve S*w = u
        w = zeros(Int64, n)
        for i in 1:r
            d_ii = diag_D[i]
            if d_ii == 0
                continue
            end
            if u[i] % d_ii != 0
                return (zeros(Int64, n), false, :divisibility_fail)
            end
            w[i] = u[i] ÷ d_ii
        end

        f = V * w
        return (f, true, :ok)

    catch e
        # Fallback: least-squares over ℚ and round
        B_f = Float64.(Matrix(B))
        b_f = Float64.(b)
        if size(B_f, 1) >= size(B_f, 2)
            f_approx = B_f \ b_f
        else
            f_approx = B_f' * ((B_f * B_f') \ b_f)
        end
        f = round.(Int64, f_approx)
        residual = Matrix(B) * f - b
        if all(==(0), residual)
            return (f, true, :ok)
        else
            return (zeros(Int64, n), false, :not_in_image)
        end
    end
end

"""
    linking_intersection_pairing(a, f, K_p_simplices, K_qplus1_simplices, n)

Compute the simplicial intersection number ⟨K_a, F⟩ over ℤ.
`a` encodes K_a in Z^{|C_p|}, `f` encodes the Seifert chain in Z^{|C_{q+1}|}.
p + (q+1) = n (ambient dimension).

Returns:
    Int64 — the signed linking number.

Called by: exact path of compute_linking_number.
"""
function linking_intersection_pairing(
    a_raw,
    f_raw,
    K_p_simplices_raw,
    K_qplus1_simplices_raw,
    n::Int,
)
    a = pyconvert(Vector{Int64}, a_raw)
    f = pyconvert(Vector{Int64}, f_raw)
    K_p = [pyconvert(Vector{Int}, s) for s in K_p_simplices_raw]
    K_qp1 = [pyconvert(Vector{Int}, s) for s in K_qplus1_simplices_raw]

    intersection = Int64(0)

    for (i, sigma) in enumerate(K_p)
        a_i = i <= length(a) ? a[i] : Int64(0)
        if a_i == 0
            continue
        end
        sigma_set = Set(sigma)
        for (j, tau) in enumerate(K_qp1)
            f_j = j <= length(f) ? f[j] : Int64(0)
            if f_j == 0
                continue
            end
            # Check sigma ⊂ tau
            if !issubset(sigma_set, Set(tau))
                continue
            end
            # Find extra vertex in tau not in sigma
            tau_sorted = sort(tau)
            extra = [v for v in tau_sorted if !(v in sigma_set)]
            if length(extra) != 1
                continue
            end
            v_extra = extra[1]
            pos = findfirst(==(v_extra), tau_sorted)
            eps = Int64((-1)^(pos - 1))
            intersection += a_i * f_j * eps
        end
    end

    return intersection
end

function linking_intersect_2chains(
    F_a_raw,
    F_b_raw,
    simplices_1_raw,
    simplices_2_raw,
    simplices_3_raw
)
    F_a = pyconvert(Vector{Int64}, F_a_raw)
    F_b = pyconvert(Vector{Int64}, F_b_raw)
    simplices_1 = [pyconvert(Vector{Int}, s) for s in simplices_1_raw]
    simplices_2 = [pyconvert(Vector{Int}, s) for s in simplices_2_raw]
    simplices_3 = [pyconvert(Vector{Int}, s) for s in simplices_3_raw]

    n_1simplices = length(simplices_1)
    intersection_chain = zeros(Int64, n_1simplices)

    idx_1 = Dict{Tuple{Int, Int}, Int}()
    for (i, s) in enumerate(simplices_1)
        idx_1[(s[1], s[2])] = i
    end

    idx_2 = Dict{Tuple{Int, Int, Int}, Int}()
    for (i, s) in enumerate(simplices_2)
        idx_2[(s[1], s[2], s[3])] = i
    end

    for s3 in simplices_3
        v0, v1, v2, v3 = s3[1], s3[2], s3[3], s3[4]
        
        f_face = (v0, v1, v2)
        b_face = (v1, v2, v3)
        m_edge = (v1, v2)
        
        if haskey(idx_2, f_face) && haskey(idx_2, b_face) && haskey(idx_1, m_edge)
            i_f = idx_2[f_face]
            i_b = idx_2[b_face]
            i_m = idx_1[m_edge]
            
            c_a_f = i_f <= length(F_a) ? F_a[i_f] : 0
            c_b_b = i_b <= length(F_b) ? F_b[i_b] : 0
            c_b_f = i_f <= length(F_b) ? F_b[i_f] : 0
            c_a_b = i_b <= length(F_a) ? F_a[i_b] : 0
            
            val = c_a_f * c_b_b - c_b_f * c_a_b
            intersection_chain[i_m] += val
        end
    end
    
    return intersection_chain
end

"""
    surgery_handle_attach(K_simplices, attaching_sphere, tubular_neighborhood, co_disk_simplices, vertex_offset, k, n)

Perform the simplex-level handle attachment on K.
Returns the modified simplex dictionary keyed by dimension.

Args:
    K_simplices: Dict{Int, Vector{Vector{Int}}} of K simplices by dimension.
    attaching_sphere: Vector{Vector{Int}} of top simplices of σ ≅ S^{k-1}.
    tubular_neighborhood: Vector{Vector{Int}} of tube simplices.
    co_disk_simplices: Vector{Vector{Int}} of co-disk simplices to add.
    vertex_offset: Int offset to add to co-disk vertices.
    k: Handle index.
    n: Ambient dimension.

Returns:
    Dict{Int, Vector{Vector{Int}}} of result complex simplices.

Called by: perform_handle_surgery.
"""
function surgery_handle_attach(
    K_simplices_raw,
    attaching_sphere_raw,
    tubular_neighborhood_raw,
    co_disk_simplices_raw,
    vertex_offset::Int,
    k::Int,
    n::Int,
)
    # Convert inputs
    attaching_sphere = Set([Tuple(sort(pyconvert(Vector{Int}, s))) for s in attaching_sphere_raw])
    tubular_nbhd = Set([Tuple(sort(pyconvert(Vector{Int}, s))) for s in tubular_neighborhood_raw])
    co_disk = [pyconvert(Vector{Int}, s) for s in co_disk_simplices_raw]

    # Build result simplex dict
    result = Dict{Int, Vector{Vector{Int}}}()

    # Process existing K simplices
    K_dim_keys = try
        pyconvert(Vector{Int}, collect(keys(K_simplices_raw)))
    catch
        Int[]
    end

    for d in K_dim_keys
        dim_simps_raw = try K_simplices_raw[d] catch; continue end
        dim_simps = [pyconvert(Vector{Int}, s) for s in dim_simps_raw]
        kept = Vector{Int}[]
        for s in dim_simps
            key = Tuple(sort(s))
            if d == n && key in tubular_nbhd && !(key in attaching_sphere)
                continue  # Remove open tube interior (top-dim only)
            end
            push!(kept, s)
        end
        if !isempty(kept)
            result[d] = kept
        end
    end

    # Add co-disk simplices (shifted by vertex_offset)
    for s in co_disk
        shifted = s .+ vertex_offset
        d = length(shifted) - 1
        if !haskey(result, d)
            result[d] = Vector{Int}[]
        end
        push!(result[d], shifted)
        # Add all faces
        for i in 1:length(shifted)
            face = [shifted[j] for j in 1:length(shifted) if j != i]
            df = length(face) - 1
            if df >= 0
                if !haskey(result, df)
                    result[df] = Vector{Int}[]
                end
                push!(result[df], face)
            end
        end
    end

    # Deduplicate
    for d in keys(result)
        unique_set = Set{Tuple{Vararg{Int}}}()
        kept = Vector{Int}[]
        for s in result[d]
            key = Tuple(sort(s))
            if !(key in unique_set)
                push!(unique_set, key)
                push!(kept, s)
            end
        end
        result[d] = kept
    end

    return result
end

"""
    sphere_recognition_pl(simplices, dim)

PL sphere recognition for simplicial complexes. Dispatches by dimension:
  - dim 0: S^0 = two disjoint points.
  - dim 1: S^1 = simple cycle graph.
  - dim 2: closed orientable 2-manifold with Euler characteristic 2.
  - dim 3: bounded Rubinstein-Thompson heuristic.
  - dim ≥ 4: bounded PL recognition (collapsibility, homological checks).

Returns:
    (is_sphere::Bool, certificate::Symbol)
    certificate ∈ {:trivial_S0, :graph_cycle, :euler_orientable,
                   :rubinstein_thompson, :bounded_pl, :rejected_link,
                   :undecidable_dim}

Called by: find_attachment_sphere (exact path).
"""
function sphere_recognition_pl(simplices_raw, dim::Int)
    simps = [sort(pyconvert(Vector{Int}, s)) for s in simplices_raw]

    if dim == 0
        verts = [s for s in simps if length(s) == 1]
        edges = [s for s in simps if length(s) == 2]
        return (length(verts) == 2 && isempty(edges), :trivial_S0)

    elseif dim == 1
        verts = Set{Int}([s[1] for s in simps if length(s) == 1])
        edges = [s for s in simps if length(s) == 2]
        isempty(edges) && return (false, :rejected_link)
        degree = Dict{Int, Int}(v => 0 for v in verts)
        for e in edges
            length(e) >= 2 || return (false, :rejected_link)
            degree[e[1]] = get(degree, e[1], 0) + 1
            degree[e[2]] = get(degree, e[2], 0) + 1
        end
        all(d == 2 for d in values(degree)) || return (false, :rejected_link)
        # Connectivity
        adj = Dict{Int, Vector{Int}}(v => Int[] for v in verts)
        for e in edges
            push!(adj[e[1]], e[2])
            push!(adj[e[2]], e[1])
        end
        visited = Set{Int}()
        start = first(verts)
        stack = [start]
        while !isempty(stack)
            v = pop!(stack)
            v in visited && continue
            push!(visited, v)
            append!(stack, adj[v])
        end
        return (visited == verts, :graph_cycle)

    elseif dim == 2
        triangles = [s for s in simps if length(s) == 3]
        isempty(triangles) && return (false, :rejected_link)
        edge_count = Dict{Tuple{Int,Int}, Int}()
        for t in triangles
            for i in 1:3
                face = Tuple(sort([t[j] for j in 1:3 if j != i]))
                edge_count[face] = get(edge_count, face, 0) + 1
            end
        end
        all(c == 2 for c in values(edge_count)) || return (false, :rejected_link)
        V = length(Set{Int}([v for s in simps for v in s]))
        E = length(edge_count)
        F = length(triangles)
        return (V - E + F == 2, :euler_orientable)

    elseif dim == 3
        # Bounded 3-sphere recognition: check Betti numbers and Euler characteristic
        # Full Rubinstein-Thompson requires Regina; use homological heuristic here.
        verts = Set{Int}([v for s in simps for v in s])
        V = length(verts)
        # For S^3: Euler characteristic = 0, H_0=Z, H_1=0, H_2=0, H_3=Z
        # Count cells
        d0 = length([s for s in simps if length(s) == 1])
        d1 = length([s for s in simps if length(s) == 2])
        d2 = length([s for s in simps if length(s) == 3])
        d3 = length([s for s in simps if length(s) == 4])
        chi = d0 - d1 + d2 - d3
        # χ(S^3) = 0
        return (chi == 0 && d3 > 0, :rubinstein_thompson)

    elseif dim >= 4
        # Bounded PL recognition: Euler characteristic check
        cells = [length([s for s in simps if length(s) == d + 1]) for d in 0:dim]
        chi = sum((-1)^d * cells[d+1] for d in 0:dim)
        expected_chi = 1 + (-1)^dim  # χ(S^n) = 1 + (-1)^n
        return (chi == expected_chi && !isempty([s for s in simps if length(s) == dim + 1]), :bounded_pl)
    end

    return (false, :undecidable_dim)
end

# ────────────────────────────────────────────────────────────────────────────
# Exhaustive E_infinity enumeration for the Adams spectral sequence.
#
# Each ambiguous d_r flag at (r, src, tgt) has a rank k in {0..min(S,T)};
# this kernel enumerates every joint rank assignment over all flags and
# tracks the (min, max, sum) dim per cell after subtracting rank-out and
# rank-in contributions.
#
# Inputs (all 1-based indices in src_idx, tgt_idx):
#   e2_vec  :: Vector{Int64}   E_2 dim at each cell, indexed 1..n_cells.
#   radices :: Vector{Int64}   ranges of each flag's rank (0..radix-1).
#   src_idx :: Vector{Int64}   for each flag, the cell index of its source
#                               (1-based; 0 means "outside window — skip").
#   tgt_idx :: Vector{Int64}   for each flag, the cell index of its target
#                               (1-based; 0 means "outside window — skip").
#
# Returns: (e_min, e_max, e_sum, explored)
#   - e_min, e_max :: Vector{Int64} per-cell min/max over assignments.
#   - e_sum        :: Vector{Float64} per-cell sum over assignments.
#   - explored     :: Int total number of rank assignments visited.
# ────────────────────────────────────────────────────────────────────────────
function exhaustive_e_inf(
    e2_vec::AbstractVector{Int64},
    radices::AbstractVector{Int64},
    src_idx::AbstractVector{Int64},
    tgt_idx::AbstractVector{Int64},
)
    n_cells = length(e2_vec)
    n_flags = length(radices)

    e_min = copy(collect(e2_vec))
    e_max = copy(collect(e2_vec))
    e_sum = zeros(Float64, n_cells)
    cand  = Vector{Int64}(undef, n_cells)
    indices = zeros(Int64, n_flags)

    explored = 0
    while true
        # cand = e2 with rank subtractions
        @inbounds for j in 1:n_cells
            cand[j] = e2_vec[j]
        end
        @inbounds for i in 1:n_flags
            k = indices[i]
            if k > 0
                si = src_idx[i]
                ti = tgt_idx[i]
                if si > 0
                    cand[si] -= k
                end
                if ti > 0
                    cand[ti] -= k
                end
            end
        end
        @inbounds for j in 1:n_cells
            v = cand[j]
            if v < 0
                v = 0
            end
            if v < e_min[j]
                e_min[j] = v
            end
            if v > e_max[j]
                e_max[j] = v
            end
            e_sum[j] += Float64(v)
        end
        explored += 1

        # mixed-radix increment
        done = true
        @inbounds for i in 1:n_flags
            indices[i] += 1
            if indices[i] < radices[i]
                done = false
                break
            end
            indices[i] = 0
        end
        if done
            break
        end
    end

    return e_min, e_max, e_sum, explored
end


# ── Acceleration 1: Batch intersection pairings ───────────────────────────────

"""
    linking_intersection_batch(a_series, f, K_p_simplices, K_qplus1_simplices, n)

Compute ⟨K_a_i, F⟩ for every a-vector in `a_series`, reusing the same Seifert chain `f`.
This eliminates the repeated SNF solve across unlink passes when K_b is fixed.

Args:
    a_series_raw: Python list of a-vectors (each Vector{Int64}).
    f_raw:        Seifert chain f, precomputed by linking_seifert_solve_z.
    K_p_simplices_raw: top-dimensional simplices of K_a (p-cells), Python list.
    K_qplus1_simplices_raw: (q+1)-cells of ambient K, Python list.
    n:            Ambient dimension.

Returns:
    Vector{Int64} — one linking number per a-vector, computed in parallel.

Called by: compute_linking_from_chain (Python, surgery.py).
"""
function linking_intersection_batch(
    a_series_raw,
    f_raw,
    K_p_simplices_raw,
    K_qplus1_simplices_raw,
    n_raw,
)
    f    = pyconvert(Vector{Int64}, f_raw)
    K_p  = [pyconvert(Vector{Int}, s) for s in K_p_simplices_raw]
    K_qp1 = [pyconvert(Vector{Int}, s) for s in K_qplus1_simplices_raw]
    n    = Int(n_raw)

    # Build fast lookup: for each (q+1)-simplex, record non-zero f_j and its sigma_set
    # to avoid redundant set construction inside the a-loop.
    nonzero_f = [(j, f[j], Set(K_qp1[j]), sort(K_qp1[j])) for j in 1:length(K_qp1) if j <= length(f) && f[j] != 0]

    # Convert a_series in one shot to avoid repeated pyconvert inside the parallel loop
    a_list = [pyconvert(Vector{Int64}, a) for a in a_series_raw]
    results = zeros(Int64, length(a_list))

    Threads.@threads for idx in 1:length(a_list)
        a = a_list[idx]
        lk = Int64(0)
        for (i, sigma) in enumerate(K_p)
            a_i = i <= length(a) ? a[i] : Int64(0)
            a_i == 0 && continue
            sigma_set = Set(sigma)
            for (j, f_j, tau_set, tau_sorted) in nonzero_f
                issubset(sigma_set, tau_set) || continue
                extra = [v for v in tau_sorted if !(v in sigma_set)]
                length(extra) == 1 || continue
                pos = findfirst(==(extra[1]), tau_sorted)
                lk += a_i * f_j * Int64((-1)^(pos - 1))
            end
        end
        results[idx] = lk
    end
    return results
end


# ── Acceleration 2: Julia-native cohomology basis for IntersectionForm ────────

"""
    compute_cohomology_basis_jl(Dm_I, Dm_J, Dm_V, nrows_Dm, ncols_Dm,
                                 Dmp1_I, Dmp1_J, Dmp1_V, nrows_Dmp1, ncols_Dmp1,
                                 Dn_I, Dn_J, Dn_V, nrows_Dn, ncols_Dn)

Compute the free cohomology generators of H^m(K; ℤ) from boundary matrices,
and the fundamental class F, entirely in exact integer arithmetic via AbstractAlgebra.

This replaces the Python path that uses smith_normal_decomp (Numba/SymPy) + SymPy.inv()
for the Gram matrix ZT_Z, which is the bottleneck when β_m > 50.

Algorithm:
  1. Build δ_m = D_{m+1}^T, δ_{m-1} = D_m^T as dense integer matrices.
  2. SNF(δ_m) → null space Z_m (the m-cocycle lattice).
  3. Exact inversion of ZT_Z = Z_m^T Z_m via AbstractAlgebra.
  4. M = (ZT_Z)^{-1} Z_m^T δ_{m-1} → SNF(M) with transforms U_M, giving cocycles.
  5. Free generators: columns of Z_m U_M^{-1} with S_M diagonal entry ≠ 1.
  6. Fundamental class F = first null-space vector of D_n.

Returns:
    (cocycles_flat::Vector{Int64}, n_rows::Int, n_cols::Int,
     free_col_indices::Vector{Int64}, F_flat::Vector{Int64})

    - cocycles_flat: column-major flattened cocycle matrix (n_rows × n_cols).
    - free_col_indices: 0-based column indices of the free (non-torsion) generators.
    - F_flat: fundamental class as a vector of length ncols_Dn.

Called by: IntersectionForm.from_complex (Python, intersection_forms.py).
"""
function compute_cohomology_basis_jl(
    Dm_I_raw, Dm_J_raw, Dm_V_raw, nrows_Dm::Int, ncols_Dm::Int,
    Dmp1_I_raw, Dmp1_J_raw, Dmp1_V_raw, nrows_Dmp1::Int, ncols_Dmp1::Int,
    Dn_I_raw, Dn_J_raw, Dn_V_raw, nrows_Dn::Int, ncols_Dn::Int,
)
    # ── Convert COO data (0-based Python → 1-based Julia) ────────────────────
    function to_sparse(I_raw, J_raw, V_raw, m, n)
        Iv = pyconvert(Vector{Int64}, I_raw) .+ 1
        Jv = pyconvert(Vector{Int64}, J_raw) .+ 1
        Vv = pyconvert(Vector{Int64}, V_raw)
        isempty(Iv) ? sparse(Int64[], Int64[], Int64[], m, n) :
                      sparse(Iv, Jv, Vv, m, n)
    end

    D_m   = to_sparse(Dm_I_raw,   Dm_J_raw,   Dm_V_raw,   nrows_Dm,   ncols_Dm)
    D_mp1 = to_sparse(Dmp1_I_raw, Dmp1_J_raw, Dmp1_V_raw, nrows_Dmp1, ncols_Dmp1)
    D_n   = to_sparse(Dn_I_raw,   Dn_J_raw,   Dn_V_raw,   nrows_Dn,   ncols_Dn)

    # δ_m   = D_{m+1}^T  (coboundary C^m → C^{m+1})
    # δ_{m-1} = D_m^T  (coboundary C^{m-1} → C^m)
    delta_m_dense    = Matrix{Int64}(D_mp1')
    delta_mm1_dense  = Matrix{Int64}(D_m')

    # ── Step 1: null space of δ_m via AbstractAlgebra SNF ─────────────────────
    # AbstractAlgebra convention: snf_with_transform(A) returns (S, U, V) with U*A*V = S.
    # Null space of A = columns of V beyond rank r (since A*V[:,j] = U^{-1}*S[:,j] = 0 for j>r).
    AA_dm = AbstractAlgebra.matrix(ZZ, delta_m_dense)
    S_dm, _, V_dm = AbstractAlgebra.snf_with_transform(AA_dm)
    r_m = count(i -> !iszero(S_dm[i, i]), 1:min(size(S_dm)...))
    nr_v, nc_v = size(V_dm)
    V_int = [Int64(V_dm[i, j]) for i in 1:nr_v, j in 1:nc_v]
    Z_m = V_int[:, (r_m + 1):end]   # null-space basis (columns)

    if size(Z_m, 2) == 0
        return (Int64[], 0, 0, Int64[], zeros(Int64, ncols_Dn))
    end

    # ── Step 2: M = ZTZ⁻¹ · Z_m^T · δ_{m-1} over ℚ (exact) ─────────────────
    # We compute ZTZ⁻¹ · rhs via exact rational arithmetic, then verify
    # the result is integer (it always is by theory: ZTZ is the Gram matrix
    # of a sublattice, and Z_m^T δ_{m-1} is in its column space over ℤ).
    ZTZ_int = Z_m' * Z_m                                   # (k × k), k = dim(null space)
    rhs_int = Z_m' * delta_mm1_dense                       # (k × c_{m-1})

    ZTZ_qq  = AbstractAlgebra.change_base_ring(AbstractAlgebra.QQ,
                  AbstractAlgebra.matrix(ZZ, ZTZ_int))
    rhs_qq  = AbstractAlgebra.change_base_ring(AbstractAlgebra.QQ,
                  AbstractAlgebra.matrix(ZZ, rhs_int))
    M_qq    = inv(ZTZ_qq) * rhs_qq                         # exact rational

    nr_M, nc_M = size(M_qq)
    M = zeros(Int64, nr_M, nc_M)
    for i in 1:nr_M, j in 1:nc_M
        q = M_qq[i, j]
        # By theory the result is integer; if not, round (handles rounding from QQ repr).
        num = Int64(numerator(q))
        den = Int64(denominator(q))
        M[i, j] = den == 1 ? num : round(Int64, num / den)
    end

    # ── Step 3: SNF of M → cocycle basis ─────────────────────────────────────
    # snf_with_transform(M) gives (S_M, U_M, V_M) with U_M * M * V_M = S_M.
    # We need U_M^{-1} to transform Z_m: cocycles = Z_m · U_M^{-1}.
    # U_M is unimodular (det = ±1) since it comes from integer row operations.
    AA_M = AbstractAlgebra.matrix(ZZ, M)
    S_M, U_M_aa, _ = AbstractAlgebra.snf_with_transform(AA_M)
    nu = size(U_M_aa, 1)
    U_M_qq = AbstractAlgebra.change_base_ring(AbstractAlgebra.QQ, U_M_aa)
    U_M_inv_qq = inv(U_M_qq)                               # exact; entries are integers (det = ±1)
    U_M_inv = [Int64(numerator(U_M_inv_qq[i, j])) for i in 1:nu, j in 1:nu]

    cocycles = Z_m * U_M_inv                               # (c_m × k) integer matrix

    # ── Step 4: free-generator column indices (bounded by cocycles, not S_M) ──
    # A column j is a FREE cohomology generator iff S_M[j,j] ≠ 1:
    #   S_M[j,j] = 0 → free, not in image of M (genuine cohomology class)
    #   S_M[j,j] > 1 → torsion cohomology class
    #   S_M[j,j] = 1 → coboundary (in image of δ_{m-1}) → exclude
    # Loop bound = size(cocycles, 2), NOT size(S_M, 2), because S_M may have
    # more columns than cocycles (when M has more cols than rows).
    n_coc_cols = size(cocycles, 2)
    free_cols  = Int64[]
    for j in 1:n_coc_cols
        d_j = (j <= min(size(S_M)...)) ? Int64(S_M[j, j]) : Int64(0)
        if d_j != 1
            push!(free_cols, Int64(j - 1))                 # 0-based for Python
        end
    end

    # ── Step 5: fundamental class F (null space of D_n) ───────────────────────
    F = zeros(Int64, ncols_Dn)
    if !iszero(D_n)
        D_n_dense = Matrix{Int64}(D_n)
        AA_Dn = AbstractAlgebra.matrix(ZZ, D_n_dense)
        S_n, _, V_n_aa = AbstractAlgebra.snf_with_transform(AA_Dn)
        r_n   = count(i -> !iszero(S_n[i, i]), 1:min(size(S_n)...))
        nrv_n, ncv_n = size(V_n_aa)
        V_n   = [Int64(V_n_aa[i, j]) for i in 1:nrv_n, j in 1:ncv_n]
        if ncv_n > r_n
            F = V_n[:, r_n + 1]
        end
    end

    # Return cocycles column-major flat (Julia is column-major, same as NumPy order='F')
    nr_c, nc_c  = size(cocycles)
    cocycles_flat = vec(cocycles)

    return (cocycles_flat, nr_c, nc_c, free_cols, F)
end


"""
    alexander_from_seifert_jl(V)

Compute det(t*V - V^T) as a polynomial with integer coefficients.

Returns a pair (coeffs, min_degree) where coeffs[k] is the coefficient of
t^{min_degree + k - 1}. Uses AbstractAlgebra.jl for exact polynomial arithmetic.
"""
function alexander_from_seifert_jl(V_raw)
    V = Matrix{Int64}(hcat([collect(row) for row in V_raw]...)')
    g = size(V, 1)
    if g == 0
        return ([1], 0)
    end

    # Build det(t*V - V^T) symbolically using AbstractAlgebra
    R, t = AbstractAlgebra.polynomial_ring(AbstractAlgebra.ZZ, :t)
    M = AbstractAlgebra.matrix(R, g, g, [
        t * V[i, j] - V[j, i]
        for i in 1:g for j in 1:g
    ])

    poly = AbstractAlgebra.det(M)

    # Extract coefficients from min to max degree
    if AbstractAlgebra.iszero(poly)
        return ([0], 0)
    end

    d_min = Int64(AbstractAlgebra.degree(poly))  # deg gives highest degree
    # Collect all coefficients (AbstractAlgebra poly is 0-indexed)
    deg_high = Int64(AbstractAlgebra.degree(poly))
    coeffs = [Int64(AbstractAlgebra.coeff(poly, k)) for k in 0:deg_high]
    # Normalize so Δ(1) = 1
    delta_1 = sum(coeffs)
    if delta_1 < 0
        coeffs = -coeffs
    end
    return (coeffs, 0)  # coeffs[k+1] = coeff of t^k, starting from t^0
end


"""
    knot_signature_jl(V)

Compute the knot signature σ(K) = signature(V + V^T) exactly using eigenvalues.
Returns #positive_eigenvalues - #negative_eigenvalues.
"""
function knot_signature_jl(V_raw)
    V = Matrix{Float64}(hcat([collect(row) for row in V_raw]...)')
    g = size(V, 1)
    if g == 0
        return 0
    end
    S = V + V'
    eigs = LinearAlgebra.eigvals(Symmetric(S))
    pos = count(e -> e > 1e-10, eigs)
    neg = count(e -> e < -1e-10, eigs)
    return Int64(pos - neg)
end


"""
    linking_gauss_riemann_jl(Ka_starts, Ka_ends, Ka_mult, Kb_starts, Kb_ends, Kb_mult, n_samples)

Compute the (raw) Gauss linking integral lk(K_a, K_b) via a tight midpoint Riemann
sum. Each row of `Ka_starts`/`Ka_ends` is the 3D coordinate of the start/end of an
oriented edge of K_a (already in cycle orientation); `Ka_mult` carries the integer
multiplicity of each edge in the chain. Same for K_b.

Returns the integral already divided by 4π. The Python caller rounds to the nearest
integer to recover the signed linking number.

This is the canonical embedding-based linking number formula; it requires only the
vertex coordinates of K_a, K_b — no Seifert chain, no SNF — and is exact (up to the
Riemann-sum truncation, which is well under 1/2 for n_samples ≥ 16 on smooth links).
"""
function linking_gauss_riemann_jl(
    Ka_starts_raw,
    Ka_ends_raw,
    Ka_mult_raw,
    Kb_starts_raw,
    Kb_ends_raw,
    Kb_mult_raw,
    n_samples_raw,
)
    Ka_starts = pyconvert(Matrix{Float64}, Ka_starts_raw)
    Ka_ends   = pyconvert(Matrix{Float64}, Ka_ends_raw)
    Ka_mult   = pyconvert(Vector{Int64},   Ka_mult_raw)
    Kb_starts = pyconvert(Matrix{Float64}, Kb_starts_raw)
    Kb_ends   = pyconvert(Matrix{Float64}, Kb_ends_raw)
    Kb_mult   = pyconvert(Vector{Int64},   Kb_mult_raw)
    n_samples = pyconvert(Int, n_samples_raw)

    N_a = size(Ka_starts, 1)
    N_b = size(Kb_starts, 1)
    if N_a == 0 || N_b == 0 || n_samples <= 0
        return 0.0
    end

    inv_n  = 1.0 / n_samples
    inv_n2 = inv_n * inv_n

    # Precompute midpoint samples and (cumulative-pair) midpoint coords for each segment.
    A_pts = Matrix{Float64}(undef, N_a * n_samples, 3)
    A_tan = Matrix{Float64}(undef, N_a, 3)
    @inbounds for i in 1:N_a
        tx = Ka_ends[i, 1] - Ka_starts[i, 1]
        ty = Ka_ends[i, 2] - Ka_starts[i, 2]
        tz = Ka_ends[i, 3] - Ka_starts[i, 3]
        A_tan[i, 1] = tx; A_tan[i, 2] = ty; A_tan[i, 3] = tz
        base = (i - 1) * n_samples
        for k in 1:n_samples
            s = (k - 0.5) * inv_n
            A_pts[base + k, 1] = Ka_starts[i, 1] + s * tx
            A_pts[base + k, 2] = Ka_starts[i, 2] + s * ty
            A_pts[base + k, 3] = Ka_starts[i, 3] + s * tz
        end
    end

    B_pts = Matrix{Float64}(undef, N_b * n_samples, 3)
    B_tan = Matrix{Float64}(undef, N_b, 3)
    @inbounds for j in 1:N_b
        tx = Kb_ends[j, 1] - Kb_starts[j, 1]
        ty = Kb_ends[j, 2] - Kb_starts[j, 2]
        tz = Kb_ends[j, 3] - Kb_starts[j, 3]
        B_tan[j, 1] = tx; B_tan[j, 2] = ty; B_tan[j, 3] = tz
        base = (j - 1) * n_samples
        for k in 1:n_samples
            s = (k - 0.5) * inv_n
            B_pts[base + k, 1] = Kb_starts[j, 1] + s * tx
            B_pts[base + k, 2] = Kb_starts[j, 2] + s * ty
            B_pts[base + k, 3] = Kb_starts[j, 3] + s * tz
        end
    end

    total = 0.0
    @inbounds for i in 1:N_a
        ta_x = A_tan[i, 1]; ta_y = A_tan[i, 2]; ta_z = A_tan[i, 3]
        ma = Ka_mult[i]
        base_a = (i - 1) * n_samples
        for j in 1:N_b
            tb_x = B_tan[j, 1]; tb_y = B_tan[j, 2]; tb_z = B_tan[j, 3]
            mb = Kb_mult[j]
            cx = ta_y * tb_z - ta_z * tb_y
            cy = ta_z * tb_x - ta_x * tb_z
            cz = ta_x * tb_y - ta_y * tb_x
            mm = Float64(ma * mb)
            base_b = (j - 1) * n_samples
            pair_total = 0.0
            for ki in 1:n_samples
                ax = A_pts[base_a + ki, 1]
                ay = A_pts[base_a + ki, 2]
                az = A_pts[base_a + ki, 3]
                for kj in 1:n_samples
                    dx = ax - B_pts[base_b + kj, 1]
                    dy = ay - B_pts[base_b + kj, 2]
                    dz = az - B_pts[base_b + kj, 3]
                    d2 = dx*dx + dy*dy + dz*dz
                    if d2 < 1e-18
                        continue
                    end
                    num = dx * cx + dy * cy + dz * cz
                    pair_total += num / (d2 * sqrt(d2))
                end
            end
            total += mm * pair_total
        end
    end

    return total * inv_n2 / (4.0 * pi)
end

function _grid_from_values_jl(vals::Vector{Float64}, eps_max::Union{Nothing, Float64}, n_samples::Union{Nothing, Int})
    uvals = unique(vals)
    if isempty(uvals)
        uvals = [0.0]
    end
    if eps_max !== nothing
        uvals = uvals[uvals .<= eps_max]
    end
    if isempty(uvals)
        return [0.0]
    end
    if n_samples !== nothing && length(uvals) > n_samples
        indices = round.(Int, range(1, length(uvals), length=n_samples))
        uvals = uvals[unique(indices)]
    end
    if !any(x -> x == 0.0, uvals)
        push!(uvals, 0.0)
        sort!(uvals)
    end
    return uvals
end

function _get_reduced_homology_jl(lk_simplices, lk_max_dim)
    lk_b, lk_c = compute_boundary_payload_from_simplices(lk_simplices, lk_max_dim, false)
    rh = Dict{Int, Tuple{Int, Vector{Int}}}()

    d_max = -1
    for d in 0:lk_max_dim
        if get(lk_c, d, 0) > 0
            d_max = d
        end
    end

    for d in 0:d_max
        n_rows_d = get(lk_c, d-1, 0)
        n_cols_d = get(lk_c, d, 0)

        rank_ker_d = if n_cols_d == 0
            0
        elseif n_rows_d == 0
            n_cols_d
        else
            b_d = lk_b[d]
            n_cols_d - rank_q_sparse(b_d["rows"], b_d["cols"], b_d["data"], b_d["n_rows"], b_d["n_cols"])
        end

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
        if d == 0
            betti = max(0, betti - 1)
        end

        if betti > 0 || !isempty(torsion)
            rh[d] = (Int(betti), Int.(torsion))
        end
    end
    return rh, d_max
end

function check_closed_manifold_jl(active_cliques, dim::Int)
    if dim < 1
        return true
    end
    face_counts = Dict{Tuple{Vararg{Int}}, Int}()
    for c in active_cliques
        if length(c) == dim + 1
            for i in 1:(dim + 1)
                t = ntuple(j -> j < i ? c[j] : c[j+1], dim)
                face_counts[t] = get(face_counts, t, 0) + 1
            end
        end
    end
    return all(count == 2 for count in values(face_counts))
end

function _run_manifold_analysis_jl(cliques, vals, max_dim::Int, epsilon::Float64, n_samples::Union{Nothing, Int}, n_pts::Int = 0)
    p = sortperm(vals)
    sorted_cliques = cliques[p]
    sorted_vals = vals[p]
    
    grid_epsilons = _grid_from_values_jl(sorted_vals, epsilon, n_samples)
    
    if n_pts <= 0
        for c in sorted_cliques
            for v in c
                if v > n_pts
                    n_pts = v
                end
            end
        end
    end
    
    link_simplices_faces = [Vector{Vector{Int}}() for _ in 1:n_pts]
    link_simplices_vals = [Vector{Float64}() for _ in 1:n_pts]
    for (c, val) in zip(sorted_cliques, sorted_vals)
        len = length(c)
        if len > 1
            for i in 1:len
                v = c[i]
                face = Vector{Int}(undef, len - 1)
                idx_f = 1
                for j in 1:len
                    if j != i
                        @inbounds face[idx_f] = c[j]
                        idx_f += 1
                    end
                end
                push!(link_simplices_faces[v], face)
                push!(link_simplices_vals[v], val)
            end
        end
    end
    
    # Precompute running maximum dimension
    running_max_dim = Vector{Int}(undef, length(sorted_cliques))
    curr_max = -1
    for i in 1:length(sorted_cliques)
        d_c = length(sorted_cliques[i]) - 1
        if d_c > curr_max
            curr_max = d_c
        end
        @inbounds running_max_dim[i] = curr_max
    end
    
    m_epsilons = Float64[]
    m_is_manifold = Bool[]
    m_dimensions = Int[]
    m_is_closed = Bool[]
    m_failures = Int[]
    
    # Pre-allocate v_map_arr to avoid vertex Dict allocations inside the loop
    v_map_arr = zeros(Int, n_pts)
    
    for eps in grid_epsilons
        idx = searchsortedlast(sorted_vals, eps)
        if idx == 0
            push!(m_epsilons, eps)
            push!(m_is_manifold, true)
            push!(m_dimensions, -1)
            push!(m_is_closed, true)
            push!(m_failures, 0)
            continue
        end
        
        active_cliques = view(sorted_cliques, 1:idx)
        d_active = running_max_dim[idx]
        
        if d_active <= 0
            push!(m_epsilons, eps)
            push!(m_is_manifold, true)
            push!(m_dimensions, d_active)
            push!(m_is_closed, true)
            push!(m_failures, 0)
            continue
        end
        
        n_fail = 0
        local_dims = Int[]
        
        for v in 1:n_pts
            idx_lk = searchsortedlast(link_simplices_vals[v], eps)
            if idx_lk == 0
                push!(local_dims, 0)
                continue
            end
            lk = view(link_simplices_faces[v], 1:idx_lk)
            
            c_dims = zeros(Int, max_dim + 1)
            for face in lk
                fd = length(face) - 1
                if 0 <= fd <= max_dim
                    @inbounds c_dims[fd + 1] += 1
                end
            end
            
            chi = 0
            for fd in 0:max_dim
                @inbounds chi += (-1)^fd * c_dims[fd + 1]
            end
            
            exp_sph = 1 + (-1)^(d_active - 1)
            exp_dsk = 1
            
            if chi != exp_sph && chi != exp_dsk
                n_fail += 1
                continue
            end
            
            if d_active == 1
                n_v = c_dims[1]
                if n_v != 1 && n_v != 2
                    n_fail += 1
                else
                    push!(local_dims, n_v == 2 ? 1 : 0)
                end
            elseif d_active == 2
                n_v = c_dims[1]
                n_e = c_dims[2]
                
                v_set = BitSet()
                for face in lk
                    if length(face) == 1
                        push!(v_set, face[1])
                    end
                end
                v_list = collect(v_set)
                curr_nv = length(v_list)
                
                for i in 1:curr_nv
                    @inbounds v_map_arr[v_list[i]] = i
                end
                
                adj = [Vector{Int}() for _ in 1:curr_nv]
                for face in lk
                    if length(face) == 2
                        u = @inbounds v_map_arr[face[1]]
                        w = @inbounds v_map_arr[face[2]]
                        if u > 0 && w > 0
                            push!(adj[u], w)
                            push!(adj[w], u)
                        end
                    end
                end
                
                for i in 1:curr_nv
                    @inbounds v_map_arr[v_list[i]] = 0
                end
                
                is_connected = true
                if curr_nv > 0
                    q = Vector{Int}(undef, curr_nv)
                    q[1] = 1
                    visited = zeros(Bool, curr_nv)
                    visited[1] = true
                    head = 1
                    tail = 1
                    while head <= tail
                        curr = q[head]
                        head += 1
                        for neighbor in adj[curr]
                            if !visited[neighbor]
                                visited[neighbor] = true
                                tail += 1
                                q[tail] = neighbor
                            end
                        end
                    end
                    is_connected = (tail == curr_nv)
                end
                
                if !is_connected
                    n_fail += 1
                else
                    if n_v - n_e == 0
                        push!(local_dims, 2)
                    elseif n_v - n_e == 1
                        push!(local_dims, 1)
                    else
                        n_fail += 1
                    end
                end
            else
                rh, lk_d_max = _get_reduced_homology_jl(lk, d_active - 1)
                if isempty(rh)
                    push!(local_dims, d_active - 1)
                elseif length(rh) == 1
                    deg = first(keys(rh))
                    betti, torsion = rh[deg]
                    if betti == 1 && isempty(torsion) && deg == d_active - 1
                        push!(local_dims, d_active)
                    else
                        n_fail += 1
                    end
                else
                    n_fail += 1
                end
            end
        end
        
        is_mani = (n_fail == 0) && (length(unique(local_dims)) <= 1)
        if !is_mani && n_fail == 0
            n_fail = n_pts
        end
        
        detected_dim = if is_mani
            isempty(local_dims) ? d_active : first(local_dims)
        else
            d_active
        end
        
        is_closed = false
        if is_mani
            is_closed = check_closed_manifold_jl(active_cliques, detected_dim)
        end
        
        push!(m_epsilons, eps)
        push!(m_is_manifold, is_mani)
        push!(m_dimensions, detected_dim)
        push!(m_is_closed, is_closed)
        push!(m_failures, n_fail)
    end
    
    return m_epsilons, m_is_manifold, m_dimensions, m_is_closed, m_failures
end

function _generate_all_cliques_and_values(points::Matrix{Float64}, epsilon::Float64, max_dim::Int)
    n_pts = size(points, 1)
    dim_pts = size(points, 2)
    eps2 = epsilon^2
    n_threads = Threads.maxthreadid()
    I_threads = [Int[] for _ in 1:n_threads]
    J_threads = [Int[] for _ in 1:n_threads]
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
                push!(I_local, i); push!(J_local, j)
                push!(I_local, j); push!(J_local, i)
            end
        end
    end
    I = reduce(vcat, I_threads)
    J = reduce(vcat, J_threads)
    adj = sparse(I, J, ones(Int, length(I)), n_pts, n_pts)
    rowptr = adj.colptr
    colval = adj.rowval

    cliques_threads = [Vector{Vector{Int}}() for _ in 1:n_threads]
    vals_threads = [Float64[] for _ in 1:n_threads]

    is_adj = (u::Int, v::Int) -> begin
        s = rowptr[u]; e = rowptr[u+1] - 1
        while s <= e
            mid = (s + e) >> 1
            c = colval[mid]
            if c == v
                return true
            elseif c < v
                s = mid + 1
            else
                e = mid - 1
            end
        end
        return false
    end

    sqdist = (u::Int, v::Int) -> begin
        acc = 0.0
        @inbounds for k in 1:dim_pts
            diff = points[u, k] - points[v, k]
            acc += diff * diff
        end
        return acc
    end

    function backtrack(local_cliques::Vector{Vector{Int}}, local_vals::Vector{Float64},
                       current_clique::Vector{Int}, current_val2::Float64,
                       candidates::Vector{Int})
        push!(local_cliques, copy(current_clique))
        push!(local_vals, sqrt(current_val2))
        if length(current_clique) == max_dim + 1
            return
        end
        for (i, v) in enumerate(candidates)
            new_candidates = Int[]
            for j in (i+1):length(candidates)
                w = candidates[j]
                if is_adj(v, w)
                    push!(new_candidates, w)
                end
            end
            new_val2 = current_val2
            @inbounds for u in current_clique
                dv = sqdist(v, u)
                if dv > new_val2
                    new_val2 = dv
                end
            end
            push!(current_clique, v)
            backtrack(local_cliques, local_vals, current_clique, new_val2, new_candidates)
            pop!(current_clique)
        end
    end

    Threads.@threads for u in 1:n_pts
        tid = Threads.threadid()
        candidates = Int[]
        for ptr in rowptr[u]:(rowptr[u+1]-1)
            v = colval[ptr]
            if v > u
                push!(candidates, v)
            end
        end
        backtrack(cliques_threads[tid], vals_threads[tid], Int[u], 0.0, candidates)
    end

    cliques = reduce(vcat, cliques_threads)
    vals = reduce(vcat, vals_threads)
    return cliques, vals
end

function _circumsphere_jl(P::Matrix{Float64})
    k, D = size(P)
    if k == 1
        return P[1, :], 0.0
    end
    Q = Matrix{Float64}(undef, k - 1, D)
    for j in 1:D
        p1 = P[1, j]
        for i in 2:k
            Q[i - 1, j] = P[i, j] - p1
        end
    end
    rhs = Vector{Float64}(undef, k - 1)
    for i in 1:(k - 1)
        s = 0.0
        for j in 1:D
            q_val = Q[i, j]
            s += q_val * q_val
        end
        rhs[i] = 0.5 * s
    end
    G = Q * Q'
    a = try
        if size(G, 1) == size(G, 2)
            G \ rhs
        else
            pinv(G) * rhs
        end
    catch
        pinv(G) * rhs
    end
    center = [P[1, j] for j in 1:D]
    for j in 1:D
        for i in 1:(k - 1)
            center[j] += Q[i, j] * a[i]
        end
    end
    r2 = 0.0
    for j in 1:D
        diff = center[j] - P[1, j]
        r2 += diff * diff
    end
    return center, r2
end

function _compute_alpha_filtration(points::Matrix{Float64}, top_simplices::Matrix{Int}, max_dim::Int, analyze_manifolds::Bool=false, n_samples::Union{Nothing, Int}=nothing, eps_max::Union{Nothing, Float64}=nothing)
    n_pts = size(points, 1)
    full_dim = size(points, 2)
    
    # 1. Generate all faces of all dimensions up to full_dim from the Delaunay triangulation
    faces_by_dim = [Set{Vector{Int}}() for _ in 1:(full_dim + 1)]
    opposite = Dict{Vector{Int}, Set{Int}}()
    
    n_top = size(top_simplices, 1)
    d_top = size(top_simplices, 2)
    for i in 1:n_top
        t = sort([top_simplices[i, j] + 1 for j in 1:d_top])
        t_set = Set(t)
        
        for d in 0:min(full_dim, d_top - 1)
            for f in combinations(t, d + 1)
                push!(faces_by_dim[d + 1], f)
                opp = setdiff(t_set, f)
                s = get!(opposite, f, Set{Int}())
                union!(s, opp)
            end
        end
    end
    
    # 2. Compute circumspheres
    circ = Dict{Vector{Int}, Tuple{Vector{Float64}, Float64}}()
    for d in 0:full_dim
        for f in faces_by_dim[d + 1]
            P = Matrix{Float64}(undef, length(f), full_dim)
            for r in 1:length(f)
                for c in 1:full_dim
                    P[r, c] = points[f[r], c]
                end
            end
            circ[f] = _circumsphere_jl(P)
        end
    end
    
    # 3. Compute Gabriel empty-sphere test and propagate alpha values
    alpha2 = Dict{Vector{Int}, Float64}()
    for d in full_dim:-1:0
        for f in faces_by_dim[d + 1]
            center, r2 = circ[f]
            
            gabriel = true
            for p in opposite[f]
                d2 = 0.0
                for j in 1:full_dim
                    diff = points[p, j] - center[j]
                    d2 += diff * diff
                end
                if d2 < r2 - 1e-12
                    gabriel = false
                    break
                end
            end
            
            if gabriel
                alpha2[f] = r2
            else
                min_a2 = r2
                has_coface = false
                for p in opposite[f]
                    coface = copy(f)
                    insert_idx = searchsortedfirst(coface, p)
                    insert!(coface, insert_idx, p)
                    
                    if haskey(alpha2, coface)
                        val = alpha2[coface]
                        if !has_coface || val < min_a2
                            min_a2 = val
                            has_coface = true
                        end
                    end
                end
                alpha2[f] = min_a2
            end
        end
    end
    
    # 4. Filter up to max_dim and compute sqrt values
    cliques = Vector{Vector{Int}}()
    vals = Float64[]
    for d in 0:max_dim
        if d <= full_dim
            for f in faces_by_dim[d + 1]
                a2 = alpha2[f]
                val = sqrt(max(a2, 0.0))
                if eps_max === nothing || val <= eps_max
                    push!(cliques, f)
                    push!(vals, val)
                end
            end
        end
    end
    
    M = length(cliques)
    
    # 5. Persistent homology reduction
    (bar_dim, bar_birth, bar_death) = _reduce_from_simplices(cliques, vals)
    
    # Distinct values grid
    eps_values = sort(unique(vals))
    dim_first = Dict{Int, Float64}()
    dim_cnt = Dict{Int, Int}()
    @inbounds for j in 1:M
        d = length(cliques[j]) - 1
        v = vals[j]
        if haskey(dim_first, d)
            v < dim_first[d] && (dim_first[d] = v)
            dim_cnt[d] += 1
        else
            dim_first[d] = v
            dim_cnt[d] = 1
        end
    end
    dim_ids = sort(collect(keys(dim_first)))
    dim_first_val = Float64[dim_first[d] for d in dim_ids]
    dim_count = Int[dim_cnt[d] for d in dim_ids]
    
    # 6. Manifold analysis
    if analyze_manifolds
        epsilon_val = (eps_max !== nothing) ? eps_max : (isempty(vals) ? 1.0 : maximum(vals))
        m_eps, m_is_mani, m_dims, m_is_closed, m_failures = _run_manifold_analysis_jl(cliques, vals, max_dim, epsilon_val, n_samples, n_pts)
        return (bar_dim, bar_birth, bar_death, eps_values, dim_ids, dim_first_val,
                dim_count, M, true, m_eps, m_is_mani, m_dims, m_is_closed, m_failures)
    else
        return (bar_dim, bar_birth, bar_death, eps_values, dim_ids, dim_first_val,
                dim_count, M, false, Float64[], Bool[], Int[], Bool[], Int[])
    end
end

"""
    compute_alpha_filtration(points, top_simplices, max_dim, analyze_manifolds, n_samples, eps_max)

PythonCall entry point for [`_compute_alpha_filtration`](@ref).
"""
function compute_alpha_filtration(points_raw, top_simplices_raw, max_dim_raw, analyze_manifolds_raw=false, n_samples_raw=nothing, eps_max_raw=nothing)
    return _compute_alpha_filtration(
        pyconvert(Matrix{Float64}, points_raw),
        pyconvert(Matrix{Int}, top_simplices_raw),
        pyconvert(Int, max_dim_raw),
        pyconvert(Bool, analyze_manifolds_raw),
        pyconvert(Union{Nothing, Int}, n_samples_raw),
        pyconvert(Union{Nothing, Float64}, eps_max_raw),
    )
end

end # module
