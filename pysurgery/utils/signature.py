from typing import Tuple

def signature_via_snf(form) -> Tuple[int, int]:
    """Computes exact (signature, rank) for an IntersectionForm.

    Prefers the exact IntersectionForm.signature() path. Since
    IntersectionForm.rank() relies on floating-point arithmetic, we compute the
    exact rank using compute_exact_sparse_snf.
    """
    from pysurgery.algebra.exact_snf_julia import compute_exact_sparse_snf
    import scipy.sparse as sp
    
    if form.matrix.size == 0:
        return 0, 0

    # Signature via SymPy's LDLdecomposition (exact)
    sig = form.signature()
    
    # Exact integer rank via Smith Normal Form
    snf_res = compute_exact_sparse_snf(sp.csr_matrix(form.matrix))
    rank = len([d for d in snf_res.diagonal if d != 0])
    
    return sig, rank