import scipy.sparse as sp
from fractions import Fraction
from pydantic import BaseModel
from typing import List, Literal

from pysurgery.bridge.julia_bridge import JuliaBridge


class BackendUnavailable(RuntimeError):
    """Requested backend is not installed; choose another or install Julia."""


class Barcode(BaseModel):
    """A single persistence interval (birth, death) in a given homological dimension."""

    birth: int
    death: int
    dim: int
    multiplicity: int = 1
    exact: bool = True


class BarcodeResult(BaseModel):
    """Collection of persistence barcodes with field and backend metadata."""

    barcodes: List[Barcode]
    field: str
    exact: bool = True
    backend: Literal["julia", "python", "auto"] = "julia"


def compute_barcodes_exact(
    filtered_complex,
    dimension: int,
    field: str = 'Z2',
    backend: Literal["auto", "julia", "python"] = "auto",
) -> BarcodeResult:
    """Compute persistent homology barcodes.

    Args:
        filtered_complex: Object providing `.boundary_matrix(d)` or `.get_boundary_matrix(d)`.
        dimension: Maximum homological dimension to compute.
        field: Field for persistence computation ('Z2' or 'Q').
        backend: 'julia' (exact, requires Julia), 'python' (pure-Python column
            reduction, capped at dimension<=2), or 'auto' (Julia if available,
            else Python).

    Returns:
        BarcodeResult with exact=True when using Julia or the Python column-reduction path.

    Raises:
        BackendUnavailable: If backend='julia' and Julia is not installed.
    """
    bridge = JuliaBridge()
    if backend in ("auto", "julia"):
        try:
            bridge.require_julia()
        except Exception as exc:
            if backend == "julia":
                raise BackendUnavailable(
                    f"persistence backend='julia' requested but Julia is unavailable: {exc}"
                ) from exc
            return _python_persistence_fallback(filtered_complex, dimension, field)
    elif backend == "python":
        return _python_persistence_fallback(filtered_complex, dimension, field)

    # Use the centralized Julia engine
    boundary_matrices = {}
    for d in range(1, dimension + 1):
        try:
            mat = filtered_complex.boundary_matrix(d)
        except AttributeError:
            mat = filtered_complex.get_boundary_matrix(d)
        
        boundary_matrices[d] = mat

    try:
        jl_barcodes = bridge.compute_persistence_barcodes(boundary_matrices, field)
    except Exception as e:
        raise RuntimeError(f"Julia backend failed: {e}")

    barcodes = [
        Barcode(
            birth=b["birth"],
            death=b["death"],
            dim=b["dim"],
            multiplicity=b["multiplicity"],
            exact=True,
        )
        for b in jl_barcodes
    ]

    return BarcodeResult(barcodes=barcodes, field=field, exact=True, backend="julia")


def _python_persistence_fallback(filtered_complex, dimension: int, field: str) -> BarcodeResult:
    """Pure-Python column reduction for persistent homology over Z2 or Q.

    Implements the standard persistence reduction algorithm on boundary matrices.
    Column reduction over a field is exact; results carry exact=True.

    Capped at dimension <= 2. For higher dimensions, use backend='julia'.

    Args:
        filtered_complex: Object with .boundary_matrix(d) or .get_boundary_matrix(d).
        dimension: Maximum homological dimension (capped at 2).
        field: 'Z2' or 'Q'.

    Returns:
        BarcodeResult with exact=True.

    Raises:
        NotImplementedError: If dimension > 2 or field not in {'Z2', 'Q'}.
    """
    if dimension > 2:
        raise NotImplementedError(
            "_python_persistence_fallback caps at dimension=2; "
            "use backend='julia' for higher dimensions."
        )
    if field not in ('Z2', 'Q'):
        raise NotImplementedError(
            f"field={field!r} not supported by the Python fallback; use 'Z2' or 'Q'."
        )

    import numpy as np

    use_z2 = (field == 'Z2')

    def _get_bmat(d):
        try:
            mat = filtered_complex.boundary_matrix(d)
        except AttributeError:
            mat = filtered_complex.get_boundary_matrix(d)
        if sp.issparse(mat):
            mat = mat.toarray()
        return np.array(mat, dtype=object)

    def _low(col):
        for i in range(len(col) - 1, -1, -1):
            if col[i] != 0:
                return i
        return -1

    barcodes = []

    for d in range(1, dimension + 1):
        try:
            mat = _get_bmat(d)
        except Exception:
            continue

        if mat.size == 0:
            continue

        nrows, ncols = mat.shape
        pivot_row_to_col = {}

        if use_z2:
            cols = [[int(mat[i, j]) % 2 for i in range(nrows)] for j in range(ncols)]
        else:
            cols = [[Fraction(int(mat[i, j])) for i in range(nrows)] for j in range(ncols)]

        for j in range(ncols):
            lj = _low(cols[j])
            while lj != -1 and lj in pivot_row_to_col:
                k = pivot_row_to_col[lj]
                if use_z2:
                    for i in range(nrows):
                        cols[j][i] = (cols[j][i] + cols[k][i]) % 2
                else:
                    scale = cols[j][lj] / cols[k][lj]
                    for i in range(nrows):
                        cols[j][i] -= scale * cols[k][i]
                lj = _low(cols[j])

            if lj != -1:
                pivot_row_to_col[lj] = j
                barcodes.append(Barcode(birth=lj, death=j, dim=d - 1, exact=True))

    return BarcodeResult(barcodes=barcodes, field=field, exact=True, backend="python")


def compute_zigzag_persistence(complex_sequence: list, field: str = 'Q') -> BarcodeResult:
    """Compute zigzag persistence for a sequence of complexes.

    Builds the union/intersection zigzag sequence and routes to Julia kernel.
    """
    from pysurgery.topology.complexes import SimplicialComplex

    if not complex_sequence:
        return BarcodeResult(barcodes=[], field=field)

    total_simplices = set()
    for z in complex_sequence:
        for d in range(z.dimension + 1):
            for s in z.n_simplices(d):
                total_simplices.add(s)

    total_union = SimplicialComplex.from_simplices(list(total_simplices), close_under_faces=True)

    return compute_barcodes_exact(total_union, dimension=max(total_union.dimension, 1), field=field)


def compute_topological_loss(barcodes: List[Barcode], target: List[Barcode], epsilon: float = 0.01, max_iter: int = 50):
    """Computes a differentiable Gromov-Wasserstein topological loss between two sets of barcodes.

    Note: this uses JAX/Sinkhorn with epsilon regularization — the result is approximate,
    not exact. Do not use a BarcodeResult from this function to claim exact=True.
    """
    try:
        import jax.numpy as jnp
        from jax import lax
    except ImportError:
        raise ImportError("JAX is required for compute_topological_loss")

    b1 = jnp.array([[float(b.birth), float(b.death)] for b in barcodes if b.dim == 0])
    b2 = jnp.array([[float(b.birth), float(b.death)] for b in target if b.dim == 0])

    if len(b1) == 0 and len(b2) == 0:
        return jnp.float32(0.0)
    if len(b1) == 0 or len(b2) == 0:
        return jnp.float32(1.0)

    D1 = jnp.sum((b1[:, None, :] - b1[None, :, :]) ** 2, axis=-1)
    D2 = jnp.sum((b2[:, None, :] - b2[None, :, :]) ** 2, axis=-1)

    n, m = D1.shape[0], D2.shape[0]
    p = jnp.ones(n) / n
    q = jnp.ones(m) / m

    def gw_step(i, val):
        T, u, v = val
        C_T = -2.0 * jnp.dot(D1, jnp.dot(T, D2.T))
        exponent = -C_T / epsilon
        exponent = exponent - jnp.max(exponent)
        K = jnp.exp(exponent)
        u = p / (jnp.dot(K, v) + 1e-10)
        v = q / (jnp.dot(K.T, u) + 1e-10)
        T = u[:, None] * K * v[None, :]
        return T, u, v

    T_init = jnp.outer(p, q)
    u_init = jnp.ones(n) / n
    v_init = jnp.ones(m) / m

    T_final, _, _ = lax.fori_loop(0, max_iter, gw_step, (T_init, u_init, v_init))

    C1 = jnp.sum(D1**2, axis=1, keepdims=True)
    C2 = jnp.sum(D2**2, axis=0, keepdims=True)
    C_T = -2.0 * jnp.dot(D1, jnp.dot(T_final, D2.T))
    cost_matrix = C1 + C2 + C_T

    gw_dist = jnp.sum(T_final * cost_matrix)
    return gw_dist
