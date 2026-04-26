import numpy as np
from .intersection_forms import IntersectionForm
from .exceptions import CharacteristicClassError


def extract_stiefel_whitney_w2(q: IntersectionForm) -> np.ndarray:
    """Evaluates the 2nd Stiefel-Whitney class w_2 in H^2(M; Z_2) from the intersection form.

    By Wu's formula, the total Stiefel-Whitney class corresponds to the
    Wu class v_2 under the Steenrod Square Sq^i.
    Specifically, over a closed 4-manifold, w_2 is the characteristic element of the
    intersection form over Z_2, meaning:
    Q(x, w_2) = Q(x, x) mod 2  for all x in H_2(M; Z).

    Args:
        q: The 4-manifold's intersection form.

    Returns:
        The Z_2 coefficient vector representing the w_2 class in the H_{k} basis.

    Raises:
        CharacteristicClassError: If the manifold dimension is odd, if it's not unimodular,
            or if inversion over Z_2 fails.
    """
    if q.dimension % 2 != 0:
        raise CharacteristicClassError(
            f"w_2 via the intersection form Wu class is defined here specifically for even-dimensional manifolds. "
            f"Received dimension {q.dimension}."
        )

    # We are looking for a vector w in {0, 1}^n such that:
    # Q(x, w) = Q(x, x) mod 2 for all x

    # Since this holds for all x, it must hold for the standard basis vectors e_i
    # For e_i, Q(e_i, e_i) is just the i-th diagonal element of Q.
    # Q(e_i, w) is the i-th element of Q * w.

    # So we need to solve the linear system: Q * w = diag(Q)  over the field GF(2).

    import sympy as sp

    n = q.matrix.shape[0]
    if n == 0:
        return np.zeros(0, dtype=int)

    det_q = q.determinant()
    if abs(det_q) != 1:
        raise CharacteristicClassError(
            f"Intersection form must be unimodular (det = +/-1) for Wu-class extraction; got det = {det_q}."
        )

    # Convert Q to mod 2
    Q_mod2 = q.matrix % 2

    # Extract diagonal mod 2
    diag_mod2 = np.diag(q.matrix) % 2

    # We solve Q_mod2 * w = diag_mod2 over GF(2)
    # Since Q is unimodular, its determinant is +/- 1, so det(Q_mod2) = 1 mod 2.
    # Thus Q_mod2 is always invertible over GF(2).

    # Find inverse over GF(2) using SymPy
    sym_Q = sp.Matrix(Q_mod2)
    det_mod2 = int(sym_Q.det()) % 2
    if det_mod2 == 0:
        raise CharacteristicClassError(
            "Intersection form is not invertible over Z_2. This contradicts unimodularity over Z and indicates degenerate input data."
        )
    try:
        sym_Q_inv = sym_Q.inv_mod(2)
        w2_sym = (sym_Q_inv * sp.Matrix(diag_mod2)) % 2
        w2 = np.array(w2_sym).astype(int).flatten()
        return w2
    except (ValueError, ZeroDivisionError) as e:
        raise CharacteristicClassError(
            f"Intersection form inversion over Z_2 failed ({e!r}). The manifold may violate closed-unimodular assumptions."
        )


def check_spin_structure(q: IntersectionForm) -> str:
    """Uses the 2nd Stiefel-Whitney class to mathematically prove if the manifold is Spin.

    A manifold admits a Spin structure if and only if w_2 = 0.

    Args:
        q: The 4-manifold's intersection form.

    Returns:
        A string describing the Spin structure result and w_2 coefficients.
    """
    w2 = extract_stiefel_whitney_w2(q)

    if np.all(w2 == 0):
        return "w_2 = 0. The manifold admits a Spin structure. (Notice that Q is Even/Type II)."
    else:
        # Format the non-zero coefficients
        basis_elements = [f"e_{i}" for i, val in enumerate(w2) if val == 1]
        return f"w_2 = {' + '.join(basis_elements)} != 0. The manifold is Non-Spin (Type I). The tangent bundle is twisted."


def extract_pontryagin_p1(q: IntersectionForm) -> int:
    """Evaluates the first Pontryagin class.

    By the Hirzebruch Signature Theorem for 4-manifolds:
    signature(M) = 1/3 * <p_1(M), [M]>.
    Therefore, the evaluation of the first Pontryagin class on the fundamental class is:
    p_1(M)[M] = 3 * signature(M).

    Args:
        q: The 4-manifold's intersection form.

    Returns:
        The evaluation of the first Pontryagin class on the fundamental class.

    Raises:
        CharacteristicClassError: If the manifold is not 4-dimensional.
    """
    if q.dimension != 4:
        raise CharacteristicClassError(
            "Hirzebruch Signature Theorem p_1 calculation is specifically formulated for 4-manifolds here."
        )
    return 3 * q.signature()


def verify_hirzebruch_signature(q: IntersectionForm, p1_eval: int) -> bool:
    """Verifies if the intersection form matches the geometric vector bundle integrations.

    Checks if 3 * signature == p1_eval.

    Args:
        q: The 4-manifold's intersection form.
        p1_eval: The integration of p_1 over the fundamental class.

    Returns:
        True if the signature theorem holds, False otherwise.
    """
    return extract_pontryagin_p1(q) == p1_eval
