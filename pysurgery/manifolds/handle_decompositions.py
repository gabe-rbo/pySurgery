import warnings
from collections import defaultdict
from typing import Dict, List, Optional, Any, Tuple, Set, Sequence

import numpy as np
import sympy as sp
from scipy.sparse import csr_matrix, lil_matrix
from pydantic import BaseModel, ConfigDict, Field

from pysurgery.topology.complexes import CWComplex
from pysurgery.core.exceptions import KirbyMoveError


class Handle(BaseModel):
    """Geometric Handle of index k attached to a manifold.

    Overview:
        A Handle corresponds to a disk D^k x D^{n-k} attached along S^{k-1} x D^{n-k}.
        It is the fundamental building block of smooth and PL manifolds,
        corresponding to critical points in Morse theory and cells in a CW complex.

    Attributes:
        index: The index k of the handle.
        cell_id: A unique identifier for the handle.
        framing: Optional framing integer (crucial for 2-handles in 4-manifolds).
        attaching_sphere_data: Topological data describing the attaching map.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    index: int
    cell_id: int
    framing: Optional[int] = None
    attaching_sphere_data: Optional[Dict[str, Any]] = None


class HandleDecomposition(BaseModel):
    """Handle Decomposition of a manifold.

    Overview:
        Represents a manifold built out of handles. Tracks the sequence of
        handles and the homological algebraic structure (boundary matrices)
        to compute invariants like homology and intersection forms.

    Attributes:
        handles: List of handles forming the decomposition.
        boundaries: Reduced (Morse) boundary matrices on the critical chain
            complex.
        exact: ``True`` only when every dimension's boundary was reduced via
            an exact integer Schur complement on the matched-matched block.
        notes: Free-form provenance for the boundary computation.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    handles: List[Handle]
    boundaries: Dict[int, csr_matrix] = Field(default_factory=dict)
    exact: bool = False
    notes: str = ""

    def attach_handle(self, index: int, boundary_vector: Optional[np.ndarray] = None, framing: Optional[int] = None) -> Handle:
        """Attach a new framed handle to the decomposition.

        What is Being Computed?:
            Algebraic tracking of a handle attachment via surgery.
            Adds a handle and updates the cellular chain complex boundaries.

        Args:
            index: The index of the new handle.
            boundary_vector: An array representing the attaching sphere in the cellular basis of (index-1)-handles.
            framing: The framing of the handle.

        Returns:
            The newly attached Handle.
        """
        existing_k = sum(1 for h in self.handles if h.index == index)
        existing_k_minus_1 = sum(1 for h in self.handles if h.index == index - 1)

        handle = Handle(index=index, cell_id=existing_k, framing=framing)
        self.handles.append(handle)

        if boundary_vector is None:
            boundary_vector = np.zeros(existing_k_minus_1, dtype=np.int64)
        else:
            boundary_vector = np.asarray(boundary_vector, dtype=np.int64)
            if len(boundary_vector) != existing_k_minus_1:
                raise ValueError(f"Boundary vector length {len(boundary_vector)} does not match {existing_k_minus_1} existing (k-1)-handles.")

        if index not in self.boundaries:
            if existing_k == 0:
                self.boundaries[index] = csr_matrix((existing_k_minus_1, 1), dtype=np.int64)
                self.boundaries[index][:, 0] = boundary_vector.reshape(-1, 1)
            else:
                self.boundaries[index] = csr_matrix((existing_k_minus_1, 0), dtype=np.int64)
        else:
            mat = self.boundaries[index].tolil()
            if mat.shape[0] < existing_k_minus_1:
                mat.resize((existing_k_minus_1, mat.shape[1]))
            new_col = lil_matrix(boundary_vector.reshape(-1, 1))
            from scipy.sparse import hstack
            self.boundaries[index] = hstack([self.boundaries[index], new_col]).tocsr()

        return handle

    def get_intersection_form(self) -> np.ndarray:
        """Compute the intersection form from 2-handle framings and attachments.

        What is Being Computed?:
            The algebraic intersection form matrix. For a 4-manifold built with one 0-handle
            and a set of 2-handles, the intersection form is given by the linking matrix of
            the attaching spheres, where diagonal entries are the framings.

        Returns:
            A symmetric numpy array.

        Raises:
            KirbyMoveError: If framings are missing or diagram is invalid.
        """
        handles_2 = [h for h in self.handles if h.index == 2]
        n = len(handles_2)
        if n == 0:
            return np.zeros((0, 0), dtype=np.int64)

        mat = np.zeros((n, n), dtype=np.int64)
        for i, hi in enumerate(handles_2):
            if hi.framing is None:
                raise KirbyMoveError(f"Handle 2-cell {hi.cell_id} is missing a framing for intersection form extraction.")
            mat[i, i] = hi.framing

            if hi.attaching_sphere_data and "linking" in hi.attaching_sphere_data:
                for j, lk in hi.attaching_sphere_data["linking"].items():
                    if j < n:
                        mat[i, j] = lk
                        mat[j, i] = lk

        return mat


def _greedy_discrete_morse_matching(
    cw: CWComplex,
) -> Tuple[List[Tuple[Tuple[int, int], Tuple[int, int]]], Set[Tuple[int, int]]]:
    """Perform a greedy acyclic matching on the cells of a CW complex.

    Algorithm:
        1. Iterate through dimensions, looking for free faces of unit
           incidence.
        2. A face F of cell C is free if C is the only unmatched coface of
           F with incidence ±1, and F is unmatched.
        3. Match F and C in the order they were discovered. The match order
           is consistent with a topological sort of the gradient flow,
           which makes the matched-matched block triangularisable for the
           downstream Schur complement reduction.

    Returns:
        matched_pairs: List of ((d-1, face_idx), (d, coface_idx)) tuples,
            in the order they were matched.
        critical_cells: Set of (dim, cell_idx) tuples that remain unmatched.
    """
    matched_set: Set[Tuple[int, int]] = set()
    matched_pairs: List[Tuple[Tuple[int, int], Tuple[int, int]]] = []
    critical: Set[Tuple[int, int]] = set()

    for d in sorted(cw.attaching_maps.keys()):
        mat = cw.attaching_maps[d].tocoo()

        face_to_cofaces: Dict[int, List[int]] = {
            i: [] for i in range(cw.cells.get(d - 1, 0))
        }
        for r, c, v in zip(mat.row, mat.col, mat.data):
            if abs(int(v)) == 1:
                face_to_cofaces[int(r)].append(int(c))

        for f, cofaces in face_to_cofaces.items():
            if (d - 1, f) in matched_set:
                continue
            unmatched_cofaces = [c for c in cofaces if (d, c) not in matched_set]

            if len(unmatched_cofaces) == 1:
                c = unmatched_cofaces[0]
                matched_set.add((d - 1, f))
                matched_set.add((d, c))
                matched_pairs.append(((d - 1, f), (d, c)))

    for d, count in cw.cells.items():
        for i in range(count):
            if (d, i) not in matched_set:
                critical.add((d, i))

    return matched_pairs, critical


def _exact_morse_boundary(
    full_mat: np.ndarray,
    crit_dm1_old: Sequence[int],
    crit_d_old: Sequence[int],
    matched_dm1_old: Sequence[int],
    matched_d_old: Sequence[int],
) -> Tuple[Optional[np.ndarray], Optional[str]]:
    """Compute the Morse boundary block via the integer Schur complement.

    What is Being Computed?:
        For one fixed pair of dimensions (d, d-1), the Morse boundary on
        the critical chain complex,

            ∂^M_d = ∂_CC - ∂_CM · ∂_MM^{-1} · ∂_MC,

        where the splits are critical-cells / matched-cells (matched in
        M_d, the pairs whose coface lives in dimension d).

    Algorithm:
        1. Extract the four blocks of ∂_d from the full incidence matrix
           using ``sympy.Matrix`` so all arithmetic is exact integer/ℚ.
        2. Invert the matched-matched block (sympy Gauss-Jordan).
        3. Compute the Schur complement as a sympy matrix.
        4. Coerce each entry to ℤ; if any entry is not an integer, treat
           the reduction as inexact for that dimension.

    Returns:
        ``(reduced_matrix, error_message)``. ``error_message`` is ``None``
        on success and the result is exact integer; on failure the matrix
        is ``None`` and the message describes the cause.
    """
    if not crit_d_old or not crit_dm1_old:
        return None, "empty critical set on at least one side"

    def block(rows: Sequence[int], cols: Sequence[int]) -> sp.Matrix:
        if not rows or not cols:
            return sp.zeros(len(rows), len(cols))
        return sp.Matrix(
            [[int(full_mat[r, c]) for c in cols] for r in rows]
        )

    bcc = block(crit_dm1_old, crit_d_old)

    if not matched_d_old or not matched_dm1_old:
        return np.array(bcc.tolist(), dtype=np.int64), None

    bcm = block(crit_dm1_old, matched_d_old)
    bmc = block(matched_dm1_old, crit_d_old)
    bmm = block(matched_dm1_old, matched_d_old)

    try:
        bmm_inv = bmm.inv()
    except sp.matrices.exceptions.NonInvertibleMatrixError as exc:
        return None, f"matched-matched block non-invertible: {exc}"
    except Exception as exc:
        return None, f"matched-matched inversion failed: {exc!r}"

    reduced_sym = bcc - bcm * bmm_inv * bmc

    int_rows: List[List[int]] = []
    for row in reduced_sym.tolist():
        int_row: List[int] = []
        for val in row:
            r = sp.nsimplify(val, rational=True)
            if r.is_Integer:
                int_row.append(int(r))
            elif r.is_Rational:
                if int(r.q) == 1:
                    int_row.append(int(r.p))
                else:
                    return None, f"Schur complement produced non-integer entry {r}"
            else:
                return None, f"Schur complement produced non-rational entry {r}"
        int_rows.append(int_row)

    return np.array(int_rows, dtype=np.int64), None


def _build_handle_decomposition_from_cw(cw: CWComplex) -> HandleDecomposition:
    """Internal worker for the CW → HandleDecomposition translation.

    What is Being Computed?:
        Builds a discrete Morse gradient field on the CW complex; the
        unmatched (critical) cells become handles of index equal to their
        cell dimension. The reduced Morse boundary on the critical chain
        complex is computed exactly via a block Schur complement in each
        dimension, which is equivalent to summing incidence-products over
        every gradient path between critical cells.

    Algorithm:
        1. Greedy acyclic matching on the Hasse diagram (free-face
           collapses) yielding a list of pairs (f, c) with dim(c) = dim(f)
           + 1 and ⟨∂c, f⟩ = ±1, plus the set of critical cells.
        2. For each dimension d, partition C_d and C_{d-1} as

               C_d   = C_d^crit       ⊕ M_d^down    (… ⊕ rest)
               C_{d-1} = C_{d-1}^crit ⊕ M_d^up      (… ⊕ rest)

           where ``M_d`` are the pairs in M whose coface sits in dimension
           d; M_d^down ⊂ C_d are the matched cofaces and M_d^up ⊂ C_{d-1}
           are the matched faces.
        3. Compute the four blocks of ∂_d on those splits and form

               ∂^M_d = ∂_CC - ∂_CM · ∂_MM^{-1} · ∂_MC,

           using exact integer/ℚ arithmetic via :mod:`sympy`. Gradient
           paths from a critical d-cell to a critical (d-1)-cell only
           traverse cells in dimensions d and d-1 and only use M_d
           pairs (Forman, 2002, §6), so the per-dimension Schur complement
           recovers the exact Morse boundary.
        4. The matched-matched block is invertible because matched pairs
           have ±1 incidences on the diagonal and the greedy matching
           produces a topologically ordered (acyclic) flow, so the block
           is unit triangularisable in that order.

    Returns:
        :class:`HandleDecomposition`. ``hd.exact`` is ``True`` iff every
        dimension's Schur complement succeeded and yielded an integer
        matrix, in which case ``hd.notes`` records the exact-Morse
        provenance. Otherwise ``hd.exact`` is ``False`` and ``hd.notes``
        records, per dimension, why the reduction fell back to projection.

    References:
        Forman, R. (1998). Morse theory for cell complexes.
            Adv. Math. 134, 90–145.
        Forman, R. (2002). A user's guide to discrete Morse theory.
            Sém. Lothar. Combin. B48c, 35 pp. (see §6 for gradient paths).
        Sköldberg, E. (2006). Morse theory from an algebraic viewpoint.
            Trans. AMS 358, 115–129. (Schur-complement formulation.)
    """
    matched_pairs, critical = _greedy_discrete_morse_matching(cw)

    pairs_by_d: Dict[int, List[Tuple[Tuple[int, int], Tuple[int, int]]]] = defaultdict(list)
    for (face, coface) in matched_pairs:
        pairs_by_d[coface[0]].append((face, coface))

    hd = HandleDecomposition(handles=[])

    for d in sorted(cw.cells.keys()):
        crit_d = sorted([i for (dim, i) in critical if dim == d])
        for new_idx, _old_idx in enumerate(crit_d):
            hd.handles.append(Handle(index=d, cell_id=new_idx))

    exact_throughout = True
    error_notes: List[str] = []

    for d in sorted(cw.attaching_maps.keys()):
        if d <= 0:
            continue
        crit_d_old = sorted([i for (dim, i) in critical if dim == d])
        crit_dm1_old = sorted([i for (dim, i) in critical if dim == d - 1])

        if not crit_d_old or not crit_dm1_old:
            continue

        attach = cw.attaching_maps.get(d)
        if attach is None:
            continue
        full_mat = attach.toarray().astype(np.int64)

        pairs_d = pairs_by_d.get(d, [])
        matched_d_old = [coface[1] for (_, coface) in pairs_d]
        matched_dm1_old = [face[1] for (face, _) in pairs_d]

        reduced, err = _exact_morse_boundary(
            full_mat,
            crit_dm1_old,
            crit_d_old,
            matched_dm1_old,
            matched_d_old,
        )

        if err is not None or reduced is None:
            exact_throughout = False
            error_notes.append(f"dim {d}: {err}")
            reduced = np.array(
                [[int(full_mat[r, c]) for c in crit_d_old] for r in crit_dm1_old],
                dtype=np.int64,
            )

        hd.boundaries[d] = csr_matrix(reduced)

    if exact_throughout:
        hd.exact = True
        hd.notes = "Exact Morse boundary via Schur complement."
    else:
        hd.exact = False
        joined = "; ".join(error_notes)
        hd.notes = (
            "Morse boundary fell back to projection in some dimensions: " + joined
        )

    return hd


def cw_complex_to_handle_decomposition(cw: CWComplex) -> HandleDecomposition:
    """Deprecated: use ``CWComplex.to_handle_decomposition()`` instead."""
    warnings.warn(
        "cw_complex_to_handle_decomposition is deprecated; use "
        "CWComplex.to_handle_decomposition()",
        DeprecationWarning,
        stacklevel=2,
    )
    return cw.to_handle_decomposition()
