from __future__ import annotations

from typing import Any

import numpy as np


def coerce_int_matrix(matrix: Any, *, name: str = "matrix") -> np.ndarray:
    """Return a 2D int64 matrix and raise clear errors for invalid input.

    Args:
        matrix (Any): The input matrix-like object to coerce.
        name (str): The name of the matrix for error messages. Defaults to "matrix".

    Returns:
        np.ndarray: A 2D array of type int64.

    Raises:
        ValueError: If the input is not a 2D array or contains non-integer values.
    """
    arr = np.asarray(matrix)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2D array-like object")
    if arr.size == 0:
        return arr.astype(np.int64)

    if np.issubdtype(arr.dtype, np.integer):
        return arr.astype(np.int64, copy=False)

    arr_float = np.asarray(arr, dtype=np.float64)
    rounded = np.rint(arr_float)
    if not np.allclose(arr_float, rounded, atol=0.0):
        raise ValueError(f"{name} must contain integer-valued entries")
    return rounded.astype(np.int64)


def normalize_word_token(token: str) -> str:
    """Normalize a free-group token into canonical g or g^-1 form.

    Args:
        token (str): The group word token to normalize.

    Returns:
        str: The normalized token.

    Raises:
        ValueError: If the token is empty or represents an invalid inverse.
    """
    t = str(token).strip()
    if not t:
        raise ValueError("Group word token cannot be empty")
    if t.endswith("^-1"):
        base = t[:-3].strip()
        if not base:
            raise ValueError("Invalid inverse token")
        return f"{base}^-1"
    if t.endswith("-1") and "^" not in t:
        base = t[:-2].strip()
        if base:
            return f"{base}^-1"
    return t


def validate_group_descriptor(descriptor: str) -> tuple[bool, str]:
    """Validate supported descriptor grammar used by high-level APIs.

    Args:
        descriptor (str): The group descriptor string to validate.

    Returns:
        tuple[bool, str]: A tuple containing a boolean indicating validity and
            a status message ("ok" or an error message).
    """
    d = str(descriptor).strip()
    if not d:
        return False, "Group descriptor is empty"
    if d in {"1", "Z", "trivial", "e"}:
        return True, "ok"

    factors = [f.strip() for f in d.split("x") if f.strip()]
    if factors and len(factors) > 1:
        for f in factors:
            ok, msg = validate_group_descriptor(f)
            if not ok:
                return False, f"Invalid product factor '{f}': {msg}"
        return True, "ok"

    if d.startswith("Z_"):
        tail = d.split("_", 1)[1]
        if tail.isdigit() and int(tail) > 1:
            return True, "ok"
        return False, "Finite cyclic descriptors must be of the form Z_n with n>1"

    return False, "Unsupported descriptor grammar"
