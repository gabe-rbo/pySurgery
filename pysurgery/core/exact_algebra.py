from __future__ import annotations

from typing import Any

import numpy as np


def coerce_int_matrix(matrix: Any, *, name: str = "matrix") -> np.ndarray:
    """Return a 2D int64 matrix and raise clear errors for invalid input.

    What is Being Computed?:
        Coerces arbitrary array-like input into a strictly typed 2D NumPy array
        with 64-bit integer entries, suitable for exact algebraic computations.

    Algorithm:
        1. Convert input to a NumPy array.
        2. Validate that the dimension is exactly 2.
        3. If already integer-typed, cast to int64.
        4. If float-typed, verify that all entries are exact integers (no fractional part)
           before casting.

    Preserved Invariants:
        None (data type transformation).

    Args:
        matrix: The input matrix-like object to coerce.
        name: The name of the matrix for error messages. Defaults to "matrix".

    Returns:
        np.ndarray: A 2D array of type int64.

    Use When:
        - Preparing data for Smith Normal Form (SNF) or other exact algorithms
        - Validating user-provided matrices in high-level APIs
        - Ensuring integer consistency across different array backends

    Example:
        M = coerce_int_matrix([[1, 2], [3, 4]])
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

    What is Being Computed?:
        Converts various string representations of group generators and their
        inverses into a standard `base` or `base^-1` format.

    Algorithm:
        1. Strip whitespace.
        2. Detect inverse markers like `^-1` or `-1`.
        3. Reformat into canonical `base^-1` if an inverse is detected.
        4. Return the base token otherwise.

    Preserved Invariants:
        - Group element identity: The normalized token represents the same group element.

    Args:
        token: The group word token to normalize.

    Returns:
        str: The normalized token in canonical form.

    Use When:
        - Parsing group presentations or word descriptors
        - Comparing tokens for equality in free group logic
        - Pre-processing user input for fundamental group definitions

    Example:
        token = normalize_word_token("a-1")  # returns "a^-1"
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

    What is Being Computed?:
        Checks if a string follows the allowed grammar for group descriptions
        (e.g., "Z x Z_2", "trivial", "1").

    Algorithm:
        1. Check for trivial group aliases ("1", "Z", "trivial", "e").
        2. Recursively split and validate product groups separated by "x".
        3. Validate finite cyclic group notation "Z_n" where n > 1.

    Preserved Invariants:
        None (grammar validation).

    Args:
        descriptor: The group descriptor string to validate.

    Returns:
        tuple[bool, str]: (is_valid, status_message) where status_message is "ok" or an error.

    Use When:
        - Validating user input in constructor methods for spaces
        - Sanitizing group definitions before passing to algebraic engines

    Example:
        is_ok, msg = validate_group_descriptor("Z x Z_3")
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
