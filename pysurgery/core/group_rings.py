from typing import Callable, Dict, Optional
import warnings
from .exceptions import GroupRingError
from .exact_algebra import normalize_word_token
from ..bridge.julia_bridge import julia_engine


class GroupRingElement:
    """An element of the integral group ring ℤ[G].

    Overview:
        A GroupRingElement represents a formal sum Σ a_g * g, where a_g ∈ ℤ and g ∈ G.
        It provides the foundation for algebraic surgery theory by allowing 
        computations over fundamental group rings, supporting addition, 
        multiplication, and the standard involution (anti-automorphism).

    Key Concepts:
        - **Formal Sum**: Linear combination of group elements with integer coefficients.
        - **Sparse Representation**: Only non-zero coefficients are stored in a dictionary.
        - **Group Law**: The multiplication g₁ * g₂ = g₃ defined by the underlying group G.
        - **Involution (Bar Map)**: The map Σ a_g * g ↦ Σ a_g * g⁻¹, essential for Hermitian forms.

    Common Workflows:
        1. **Creation** → `GroupRingElement({'a': 1, 'b': -2})`
        2. **Algebra** → `z3 = z1 * z2 + z1.involution()`
        3. **Form Evaluation** → Used in `IntersectionForm` over ℤ[π₁].

    Coefficient Ring:
        - ℤ (Integers): Standard for integral group rings in surgery theory.

    Attributes:
        coeffs (Dict[str, int]): Mapping from group element labels to their integer coefficients.
        group_order (Optional[int]): The order of G if it is a cyclic group (ℤ/nℤ).
        group_law (Optional[Callable]): Functional definition of the group multiplication.
        inverse_law (Optional[Callable]): Functional definition of the group inverse.
        mul_table (Optional[Dict]): Precomputed multiplication table for finite groups.
    """

    @staticmethod
    def _normalize_key(g: str) -> str:
        """Normalize group-element labels to a canonical key.

        Args:
            g (str): The group element label.

        Returns:
            str: The normalized key.
        """
        gs = str(g).strip()
        try:
            gs = normalize_word_token(gs)
        except ValueError:
            pass
        if gs in {"e", "1", "g0", "g_0"}:
            return "1"
        return gs

    @classmethod
    def _parse_cyclic_power(cls, g: str, group_order: int) -> int:
        """Parse a cyclic generator label and return exponent modulo group order.

        Args:
            g (str): The generator label.
            group_order (int): The order of the cyclic group.

        Returns:
            int: The exponent modulo group order.

        Raises:
            GroupRingError: If the generator label is unsupported.
        """
        gn = cls._normalize_key(g)
        if gn == "1":
            return 0
        if gn.startswith("g_"):
            return int(gn.split("_", 1)[1]) % group_order
        if gn.startswith("g"):
            return int(gn[1:]) % group_order
        raise GroupRingError(f"Unsupported cyclic generator label '{g}'.")

    def __init__(
        self,
        coeffs: Dict[str, int],
        group_order: Optional[int] = None,
        group_law: Optional[Callable[[str, str], str]] = None,
        inverse_law: Optional[Callable[[str], str]] = None,
        mul_table: Optional[Dict[str, Dict[str, str]]] = None,
    ):
        """Create a sparse group-ring element with normalized coefficients.

        Args:
            coeffs (Dict[str, int]): Mapping from group elements to coefficients.
            group_order (Optional[int]): The order of the group, if cyclic.
            group_law (Optional[Callable[[str, str], str]]): The group multiplication law.
            inverse_law (Optional[Callable[[str], str]]): The group inverse law.
            mul_table (Optional[Dict[str, Dict[str, str]]]): Multiplication table for finite groups.
        """
        normalized = {}
        for g, c in coeffs.items():
            if c == 0:
                continue
            key = self._normalize_key(g)
            normalized[key] = normalized.get(key, 0) + int(c)
        self.coeffs = {g: c for g, c in normalized.items() if c != 0}
        self.group_order = group_order
        self.group_law = group_law
        self.inverse_law = inverse_law
        self.mul_table = mul_table

    def __add__(self, other: "GroupRingElement") -> "GroupRingElement":
        """Add two group ring elements.

        What is Being Computed?:
            The pointwise sum of coefficients: (Σ a_g * g) + (Σ b_g * g) = Σ (a_g + b_g) * g.

        Algorithm:
            1. Copy the coefficients of the first element.
            2. Iterate through the second element's coefficients, adding them to the copy.
            3. Filter out zero coefficients to maintain sparsity.

        Preserved Invariants:
            - Module structure over ℤ.
            - The underlying group G remains unchanged.

        Args:
            other (GroupRingElement): The element to add.

        Returns:
            GroupRingElement: The resulting sum.

        Raises:
            GroupRingError: If the elements belong to different group rings.

        Example:
            z1 = GroupRingElement({'a': 1})
            z2 = GroupRingElement({'a': 1, 'b': 1})
            z3 = z1 + z2  # Represents 2a + b
        """
        if self.group_order != other.group_order:
            raise GroupRingError(
                f"Cannot add elements from different group rings. Group orders |G|={self.group_order} and |H|={other.group_order} do not match."
            )
        if self.group_law is not other.group_law:
            raise GroupRingError(
                "Cannot add elements with different group-law definitions."
            )
        res = self.coeffs.copy()
        for g, c in other.coeffs.items():
            res[g] = res.get(g, 0) + c
        return GroupRingElement(
            res, self.group_order, self.group_law, self.inverse_law, self.mul_table
        )

    def __mul__(self, other: "GroupRingElement") -> "GroupRingElement":
        """Multiply two group ring elements.

        What is Being Computed?:
            The Cauchy product in the group ring: (Σ a_g * g) * (Σ b_h * h) = Σ (a_g * b_h) * (g * h).

        Algorithm:
            Delegates to `multiply()` which selects the backend (Julia or Python fallback).
            It iterates over all pairs (g, h) from the supports, computes g*h, 
            and accumulates the products of their coefficients.

        Preserved Invariants:
            - Ring structure of ℤ[G].
            - Identity element '1' (if present in the group law).

        Args:
            other (GroupRingElement): The element to multiply by.

        Returns:
            GroupRingElement: The resulting product.

        Use When:
            - Computing intersection forms with ℤ[π₁] coefficients.
            - Evaluating L-theory obstructions.
            - General algebraic manipulation of group ring elements.

        Example:
            z1 = GroupRingElement({'a': 1})
            z2 = GroupRingElement({'a_inv': 1})
            z3 = z1 * z2  # Represents the identity '1' if a * a_inv = 1
        """
        if self.group_order != other.group_order:
            raise GroupRingError("Cannot multiply elements from different group rings.")
        if self.group_law is not other.group_law:
            raise GroupRingError(
                "Cannot multiply elements with different group-law definitions."
            )
        if self.mul_table is not other.mul_table:
            raise GroupRingError(
                "Cannot multiply elements with different finite-group multiplication tables."
            )
        return self.multiply(other, backend="auto")

    def multiply(self, other: "GroupRingElement", backend: str = "auto") -> "GroupRingElement":
        """Multiply two group ring elements with backend selection."""
        # Normalize backend
        backend_norm = str(backend).lower().strip()
        use_julia = (backend_norm == "julia") or (backend_norm == "auto" and julia_engine.available)

        if self.group_law is not None:
            res: Dict[str, int] = {}
            for g1, c1 in self.coeffs.items():
                for g2, c2 in other.coeffs.items():
                    g = self._normalize_key(self.group_law(g1, g2))
                    res[g] = res.get(g, 0) + c1 * c2
            return GroupRingElement(
                res, self.group_order, self.group_law, self.inverse_law, self.mul_table
            )

        if self.mul_table is not None:
            res: Dict[str, int] = {}
            for g1, c1 in self.coeffs.items():
                row = self.mul_table.get(self._normalize_key(g1), {})
                for g2, c2 in other.coeffs.items():
                    g = self._normalize_key(row.get(self._normalize_key(g2), ""))
                    if not g:
                        raise GroupRingError(
                            f"Missing multiplication table entry for ({g1}, {g2})."
                        )
                    res[g] = res.get(g, 0) + c1 * c2
            return GroupRingElement(
                res, self.group_order, self.group_law, self.inverse_law, self.mul_table
            )

        if self.group_order is None:
            raise GroupRingError(
                "Group order must be specified for exact ring multiplication, unless group_law is provided."
            )

        if use_julia:
            try:
                res_coeffs = julia_engine.group_ring_multiply(
                    self.coeffs, other.coeffs, self.group_order
                )
                return GroupRingElement(
                    res_coeffs,
                    self.group_order,
                    self.group_law,
                    self.inverse_law,
                    self.mul_table,
                )
            except Exception as e:
                if backend_norm == "julia":
                    raise e

        # Python fallback...

        warnings.warn(
            "Group-ring multiplication fallback in `GroupRingElement.__mul__`: Julia backend unavailable; "
            "install/enable Julia for significantly faster exact multiplication on larger supports."
        )

        # Pure Python fallback for cyclic groups generated by g.
        res: Dict[str, int] = {}
        try:
            for g1, c1 in self.coeffs.items():
                p1 = self._parse_cyclic_power(g1, self.group_order)
                for g2, c2 in other.coeffs.items():
                    p2 = self._parse_cyclic_power(g2, self.group_order)
                    p = (p1 + p2) % self.group_order
                    key = "1" if p == 0 else f"g_{p}"
                    res[key] = res.get(key, 0) + c1 * c2
            return GroupRingElement(
                res, self.group_order, self.group_law, self.inverse_law, self.mul_table
            )
        except GroupRingError as e:
            raise GroupRingError(
                "Non-cyclic exact Group Ring multiplication requires Julia bridge. "
                f"Details: {e!r}"
            )

    def involution(self) -> "GroupRingElement":
        """Compute the standard involution (bar map) on the group ring.

        What is Being Computed?:
            The anti-automorphism z ↦ z̄ defined by Σ a_g * g ↦ Σ a_g * g⁻¹.

        Algorithm:
            1. Iterate through each group element g in the support.
            2. Compute its inverse g⁻¹ using `inverse_law` or cyclic group arithmetic.
            3. Construct a new GroupRingElement with the same coefficients but inverted group elements.

        Preserved Invariants:
            - Norms in Hermitian forms (z * z̄).
            - Real parts (fixed points of the involution).

        Returns:
            GroupRingElement: The involuted element (z̄).

        Raises:
            GroupRingError: If no inverse law or group order is provided for non-identity elements.

        Use When:
            - Defining Hermitian or anti-Hermitian forms.
            - Computing the adjoint of a matrix over ℤ[G].
            - Checking for self-adjoint elements.

        Example:
            z = GroupRingElement({'g': 1, 'h': 2})
            z_bar = z.involution()  # Represents g⁻¹ + 2h⁻¹
        """
        if self.inverse_law is None and self.group_order is None:
            raise GroupRingError(
                "Involution a -> a_bar requires either an explicit inverse law or a cyclic group order. "
                "This is mathematically mandatory for defining Hermitian forms over group rings."
            )

        result = {}
        for g, c in self.coeffs.items():
            gn = self._normalize_key(g)
            if gn == "1":
                result["1"] = result.get("1", 0) + c
            elif self.inverse_law is not None:
                inv_g = self._normalize_key(self.inverse_law(gn))
                result[inv_g] = result.get(inv_g, 0) + c
            elif gn.startswith("g"):
                try:
                    power = int(gn.split("_")[1] if "_" in gn else gn[1:])
                    inv_power = (self.group_order - power) % self.group_order
                    inv_g = (
                        "1"
                        if inv_power == 0
                        else (f"g_{inv_power}" if "_" in gn else f"g{inv_power}")
                    )
                    result[inv_g] = result.get(inv_g, 0) + c
                except (ValueError, IndexError, GroupRingError):
                    raise GroupRingError(
                        f"Cannot compute involution for generator '{gn}' without explicit group inverse structure."
                    )
            else:
                raise GroupRingError(
                    f"Cannot compute involution for non-cyclic generator '{gn}' without explicit inverse law."
                )
        return GroupRingElement(
            result, self.group_order, self.group_law, self.inverse_law, self.mul_table
        )
