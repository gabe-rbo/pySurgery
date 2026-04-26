from typing import Callable, Dict, Optional
import warnings
from .exceptions import GroupRingError
from .exact_algebra import normalize_word_token
from ..bridge.julia_bridge import julia_engine


class GroupRingElement:
    """Element of the group ring Z[G].

    Represented as a mapping from group elements to integer coefficients.
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
        """Add two elements from the same group ring.

        Args:
            other (GroupRingElement): The other group-ring element.

        Returns:
            GroupRingElement: The sum of the two elements.

        Raises:
            GroupRingError: If the elements are from different group rings.
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
        """Multiply two group-ring elements using exact backend or cyclic fallback.

        Args:
            other (GroupRingElement): The other group-ring element.

        Returns:
            GroupRingElement: The product of the two elements.

        Raises:
            GroupRingError: If the elements are from different group rings or
                if multiplication is not defined.
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

        if julia_engine.available:
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
        """The standard involution bar: Z[G] -> Z[G] mapping g to g^-1.

        Used to define Hermitian forms over group rings.

        Returns:
            GroupRingElement: The involuted group-ring element.

        Raises:
            GroupRingError: If the inverse structure is missing.
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
