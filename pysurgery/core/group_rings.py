from typing import Dict, Optional
from .exceptions import GroupRingError
from ..bridge.julia_bridge import julia_engine

class GroupRingElement:
    """
    Element of the group ring Z[G].
    Represented as a mapping from group elements to integer coefficients.
    """
    def __init__(self, coeffs: Dict[str, int], group_order: Optional[int] = None):
        self.coeffs = {g: c for g, c in coeffs.items() if c != 0}
        self.group_order = group_order

    def __add__(self, other: 'GroupRingElement') -> 'GroupRingElement':
        if self.group_order != other.group_order:
            raise GroupRingError(f"Cannot add elements from different group rings. Group orders |G|={self.group_order} and |H|={other.group_order} do not match.")
        res = self.coeffs.copy()
        for g, c in other.coeffs.items():
            res[g] = res.get(g, 0) + c
        return GroupRingElement(res, self.group_order)
        
    def __mul__(self, other: 'GroupRingElement') -> 'GroupRingElement':
        if self.group_order != other.group_order:
            raise GroupRingError("Cannot multiply elements from different group rings.")
        if self.group_order is None:
            raise GroupRingError("Group order must be specified for exact ring multiplication.")
        
        if julia_engine.available:
            res_coeffs = julia_engine.group_ring_multiply(self.coeffs, other.coeffs, self.group_order)
            return GroupRingElement(res_coeffs, self.group_order)
        else:
            raise GroupRingError("Exact Group Ring multiplication requires Julia bridge.")

    def involution(self) -> 'GroupRingElement':
        """
        The standard involution bar: Z[G] -> Z[G] mapping g to g^-1.
        Used to define Hermitian forms over group rings.
        """
        if self.group_order is None:
            raise GroupRingError("Involution a -> a_bar requires the group order to establish the mapping g -> g^-1. "
                                 "This is mathematically mandatory for defining Hermitian forms over group rings.")
        
        result = {}
        for g, c in self.coeffs.items():
            if g == 'e' or g == '1':
                result[g] = c
            elif g.startswith('g'):
                try:
                    power = int(g[1:])
                    inv_power = (self.group_order - power) % self.group_order
                    inv_g = '1' if inv_power == 0 else f'g{inv_power}'
                    result[inv_g] = result.get(inv_g, 0) + c
                except (ValueError, GroupRingError):
                    # For non-cyclic presentations, involution evaluates dynamically.
                    result[f"{g}^-1"] = c
            else:
                # Base element involution
                result[f"{g}^-1"] = c
        return GroupRingElement(result, self.group_order)
