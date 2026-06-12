"""Point cloud representation, spatial partitioning, and geometric querying tools."""

import numpy as np
from typing import Dict, Tuple, Optional, Union, Any, List


class SpaceBlock:
    """Represents a D-dimensional block (hyperrectangle) of space.
    
    A SpaceBlock defines a bounded region in D-dimensional Euclidean space using
    axis-aligned lower and upper bounds. It is used to define spatial partitions or
    masks for selective geometric transformations in a PointCloud.
    
    Mathematical Foundations:
        A point x = (x_1, x_2, ..., x_D) in R^D is inside the SpaceBlock if and only if
        for every dimension j in {1, 2, ..., D}:
            min_bounds[j] <= x_j <= max_bounds[j]
            
        For a set of N points, this evaluates to a boolean mask of shape (N,) where
        each element is the logical AND across all dimensions of the individual coordinate
        interval containment checks.
    """

    def __init__(self, min_bounds: Any, max_bounds: Any):
        """Initializes a SpaceBlock with coordinate bounds for each dimension.

        Args:
            min_bounds (Any): Array-like of shape (D,) representing the lower coordinate bounds
                for each dimension. Will be cast to a 1D float numpy array.
            max_bounds (Any): Array-like of shape (D,) representing the upper coordinate bounds
                for each dimension. Will be cast to a 1D float numpy array.

        Raises:
            ValueError: If min_bounds and max_bounds have mismatching shapes or are not 1D.
        """
        self.min_bounds = np.asarray(min_bounds, dtype=float)
        self.max_bounds = np.asarray(max_bounds, dtype=float)
        if self.min_bounds.shape != self.max_bounds.shape or self.min_bounds.ndim != 1:
            raise ValueError("Bounds must be 1D arrays of the same shape.")

    def contains(self, points: Any) -> np.ndarray:
        """Determines which points lie within the bounds of this space block.

        Args:
            points (Any): An (N, D) array-like of coordinate points or a PointCloud object to check.
                Single points of shape (D,) are automatically reshaped to (1, D).

        Returns:
            np.ndarray: A boolean array of shape (N,) where entry `i` is True if the `i`-th point
            satisfies the condition `min_bounds[j] <= points[i, j] <= max_bounds[j]` for all dimensions `j`.

        Raises:
            ValueError: If the coordinate dimension of the input points does not match the block dimension.
        """
        pts = np.asarray(points, dtype=float)
        if pts.ndim == 1:
            pts = pts[np.newaxis, :]
        if pts.shape[1] != self.min_bounds.shape[0]:
            raise ValueError(
                f"Points dimension {pts.shape[1]} does not match block dimension {self.min_bounds.shape[0]}"
            )
        return np.all((pts >= self.min_bounds) & (pts <= self.max_bounds), axis=1)

    def __contains__(self, point: Any) -> bool:
        """Enables membership testing using the `in` operator (e.g., `point in space_block`).

        Args:
            point (Any): A single coordinate point of shape (D,).

        Returns:
            bool: True if the point is within the bounds of this space block in all dimensions, otherwise False.
        """
        pt = np.asarray(point, dtype=float)
        if pt.shape != self.min_bounds.shape:
            return False
        return bool(np.all((pt >= self.min_bounds) & (pt <= self.max_bounds)))

    def __repr__(self) -> str:
        """Returns a string representation of the SpaceBlock showing its min and max bounds."""
        return f"SpaceBlock(min_bounds={self.min_bounds.tolist()}, max_bounds={self.max_bounds.tolist()})"


class PointCloud:
    """A D-dimensional point cloud representation supporting advanced deformations and history tracking.

    A PointCloud holds a collection of N points in D-dimensional space, supporting operations like
    rotation, scaling, shearing, twisting, bending, unbending, and tapering. Each operation can be
    confined to specific regions of space using `SpaceBlock` instances. The class also maintains
    a transformation history, enabling sequential undoing and mathematical reversion of operations.

    Mathematical Foundations & Masking:
        Deformations are applied as continuous mappings f: R^D -> R^D.
        When selective masking is enabled, a boolean mask M of shape (N,) is computed using:
            M = (Union_{B in movable_blocks} (B.contains(points))) AND NOT (Union_{S in static_blocks} (S.contains(points)))
        For each point p_i:
            p_i' = f(p_i)   if M[i] is True
            p_i' = p_i      if M[i] is False

        This masking pattern is tracked inside the history log to ensure that the mathematical
        inverse of the transformation is applied only to the points that were actually deformed during
        the forward pass.
    """

    def __init__(
        self,
        points: Any,
        parent: Optional[Any] = None,
        *,
        history: Optional[List[Dict[str, Any]]] = None,
        original_points: Optional[np.ndarray] = None
    ):
        """Initializes a PointCloud instance.

        Args:
            points (Any): An (N, D) array-like structure of floating-point coordinates.
            parent (Optional[Any]): A parent topological structure (e.g., SimplicialComplex)
                to sync coordinate changes to.
            history (Optional[List[Dict[str, Any]]]): A list of dicts detailing prior transformations
                applied to this point cloud.
            original_points (Optional[np.ndarray]): An (N, D) array representing the undeformed
                state of the point cloud coordinates, used for undo operations.

        Raises:
            ValueError: If the input points array is not 2-dimensional.
        """
        self.points = np.asarray(points, dtype=float)
        if self.points.ndim != 2:
            raise ValueError(
                f"Points must be a 2D array of shape (N, D), got shape {self.points.shape}"
            )
        self._parent = parent
        self._history = list(history) if history is not None else []
        self._original_points = (
            np.asarray(original_points, dtype=float).copy()
            if original_points is not None
            else self.points.copy()
        )

    def _update_parent(self, new_points: np.ndarray) -> None:
        """Propagates updated coordinate values to the parent structure if present.

        This method updates the parent's coordinate cache and recalculates any needed
        topological/geometric mappings (e.g., for alpha complexes).

        Args:
            new_points (np.ndarray): The new (N, D) coordinate array to write to the parent.
        """
        if self._parent is not None:
            self._parent._coordinates = new_points
            if hasattr(self._parent, "_generate_point_cloud_mappings"):
                self._parent._generate_point_cloud_mappings(new_points)

    def _get_movable_mask(
        self,
        movable_blocks: Optional[Union[SpaceBlock, List[SpaceBlock], Tuple[SpaceBlock, ...]]] = None,
        static_blocks: Optional[Union[SpaceBlock, List[SpaceBlock], Tuple[SpaceBlock, ...]]] = None
    ) -> np.ndarray:
        """Computes a boolean mask indicating which points are subject to deformation.

        Mathematical Foundations:
            Let M_mov be the union of regions defined by `movable_blocks`. If `movable_blocks` is None,
            M_mov defaults to the entire space R^D.
            Let M_stat be the union of regions defined by `static_blocks`. If `static_blocks` is None,
            M_stat defaults to the empty set.
            The final mask is defined as M = M_mov \\ M_stat.

        Args:
            movable_blocks (Optional[Union[SpaceBlock, List[SpaceBlock], Tuple[SpaceBlock, ...]]]):
                Blocks defining regions where points are allowed to deform.
            static_blocks (Optional[Union[SpaceBlock, List[SpaceBlock], Tuple[SpaceBlock, ...]]]):
                Blocks defining regions where points must remain stationary.

        Returns:
            np.ndarray: A boolean mask of shape (N,) where True indicates a point is movable.
        """
        mask = np.ones(self.num_points, dtype=bool)

        if movable_blocks is not None:
            if isinstance(movable_blocks, SpaceBlock):
                mov_list = [movable_blocks]
            else:
                mov_list = list(movable_blocks)

            mov_mask = np.zeros(self.num_points, dtype=bool)
            for block in mov_list:
                mov_mask |= block.contains(self.points)
            mask &= mov_mask

        if static_blocks is not None:
            if isinstance(static_blocks, SpaceBlock):
                stat_list = [static_blocks]
            else:
                stat_list = list(static_blocks)

            stat_mask = np.zeros(self.num_points, dtype=bool)
            for block in stat_list:
                stat_mask |= block.contains(self.points)
            mask &= ~stat_mask

        return mask

    def _create_deformed_pc(
        self,
        new_points: np.ndarray,
        method_name: str,
        args: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> "PointCloud":
        """Updates coordinates and parent in-place, logs transformation history, and returns self.

        Args:
            new_points (np.ndarray): The deformed coordinate array.
            method_name (str): The name of the deformation method called.
            args (Dict[str, Any]): The arguments supplied to the deformation.
            metadata (Optional[Dict[str, Any]]): Internal execution details (e.g. resolved anchors,
                movable masks) needed for reversion.

        Returns:
            PointCloud: The current PointCloud object mutated in-place.
        """
        self.points = new_points
        self._update_parent(new_points)
        history_entry = {
            "method": method_name,
            "args": args,
        }
        if metadata is not None:
            history_entry["metadata"] = metadata
        self._history.append(history_entry)
        return self

    def list_transformations(self) -> List[Dict[str, Any]]:
        """Lists all transformations applied to this point cloud since its original state.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries, each containing the "method" name
            and the "args" dict used in the forward transformation.
        """
        import copy
        return [{"method": item["method"], "args": copy.deepcopy(item["args"])} for item in self._history]

    def undo(self, indices: Optional[Union[int, List[int]]] = None) -> "PointCloud":
        """Undoes one or more transformations in the history log by mutating coordinates and history in-place.

        An undo is performed by removing the specified transformation entries from the
        history log, resetting the point cloud coordinates to the original coordinates,
        and sequentially re-applying all remaining transformations in the log.

        Args:
            indices (Optional[Union[int, List[int]]]): The index or list of indices of the
                transformations to remove from the history. Supports negative indexing.
                If None, defaults to undoing the very last transformation (-1).

        Returns:
            PointCloud: The current PointCloud object mutated in-place.

        Raises:
            ValueError: If the transformation history is empty.
            IndexError: If any index is out of bounds for the history log.
        """
        if not self._history:
            raise ValueError("No transformations to undo.")

        if indices is None:
            indices_to_remove = {len(self._history) - 1}
        elif isinstance(indices, int):
            idx = indices if indices >= 0 else len(self._history) + indices
            if idx < 0 or idx >= len(self._history):
                raise IndexError("Transformation index out of range.")
            indices_to_remove = {idx}
        else:
            indices_to_remove = set()
            for i in indices:
                idx = i if i >= 0 else len(self._history) + i
                if idx < 0 or idx >= len(self._history):
                    raise IndexError("Transformation index out of range.")
                indices_to_remove.add(idx)

        new_history_entries = [
            item for idx, item in enumerate(self._history)
            if idx not in indices_to_remove
        ]

        # Start from original points on a temporary PointCloud to avoid mutating intermediate states
        temp_pc = PointCloud(self._original_points.copy(), original_points=self._original_points)

        # Re-apply remaining transformations sequentially
        for item in new_history_entries:
            method_name = item["method"]
            args = item["args"]
            method = getattr(temp_pc, method_name)
            method(**args)

        self.points = temp_pc.points
        self._history = new_history_entries
        self._update_parent(self.points)
        return self

    def revert(self) -> "PointCloud":
        """Reverts all transformations by mathematically inverting them in reverse order, mutating self in-place.

        Instead of re-applying transformations from the original state (like `undo`), `revert`
        directly applies the analytical mathematical inverse of each transformation step in
        reverse chronological order. This is highly efficient and operates on the active coordinate
        representation, keeping track of point-level masks to only invert translations on points
        that were actually modified in that step.

        Mathematical Foundations:
            For each transformation step in reversed(history), the inverse function f^-1: R^D -> R^D is
            computed and applied selectively using the stored movable mask:
                p_i^{prev} = f^-1(p_i^{current})   if M[i] is True
                p_i^{prev} = p_i^{current}          if M[i] is False

        Returns:
            PointCloud: The current PointCloud instance reverted in-place with reverted coordinates and an empty history log.

        Raises:
            ValueError: If any transformation in history cannot be mathematically inverted
                (e.g., scaling to diameter with factor=0, spherization with factor=1.0,
                or arbitrary mapping functions via `apply_mapping`).
        """
        if not self._history:
            return self

        current_points = self.points.copy()

        for item in reversed(self._history):
            method = item["method"]
            args = item["args"]
            metadata = item.get("metadata", {})

            inv_points = current_points.copy()

            if method == "translate":
                v = np.asarray(args["translation_vector"], dtype=float)
                inv_points = inv_points - v

            elif method == "rotate":
                angle = args["angle"]
                plane = args["plane"]
                use_degrees = args.get("use_degrees", True)
                resolved_center = metadata["resolved_center"]

                if isinstance(plane, str):
                    plane_lower = plane.lower()
                    mapping = {"xy": (0, 1), "yz": (1, 2), "zx": (2, 0), "xz": (0, 2)}
                    axis1, axis2 = mapping[plane_lower]
                else:
                    axis1, axis2 = plane

                theta = np.radians(-angle) if use_degrees else -angle
                cos_t, sin_t = np.cos(theta), np.sin(theta)
                R = np.array([[cos_t, -sin_t], [sin_t, cos_t]])

                shifted = inv_points - resolved_center
                coords_plane = shifted[:, [axis1, axis2]]
                rotated_plane = coords_plane @ R.T
                inv_points[:, [axis1, axis2]] = rotated_plane + resolved_center[[axis1, axis2]]

            elif method == "scale_to_diameter":
                scale_factor = metadata["scale_factor"]
                resolved_anchor_pt = metadata["resolved_anchor_pt"]
                axis = args.get("axis", None)

                if np.isclose(scale_factor, 0.0):
                    raise ValueError("Cannot revert scale_to_diameter: scale factor is zero.")

                inv_factor = 1.0 / scale_factor
                if axis is None:
                    inv_points = resolved_anchor_pt + inv_factor * (inv_points - resolved_anchor_pt)
                else:
                    coords = inv_points[:, axis]
                    inv_points[:, axis] = resolved_anchor_pt[axis] + inv_factor * (coords - resolved_anchor_pt[axis])

            elif method == "shear":
                factor = args["factor"]
                axis = args["axis"]
                control_axis = args["control_axis"]
                resolved_anchor_pt = metadata["resolved_anchor_pt"]

                control_coords = inv_points[:, control_axis]
                displacement = factor * (control_coords - resolved_anchor_pt[control_axis])
                inv_points[:, axis] -= displacement

            elif method == "twist":
                rate = args["rate"]
                plane = args["plane"]
                control_axis = args.get("control_axis", 2)
                use_degrees = args.get("use_degrees", True)
                resolved_anchor_pt = metadata["resolved_anchor_pt"]

                if isinstance(plane, str):
                    plane_lower = plane.lower()
                    mapping = {"xy": (0, 1), "yz": (1, 2), "zx": (2, 0), "xz": (0, 2)}
                    axis1, axis2 = mapping[plane_lower]
                else:
                    axis1, axis2 = plane

                dist_along_control = inv_points[:, control_axis] - resolved_anchor_pt[control_axis]
                angles = -rate * dist_along_control
                if use_degrees:
                    angles = np.radians(angles)

                cos_vals = np.cos(angles)
                sin_vals = np.sin(angles)

                x = inv_points[:, axis1] - resolved_anchor_pt[axis1]
                y = inv_points[:, axis2] - resolved_anchor_pt[axis2]

                inv_points[:, axis1] = resolved_anchor_pt[axis1] + x * cos_vals - y * sin_vals
                inv_points[:, axis2] = resolved_anchor_pt[axis2] + x * sin_vals + y * cos_vals

            elif method == "bend":
                curvature = args["curvature"]
                axis = args.get("axis", 0)
                control_axis = args.get("control_axis", 1)
                resolved_anchor_pt = metadata["resolved_anchor_pt"]
                theta_seam = metadata.get("theta_seam", np.pi)

                X = inv_points[:, axis] - resolved_anchor_pt[axis]
                Y = inv_points[:, control_axis] - resolved_anchor_pt[control_axis]

                if np.abs(curvature) < 1e-8:
                    inv_points[:, axis] = resolved_anchor_pt[axis] + X + curvature * X * Y
                    inv_points[:, control_axis] = resolved_anchor_pt[control_axis] + Y - 0.5 * curvature * (X**2)
                else:
                    r = 1.0 / curvature
                    sign_c = np.sign(curvature)
                    theta = np.arctan2(X * sign_c, (r - Y) * sign_c)
                    d = np.sqrt(X**2 + (r - Y)**2)
                    theta_shifted = (theta - theta_seam) % (2 * np.pi) - np.pi
                    x_unbent = theta_shifted / curvature
                    y_unbent = r - sign_c * d

                    inv_points[:, axis] = resolved_anchor_pt[axis] + x_unbent
                    inv_points[:, control_axis] = resolved_anchor_pt[control_axis] + y_unbent

            elif method == "unbend":
                curvature = args["curvature"]
                axis = args["axis"]
                control_axis = args["control_axis"]
                resolved_anchor_pt = metadata["resolved_anchor_pt"]
                theta_seam = metadata.get("theta_seam", np.pi)

                x_unbent = inv_points[:, axis] - resolved_anchor_pt[axis]
                y_unbent = inv_points[:, control_axis] - resolved_anchor_pt[control_axis]

                if np.abs(curvature) < 1e-8:
                    inv_points[:, axis] = resolved_anchor_pt[axis] + x_unbent - curvature * x_unbent * y_unbent
                    inv_points[:, control_axis] = resolved_anchor_pt[control_axis] + y_unbent + 0.5 * curvature * (x_unbent**2)
                else:
                    r = 1.0 / curvature
                    theta = curvature * x_unbent + theta_seam + np.pi
                    inv_points[:, axis] = resolved_anchor_pt[axis] + (r - y_unbent) * np.sin(theta)
                    inv_points[:, control_axis] = resolved_anchor_pt[control_axis] + r - (r - y_unbent) * np.cos(theta)

            elif method == "taper":
                factor = args["factor"]
                axis_arg = args.get("axis", None)
                control_axis = args.get("control_axis", 2)
                resolved_anchor_pt = metadata["resolved_anchor_pt"]

                if axis_arg is None:
                    axes_to_scale = [d for d in range(self.dimension) if d != control_axis]
                elif isinstance(axis_arg, int):
                    axes_to_scale = [axis_arg]
                else:
                    axes_to_scale = list(axis_arg)

                h = inv_points[:, control_axis] - resolved_anchor_pt[control_axis]
                multipliers = 1.0 + factor * h
                if np.any(np.isclose(multipliers, 0.0)):
                    raise ValueError("Cannot revert taper: some scale multipliers are zero.")

                for d in axes_to_scale:
                    inv_points[:, d] = resolved_anchor_pt[d] + (inv_points[:, d] - resolved_anchor_pt[d]) / multipliers

            elif method == "radial_scale":
                factor = args["factor"]
                resolved_center = metadata["resolved_center"]

                if np.isclose(factor, 0.0):
                    raise ValueError("Cannot revert radial_scale: scale factor is zero.")

                inv_points = resolved_center + (inv_points - resolved_center) / factor

            elif method == "spherize":
                factor = args["factor"]
                resolved_center = metadata["resolved_center"]
                resolved_radius = metadata["resolved_radius"]

                if np.isclose(factor, 1.0):
                    raise ValueError("Spherization with factor=1.0 is not mathematically invertible.")

                diff = inv_points - resolved_center
                distances_prime = np.linalg.norm(diff, axis=1)

                original_distances = (distances_prime - factor * resolved_radius) / (1.0 - factor)

                valid_indices = distances_prime > 0
                reconstructed_points = inv_points.copy()
                scale_ratio = original_distances[valid_indices] / distances_prime[valid_indices]
                reconstructed_points[valid_indices] = resolved_center + diff[valid_indices] * scale_ratio[:, np.newaxis]
                inv_points = reconstructed_points

            elif method == "apply_mapping":
                raise ValueError("apply_mapping is an arbitrary function and cannot be mathematically inverted.")

            else:
                raise ValueError(f"Unknown transformation method: {method}")

            movable_mask = metadata.get("movable_mask", None)
            if movable_mask is not None:
                current_points = np.where(movable_mask[:, np.newaxis], inv_points, current_points)
            else:
                current_points = inv_points

        self.points = current_points
        self._history = []
        self._update_parent(self.points)
        return self


    def block_division(
        self,
        num_blocks: Optional[Union[int, List[int], Tuple[int, ...]]] = None
    ) -> List[SpaceBlock]:
        """Divides the bounding box of the point cloud into separate SpaceBlock regions.

        Mathematical Foundations & Subdivision Logic:
            1. Quadrant/Octant Mode (num_blocks is None):
               Uses the center of mass C = (c_1, c_2, ..., c_D) as the origin.
               The space is partitioned into 2^D regions by defining bounds along each dimension
               as either (-inf, c_j] or [c_j, inf). The Cartesian product of these intervals yields
               the 2^D axis-aligned quadrants/octants.
            2. Single-Axis Division (num_blocks is an int):
               Identifies the axis with the largest coordinate span:
                   j_max = argmax_j (max(p_j) - min(p_j))
               Divides the range [min(p_{j_max}), max(p_{j_max})] into `num_blocks` equal sub-intervals.
               Other dimensions keep their full bounding box range [min(p_d), max(p_d)].
            3. Grid Division (num_blocks is a sequence of D ints):
               For each dimension j, divides [min(p_j), max(p_j)] into `num_blocks[j]` equal sub-intervals.
               The Cartesian product of these intervals across all dimensions yields the grid of blocks.

        Args:
            num_blocks (Optional[Union[int, List[int], Tuple[int, ...]]]):
                - None: Divides the space into 2^D quadrants/octants relative to the center of mass.
                - int: Divides the bounding box of the point cloud into `num_blocks` equal subblocks
                       along the axis with the largest coordinate span.
                - list/tuple of D ints: Divides the bounding box into a grid of subblocks according
                       to the specified divisions along each coordinate axis.

        Returns:
            List[SpaceBlock]: A list of SpaceBlock objects representing the divided subblocks.

        Raises:
            ValueError: If the point cloud is empty, if `num_blocks` is non-positive,
                or if the grid divisions list length does not match the space dimension.
        """
        if self.num_points == 0:
            return []

        D = self.dimension

        if num_blocks is None:
            # 2^D quadrants/octants from the center of mass
            c = self.center_of_mass
            intervals = []
            for j in range(D):
                intervals.append([(-np.inf, c[j]), (c[j], np.inf)])

            import itertools
            blocks = []
            for combo in itertools.product(*intervals):
                min_b = [bound[0] for bound in combo]
                max_b = [bound[1] for bound in combo]
                blocks.append(SpaceBlock(min_b, max_b))
            return blocks

        elif isinstance(num_blocks, int):
            if num_blocks <= 0:
                raise ValueError("Number of blocks must be positive.")
            min_coords = np.min(self.points, axis=0)
            max_coords = np.max(self.points, axis=0)
            spans = max_coords - min_coords
            longest_axis = np.argmax(spans)

            edges = np.linspace(min_coords[longest_axis], max_coords[longest_axis], num_blocks + 1)
            blocks = []
            for i in range(num_blocks):
                min_b = min_coords.copy()
                max_b = max_coords.copy()
                min_b[longest_axis] = edges[i]
                max_b[longest_axis] = edges[i+1]
                blocks.append(SpaceBlock(min_b, max_b))
            return blocks

        else:
            divisions = list(num_blocks)
            if len(divisions) != D:
                raise ValueError(f"Grid divisions length ({len(divisions)}) must match dimension ({D}).")
            for div in divisions:
                if div <= 0:
                    raise ValueError("Grid divisions must be positive integers.")

            min_coords = np.min(self.points, axis=0)
            max_coords = np.max(self.points, axis=0)

            axis_intervals = []
            for j in range(D):
                edges = np.linspace(min_coords[j], max_coords[j], divisions[j] + 1)
                intervals_j = []
                for i in range(divisions[j]):
                    intervals_j.append((edges[i], edges[i+1]))
                axis_intervals.append(intervals_j)

            import itertools
            blocks = []
            for combo in itertools.product(*axis_intervals):
                min_b = [bound[0] for bound in combo]
                max_b = [bound[1] for bound in combo]
                blocks.append(SpaceBlock(min_b, max_b))
            return blocks

        return []

    @property
    def num_points(self) -> int:
        """Returns the number of points in the point cloud.

        Returns:
            int: The size of the point cloud along the first dimension.
        """
        return self.points.shape[0]

    @property
    def dimension(self) -> int:
        """Returns the spatial dimension of the point cloud.

        Returns:
            int: The number of coordinate axes (dimension of the space).
        """
        return self.points.shape[1]

    @property
    def center_of_mass(self) -> np.ndarray:
        """Calculates the center of mass (centroid) of all points in the cloud.

        Mathematical Foundations:
            Let P be the (N, D) matrix of coordinates. The center of mass C in R^D is:
                C = (1 / N) * sum_{i=1}^N p_i

        Returns:
            np.ndarray: A 1D array of shape (D,) containing the centroid coordinates.
        """
        if self.num_points == 0:
            return np.zeros(self.dimension)
        return np.mean(self.points, axis=0)

    def get_extreme_points(
        self,
        axis: Optional[int] = None,
        extreme: Optional[str] = None
    ) -> Union[Dict[str, Any], Tuple[np.ndarray, np.ndarray], np.ndarray]:
        """Retrieves the coordinates of the extreme points in the point cloud.

        Args:
            axis (Optional[int]): The coordinate axis to query. If None, queries all axes.
            extreme (Optional[str]): The type of extreme to find: "min" or "max". If None, finds both.

        Returns:
            Union[Dict[str, Any], Tuple[np.ndarray, np.ndarray], np.ndarray]:
                - If axis is None and extreme is None:
                  A dictionary containing:
                    "min": np.ndarray of shape (D, D) where row `d` is the point minimizing axis `d`.
                    "max": np.ndarray of shape (D, D) where row `d` is the point maximizing axis `d`.
                    "min_indices": np.ndarray of shape (D,) containing indices of the min points.
                    "max_indices": np.ndarray of shape (D,) containing indices of the max points.
                - If axis is specified and extreme is None:
                  A tuple (min_point, max_point) of shape (D,) along the specified axis.
                - If axis is specified and extreme is specified:
                  The single extreme point (shape (D,)) that minimizes/maximizes the specified axis.
                - If axis is None and extreme is specified:
                  An array of shape (D, D) containing the extreme points of the specified type for each axis.

        Raises:
            ValueError: If the point cloud is empty, if the axis is out of bounds, or if the extreme parameter is invalid.
        """
        if self.num_points == 0:
            raise ValueError("Cannot get extreme points of an empty point cloud.")

        if axis is not None and (axis < 0 or axis >= self.dimension):
            raise ValueError(f"Axis {axis} is out of bounds for dimension {self.dimension}")

        if extreme is not None and extreme not in ("min", "max"):
            raise ValueError(f"Extreme must be 'min' or 'max', got {extreme}")

        # Compute argmin/argmax across all axes
        min_indices = np.argmin(self.points, axis=0)
        max_indices = np.argmax(self.points, axis=0)

        if axis is None and extreme is None:
            return {
                "min": self.points[min_indices],
                "max": self.points[max_indices],
                "min_indices": min_indices,
                "max_indices": max_indices,
            }
        elif axis is not None and extreme is None:
            return (self.points[min_indices[axis]], self.points[max_indices[axis]])
        elif axis is not None and extreme is not None:
            idx = min_indices[axis] if extreme == "min" else max_indices[axis]
            return self.points[idx]
        else:  # axis is None and extreme is not None
            indices = min_indices if extreme == "min" else max_indices
            return self.points[indices]

    def get_rotation_lines(
        self,
        plane: Union[str, Tuple[int, int]] = "xy",
        angles: Union[int, np.ndarray, list] = 360,
        length: float = 1.0,
        use_degrees: bool = True
    ) -> np.ndarray:
        """Generates visual reference line segments radiating from the center of mass along specified angles.

        Mathematical Foundations:
            Let `c` be the center of mass in R^D.
            Let `u` and `v` be the plane axes.
            For each angle theta, a unit vector `d` is created:
                d_u = cos(theta)
                d_v = sin(theta)
                d_j = 0  for j not in {u, v}
            A line segment is returned as two points: [c, c + length * d].

        Args:
            plane (Union[str, Tuple[int, int]]): The coordinate plane for the angles.
                Can be a string (e.g., "xy") or a tuple of axis indices (e.g., (0, 1)).
            angles (Union[int, np.ndarray, list]):
                - int: Automatically generates that many equally spaced angles around a full circle.
                - np.ndarray or list: An array of specific angles.
            length (float): The length of the radiating lines.
            use_degrees (bool): If True, interprets and generates angles in degrees. If False, in radians.

        Returns:
            np.ndarray: An array of shape (num_angles, 2, D) where segment[i, 0] is the center of mass
            and segment[i, 1] is the outer point of the i-th line segment.

        Raises:
            ValueError: If the plane axes or angles parameters are invalid.
        """
        if isinstance(plane, str):
            plane_lower = plane.lower()
            mapping = {
                "xy": (0, 1),
                "yz": (1, 2),
                "zx": (2, 0),
                "xz": (0, 2)
            }
            if plane_lower not in mapping:
                raise ValueError(
                    f"Unknown plane: {plane}. Choose from 'xy', 'yz', 'zx', 'xz' or specify a tuple of axis indices."
                )
            axis1, axis2 = mapping[plane_lower]
        elif isinstance(plane, tuple) and len(plane) == 2:
            axis1, axis2 = plane
        else:
            raise ValueError("Plane must be a string or a tuple of two axis indices.")

        if axis1 < 0 or axis1 >= self.dimension or axis2 < 0 or axis2 >= self.dimension:
            raise ValueError(
                f"Plane axes {(axis1, axis2)} exceed point cloud dimension {self.dimension}"
            )

        if isinstance(angles, int):
            if angles <= 0:
                raise ValueError("Number of angles must be positive.")
            if use_degrees:
                angles_rad = np.radians(np.linspace(0, 360, angles, endpoint=False))
            else:
                angles_rad = np.linspace(0, 2 * np.pi, angles, endpoint=False)
        else:
            angles_arr = np.asarray(angles, dtype=float)
            if use_degrees:
                angles_rad = np.radians(angles_arr)
            else:
                angles_rad = angles_arr

        num_angles = len(angles_rad)
        segments = np.zeros((num_angles, 2, self.dimension))
        c = self.center_of_mass

        for idx, theta in enumerate(angles_rad):
            direction = np.zeros(self.dimension)
            direction[axis1] = np.cos(theta)
            direction[axis2] = np.sin(theta)

            segments[idx, 0] = c
            segments[idx, 1] = c + length * direction

        return segments

    def scale_to_diameter(
        self,
        target_diameter: float,
        axis: Optional[int] = None,
        anchor: Optional[Union[str, np.ndarray, List[float], Tuple[float, ...]]] = None,
        movable_blocks: Optional[Any] = None,
        static_blocks: Optional[Any] = None
    ) -> "PointCloud":
        """Scales the point cloud coordinates so that its diameter matches a target value.

        Mathematical Foundations:
            1. Uniform Scaling (axis is None):
               - The current diameter is the maximum pairwise Euclidean distance:
                 D_curr = max_{i, j} ||p_i - p_j||_2
               - The scaling factor is s = D_target / D_curr.
               - Points are mapped relative to the anchor point p_anchor:
                 p_i' = p_anchor + s * (p_i - p_anchor)
               - The inverse transformation is:
                 p_i = p_anchor + (1/s) * (p_i' - p_anchor)

            2. Axis-Specific Scaling (axis is an integer):
               - The current diameter along the specified axis `d` is:
                 D_curr = max_i(p_{i, d}) - min_i(p_{i, d})
               - The scaling factor is s = D_target / D_curr.
               - Coordinates along axis `d` are scaled relative to the anchor's `d`-th component:
                 p'_{i, d} = p_{anchor, d} + s * (p_{i, d} - p_{anchor, d})
                 p'_{i, j} = p_{i, j}  for j != d
               - The inverse transformation is:
                 p_{i, d} = p_{anchor, d} + (1/s) * (p'_{i, d} - p_{anchor, d})

        Args:
            target_diameter (float): The desired size/diameter of the point cloud (must be non-negative).
            axis (Optional[int]): The axis index along which to scale. If None, scales uniformly in all dimensions.
            anchor (Optional[Union[str, np.ndarray, List[float], Tuple[float, ...]]]):
                The anchor point that remains stationary during scaling.
                - "center" or None: Uses the center of mass.
                - "min": Uses the extreme point minimizing the coordinates along `axis` (or axis 0 if `axis` is None).
                - "max": Uses the extreme point maximizing the coordinates along `axis` (or axis 0 if `axis` is None).
                - Array-like of shape (D,): A custom coordinate point.
            movable_blocks (Optional[Any]): SpaceBlock or sequence of SpaceBlocks. Only points within will scale.
            static_blocks (Optional[Any]): SpaceBlock or sequence of SpaceBlocks. Points within will not scale.

        Returns:
            PointCloud: A new PointCloud containing the scaled coordinates.

        Raises:
            ValueError: If `target_diameter` is negative, the point cloud is empty/too small,
                the current diameter is zero, the axis index is invalid, or the anchor is invalid.
        """
        if target_diameter < 0:
            raise ValueError("Target diameter must be non-negative.")

        if self.num_points == 0:
            raise ValueError("Cannot scale an empty point cloud.")

        M = self._get_movable_mask(movable_blocks, static_blocks)

        # Determine anchor point coordinates
        if anchor is None or (isinstance(anchor, str) and anchor == "center"):
            anchor_pt = self.center_of_mass
        elif isinstance(anchor, str) and anchor in ("min", "max"):
            ext_axis = axis if axis is not None else 0
            anchor_pt = self.get_extreme_points(axis=ext_axis, extreme=anchor)
        elif isinstance(anchor, (np.ndarray, list, tuple)):
            anchor_arr = np.asarray(anchor, dtype=float)
            if anchor_arr.shape != (self.dimension,):
                raise ValueError(f"Anchor shape must be {(self.dimension,)}, got {anchor_arr.shape}")
            anchor_pt = anchor_arr
        else:
            raise ValueError(f"Invalid anchor: {anchor}")

        if axis is None:
            if self.num_points < 2:
                raise ValueError("Cannot compute overall diameter for point cloud with fewer than 2 points.")

            from scipy.spatial.distance import pdist
            dists = pdist(self.points)
            current_diameter = np.max(dists) if dists.size > 0 else 0.0

            if current_diameter == 0.0:
                raise ValueError("Current overall diameter is 0 (all points are identical), cannot scale.")

            scale_factor = target_diameter / current_diameter
            new_points = anchor_pt + scale_factor * (self.points - anchor_pt)
        else:
            if axis < 0 or axis >= self.dimension:
                raise ValueError(f"Axis {axis} is out of bounds for dimension {self.dimension}")

            coords = self.points[:, axis]
            current_diameter = np.max(coords) - np.min(coords)

            if current_diameter == 0.0:
                raise ValueError(f"Current diameter along axis {axis} is 0, cannot scale.")

            scale_factor = target_diameter / current_diameter

            new_points = self.points.copy()
            new_points[:, axis] = anchor_pt[axis] + scale_factor * (coords - anchor_pt[axis])

        new_points = np.where(M[:, np.newaxis], new_points, self.points)
        return self._create_deformed_pc(
            new_points,
            "scale_to_diameter",
            args={"target_diameter": target_diameter, "axis": axis, "anchor": anchor, "movable_blocks": movable_blocks, "static_blocks": static_blocks},
            metadata={"scale_factor": scale_factor, "resolved_anchor_pt": anchor_pt, "movable_mask": M}
        )

    def rotate(
        self,
        angle: float,
        plane: Union[str, Tuple[int, int]] = "xy",
        center: Optional[Union[str, np.ndarray, List[float], Tuple[float, ...]]] = None,
        use_degrees: bool = True,
        movable_blocks: Optional[Any] = None,
        static_blocks: Optional[Any] = None
    ) -> "PointCloud":
        """Rotates the point cloud coordinates in a specified 2D plane.

        Mathematical Foundations:
            Let `u` and `v` be the coordinate axes of the rotation plane (e.g., axis 0 and axis 1 for 'xy').
            Let `c = (c_u, c_v)` be the projection of the rotation center onto the plane.
            Let theta be the rotation angle (in radians).
            For each point, coordinates `x` are shifted relative to the center:
                u_shifted = u - c_u
                v_shifted = v - c_v
            The rotated coordinates are computed as:
                u' = c_u + u_shifted * cos(theta) - v_shifted * sin(theta)
                v' = c_v + u_shifted * sin(theta) + v_shifted * cos(theta)
            All other coordinate components remain unchanged.

            The inverse transformation (used by `revert`) is rotation by `-theta` about the same center:
                u = c_u + u'_shifted * cos(-theta) - v'_shifted * sin(-theta)
                v = c_v + u'_shifted * sin(-theta) + v'_shifted * cos(-theta)
                where u'_shifted = u' - c_u, v'_shifted = v' - c_v.

        Args:
            angle (float): The angle of rotation.
            plane (Union[str, Tuple[int, int]]): The 2D plane of rotation.
                Can be a string (e.g. "xy", "yz", "zx", "xz") or a tuple of two axis indices.
            center (Optional[Union[str, np.ndarray, List[float], Tuple[float, ...]]]):
                The center of rotation.
                - "center" or None: Uses the center of mass.
                - "min": Uses the extreme point minimizing the coordinates along `plane[0]`.
                - "max": Uses the extreme point maximizing the coordinates along `plane[0]`.
                - Array-like of shape (D,): A custom coordinate point.
            use_degrees (bool): If True, interprets `angle` in degrees. If False, in radians.
            movable_blocks (Optional[Any]): SpaceBlock or sequence of SpaceBlocks. Only points within will rotate.
            static_blocks (Optional[Any]): SpaceBlock or sequence of SpaceBlocks. Points within will not rotate.

        Returns:
            PointCloud: A new PointCloud containing the rotated coordinates.

        Raises:
            ValueError: If the plane identifier or axes are invalid, or if the center coordinate is invalid.
        """
        if isinstance(plane, str):
            plane_lower = plane.lower()
            mapping = {
                "xy": (0, 1),
                "yz": (1, 2),
                "zx": (2, 0),
                "xz": (0, 2)
            }
            if plane_lower not in mapping:
                raise ValueError(f"Unknown plane: {plane}")
            axis1, axis2 = mapping[plane_lower]
        elif isinstance(plane, tuple) and len(plane) == 2:
            axis1, axis2 = plane
        else:
            raise ValueError("Plane must be a string or a tuple of two axis indices.")

        if axis1 < 0 or axis1 >= self.dimension or axis2 < 0 or axis2 >= self.dimension:
            raise ValueError(f"Plane axes exceed point cloud dimension.")

        M = self._get_movable_mask(movable_blocks, static_blocks)

        if center is None or (isinstance(center, str) and center == "center"):
            c = self.center_of_mass
        elif isinstance(center, str) and center in ("min", "max"):
            c = self.get_extreme_points(axis=axis1, extreme=center)
        elif isinstance(center, (np.ndarray, list, tuple)):
            center_arr = np.asarray(center, dtype=float)
            if center_arr.shape != (self.dimension,):
                raise ValueError("Center shape must match dimension.")
            c = center_arr
        else:
            raise ValueError(f"Invalid center: {center}")

        theta = np.radians(angle) if use_degrees else angle
        cos_t, sin_t = np.cos(theta), np.sin(theta)

        R = np.array([
            [cos_t, -sin_t],
            [sin_t, cos_t]
        ])

        new_points = self.points.copy()
        shifted = self.points - c
        coords_plane = shifted[:, [axis1, axis2]]
        rotated_plane = coords_plane @ R.T
        new_points[:, [axis1, axis2]] = rotated_plane + c[[axis1, axis2]]

        new_points = np.where(M[:, np.newaxis], new_points, self.points)
        return self._create_deformed_pc(
            new_points,
            "rotate",
            args={"angle": angle, "plane": plane, "center": center, "use_degrees": use_degrees, "movable_blocks": movable_blocks, "static_blocks": static_blocks},
            metadata={"resolved_center": c, "movable_mask": M}
        )

    def translate(
        self,
        translation_vector: Any,
        movable_blocks: Optional[Any] = None,
        static_blocks: Optional[Any] = None
    ) -> "PointCloud":
        """Translates the point cloud coordinates by a given translation vector.

        Mathematical Foundations:
            Let `v` in R^D be the translation vector.
            Each coordinate vector `p_i` is mapped to:
                p_i' = p_i + v
            The inverse transformation is:
                p_i = p_i' - v

        Args:
            translation_vector (Any): Array-like of shape (D,) representing the displacement in each dimension.
            movable_blocks (Optional[Any]): SpaceBlock or sequence of SpaceBlocks. Only points within will translate.
            static_blocks (Optional[Any]): SpaceBlock or sequence of SpaceBlocks. Points within will not translate.

        Returns:
            PointCloud: A new PointCloud containing the translated coordinates.

        Raises:
            ValueError: If the translation vector shape does not match the space dimension.
        """
        v = np.asarray(translation_vector, dtype=float)
        if v.shape != (self.dimension,):
            raise ValueError(f"Translation vector must have shape {(self.dimension,)}, got {v.shape}")

        M = self._get_movable_mask(movable_blocks, static_blocks)
        new_points = self.points + v
        new_points = np.where(M[:, np.newaxis], new_points, self.points)
        return self._create_deformed_pc(
            new_points,
            "translate",
            args={"translation_vector": translation_vector, "movable_blocks": movable_blocks, "static_blocks": static_blocks},
            metadata={"movable_mask": M}
        )

    def shear(
        self,
        factor: float,
        axis: int,
        control_axis: int,
        anchor: Optional[Union[str, np.ndarray, List[float], Tuple[float, ...]]] = None,
        movable_blocks: Optional[Any] = None,
        static_blocks: Optional[Any] = None
    ) -> "PointCloud":
        """Applies a linear shear transformation along a specified axis.

        Mathematical Foundations:
            Let `a` be the shear axis (direction of displacement) and `c` be the control axis.
            Let `p_anchor` be the anchor point.
            The sheared coordinate for each point is:
                p'_{i, a} = p_{i, a} + factor * (p_{i, c} - p_{anchor, c})
                p'_{i, j} = p_{i, j}  for j != a
            Since the control axis `c` is not modified (`a != c`), the inverse transformation is:
                p_{i, a} = p'_{i, a} - factor * (p'_{i, c} - p_{anchor, c})

        Args:
            factor (float): The shear scaling factor.
            axis (int): The coordinate index that will be displaced by the shear.
            control_axis (int): The coordinate index that determines the magnitude of displacement.
            anchor (Optional[Union[str, np.ndarray, List[float], Tuple[float, ...]]]):
                The anchor point defining the origin of the control axis displacement.
                - "center" or None: Uses the center of mass.
                - "min": Uses the extreme point minimizing the control axis coordinates.
                - "max": Uses the extreme point maximizing the control axis coordinates.
                - Array-like of shape (D,): A custom coordinate point.
            movable_blocks (Optional[Any]): SpaceBlock or sequence of SpaceBlocks. Only points within will shear.
            static_blocks (Optional[Any]): SpaceBlock or sequence of SpaceBlocks. Points within will not shear.

        Returns:
            PointCloud: A new PointCloud containing the sheared coordinates.

        Raises:
            ValueError: If axis indices are invalid or equal, or if the anchor is invalid.
        """
        if axis < 0 or axis >= self.dimension or control_axis < 0 or control_axis >= self.dimension:
            raise ValueError("Axes indices are out of bounds.")

        M = self._get_movable_mask(movable_blocks, static_blocks)

        if anchor is None or (isinstance(anchor, str) and anchor == "center"):
            anchor_pt = self.center_of_mass
        elif isinstance(anchor, str) and anchor in ("min", "max"):
            anchor_pt = self.get_extreme_points(axis=control_axis, extreme=anchor)
        elif isinstance(anchor, (np.ndarray, list, tuple)):
            anchor_arr = np.asarray(anchor, dtype=float)
            if anchor_arr.shape != (self.dimension,):
                raise ValueError("Anchor shape must match dimension.")
            anchor_pt = anchor_arr
        else:
            raise ValueError(f"Invalid anchor: {anchor}")

        new_points = self.points.copy()
        control_coords = self.points[:, control_axis]
        displacement = factor * (control_coords - anchor_pt[control_axis])
        new_points[:, axis] += displacement

        new_points = np.where(M[:, np.newaxis], new_points, self.points)
        return self._create_deformed_pc(
            new_points,
            "shear",
            args={"factor": factor, "axis": axis, "control_axis": control_axis, "anchor": anchor, "movable_blocks": movable_blocks, "static_blocks": static_blocks},
            metadata={"resolved_anchor_pt": anchor_pt, "movable_mask": M}
        )

    def apply_mapping(
        self,
        func: Any,
        movable_blocks: Optional[Any] = None,
        static_blocks: Optional[Any] = None
    ) -> "PointCloud":
        """Applies an arbitrary coordinate mapping function to the point cloud.

        Mathematical Foundations:
            Let `func` be a function mapping R^D -> R^D.
            For each point:
                p'_i = func(p_i)
            This transformation is arbitrary and cannot be automatically inverted. Revert operations
            on point clouds containing `apply_mapping` in their history will fail.

        Args:
            func (Any): A callable that takes an (N, D) numpy array of coordinates and returns
                an (N, D) numpy array of transformed coordinates.
            movable_blocks (Optional[Any]): SpaceBlock or sequence of SpaceBlocks. Only points within will be mapped.
            static_blocks (Optional[Any]): SpaceBlock or sequence of SpaceBlocks. Points within will not be mapped.

        Returns:
            PointCloud: A new PointCloud containing the transformed coordinates.

        Raises:
            TypeError: If `func` is not callable.
        """
        if not callable(func):
            raise TypeError("func must be a callable.")

        M = self._get_movable_mask(movable_blocks, static_blocks)
        transformed = func(self.points)
        transformed = np.where(M[:, np.newaxis], transformed, self.points)
        return self._create_deformed_pc(
            transformed,
            "apply_mapping",
            args={"func": func, "movable_blocks": movable_blocks, "static_blocks": static_blocks},
            metadata={"movable_mask": M}
        )

    def twist(
        self,
        rate: float,
        plane: Union[str, Tuple[int, int]] = "xy",
        control_axis: int = 2,
        anchor: Optional[Union[str, np.ndarray, List[float], Tuple[float, ...]]] = None,
        use_degrees: bool = True,
        movable_blocks: Optional[Any] = None,
        static_blocks: Optional[Any] = None
    ) -> "PointCloud":
        """Twists the point cloud by rotating coordinates along a plane proportionally to their position on a control axis.

        Mathematical Foundations:
            Let `u` and `v` be the rotation plane axes, and `c` be the control axis.
            Let `p_anchor` be the anchor point defining the twist origin.
            For each point, the distance along the control axis is:
                h = p_{i, c} - p_{anchor, c}
            The dynamic angle of rotation is:
                alpha = rate * h
            Coordinates on the rotation plane are shifted and rotated:
                u_shifted = p_{i, u} - p_{anchor, u}
                v_shifted = p_{i, v} - p_{anchor, v}
                u' = p_{anchor, u} + u_shifted * cos(alpha) - v_shifted * sin(alpha)
                v' = p_{anchor, v} + u_shifted * sin(alpha) + v_shifted * cos(alpha)
            All other coordinate components (including control axis `c`) remain unchanged.

            The inverse transformation (used by `revert`) rotates plane coordinates by `-alpha`:
                u = p_{anchor, u} + u'_shifted * cos(-alpha) - v'_shifted * sin(-alpha)
                v = p_{anchor, v} + u'_shifted * sin(-alpha) + v'_shifted * cos(-alpha)
                where u'_shifted = u' - p_{anchor, u}, v'_shifted = v' - p_{anchor, v}.

        Args:
            rate (float): The rate of rotation per unit distance along the control axis.
            plane (Union[str, Tuple[int, int]]): The coordinate plane in which rotation occurs.
                Can be a string (e.g. "xy", "yz") or a tuple of two axis indices.
            control_axis (int): The axis coordinate that determines the angle of rotation.
            anchor (Optional[Union[str, np.ndarray, List[float], Tuple[float, ...]]]):
                The anchor point defining the twist origin.
                - "center" or None: Uses the center of mass.
                - "min": Uses the extreme point minimizing the control axis coordinates.
                - "max": Uses the extreme point maximizing the control axis coordinates.
                - Array-like of shape (D,): A custom coordinate point.
            use_degrees (bool): If True, interprets `rate` and angles in degrees/unit distance.
                If False, in radians/unit distance.
            movable_blocks (Optional[Any]): SpaceBlock or sequence of SpaceBlocks. Only points within will twist.
            static_blocks (Optional[Any]): SpaceBlock or sequence of SpaceBlocks. Points within will not twist.

        Returns:
            PointCloud: A new PointCloud containing the twisted coordinates.

        Raises:
            ValueError: If plane axes or control axis indices are invalid/out of bounds,
                or if the anchor is invalid.
        """
        if isinstance(plane, str):
            plane_lower = plane.lower()
            mapping = {"xy": (0, 1), "yz": (1, 2), "zx": (2, 0), "xz": (0, 2)}
            if plane_lower not in mapping:
                raise ValueError(f"Unknown plane: {plane}")
            axis1, axis2 = mapping[plane_lower]
        elif isinstance(plane, tuple) and len(plane) == 2:
            axis1, axis2 = plane
        else:
            raise ValueError("Plane must be a string or a tuple of two axis indices.")

        if axis1 < 0 or axis1 >= self.dimension or axis2 < 0 or axis2 >= self.dimension:
            raise ValueError("Plane axes indices exceed point cloud dimension.")
        if control_axis < 0 or control_axis >= self.dimension:
            raise ValueError("Control axis is out of bounds.")

        M = self._get_movable_mask(movable_blocks, static_blocks)

        if anchor is None or (isinstance(anchor, str) and anchor == "center"):
            anchor_pt = self.center_of_mass
        elif isinstance(anchor, str) and anchor in ("min", "max"):
            anchor_pt = self.get_extreme_points(axis=control_axis, extreme=anchor)
        elif isinstance(anchor, (np.ndarray, list, tuple)):
            anchor_arr = np.asarray(anchor, dtype=float)
            if anchor_arr.shape != (self.dimension,):
                raise ValueError("Anchor shape must match dimension.")
            anchor_pt = anchor_arr
        else:
            raise ValueError(f"Invalid anchor: {anchor}")

        dist_along_control = self.points[:, control_axis] - anchor_pt[control_axis]
        angles = rate * dist_along_control
        if use_degrees:
            angles = np.radians(angles)

        cos_vals = np.cos(angles)
        sin_vals = np.sin(angles)

        new_points = self.points.copy()
        x = self.points[:, axis1] - anchor_pt[axis1]
        y = self.points[:, axis2] - anchor_pt[axis2]

        new_points[:, axis1] = anchor_pt[axis1] + x * cos_vals - y * sin_vals
        new_points[:, axis2] = anchor_pt[axis2] + x * sin_vals + y * cos_vals

        new_points = np.where(M[:, np.newaxis], new_points, self.points)
        return self._create_deformed_pc(
            new_points,
            "twist",
            args={"rate": rate, "plane": plane, "control_axis": control_axis, "anchor": anchor, "use_degrees": use_degrees, "movable_blocks": movable_blocks, "static_blocks": static_blocks},
            metadata={"resolved_anchor_pt": anchor_pt, "movable_mask": M}
        )

    def bend(
        self,
        curvature: float,
        axis: int = 0,
        control_axis: int = 1,
        anchor: Optional[Union[str, np.ndarray, List[float], Tuple[float, ...]]] = None,
        open_at: Optional[Union[int, List[int], np.ndarray]] = None,
        movable_blocks: Optional[Any] = None,
        static_blocks: Optional[Any] = None
    ) -> "PointCloud":
        """Bends the point cloud coordinates along a specified axis with a given curvature.

        Mathematical Foundations:
            Let `x = p_{i, axis} - p_{anchor, axis}` and `y = p_{i, control_axis} - p_{anchor, control_axis}`.
            
            1. Non-zero Curvature (|curvature| >= 1e-8):
               The transformation wraps the `axis` coordinate into a circular arc of radius `R = 1 / curvature`.
               - The angle of wrap is theta = curvature * x.
               - If open_at is provided, a theta_seam shift is applied: theta = curvature * x + theta_seam.
               - The new coordinates relative to the anchor are:
                 p'_{i, axis} = p_{anchor, axis} + (R - y) * sin(theta)
                 p'_{i, control_axis} = p_{anchor, control_axis} + R - (R - y) * cos(theta)
               - The inverse transformation (mathematical unbending) is:
                 x = (theta - theta_seam) / curvature
                 y = R - sign(curvature) * sqrt(x'^2 + (R - y')^2)
                 where x' = p'_{i, axis} - p_{anchor, axis}, y' = p'_{i, control_axis} - p_{anchor, control_axis},
                 and theta = arctan2(x' * sign_c, (R - y') * sign_c) with sign_c = sign(curvature).

            2. Zero or Near-Zero Curvature (|curvature| < 1e-8):
               To avoid division by zero, a first-order Taylor expansion approximation is used:
                 p'_{i, axis} = p_{anchor, axis} + x - curvature * x * y
                 p'_{i, control_axis} = p_{anchor, control_axis} + y + 0.5 * curvature * x^2
               - The inverse transformation for near-zero curvature is:
                 x = X + curvature * X * Y
                 y = Y - 0.5 * curvature * X^2
                 where X = p'_{i, axis} - p_{anchor, axis}, Y = p'_{i, control_axis} - p_{anchor, control_axis}.

        Args:
            curvature (float): The curvature of the bend (1/radius of the target circular arc).
            axis (int): The coordinate axis index to bend (along which the arc is formed).
            control_axis (int): The coordinate axis index perpendicular to the bending axis
                that represents the radial thickness.
            anchor (Optional[Union[str, np.ndarray, List[float], Tuple[float, ...]]]):
                The anchor point about which bending is centered.
                - "center" or None: Uses the center of mass.
                - "min": Uses the extreme point minimizing the bending axis coordinate.
                - "max": Uses the extreme point maximizing the bending axis coordinate.
                - Array-like of shape (D,): A custom coordinate point.
            open_at (Optional[Union[int, List[int], np.ndarray]]): Optional seam / cut reference to align.
                In bend(), this must be a coordinate point or direction of shape (D,) representing
                where the seam meets on the target circle/torus.
            movable_blocks (Optional[Any]): SpaceBlock or sequence of SpaceBlocks. Only points within will bend.
            static_blocks (Optional[Any]): SpaceBlock or sequence of SpaceBlocks. Points within will not bend.

        Returns:
            PointCloud: A new PointCloud containing the bent coordinates.

        Raises:
            ValueError: If axis indices are invalid or identical, or if the anchor or open_at is invalid.
        """
        if axis < 0 or axis >= self.dimension or control_axis < 0 or control_axis >= self.dimension:
            raise ValueError("Axis indices are out of bounds.")
        if axis == control_axis:
            raise ValueError("Bend axis and control axis must be distinct.")

        M = self._get_movable_mask(movable_blocks, static_blocks)

        if anchor is None or (isinstance(anchor, str) and anchor == "center"):
            anchor_pt = self.center_of_mass
        elif isinstance(anchor, str) and anchor in ("min", "max"):
            anchor_pt = self.get_extreme_points(axis=axis, extreme=anchor)
        elif isinstance(anchor, (np.ndarray, list, tuple)):
            anchor_arr = np.asarray(anchor, dtype=float)
            if anchor_arr.shape != (self.dimension,):
                raise ValueError("Anchor shape must match dimension.")
            anchor_pt = anchor_arr
        else:
            raise ValueError(f"Invalid anchor: {anchor}")

        x = self.points[:, axis] - anchor_pt[axis]
        y = self.points[:, control_axis] - anchor_pt[control_axis]

        new_points = self.points.copy()
        theta_seam = np.pi

        if np.abs(curvature) < 1e-8:
            new_points[:, axis] = anchor_pt[axis] + x - curvature * x * y
            new_points[:, control_axis] = anchor_pt[control_axis] + y + 0.5 * curvature * (x**2)
        else:
            r = 1.0 / curvature
            if open_at is not None:
                # In bend, starting coordinates are straight, so we cannot resolve indices.
                # It must be a coordinate point or direction of shape (D,).
                if isinstance(open_at, (int, np.integer, list, np.ndarray)) and not isinstance(open_at, np.ndarray) and not (isinstance(open_at, (list, np.ndarray)) and isinstance(open_at[0], (float, np.floating))):
                    raise ValueError("Indices cannot be used for open_at in bend() directly, as starting coordinates are straight. Use a coordinate point or direction.")
                else:
                    pt = np.asarray(open_at, dtype=float)
                    if pt.ndim == 1 and pt.shape[0] == self.dimension:
                        X_pt = pt[axis] - anchor_pt[axis]
                        Y_pt = pt[control_axis] - anchor_pt[control_axis]
                        sign_c = np.sign(curvature)
                        theta_seam = np.arctan2(X_pt * sign_c, (r - Y_pt) * sign_c)
                    else:
                        raise ValueError("open_at in bend must be a coordinate point of shape (D,).")
            
            theta = curvature * x + theta_seam + np.pi
            new_points[:, axis] = anchor_pt[axis] + (r - y) * np.sin(theta)
            new_points[:, control_axis] = anchor_pt[control_axis] + r - (r - y) * np.cos(theta)

        new_points = np.where(M[:, np.newaxis], new_points, self.points)
        return self._create_deformed_pc(
            new_points,
            "bend",
            args={"curvature": curvature, "axis": axis, "control_axis": control_axis, "anchor": anchor, "open_at": open_at, "movable_blocks": movable_blocks, "static_blocks": static_blocks},
            metadata={"resolved_anchor_pt": anchor_pt, "theta_seam": theta_seam, "movable_mask": M}
        )

    def unbend(
        self,
        curvature: float,
        axis: int,
        control_axis: int,
        anchor: Optional[Union[str, np.ndarray, List[float], Tuple[float, ...]]] = None,
        open_at: Optional[Union[int, List[int], np.ndarray]] = None,
        movable_blocks: Optional[Any] = None,
        static_blocks: Optional[Any] = None
    ) -> "PointCloud":
        """Unbends the point cloud coordinates, straightening a circular curve of a given curvature.

        Mathematical Foundations:
            Let `X = p_{i, axis} - p_{anchor, axis}` and `Y = p_{i, control_axis} - p_{anchor, control_axis}`.

            1. Non-zero Curvature (|curvature| >= 1e-8):
               Straightens points that lie on an arc of radius `R = 1 / curvature` centered at `(0, R)`.
               - The wrap angle is theta = arctan2(X * sign_c, (R - Y) * sign_c), where sign_c = sign(curvature).
               - If open_at is specified, shift the wrap angle so that the seam maps to +/-pi:
                 theta_shifted = (theta - theta_seam) % (2 * pi) - pi
               - The distance from the center of curvature is d = sqrt(X^2 + (R - Y)^2).
               - The straightened coordinates are:
                 p'_{i, axis} = p_{anchor, axis} + theta_shifted / curvature
                 p'_{i, control_axis} = p_{anchor, control_axis} + R - sign_c * d
               - The inverse transformation (re-bending, used by `revert`) is:
                 X = (R - y') * sin(k * x' + theta_seam)
                 Y = R - (R - y') * cos(k * x' + theta_seam)
                 where x' = p'_{i, axis} - p_{anchor, axis}, y' = p'_{i, control_axis} - p_{anchor, control_axis}, and k = curvature.

            2. Zero or Near-Zero Curvature (|curvature| < 1e-8):
               Uses a first-order Taylor expansion approximation for unbending:
                 p'_{i, axis} = p_{anchor, axis} + X + curvature * X * Y
                 p'_{i, control_axis} = p_{anchor, control_axis} + Y - 0.5 * curvature * X^2
               - The inverse transformation is:
                 X = x' - curvature * x' * y'
                 Y = y' + 0.5 * curvature * x'^2.

        Args:
            curvature (float): The curvature of the bend to flatten (1/radius of the circular arc).
            axis (int): The coordinate axis index to straighten.
            control_axis (int): The coordinate axis index perpendicular to the straightened axis
                representing the radial direction.
            anchor (Optional[Union[str, np.ndarray, List[float], Tuple[float, ...]]]):
                The anchor point about which unbending is centered.
                - "center" or None: Uses the center of mass.
                - "min": Uses the extreme point minimizing the axis coordinate.
                - "max": Uses the extreme point maximizing the axis coordinate.
                - Array-like of shape (D,): A custom coordinate point.
            open_at (Optional[Union[int, List[int], np.ndarray]]): Optional seam / cut reference to align.
                In unbend(), this can be:
                - An integer index, or list/array of integer indices of vertices representing the seam.
                - A coordinate point of shape (D,) representing the seam location.
            movable_blocks (Optional[Any]): SpaceBlock or sequence of SpaceBlocks. Only points within will unbend.
            static_blocks (Optional[Any]): SpaceBlock or sequence of SpaceBlocks. Points within will not unbend.

        Returns:
            PointCloud: A new PointCloud containing the unbent coordinates.

        Raises:
            ValueError: If axis indices are invalid or identical, or if the anchor or open_at is invalid.
        """
        if axis < 0 or axis >= self.dimension or control_axis < 0 or control_axis >= self.dimension:
            raise ValueError("Axis indices are out of bounds.")
        if axis == control_axis:
            raise ValueError("Bend axis and control axis must be distinct.")

        M = self._get_movable_mask(movable_blocks, static_blocks)

        if anchor is None or (isinstance(anchor, str) and anchor == "center"):
            anchor_pt = self.center_of_mass
        elif isinstance(anchor, str) and anchor in ("min", "max"):
            anchor_pt = self.get_extreme_points(axis=axis, extreme=anchor)
        elif isinstance(anchor, (np.ndarray, list, tuple)):
            anchor_arr = np.asarray(anchor, dtype=float)
            if anchor_arr.shape != (self.dimension,):
                raise ValueError("Anchor shape must match dimension.")
            anchor_pt = anchor_arr
        else:
            raise ValueError(f"Invalid anchor: {anchor}")

        X = self.points[:, axis] - anchor_pt[axis]
        Y = self.points[:, control_axis] - anchor_pt[control_axis]

        new_points = self.points.copy()
        theta_seam = np.pi

        if np.abs(curvature) < 1e-8:
            new_points[:, axis] = anchor_pt[axis] + X + curvature * X * Y
            new_points[:, control_axis] = anchor_pt[control_axis] + Y - 0.5 * curvature * (X**2)
        else:
            r = 1.0 / curvature
            sign_c = np.sign(curvature)
            theta = np.arctan2(X * sign_c, (r - Y) * sign_c)
            d = np.sqrt(X**2 + (r - Y)**2)

            if open_at is not None:
                # Determine theta_seam
                if isinstance(open_at, (int, np.integer)):
                    indices = [int(open_at)]
                elif isinstance(open_at, (list, np.ndarray)) and len(open_at) > 0 and isinstance(open_at[0], (int, np.integer, int)):
                    indices = [int(idx) for idx in open_at]
                elif isinstance(open_at, np.ndarray) and np.issubdtype(open_at.dtype, np.integer):
                    indices = [int(idx) for idx in open_at]
                else:
                    # Treat as coordinate point
                    pt = np.asarray(open_at, dtype=float)
                    if pt.ndim == 1 and pt.shape[0] == self.dimension:
                        indices = None
                        X_pt = pt[axis] - anchor_pt[axis]
                        Y_pt = pt[control_axis] - anchor_pt[control_axis]
                        theta_seam = np.arctan2(X_pt * sign_c, (r - Y_pt) * sign_c)
                    else:
                        raise ValueError("open_at must be an index, list of indices, or a coordinate point of shape (D,).")
                
                if indices is not None:
                    X_v = self.points[indices, axis] - anchor_pt[axis]
                    Y_v = self.points[indices, control_axis] - anchor_pt[control_axis]
                    theta_v = np.arctan2(X_v * sign_c, (r - Y_v) * sign_c)
                    mean_cos = np.mean(np.cos(theta_v))
                    mean_sin = np.mean(np.sin(theta_v))
                    theta_seam = np.arctan2(mean_sin, mean_cos)
            
            theta_shifted = (theta - theta_seam) % (2 * np.pi) - np.pi
            x = theta_shifted / curvature
            y = r - sign_c * d

            new_points[:, axis] = anchor_pt[axis] + x
            new_points[:, control_axis] = anchor_pt[control_axis] + y

        new_points = np.where(M[:, np.newaxis], new_points, self.points)
        return self._create_deformed_pc(
            new_points,
            "unbend",
            args={"curvature": curvature, "axis": axis, "control_axis": control_axis, "anchor": anchor, "open_at": open_at, "movable_blocks": movable_blocks, "static_blocks": static_blocks},
            metadata={"resolved_anchor_pt": anchor_pt, "theta_seam": theta_seam, "movable_mask": M}
        )

    def taper(
        self,
        factor: float,
        axis: Optional[Union[int, List[int], Tuple[int, ...]]] = None,
        control_axis: int = 2,
        anchor: Optional[Union[str, np.ndarray, List[float], Tuple[float, ...]]] = None,
        movable_blocks: Optional[Any] = None,
        static_blocks: Optional[Any] = None
    ) -> "PointCloud":
        """Tapers the point cloud by scaling coordinates along specified axes proportionally to the position along a control axis.

        Mathematical Foundations:
            Let `c` be the control axis, and `A_scale` be the list of axes to scale.
            Let `p_anchor` be the anchor point.
            For each point, the displacement along the control axis is:
                h = p_{i, c} - p_{anchor, c}
            The scale multiplier is:
                m = 1.0 + factor * h
            For each axis `d` in `A_scale`:
                p'_{i, d} = p_{anchor, d} + m * (p_{i, d} - p_{anchor, d})
            All other coordinate components (including control axis `c`) remain unchanged.

            The inverse transformation (used by `revert`) scales coordinate differences by `1 / m`:
                p_{i, d} = p_{anchor, d} + (1 / m) * (p'_{i, d} - p_{anchor, d})
                where m = 1.0 + factor * (p'_{i, c} - p_{anchor, c}).

        Args:
            factor (float): The tapering factor. A positive factor increases scale in the positive
                direction of the control axis, while a negative factor decreases it.
            axis (Optional[Union[int, List[int], Tuple[int, ...]]]): The index or indices of axes to scale.
                If None, scales all axes perpendicular to the control axis.
            control_axis (int): The index of the axis along which tapering varies.
            anchor (Optional[Union[str, np.ndarray, List[float], Tuple[float, ...]]]):
                The anchor point defining the taper origin.
                - "center" or None: Uses the center of mass.
                - "min": Uses the extreme point minimizing the control axis coordinates.
                - "max": Uses the extreme point maximizing the control axis coordinates.
                - Array-like of shape (D,): A custom coordinate point.
            movable_blocks (Optional[Any]): SpaceBlock or sequence of SpaceBlocks. Only points within will taper.
            static_blocks (Optional[Any]): SpaceBlock or sequence of SpaceBlocks. Points within will not taper.

        Returns:
            PointCloud: A new PointCloud containing the tapered coordinates.

        Raises:
            ValueError: If control axis or scale axes are invalid, if the anchor is invalid,
                or if any scale multiplier is zero during reversion.
        """
        if control_axis < 0 or control_axis >= self.dimension:
            raise ValueError("Control axis index is out of bounds.")

        # Determine axes to scale
        if axis is None:
            axes_to_scale = [d for d in range(self.dimension) if d != control_axis]
        elif isinstance(axis, int):
            axes_to_scale = [axis]
        else:
            axes_to_scale = list(axis)

        for d in axes_to_scale:
            if d < 0 or d >= self.dimension:
                raise ValueError(f"Axis index {d} is out of bounds.")

        M = self._get_movable_mask(movable_blocks, static_blocks)

        if anchor is None or (isinstance(anchor, str) and anchor == "center"):
            anchor_pt = self.center_of_mass
        elif isinstance(anchor, str) and anchor in ("min", "max"):
            anchor_pt = self.get_extreme_points(axis=control_axis, extreme=anchor)
        elif isinstance(anchor, (np.ndarray, list, tuple)):
            anchor_arr = np.asarray(anchor, dtype=float)
            if anchor_arr.shape != (self.dimension,):
                raise ValueError("Anchor shape must match dimension.")
            anchor_pt = anchor_arr
        else:
            raise ValueError(f"Invalid anchor: {anchor}")

        h = self.points[:, control_axis] - anchor_pt[control_axis]
        multipliers = 1.0 + factor * h

        new_points = self.points.copy()
        for d in axes_to_scale:
            new_points[:, d] = anchor_pt[d] + multipliers * (self.points[:, d] - anchor_pt[d])

        new_points = np.where(M[:, np.newaxis], new_points, self.points)
        return self._create_deformed_pc(
            new_points,
            "taper",
            args={"factor": factor, "axis": axis, "control_axis": control_axis, "anchor": anchor, "movable_blocks": movable_blocks, "static_blocks": static_blocks},
            metadata={"resolved_anchor_pt": anchor_pt, "movable_mask": M}
        )

    def radial_scale(
        self,
        factor: float,
        center: Optional[Union[str, np.ndarray, List[float], Tuple[float, ...]]] = None,
        movable_blocks: Optional[Any] = None,
        static_blocks: Optional[Any] = None
    ) -> "PointCloud":
        """Scales points radially outward or inward relative to a center point.

        Mathematical Foundations:
            Let `c` be the center of scaling.
            The coordinates of each point are mapped as:
                p'_i = c + factor * (p_i - c)
            The inverse transformation (used by `revert`) is:
                p_i = c + (1 / factor) * (p'_i - c)

        Args:
            factor (float): The radial scaling multiplier.
            center (Optional[Union[str, np.ndarray, List[float], Tuple[float, ...]]]):
                The center point of the scaling.
                - "center" or None: Uses the center of mass.
                - Array-like of shape (D,): A custom coordinate point.
            movable_blocks (Optional[Any]): SpaceBlock or sequence of SpaceBlocks. Only points within will scale.
            static_blocks (Optional[Any]): SpaceBlock or sequence of SpaceBlocks. Points within will not scale.

        Returns:
            PointCloud: A new PointCloud containing the radially scaled coordinates.

        Raises:
            ValueError: If the center coordinate is invalid, or if `factor` is zero during reversion.
        """
        if center is None or (isinstance(center, str) and center == "center"):
            c = self.center_of_mass
        elif isinstance(center, (np.ndarray, list, tuple)):
            center_arr = np.asarray(center, dtype=float)
            if center_arr.shape != (self.dimension,):
                raise ValueError("Center shape must match dimension.")
            c = center_arr
        else:
            raise ValueError(f"Invalid center: {center}")

        M = self._get_movable_mask(movable_blocks, static_blocks)
        new_points = c + factor * (self.points - c)
        new_points = np.where(M[:, np.newaxis], new_points, self.points)
        return self._create_deformed_pc(
            new_points,
            "radial_scale",
            args={"factor": factor, "center": center, "movable_blocks": movable_blocks, "static_blocks": static_blocks},
            metadata={"resolved_center": c, "movable_mask": M}
        )

    def spherize(
        self,
        factor: float,
        radius: Optional[float] = None,
        center: Optional[Union[str, np.ndarray, List[float], Tuple[float, ...]]] = None,
        movable_blocks: Optional[Any] = None,
        static_blocks: Optional[Any] = None
    ) -> "PointCloud":
        """Blends the point cloud coordinates between their current layout and their projection onto a sphere.

        Mathematical Foundations:
            Let `c` be the center of spherization, and `R` be the sphere radius.
            For each point, let the radial vector be v_i = p_i - c, and its distance be d_i = ||v_i||_2.
            
            The spherized distance is:
                d'_i = (1.0 - factor) * d_i + factor * R
            The spherized coordinates are:
                p'_i = c + v_i * (d'_i / d_i)   if d_i > 0
                p'_i = p_i                     if d_i == 0

            The inverse transformation (used by `revert`) reconstructs the original distance `d_i` from the spherized distance `d'_i`:
                d_i = (d'_i - factor * R) / (1.0 - factor)
            The reconstructed coordinate is:
                p_i = c + (p'_i - c) * (d_i / d'_i)   if d'_i > 0
            This reversion is valid for factor in [0.0, 1.0). If factor == 1.0, radial information is entirely collapsed, making reversion impossible.

        Args:
            factor (float): The blending factor between 0.0 (no change) and 1.0 (full projection onto the sphere).
            radius (Optional[float]): The radius of the target sphere. If None, defaults to the mean distance of all points to the center.
            center (Optional[Union[str, np.ndarray, List[float], Tuple[float, ...]]]):
                The center of the sphere.
                - "center" or None: Uses the center of mass.
                - Array-like of shape (D,): A custom coordinate point.
            movable_blocks (Optional[Any]): SpaceBlock or sequence of SpaceBlocks. Only points within will spherize.
            static_blocks (Optional[Any]): SpaceBlock or sequence of SpaceBlocks. Points within will not spherize.

        Returns:
            PointCloud: A new PointCloud containing the spherized coordinates.

        Raises:
            ValueError: If `factor` is not in [0.0, 1.0], if `radius` is negative, if the center coordinate is invalid,
                or if `factor` is 1.0 during reversion.
        """
        if factor < 0.0 or factor > 1.0:
            raise ValueError("Spherize factor must be between 0.0 and 1.0.")

        if center is None or (isinstance(center, str) and center == "center"):
            c = self.center_of_mass
        elif isinstance(center, (np.ndarray, list, tuple)):
            center_arr = np.asarray(center, dtype=float)
            if center_arr.shape != (self.dimension,):
                raise ValueError("Center shape must match dimension.")
            c = center_arr
        else:
            raise ValueError(f"Invalid center: {center}")

        M = self._get_movable_mask(movable_blocks, static_blocks)

        diff = self.points - c
        distances = np.linalg.norm(diff, axis=1)

        if radius is None:
            radius = np.mean(distances) if len(distances) > 0 else 1.0
        elif radius < 0.0:
            raise ValueError("Radius must be non-negative.")

        scale = np.ones_like(distances)
        valid_indices = distances > 0
        scale[valid_indices] = (1.0 - factor) + factor * (radius / distances[valid_indices])

        new_points = c + diff * scale[:, np.newaxis]
        new_points = np.where(M[:, np.newaxis], new_points, self.points)
        return self._create_deformed_pc(
            new_points,
            "spherize",
            args={"factor": factor, "radius": radius, "center": center, "movable_blocks": movable_blocks, "static_blocks": static_blocks},
            metadata={"resolved_center": c, "resolved_radius": radius, "movable_mask": M}
        )

    def __getitem__(self, key: Any) -> np.ndarray:
        """Allows slicing and indexing the point cloud coordinates directly.

        Args:
            key (Any): Slicing or indexing keys.

        Returns:
            np.ndarray: Slices or subsets of coordinates.
        """
        return self.points[key]

    def __setitem__(self, key: Any, value: Any) -> None:
        """Allows setting coordinate values directly, updating the parent complex in sync.

        Args:
            key (Any): Slicing or indexing keys.
            value (Any): New coordinate values.
        """
        self.points[key] = value
        self._update_parent(self.points)

    def __len__(self) -> int:
        """Returns the number of points (size along the first dimension).

        Returns:
            int: The number of points in the point cloud.
        """
        return len(self.points)

    def __iter__(self) -> Any:
        """Allows iterating over the points in the cloud.

        Returns:
            Any: Iterator over the coordinate rows.
        """
        return iter(self.points)

    def __array__(self, dtype: Optional[Any] = None, copy: Optional[bool] = None) -> np.ndarray:
        """NumPy array interface protocol to allow seamless conversion to numpy arrays.

        Args:
            dtype (Optional[Any]): The desired data-type for the array.
            copy (Optional[bool]): Unused, kept for numpy compatibility.

        Returns:
            np.ndarray: The coordinate array.
        """
        if dtype is not None:
            return np.asarray(self.points, dtype=dtype)
        return np.asarray(self.points)

    def __repr__(self) -> str:
        """String representation of the PointCloud.

        Returns:
            str: Description of point count and dimension.
        """
        return f"PointCloud(num_points={self.num_points}, dimension={self.dimension})"

    @property
    def shape(self) -> Tuple[int, ...]:
        """Returns the shape of the point cloud array.

        Returns:
            Tuple[int, ...]: A tuple of integers representing (num_points, dimension).
        """
        return self.points.shape

    @property
    def ndim(self) -> int:
        """Returns the number of dimensions of the point cloud array (always 2).

        Returns:
            int: The number of axes of the point array (always 2).
        """
        return self.points.ndim

    @property
    def dtype(self) -> np.dtype:
        """Returns the data type of the coordinates.

        Returns:
            np.dtype: The coordinate data type.
        """
        return self.points.dtype

    def frames(
        self,
        method_name: str,
        steps: int = 10,
        **kwargs
    ) -> Any:
        """Generates intermediate PointCloud frames representing the deformation homotopy.

        For each step, this method yields the interpolation parameter `t` (from 0.0 to 1.0)
        and a new, isolated PointCloud representing the state of the deformation at that step.

        Args:
            method_name (str): The name of the deformation method (e.g. 'rotate', 'unbend').
            steps (int): The number of intermediate steps. Defaults to 10.
            **kwargs: Arguments to be passed to the deformation method.

        Yields:
            Tuple[float, PointCloud]: A tuple of the interpolation parameter t and the PointCloud frame.
        """
        # Determine the initial state (t = 0.0)
        yield 0.0, PointCloud(
            self.points.copy(),
            parent=None,
            history=self._history.copy(),
            original_points=self._original_points
        )

        for i in range(1, steps + 1):
            t = float(i) / steps
            frame_pc = PointCloud(
                self.points.copy(),
                parent=None,
                history=self._history.copy(),
                original_points=self._original_points
            )

            kwargs_t = kwargs.copy()
            if method_name == "translate":
                if "translation_vector" in kwargs:
                    v = np.asarray(kwargs["translation_vector"], dtype=float)
                    kwargs_t["translation_vector"] = (t * v).tolist()
            elif method_name == "rotate":
                if "angle" in kwargs:
                    kwargs_t["angle"] = t * kwargs["angle"]
            elif method_name == "shear":
                if "factor" in kwargs:
                    kwargs_t["factor"] = t * kwargs["factor"]
            elif method_name == "twist":
                if "rate" in kwargs:
                    kwargs_t["rate"] = t * kwargs["rate"]
            elif method_name == "bend":
                if "curvature" in kwargs:
                    kwargs_t["curvature"] = t * kwargs["curvature"]
            elif method_name == "unbend":
                if "curvature" in kwargs:
                    kwargs_t["curvature"] = t * kwargs["curvature"]
            elif method_name == "taper":
                if "factor" in kwargs:
                    kwargs_t["factor"] = t * kwargs["factor"]
            elif method_name == "spherize":
                if "factor" in kwargs:
                    kwargs_t["factor"] = t * kwargs["factor"]
            elif method_name == "radial_scale":
                if "factor" in kwargs:
                    kwargs_t["factor"] = 1.0 + t * (kwargs["factor"] - 1.0)
            elif method_name == "scale_to_diameter":
                if "diameter" in kwargs:
                    axis_arg = kwargs.get("axis", None)
                    control_axis = kwargs.get("control_axis", 2)
                    if axis_arg is None:
                        axes = [d for d in range(self.dimension) if d != control_axis]
                    elif isinstance(axis_arg, int):
                        axes = [axis_arg]
                    else:
                        axes = list(axis_arg)
                    min_coords = np.min(self.points[:, axes], axis=0)
                    max_coords = np.max(self.points[:, axes], axis=0)
                    current_diameter = float(np.max(max_coords - min_coords))
                    target_diameter = float(kwargs["diameter"])
                    kwargs_t["diameter"] = current_diameter + t * (target_diameter - current_diameter)
            elif method_name == "apply_mapping":
                final_pc = PointCloud(
                    self.points.copy(),
                    parent=None,
                    history=self._history.copy(),
                    original_points=self._original_points
                )
                final_pc.apply_mapping(**kwargs)
                frame_pc.points = (1.0 - t) * self.points + t * final_pc.points
                yield t, frame_pc
                continue

            # Run the method with interpolated arguments
            method = getattr(frame_pc, method_name)
            method(**kwargs_t)
            yield t, frame_pc

    def min_distance_to(self, other: Union["PointCloud", np.ndarray]) -> float:
        """Computes the minimum Euclidean distance between this point cloud and another point cloud.

        This method uses a KD-Tree to efficiently query the minimum pairwise distance between
        the coordinate points.

        Args:
            other (Union[PointCloud, np.ndarray]): The other point cloud or array of coordinates.

        Returns:
            float: The minimum distance between any point in self and any point in other.
        """
        from scipy.spatial import cKDTree
        if isinstance(other, PointCloud):
            other_pts = other.points
        else:
            other_pts = np.asarray(other, dtype=float)

        tree1 = cKDTree(self.points)
        distances, _ = tree1.query(other_pts, k=1)
        return float(np.min(distances))

    def verify_isotopy_clearance(
        self,
        obstacle: Union["PointCloud", np.ndarray],
        actions: List[Tuple[str, Dict[str, Any]]],
        steps_per_action: int = 10,
        safety_margin: float = 0.0
    ) -> List[Tuple[float, str, float]]:
        """Verifies if a composed sequence of actions keeps the point cloud clear of an obstacle.

        This method steps through the frames of each action sequentially. At each frame,
        it calculates the minimum distance between the current frame's coordinates and the
        obstacle coordinates.

        Args:
            obstacle (Union[PointCloud, np.ndarray]): The obstacle point cloud or coordinates.
            actions (List[Tuple[str, Dict[str, Any]]]): A list of tuples, where each tuple contains
                the method name (str) and a dictionary of keyword arguments.
            steps_per_action (int): The number of frames to interpolate for each action.
            safety_margin (float): The minimum distance that must be maintained.

        Returns:
            List[Tuple[float, str, float]]: A list of violations, where each violation is a tuple:
                - time (float): The continuous time parameter (action_index + t).
                - action_name (str): The name of the action being performed.
                - min_distance (float): The actual minimum distance.
        """
        violations = []
        current_pc = PointCloud(
            self.points.copy(),
            parent=None,
            history=self._history.copy(),
            original_points=self._original_points
        )

        for action_idx, (method_name, kwargs) in enumerate(actions):
            for t, frame_pc in current_pc.frames(method_name, steps=steps_per_action, **kwargs):
                if action_idx > 0 and t == 0.0:
                    continue
                
                dist = frame_pc.min_distance_to(obstacle)
                if dist < safety_margin:
                    continuous_time = float(action_idx) + t
                    violations.append((continuous_time, method_name, dist))
            
            getattr(current_pc, method_name)(**kwargs)

        return violations
