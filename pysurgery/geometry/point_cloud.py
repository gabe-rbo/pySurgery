"""Point cloud representation and geometric querying tools."""

import numpy as np
from typing import Dict, Tuple, Optional, Union, Any, List

class SpaceBlock:
    """Represents a D-dimensional block of space defined by min and max coordinate bounds."""

    def __init__(self, min_bounds: Any, max_bounds: Any):
        """Initialize a SpaceBlock with min and max bounds for each dimension.
        
        Args:
            min_bounds: Array-like of shape (D,) representing lower bounds.
            max_bounds: Array-like of shape (D,) representing upper bounds.
        """
        self.min_bounds = np.asarray(min_bounds, dtype=float)
        self.max_bounds = np.asarray(max_bounds, dtype=float)
        if self.min_bounds.shape != self.max_bounds.shape or self.min_bounds.ndim != 1:
            raise ValueError("Bounds must be 1D arrays of the same shape.")

    def contains(self, points: Any) -> np.ndarray:
        """Returns a boolean mask of shape (N,) indicating which points are inside this block.
        
        Args:
            points: Array-like of shape (N, D) or a PointCloud.
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
        """Allows syntax like `point in space_block` for a single point of shape (D,)."""
        pt = np.asarray(point, dtype=float)
        if pt.shape != self.min_bounds.shape:
            return False
        return bool(np.all((pt >= self.min_bounds) & (pt <= self.max_bounds)))

    def __repr__(self) -> str:
        return f"SpaceBlock(min_bounds={self.min_bounds.tolist()}, max_bounds={self.max_bounds.tolist()})"

class PointCloud:
    """A class representing a point cloud in D-dimensional space.
    
    Provides utility methods to query extreme points, calculate center of mass,
    generate angular reference lines for plotting, and apply continuous deformations.
    """

    def __init__(
        self,
        points: Any,
        parent: Optional[Any] = None,
        *,
        history: Optional[List[Dict[str, Any]]] = None,
        original_points: Optional[np.ndarray] = None
    ):
        """Initialize the PointCloud with an (N, D) array-like of points.
        
        Args:
            points: An array-like object of shape (N, D) containing point coordinates.
            parent: Optional parent object (e.g., SimplicialComplex) that holds these points.
            history: Optional list of transformations applied.
            original_points: Optional original coordinates array.
            
        Raises:
            ValueError: If points cannot be cast to a 2D float array.
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
        """Updates the parent object coordinates and mappings if a parent is set."""
        if self._parent is not None:
            self._parent._coordinates = new_points
            if hasattr(self._parent, "_generate_point_cloud_mappings"):
                self._parent._generate_point_cloud_mappings(new_points)

    def _get_movable_mask(
        self,
        movable_blocks: Optional[Union[SpaceBlock, List[SpaceBlock], Tuple[SpaceBlock, ...]]] = None,
        static_blocks: Optional[Union[SpaceBlock, List[SpaceBlock], Tuple[SpaceBlock, ...]]] = None
    ) -> np.ndarray:
        """Computes a boolean mask indicating which points are allowed to move."""
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
        """Updates the parent and returns a new PointCloud with updated history."""
        self._update_parent(new_points)
        history_entry = {
            "method": method_name,
            "args": args,
        }
        if metadata is not None:
            history_entry["metadata"] = metadata
        new_history = self._history + [history_entry]
        return PointCloud(
            new_points,
            parent=self._parent,
            history=new_history,
            original_points=self._original_points
        )

    def list_transformations(self) -> List[Dict[str, Any]]:
        """Returns a copy of the transformation history log, containing method names and arguments."""
        import copy
        return [{"method": item["method"], "args": copy.deepcopy(item["args"])} for item in self._history]

    def undo(self, indices: Optional[Union[int, List[int]]] = None) -> "PointCloud":
        """Undoes specified transformations by removing them from history and re-applying the rest from the original points."""
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
        
        # Start from original points
        current_pc = PointCloud(self._original_points.copy(), parent=self._parent, original_points=self._original_points)
        
        # Re-apply remaining transformations sequentially
        for item in new_history_entries:
            method_name = item["method"]
            args = item["args"]
            method = getattr(current_pc, method_name)
            current_pc = method(**args)
            
        self._update_parent(current_pc.points)
        return current_pc

    def revert(self) -> "PointCloud":
        """Reverts all transformations by mathematically inverting them in reverse order."""
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
                    x_unbent = theta / curvature
                    y_unbent = r - sign_c * d
                    
                    inv_points[:, axis] = resolved_anchor_pt[axis] + x_unbent
                    inv_points[:, control_axis] = resolved_anchor_pt[control_axis] + y_unbent
                    
            elif method == "unbend":
                curvature = args["curvature"]
                axis = args["axis"]
                control_axis = args["control_axis"]
                resolved_anchor_pt = metadata["resolved_anchor_pt"]
                
                x_unbent = inv_points[:, axis] - resolved_anchor_pt[axis]
                y_unbent = inv_points[:, control_axis] - resolved_anchor_pt[control_axis]
                
                if np.abs(curvature) < 1e-8:
                    inv_points[:, axis] = resolved_anchor_pt[axis] + x_unbent - curvature * x_unbent * y_unbent
                    inv_points[:, control_axis] = resolved_anchor_pt[control_axis] + y_unbent + 0.5 * curvature * (x_unbent**2)
                else:
                    r = 1.0 / curvature
                    theta = curvature * x_unbent
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
                
        self._update_parent(current_points)
        return PointCloud(
            current_points,
            parent=self._parent,
            history=[],
            original_points=self._original_points
        )

    def block_division(
        self,
        num_blocks: Optional[Union[int, List[int], Tuple[int, ...]]] = None
    ) -> List[SpaceBlock]:
        """Divides the space occupied by the point cloud into SpaceBlock objects.
        
        Args:
            num_blocks:
                - None: Divides the space into 2^D quadrants/octants relative to the center of mass.
                - int: Divides the bounding box of the point cloud into `num_blocks` equal subblocks
                       along the axis with the largest coordinate span.
                - list/tuple of D ints: Divides the bounding box into a grid of subblocks according
                       to the specified divisions along each coordinate axis.
                       
        Returns:
            A list of SpaceBlock objects.
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

    @property
    def num_points(self) -> int:
        """Returns the number of points in the cloud."""
        return self.points.shape[0]

    @property
    def dimension(self) -> int:
        """Returns the dimension of the space the points lie in."""
        return self.points.shape[1]

    @property
    def center_of_mass(self) -> np.ndarray:
        """Calculates the center of mass (centroid) of the point cloud."""
        if self.num_points == 0:
            return np.zeros(self.dimension)
        return np.mean(self.points, axis=0)

    def get_extreme_points(
        self,
        axis: Optional[int] = None,
        extreme: Optional[str] = None
    ) -> Union[Dict[str, Any], Tuple[np.ndarray, np.ndarray], np.ndarray]:
        """Retrieve the extreme points of the point cloud.
        
        Args:
            axis: Optional axis index (0 for X, 1 for Y, 2 for Z, etc.).
                  If None, retrieves extreme points for all axes.
            extreme: Optional string ('min' or 'max').
                     If None, retrieves both min and max.
                     
        Returns:
            - If axis is None and extreme is None:
              A dictionary containing:
                "min": np.ndarray of shape (D, D) where row `d` is the point minimizing axis `d`.
                "max": np.ndarray of shape (D, D) where row `d` is the point maximizing axis `d`.
                "min_indices": np.ndarray of shape (D,) containing indices of the min points.
                "max_indices": np.ndarray of shape (D,) containing indices of the max points.
            - If axis is specified and extreme is None:
              A tuple (min_point, max_point) of shape (D,) each along the specified axis.
            - If axis is specified and extreme is specified:
              The single extreme point (shape (D,)) that minimizes/maximizes the specified axis.
            - If axis is None and extreme is specified:
              An array of shape (D, D) containing all extreme points of the specified type.
              
        Raises:
            ValueError: If the point cloud is empty.
            ValueError: If `axis` is out of bounds or `extreme` is invalid.
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
        """Generates line segments parting from the center of mass along specified angles.
        
        Args:
            plane: The coordinate plane for the angles. Can be a string ('xy', 'yz', 'zx', 'xz')
                   or a tuple of axis indices (e.g., (0, 1)).
            angles: If an int, divides the plane into that many equal angles (e.g., 360).
                    If an array or list, uses those specific angles.
            length: The length of the line segments.
            use_degrees: If True, angles are interpreted as/generated in degrees.
                         If False, in radians.
                         
        Returns:
            np.ndarray of shape (num_angles, 2, D) representing the line segments.
            For each segment:
              - segment[0] is the center of mass (origin).
              - segment[1] is the outer point at the specified angle.
              
        Raises:
            ValueError: If the plane is invalid or coordinates are out of bounds.
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
        anchor: Optional[Union[str, np.ndarray]] = None,
        movable_blocks: Optional[Any] = None,
        static_blocks: Optional[Any] = None
    ) -> "PointCloud":
        """Stretches or compresses the point cloud to a target diameter.
        
        Args:
            target_diameter: The desired target diameter (must be non-negative).
            axis: If None, performs uniform scaling in all dimensions based on the 
                  overall pairwise Euclidean diameter.
                  If an integer, scales only the coordinates along the specified axis 
                  to match the target diameter span.
            anchor: The point that remains stationary.
            movable_blocks: Optional block or collection of SpaceBlock. Only points inside will be deformed.
            static_blocks: Optional block or collection of SpaceBlock. Points inside will remain static.
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
        elif isinstance(anchor, np.ndarray):
            if anchor.shape != (self.dimension,):
                raise ValueError(f"Anchor shape must be {(self.dimension,)}, got {anchor.shape}")
            anchor_pt = anchor
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
        center: Optional[Union[str, np.ndarray]] = None,
        use_degrees: bool = True,
        movable_blocks: Optional[Any] = None,
        static_blocks: Optional[Any] = None
    ) -> "PointCloud":
        """Rotates the point cloud in the specified coordinate plane."""
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
        elif isinstance(center, np.ndarray):
            if center.shape != (self.dimension,):
                raise ValueError("Center shape must match dimension.")
            c = center
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
        """Translates the point cloud by a given vector."""
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
        anchor: Optional[Union[str, np.ndarray]] = None,
        movable_blocks: Optional[Any] = None,
        static_blocks: Optional[Any] = None
    ) -> "PointCloud":
        """Applies a shear deformation to the point cloud."""
        if axis < 0 or axis >= self.dimension or control_axis < 0 or control_axis >= self.dimension:
            raise ValueError("Axes indices are out of bounds.")

        M = self._get_movable_mask(movable_blocks, static_blocks)

        if anchor is None or (isinstance(anchor, str) and anchor == "center"):
            anchor_pt = self.center_of_mass
        elif isinstance(anchor, str) and anchor in ("min", "max"):
            anchor_pt = self.get_extreme_points(axis=control_axis, extreme=anchor)
        elif isinstance(anchor, np.ndarray):
            if anchor.shape != (self.dimension,):
                raise ValueError("Anchor shape must match dimension.")
            anchor_pt = anchor
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
        """Applies an arbitrary coordinate mapping function to the point cloud."""
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
        anchor: Optional[Union[str, np.ndarray]] = None,
        use_degrees: bool = True,
        movable_blocks: Optional[Any] = None,
        static_blocks: Optional[Any] = None
    ) -> "PointCloud":
        """Applies a twist deformation by rotating points dynamically along a control axis."""
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
        elif isinstance(anchor, np.ndarray):
            if anchor.shape != (self.dimension,):
                raise ValueError("Anchor shape must match dimension.")
            anchor_pt = anchor
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
        anchor: Optional[Union[str, np.ndarray]] = None,
        movable_blocks: Optional[Any] = None,
        static_blocks: Optional[Any] = None
    ) -> "PointCloud":
        """Bends the point cloud along a specified axis with a given curvature."""
        if axis < 0 or axis >= self.dimension or control_axis < 0 or control_axis >= self.dimension:
            raise ValueError("Axis indices are out of bounds.")
        if axis == control_axis:
            raise ValueError("Bend axis and control axis must be distinct.")

        M = self._get_movable_mask(movable_blocks, static_blocks)

        if anchor is None or (isinstance(anchor, str) and anchor == "center"):
            anchor_pt = self.center_of_mass
        elif isinstance(anchor, str) and anchor in ("min", "max"):
            anchor_pt = self.get_extreme_points(axis=axis, extreme=anchor)
        elif isinstance(anchor, np.ndarray):
            if anchor.shape != (self.dimension,):
                raise ValueError("Anchor shape must match dimension.")
            anchor_pt = anchor
        else:
            raise ValueError(f"Invalid anchor: {anchor}")

        x = self.points[:, axis] - anchor_pt[axis]
        y = self.points[:, control_axis] - anchor_pt[control_axis]

        new_points = self.points.copy()

        if np.abs(curvature) < 1e-8:
            new_points[:, axis] = anchor_pt[axis] + x - curvature * x * y
            new_points[:, control_axis] = anchor_pt[control_axis] + y + 0.5 * curvature * (x**2)
        else:
            r = 1.0 / curvature
            theta = curvature * x
            new_points[:, axis] = anchor_pt[axis] + (r - y) * np.sin(theta)
            new_points[:, control_axis] = anchor_pt[control_axis] + r - (r - y) * np.cos(theta)

        new_points = np.where(M[:, np.newaxis], new_points, self.points)
        return self._create_deformed_pc(
            new_points,
            "bend",
            args={"curvature": curvature, "axis": axis, "control_axis": control_axis, "anchor": anchor, "movable_blocks": movable_blocks, "static_blocks": static_blocks},
            metadata={"resolved_anchor_pt": anchor_pt, "movable_mask": M}
        )

    def unbend(
        self,
        curvature: float,
        axis: int,
        control_axis: int,
        anchor: Optional[Any] = None,
        movable_blocks: Optional[Any] = None,
        static_blocks: Optional[Any] = None
    ) -> "PointCloud":
        """Reverses a bending transformation along a specified axis."""
        if axis < 0 or axis >= self.dimension or control_axis < 0 or control_axis >= self.dimension:
            raise ValueError("Axis indices are out of bounds.")
        if axis == control_axis:
            raise ValueError("Bend axis and control axis must be distinct.")

        M = self._get_movable_mask(movable_blocks, static_blocks)

        if anchor is None or (isinstance(anchor, str) and anchor == "center"):
            anchor_pt = self.center_of_mass
        elif isinstance(anchor, str) and anchor in ("min", "max"):
            anchor_pt = self.get_extreme_points(axis=axis, extreme=anchor)
        elif isinstance(anchor, np.ndarray):
            if anchor.shape != (self.dimension,):
                raise ValueError("Anchor shape must match dimension.")
            anchor_pt = anchor
        else:
            raise ValueError(f"Invalid anchor: {anchor}")

        X = self.points[:, axis] - anchor_pt[axis]
        Y = self.points[:, control_axis] - anchor_pt[control_axis]

        new_points = self.points.copy()

        if np.abs(curvature) < 1e-8:
            new_points[:, axis] = anchor_pt[axis] + X + curvature * X * Y
            new_points[:, control_axis] = anchor_pt[control_axis] + Y - 0.5 * curvature * (X**2)
        else:
            r = 1.0 / curvature
            sign_c = np.sign(curvature)
            theta = np.arctan2(X * sign_c, (r - Y) * sign_c)
            d = np.sqrt(X**2 + (r - Y)**2)
            x = theta / curvature
            y = r - sign_c * d
            
            new_points[:, axis] = anchor_pt[axis] + x
            new_points[:, control_axis] = anchor_pt[control_axis] + y

        new_points = np.where(M[:, np.newaxis], new_points, self.points)
        return self._create_deformed_pc(
            new_points,
            "unbend",
            args={"curvature": curvature, "axis": axis, "control_axis": control_axis, "anchor": anchor, "movable_blocks": movable_blocks, "static_blocks": static_blocks},
            metadata={"resolved_anchor_pt": anchor_pt, "movable_mask": M}
        )

    def taper(
        self,
        factor: float,
        axis: Optional[Union[int, List[int], Tuple[int, ...]]] = None,
        control_axis: int = 2,
        anchor: Optional[Union[str, np.ndarray]] = None,
        movable_blocks: Optional[Any] = None,
        static_blocks: Optional[Any] = None
    ) -> "PointCloud":
        """Scales coordinates along specified axes proportionally to distance along a control axis."""
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
        elif isinstance(anchor, np.ndarray):
            if anchor.shape != (self.dimension,):
                raise ValueError("Anchor shape must match dimension.")
            anchor_pt = anchor
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
        center: Optional[Union[str, np.ndarray]] = None,
        movable_blocks: Optional[Any] = None,
        static_blocks: Optional[Any] = None
    ) -> "PointCloud":
        """Scales points radially outward or inward from a given center."""
        if center is None or (isinstance(center, str) and center == "center"):
            c = self.center_of_mass
        elif isinstance(center, np.ndarray):
            if center.shape != (self.dimension,):
                raise ValueError("Center shape must match dimension.")
            c = center
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
        center: Optional[Union[str, np.ndarray]] = None,
        movable_blocks: Optional[Any] = None,
        static_blocks: Optional[Any] = None
    ) -> "PointCloud":
        """Blends points between their current layout and their projection onto a sphere."""
        if factor < 0.0 or factor > 1.0:
            raise ValueError("Spherize factor must be between 0.0 and 1.0.")

        if center is None or (isinstance(center, str) and center == "center"):
            c = self.center_of_mass
        elif isinstance(center, np.ndarray):
            if center.shape != (self.dimension,):
                raise ValueError("Center shape must match dimension.")
            c = center
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
        """Allows slicing and indexing the point cloud coordinates directly."""
        return self.points[key]

    def __setitem__(self, key: Any, value: Any) -> None:
        """Allows setting coordinate values directly, updating the parent complex in sync."""
        self.points[key] = value
        self._update_parent(self.points)

    def __len__(self) -> int:
        """Returns the number of points (size along the first dimension)."""
        return len(self.points)

    def __iter__(self) -> Any:
        """Allows iterating over the points in the cloud."""
        return iter(self.points)

    def __array__(self, dtype: Optional[Any] = None, copy: Optional[bool] = None) -> np.ndarray:
        """NumPy array interface protocol to allow seamless conversion to numpy arrays."""
        if dtype is not None:
            return np.asarray(self.points, dtype=dtype)
        return np.asarray(self.points)

    def __repr__(self) -> str:
        """String representation of the PointCloud."""
        return f"PointCloud(num_points={self.num_points}, dimension={self.dimension})"

    @property
    def shape(self) -> Tuple[int, ...]:
        """Returns the shape of the point cloud array."""
        return self.points.shape

    @property
    def ndim(self) -> int:
        """Returns the number of dimensions of the point cloud array (always 2)."""
        return self.points.ndim

    @property
    def dtype(self) -> np.dtype:
        """Returns the data type of the coordinates."""
        return self.points.dtype

