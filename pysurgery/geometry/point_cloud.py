"""Point cloud representation and geometric querying tools."""

import numpy as np
from typing import Dict, Tuple, Optional, Union, Any, List

class PointCloud:
    """A class representing a point cloud in D-dimensional space.
    
    Provides utility methods to query extreme points, calculate center of mass,
    generate angular reference lines for plotting, and apply continuous deformations.
    """

    def __init__(self, points: Any, parent: Optional[Any] = None):
        """Initialize the PointCloud with an (N, D) array-like of points.
        
        Args:
            points: An array-like object of shape (N, D) containing point coordinates.
            parent: Optional parent object (e.g., SimplicialComplex) that holds these points.
            
        Raises:
            ValueError: If points cannot be cast to a 2D float array.
        """
        self.points = np.asarray(points, dtype=float)
        if self.points.ndim != 2:
            raise ValueError(
                f"Points must be a 2D array of shape (N, D), got shape {self.points.shape}"
            )
        self._parent = parent

    def _update_parent(self, new_points: np.ndarray) -> None:
        """Updates the parent object coordinates and mappings if a parent is set."""
        if self._parent is not None:
            self._parent._coordinates = new_points
            if hasattr(self._parent, "_generate_point_cloud_mappings"):
                self._parent._generate_point_cloud_mappings(new_points)

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
        anchor: Optional[Union[str, np.ndarray]] = None
    ) -> "PointCloud":
        """Stretches or compresses the point cloud to a target diameter.
        
        Mathematical Formulation:
            Let P_i be the coordinate vector of point i, and 'a' be the coordinate of the anchor point.
            
            1. Uniform Scaling (axis = None):
               We calculate the current Euclidean diameter:
                   D_curr = max_{i, j} || P_i - P_j ||_2
               The uniform scale factor is:
                   s = D_target / D_curr
               Each transformed point is computed as:
                   P'_i = a + s * (P_i - a)

            2. Axis-Specific Scaling (axis = d):
               We calculate the current coordinate span along axis d:
                   D_curr = max_i(P_{i, d}) - min_i(P_{i, d})
               The scale factor along axis d is:
                   s = D_target / D_curr
               Each transformed coordinate is computed as:
                   P'_{i, d} = a_d + s * (P_{i, d} - a_d)
                   P'_{i, j} = P_{i, j}  for all coordinate axes j != d
        
        Args:
            target_diameter: The desired target diameter (must be non-negative).
            axis: If None, performs uniform scaling in all dimensions based on the 
                  overall pairwise Euclidean diameter.
                  If an integer, scales only the coordinates along the specified axis 
                  to match the target diameter span.
            anchor: The point that remains stationary. Can be:
                    - None: Defaults to the center of mass.
                    - "center": Center of mass (centroid).
                    - "min": The minimum extreme point along the scaling axis (if axis is specified) 
                             or the overall minimum point.
                    - "max": The maximum extreme point along the scaling axis (if axis is specified) 
                             or the overall maximum point.
                    - np.ndarray: A custom coordinate array of shape (D,).
                    
        Returns:
            A new PointCloud instance with transformed points.
            
        Raises:
            ValueError: If target_diameter is negative.
            ValueError: If the current diameter is zero (cannot scale).
        """
        if target_diameter < 0:
            raise ValueError("Target diameter must be non-negative.")
            
        if self.num_points == 0:
            raise ValueError("Cannot scale an empty point cloud.")

        # Determine anchor point coordinates
        if anchor is None or (isinstance(anchor, str) and anchor == "center"):
            anchor_pt = self.center_of_mass
        elif isinstance(anchor, str) and anchor in ("min", "max"):
            # If axis is specified, get extreme point along that axis
            ext_axis = axis if axis is not None else 0
            anchor_pt = self.get_extreme_points(axis=ext_axis, extreme=anchor)
        elif isinstance(anchor, np.ndarray):
            if anchor.shape != (self.dimension,):
                raise ValueError(f"Anchor shape must be {(self.dimension,)}, got {anchor.shape}")
            anchor_pt = anchor
        else:
            raise ValueError(f"Invalid anchor: {anchor}")

        if axis is None:
            # Uniform scaling based on overall pairwise Euclidean diameter
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
            # Non-uniform scaling along specified axis
            if axis < 0 or axis >= self.dimension:
                raise ValueError(f"Axis {axis} is out of bounds for dimension {self.dimension}")
                
            coords = self.points[:, axis]
            current_diameter = np.max(coords) - np.min(coords)
            
            if current_diameter == 0.0:
                raise ValueError(f"Current diameter along axis {axis} is 0, cannot scale.")
                
            scale_factor = target_diameter / current_diameter
            
            new_points = self.points.copy()
            new_points[:, axis] = anchor_pt[axis] + scale_factor * (coords - anchor_pt[axis])

        self._update_parent(new_points)
        return PointCloud(new_points, parent=self._parent)

    def rotate(
        self,
        angle: float,
        plane: Union[str, Tuple[int, int]] = "xy",
        center: Optional[Union[str, np.ndarray]] = None,
        use_degrees: bool = True
    ) -> "PointCloud":
        """Rotates the point cloud in the specified coordinate plane.
        
        Mathematical Formulation:
            Let theta be the rotation angle in radians, and 'c' be the center of rotation.
            For coordinate plane spanned by axes (a_1, a_2), we define the 2D rotation matrix:
                R = [[cos(theta), -sin(theta)],
                     [sin(theta),  cos(theta)]]
                     
            For each point P_i:
                1. Project and shift points to the rotation center:
                   u_i = [[ P_{i, a_1} - c_{a_1} ],
                          [ P_{i, a_2} - c_{a_2} ]]
                2. Apply the 2D rotation:
                   u'_i = R * u_i
                3. Update coordinates:
                   P'_{i, a_1} = c_{a_1} + (P_{i, a_1} - c_{a_1})*cos(theta) - (P_{i, a_2} - c_{a_2})*sin(theta)
                   P'_{i, a_2} = c_{a_2} + (P_{i, a_1} - c_{a_1})*sin(theta) + (P_{i, a_2} - c_{a_2})*cos(theta)
                   P'_{i, j} = P_{i, j}  for all coordinate axes j not in {a_1, a_2}
        
        Args:
            angle: The rotation angle.
            plane: The coordinate plane for rotation (e.g., 'xy', 'yz', or tuple of indices).
            center: The center of rotation. Can be:
                    - None / "center": Defaults to the center of mass.
                    - "min": The minimum extreme point along plane axis 1.
                    - "max": The maximum extreme point along plane axis 1.
                    - np.ndarray: A custom center coordinates array.
            use_degrees: If True, angle is interpreted as degrees. If False, as radians.
            
        Returns:
            A new PointCloud instance with rotated points.
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

        self._update_parent(new_points)
        return PointCloud(new_points, parent=self._parent)

    def translate(self, translation_vector: Any) -> "PointCloud":
        """Translates the point cloud by a given vector.
        
        Mathematical Formulation:
            Let P_i be the coordinate vector of point i, and v be the D-dimensional translation vector.
            The transformed point is computed as:
                P'_i = P_i + v
        
        Args:
            translation_vector: Array-like of shape (D,) representing translation.
            
        Returns:
            A new PointCloud instance with translated points.
        """
        v = np.asarray(translation_vector, dtype=float)
        if v.shape != (self.dimension,):
            raise ValueError(f"Translation vector must have shape {(self.dimension,)}, got {v.shape}")
        
        new_points = self.points + v
        self._update_parent(new_points)
        return PointCloud(new_points, parent=self._parent)

    def shear(
        self,
        factor: float,
        axis: int,
        control_axis: int,
        anchor: Optional[Union[str, np.ndarray]] = None
    ) -> "PointCloud":
        """Applies a shear deformation to the point cloud.
        
        Mathematical Formulation:
            Let k be the shear factor, and 'a' be the anchor coordinate vector.
            The coordinates along 'axis' (d_1) are shifted proportionally to the distance 
            from the anchor along the 'control_axis' (d_2):
                P'_{i, d_1} = P_{i, d_1} + k * (P_{i, d_2} - a_{d_2})
                P'_{i, j} = P_{i, j}  for all coordinate axes j != d_1
        
        Args:
            factor: The shear factor.
            axis: The axis along which displacement occurs.
            control_axis: The axis that controls the displacement.
            anchor: The anchor point where no displacement occurs. Can be:
                    - None / "center": Defaults to the center of mass.
                    - "min" / "max": Extreme point along the control axis.
                    - np.ndarray: Custom coordinates array.
                    
        Returns:
            A new PointCloud instance with sheared points.
        """
        if axis < 0 or axis >= self.dimension or control_axis < 0 or control_axis >= self.dimension:
            raise ValueError("Axes indices are out of bounds.")

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

        self._update_parent(new_points)
        return PointCloud(new_points, parent=self._parent)

    def apply_mapping(self, func: Any) -> "PointCloud":
        """Applies an arbitrary coordinate mapping function to the point cloud.
        
        Mathematical Formulation:
            Let P be the coordinate matrix of shape (N, D), and f be the mapping function.
            The transformed point cloud coordinate matrix is:
                P' = f(P)
        
        Args:
            func: A callable that accepts a NumPy array of shape (N, D) 
                  and returns a transformed NumPy array of shape (N, D).
                  
        Returns:
            A new PointCloud instance with the transformed points.
        """
        if not callable(func):
            raise TypeError("func must be a callable.")
        transformed = func(self.points)
        
        self._update_parent(transformed)
        return PointCloud(transformed, parent=self._parent)

    def twist(
        self,
        rate: float,
        plane: Union[str, Tuple[int, int]] = "xy",
        control_axis: int = 2,
        anchor: Optional[Union[str, np.ndarray]] = None,
        use_degrees: bool = True
    ) -> "PointCloud":
        """Applies a twist deformation by rotating points dynamically along a control axis.
        
        Mathematical Formulation:
            Let theta_i be the angle of rotation for point i, and 'a' be the anchor coordinate vector.
            For coordinate plane spanned by axes (a_1, a_2) and control axis c_ax:
                theta_i = rate * (P_{i, c_ax} - a_{c_ax})
                
            For each point P_i, we compute its rotation matrix R_i using theta_i:
                R_i = [[cos(theta_i), -sin(theta_i)],
                       [sin(theta_i),  cos(theta_i)]]
                       
            The coordinates are updated by applying rotation R_i in the plane:
                P'_{i, a_1} = a_{a_1} + (P_{i, a_1} - a_{a_1})*cos(theta_i) - (P_{i, a_2} - a_{a_2})*sin(theta_i)
                P'_{i, a_2} = a_{a_2} + (P_{i, a_1} - a_{a_1})*sin(theta_i) + (P_{i, a_2} - a_{a_2})*cos(theta_i)
                P'_{i, j} = P_{i, j}  for all coordinate axes j not in {a_1, a_2}
                
        Args:
            rate: The twist rate (rotation angle per unit distance along control axis).
            plane: The coordinate plane in which rotation occurs (e.g., 'xy', 'yz', or tuple of indices).
            control_axis: The axis that determines the angle of rotation.
            anchor: The anchor point serving as the center of rotation and zero-twist reference.
                    Can be:
                    - None / "center": Defaults to the center of mass.
                    - "min" / "max": Extreme point along the control axis.
                    - np.ndarray: Custom coordinates array.
            use_degrees: If True, rate is interpreted in degrees/unit distance.
                         If False, in radians/unit distance.
                         
        Returns:
            A new PointCloud instance with twisted points.
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

        # Determine anchor point coordinates
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

        # Compute rotation angles for each point
        dist_along_control = self.points[:, control_axis] - anchor_pt[control_axis]
        angles = rate * dist_along_control
        if use_degrees:
            angles = np.radians(angles)

        cos_vals = np.cos(angles)
        sin_vals = np.sin(angles)

        new_points = self.points.copy()
        
        # Center coordinates in rotation plane
        x = self.points[:, axis1] - anchor_pt[axis1]
        y = self.points[:, axis2] - anchor_pt[axis2]

        # Apply rotation
        new_points[:, axis1] = anchor_pt[axis1] + x * cos_vals - y * sin_vals
        new_points[:, axis2] = anchor_pt[axis2] + x * sin_vals + y * cos_vals

        self._update_parent(new_points)
        return PointCloud(new_points, parent=self._parent)

    def bend(
        self,
        curvature: float,
        axis: int = 0,
        control_axis: int = 1,
        anchor: Optional[Union[str, np.ndarray]] = None
    ) -> "PointCloud":
        """Bends the point cloud along a specified axis with a given curvature.
        
        Mathematical Formulation:
            Let c be the curvature (c = 1/R, where R is the bending radius).
            For bend axis d_1 and perpendicular control axis d_2, and anchor point 'a':
            For each point P_i, let:
                x = P_{i, d_1} - a_{d_1}
                y = P_{i, d_2} - a_{d_2}
                
            If curvature is non-zero (or very small):
                theta = c * x
                P'_{i, d_1} = a_{d_1} + (1/c - y) * sin(theta)
                P'_{i, d_2} = a_{d_2} + 1/c - (1/c - y) * cos(theta)
            If curvature is zero (or near zero), we use first-order Taylor expansion:
                P'_{i, d_1} = a_{d_1} + x - c * x * y
                P'_{i, d_2} = a_{d_2} + y + 0.5 * c * x^2
                
            For all other coordinate axes j not in {d_1, d_2}:
                P'_{i, j} = P_{i, j}
                
        Args:
            curvature: Bending curvature. Positive curvatures bend towards positive control axis.
            axis: The primary axis of extension along which bending occurs.
            control_axis: The axis perpendicular to the bend axis in which displacement occurs.
            anchor: The anchor point marking the start/origin of the bend. Can be:
                    - None / "center": Defaults to the center of mass.
                    - "min" / "max": Extreme point along the bend axis.
                    - np.ndarray: Custom coordinates array.
                    
        Returns:
            A new PointCloud instance with bent points.
        """
        if axis < 0 or axis >= self.dimension or control_axis < 0 or control_axis >= self.dimension:
            raise ValueError("Axis indices are out of bounds.")
        if axis == control_axis:
            raise ValueError("Bend axis and control axis must be distinct.")

        # Determine anchor point coordinates
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

        # Handle very small curvature to prevent division by zero using Taylor series approximation
        if np.abs(curvature) < 1e-8:
            new_points[:, axis] = anchor_pt[axis] + x - curvature * x * y
            new_points[:, control_axis] = anchor_pt[control_axis] + y + 0.5 * curvature * (x**2)
        else:
            r = 1.0 / curvature
            theta = curvature * x
            new_points[:, axis] = anchor_pt[axis] + (r - y) * np.sin(theta)
            new_points[:, control_axis] = anchor_pt[control_axis] + r - (r - y) * np.cos(theta)

        self._update_parent(new_points)
        return PointCloud(new_points, parent=self._parent)

    def taper(
        self,
        factor: float,
        axis: Optional[Union[int, List[int], Tuple[int, ...]]] = None,
        control_axis: int = 2,
        anchor: Optional[Union[str, np.ndarray]] = None
    ) -> "PointCloud":
        """Scales coordinates along specified axes proportionally to distance along a control axis.
        
        Mathematical Formulation:
            Let k be the tapering factor, and 'a' be the anchor coordinate vector.
            For each point P_i, the distance along the control axis c_ax is:
                h = P_{i, c_ax} - a_{c_ax}
                
            The coordinate multiplier is defined as:
                m(P_i) = 1 + k * h
                
            For each specified axis d:
                P'_{i, d} = a_d + m(P_i) * (P_{i, d} - a_d)
                
            For all other coordinate axes (including the control axis):
                P'_{i, j} = P_{i, j}
                
        Args:
            factor: The tapering scale factor.
            axis: The axis or list of axes to scale. If None, scales all axes except the control axis.
            control_axis: The axis along which the scale factor varies.
            anchor: The anchor point where scale factor is 1.0. Can be:
                    - None / "center": Defaults to the center of mass.
                    - "min" / "max": Extreme point along the control axis.
                    - np.ndarray: Custom coordinates array.
                    
        Returns:
            A new PointCloud instance with tapered points.
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

        # Determine anchor point coordinates
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

        self._update_parent(new_points)
        return PointCloud(new_points, parent=self._parent)

    def radial_scale(
        self,
        factor: float,
        center: Optional[Union[str, np.ndarray]] = None
    ) -> "PointCloud":
        """Scales points radially outward or inward from a given center.
        
        Mathematical Formulation:
            Let k be the scale factor, and 'c' be the center point coordinate.
            Each transformed point is computed as:
                P'_i = c + k * (P_i - c)
                
        Args:
            factor: Radial scale factor.
            center: The center of scaling. Can be:
                    - None / "center": Defaults to the center of mass.
                    - np.ndarray: Custom coordinates array.
                    
        Returns:
            A new PointCloud instance with radially scaled points.
        """
        if center is None or (isinstance(center, str) and center == "center"):
            c = self.center_of_mass
        elif isinstance(center, np.ndarray):
            if center.shape != (self.dimension,):
                raise ValueError("Center shape must match dimension.")
            c = center
        else:
            raise ValueError(f"Invalid center: {center}")

        new_points = c + factor * (self.points - c)
        self._update_parent(new_points)
        return PointCloud(new_points, parent=self._parent)

    def spherize(
        self,
        factor: float,
        radius: Optional[float] = None,
        center: Optional[Union[str, np.ndarray]] = None
    ) -> "PointCloud":
        """Blends points between their current layout and their projection onto a sphere.
        
        Mathematical Formulation:
            Let f be the spherization factor (0.0 to 1.0), R be the sphere radius, 
            and 'c' be the sphere center.
            For each point P_i, let:
                v_i = P_i - c
                d_i = || v_i ||_2 (Euclidean distance from center)
                
            If d_i > 0, the spherized projection P_{sph, i} is:
                P_{sph, i} = c + R * (v_i / d_i)
                
            The final blended coordinate is:
                P'_i = (1 - f) * P_i + f * P_{sph, i} = c + v_i * (1 - f + f * R / d_i)
                
            If d_i = 0, the point remains static:
                P'_i = P_i
                
        Args:
            factor: The blending factor between 0.0 (no change) and 1.0 (perfect sphere).
            radius: The target sphere radius. If None, defaults to the average distance 
                    of the point cloud from the center.
            center: The sphere center. Can be:
                    - None / "center": Defaults to the center of mass.
                    - np.ndarray: Custom coordinates array.
                    
        Returns:
            A new PointCloud instance with spherized points.
        """
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

        diff = self.points - c
        distances = np.linalg.norm(diff, axis=1)

        if radius is None:
            # Default to average distance
            radius = np.mean(distances) if len(distances) > 0 else 1.0
        elif radius < 0.0:
            raise ValueError("Radius must be non-negative.")

        # Compute scaling factors for each point
        scale = np.ones_like(distances)
        valid_indices = distances > 0
        
        # Formula: 1 - f + f * (radius / d_i)
        scale[valid_indices] = (1.0 - factor) + factor * (radius / distances[valid_indices])

        new_points = c + diff * scale[:, np.newaxis]
        self._update_parent(new_points)
        return PointCloud(new_points, parent=self._parent)
