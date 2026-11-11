"""ROS-independent pinhole registration. All distances are metres."""

import numpy as np


def rigid_transform(translation, quaternion):
    """Build a homogeneous transform from XYZ translation and an XYZW quaternion."""
    x, y, z, w = np.asarray(quaternion, dtype=float)
    norm = np.linalg.norm([x, y, z, w])
    if norm < 1e-12:
        raise ValueError("Invalid TF quaternion")
    x, y, z, w = np.array([x, y, z, w]) / norm
    matrix = np.eye(4)
    matrix[:3, :3] = [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]
    matrix[:3, 3] = translation
    return matrix


def register_depth(
    depth, depth_k, color_k, depth_to_color, color_shape, minimum, maximum
):
    """Backproject depth, transform, project, retaining the nearest Z per pixel."""
    depth_intrinsics = np.asarray(depth_k).reshape(3, 3)
    color_intrinsics = np.asarray(color_k).reshape(3, 3)
    fx, fy = depth_intrinsics[0, 0], depth_intrinsics[1, 1]
    cx, cy = depth_intrinsics[0, 2], depth_intrinsics[1, 2]
    if min(fx, fy, color_intrinsics[0, 0], color_intrinsics[1, 1]) <= 0:
        raise ValueError("Invalid camera focal length")
    valid_depth = np.isfinite(depth) & (depth >= minimum) & (depth <= maximum)
    depth_rows, depth_cols = np.nonzero(valid_depth)
    z = depth[depth_rows, depth_cols]
    depth_points = np.column_stack(
        ((depth_cols - cx) * z / fx, (depth_rows - cy) * z / fy, z)
    )

    color_points = depth_points @ depth_to_color[:3, :3].T + depth_to_color[:3, 3]
    in_front = np.isfinite(color_points).all(axis=1) & (color_points[:, 2] > 0)
    color_points = color_points[in_front]
    height, width = color_shape
    registered = np.full(height * width, np.inf)
    if len(color_points):
        projected = color_points @ color_intrinsics.T
        cols = np.rint(projected[:, 0] / projected[:, 2]).astype(int)
        rows = np.rint(projected[:, 1] / projected[:, 2]).astype(int)
        inside = (cols >= 0) & (cols < width) & (rows >= 0) & (rows < height)
        pixel_indices = rows[inside] * width + cols[inside]
        np.minimum.at(registered, pixel_indices, color_points[inside, 2])
    return registered.reshape(height, width)


def sample_target(registered, box, color_k, min_pixels=3):
    """Estimate a visible surface point from depth samples inside the box."""
    x1, y1, x2, y2 = box
    h, w = registered.shape
    left = max(0, int(np.ceil(x1 + (x2 - x1) * 0.25)))
    right = min(w, int(np.ceil(x2 - (x2 - x1) * 0.25)))
    top = max(0, int(np.ceil(y1 + (y2 - y1) * 0.25)))
    bottom = min(h, int(np.ceil(y2 - (y2 - y1) * 0.25)))
    roi = registered[top:bottom, left:right]
    vv, uu = np.nonzero(np.isfinite(roi))
    if len(uu) < min_pixels:
        raise ValueError("Not enough valid registered depth pixels inside target")
    z = roi[vv, uu]
    k = np.asarray(color_k).reshape(3, 3)
    points = np.column_stack(
        ((uu + left - k[0, 2]) * z / k[0, 0], (vv + top - k[1, 2]) * z / k[1, 1], z)
    )
    return np.median(points, axis=0), len(points)
