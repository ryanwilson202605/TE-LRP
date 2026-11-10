"""Geometric preprocessing for AutoBot scene inputs."""
import numpy as np
from scipy.interpolate import interp1d

def get_polyline_dir(polyline):
    """Return normalized segment directions for a polyline (ndarray)."""
    polyline_pre = np.roll(polyline, shift=1, axis=0)
    polyline_pre[0] = polyline[0]
    diff = polyline - polyline_pre
    polyline_dir = diff / np.clip(np.linalg.norm(diff, axis=-1)[:, np.newaxis], a_min=1e-6, a_max=1000000000)
    return polyline_dir


def rotate_points_along_z(points, angle):
    """
    Rotate points around the Z-axis using the given angle.

    Args:
        points: ndarray of shape (B, N, 3 + C) - B batches, N points per batch, 3 coordinates (x, y, z) + C extra channels
        angle: ndarray of shape (B,) - angles for each batch in radians

    Returns:
        Rotated points as an ndarray.
    """

    # Checking if the input is 2D or 3D points
    is_2d = points.shape[-1] == 2

    # Cosine and sine of the angles
    cosa = np.cos(angle)
    sina = np.sin(angle)

    if is_2d:
        # Rotation matrix for 2D
        rot_matrix = np.stack((
            cosa, sina,
            -sina, cosa
        ), axis=1).reshape(-1, 2, 2)

        # Apply rotation
        points_rot = np.matmul(points, rot_matrix)
    else:
        # Rotation matrix for 3D
        rot_matrix = np.stack((
            cosa, sina, np.zeros_like(angle),
            -sina, cosa, np.zeros_like(angle),
            np.zeros_like(angle), np.zeros_like(angle), np.ones_like(angle)
        ), axis=1).reshape(-1, 3, 3)

        # Apply rotation to the first 3 dimensions
        points_rot = np.matmul(points[:, :, :3], rot_matrix)

        # Concatenate any additional dimensions back
        if points.shape[-1] > 3:
            points_rot = np.concatenate((points_rot, points[:, :, 3:]), axis=-1)

    return points_rot


def generate_mask(current_index, total_length, interval):
    """Return a sampling mask (ndarray) for current_index, total_length, and interval (int)."""
    mask = []
    for i in range(total_length):
        # Check if the position is a multiple of the frequency starting from current_index
        if (i - current_index) % interval == 0:
            mask.append(1)
        else:
            mask.append(0)

    return np.array(mask)


def find_true_segments(mask):
    """Return lists of contiguous valid indices from a Boolean mask (ndarray)."""
    # Find the indices where `True` changes to `False` and vice versa
    change_points = np.where(np.diff(mask))[0] + 1

    # Add the start and end indices
    indices = np.concatenate(([0], change_points, [len(mask)]))

    # Extract the segments of continuous `True`
    segments = [list(range(indices[i], indices[i + 1])) for i in range(len(indices) - 1) if mask[indices[i]]]

    return segments


def interpolate_polyline(polyline, step=0.5):
    """Resample a polyline at uniform arc-length intervals.

    Inputs: polyline (ndarray): source points; step (float): spacing in meters.
    Returns: ndarray: interpolated xy coordinates with zero z, or the single input point."""
    # Calculate the cumulative distance along the polyline
    if polyline.shape[0] == 1:
        return polyline
    polyline = polyline[:, :2]
    distances = np.cumsum(np.sqrt(np.sum(np.diff(polyline, axis=0) ** 2, axis=1)))
    distances = np.insert(distances, 0, 0)  # start with a distance of 0

    # Create the new distance array
    max_distance = distances[-1]
    new_distances = np.arange(0, max_distance, step)

    # Interpolate for x, y, z
    new_polyline = []
    for dim in range(polyline.shape[1]):
        interp_func = interp1d(distances, polyline[:, dim], kind='linear')
        new_polyline.append(interp_func(new_distances))

    new_polyline = np.column_stack(new_polyline)
    # add the third dimension back with zeros
    new_polyline = np.concatenate((new_polyline, np.zeros((new_polyline.shape[0], 1))), axis=1)
    return new_polyline
