from typing import Tuple
import numpy as np


def normalize_image(image: np.ndarray) -> np.ndarray:
    return (image - np.min(image)) / (np.max(image) - np.min(image))


def create_difference_map(_img_1: np.ndarray, _img_2: np.ndarray) -> np.ndarray:
    """Create a difference map between images."""
    try:
        if _img_1.shape != _img_2.shape or _img_1 is None or _img_2 is None:
            raise ValueError("Images must have the same shape.")
        difference_map = _img_1.astype(np.float32) - _img_2.astype(np.float32)
        return difference_map
    except Exception as e:
        print(f"Error in create_difference_map: {str(e)}")


def line_profile(
    img: np.ndarray,
    line_coords: Tuple[Tuple[int, int], Tuple[int, int]],
    layer_index: int
) -> np.ndarray:
    """
    Extracts and normalizes the intensity profile along a line in a specific layer of a 3D image.
    Parameters:
    - img: 3D NumPy array (C x H x W) representing the image.
    - line_coords: Coordinates of the start and end points of the line.
    - layer_index: Index of the layer to extract the profile from.

    Returns:
    - A normalized NumPy array containing the intensity profile along the line.
    """
    try:
        if layer_index >= img.shape[0]:
            raise IndexError("Layer index out of bounds.")

        profile_values = []
        (start_x, start_y), (end_x, end_y) = line_coords
        start_x, start_y, end_x, end_y = int(start_x), int(start_y), int(end_x), int(end_y)
        if (0 <= start_x < img.shape[2] and 0 <= start_y < img.shape[1] and
                0 <= end_x < img.shape[2] and 0 <= end_y < img.shape[1]):
            num_points = max(abs(end_x - start_x), abs(end_y - start_y)) + 1
            x_values = np.linspace(start_x, end_x, num_points).astype(int)
            y_values = np.linspace(start_y, end_y, num_points).astype(int)
            for x, y in zip(x_values, y_values):
                profile_values.append(img[layer_index, y, x])
        else:
            raise ValueError(
                f"Coordinates ({start_x}, {start_y}) to ({end_x}, {end_y}) out of bounds for image of shape {img.shape}.")
        return np.array(profile_values)

    except (IndexError, ValueError, Exception) as e:
        print(f"Error in line_profile: {str(e)}")
        raise


def extract_segment_from_layer(array_3d: np.ndarray, layer_index: int, segment_coords: Tuple[Tuple[int, int], Tuple[int, int]]) -> np.ndarray:
    """
    Extract a segment from a specific layer in a 3D numpy array.
    Parameters:
    - array_3d: 3D numpy array to extract from (shape: (layers, height, width))
    - layer_index: The index of the layer to extract from
    - x_start, x_end: Range of rows (height) to extract
    - y_start, y_end: Range of columns (width) to extract
    Returns:
    - segment: The extracted 2D numpy array (segment from the specified layer)
    """
    try:
        if layer_index >= array_3d.shape[0]:
            raise IndexError("Layer index out of bounds.")
        segment = array_3d[layer_index, segment_coords[0][0]:segment_coords[1][0], segment_coords[0][1]:segment_coords[1][1]]
        return segment
    except (ValueError, Exception) as e:
        print(f"Error in extract_segment_from_layer: {str(e)}")
        raise


def remove_frames(img: np.ndarray, remove_start: int, remove_end: int) -> np.ndarray:
    """
    Remove the specified number of frames from the start and end along the depth axis.
    Parameters:
    - img: 3D numpy array (shape: depth, height, width)
    - remove_start: Number of frames to remove from the start (depth axis).
    - remove_end: Number of frames to remove from the end (depth axis).
    Returns:
    - Cropped image as a 3D numpy array.
    """
    return img[remove_start:img.shape[0] - remove_end, :, :]


def extract_plane(img: np.ndarray, index: int, axis: int) -> np.ndarray:
    """
    Extracts a specific plane from a 3D image based on the given axis.

    Parameters:
    - img (np.ndarray): 3D input image of shape (layers, rows, cols).
    - index (int): The row, column, or layer index to extract.
    - axis (int): Extraction axis:
        - 0 -> Extracts row plane
        - 1 -> Extracts column plane
        - 2 -> Extracts layer plane

    Returns:
    - np.ndarray: Extracted 2D plane.
    """
    if axis not in (0, 1, 2):
        raise ValueError("Invalid axis. Choose from 0 (row), 1 (column), or 2 (layer).")

    return np.copy(np.take(img, index, axis=axis))

