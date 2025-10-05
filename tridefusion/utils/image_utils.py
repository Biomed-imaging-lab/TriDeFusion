import asyncio
import os
from typing import List, Tuple
import numpy as np
import cv2
from PIL.Image import Image
import torch
import tifffile


def normalize_image(img: np.ndarray) -> np.ndarray:
    """Normalize image to [0,1] float32"""
    img_min = np.min(img)
    img_max = np.max(img)
    if img_max - img_min < 1e-12:
        return np.zeros_like(img, dtype=np.float32)
    return ((img - img_min) / (img_max - img_min)).astype(np.float32)

def float32_to_uint8(img: np.ndarray) -> np.ndarray:
    return (img * 255).astype(np.uint8)

def uint8_to_float32(img: np.ndarray) -> np.ndarray:
    return img.astype(np.float32) / 255.0 

def load_image():
    pass

def save_image(image_numpy: np.ndarray, image_path: str) -> None:
    """Save a numpy image to the disk

    Parameters:
        image_numpy (numpy array) -- input numpy array
        image_path (str)          -- the path of the image
    """
    try:
        if image_numpy is None or image_path is None:
            tifffile.imsave(image_path, image_numpy)
    except Exception as e:
        print(f"Error in tensor2im: {str(e)}")    

def print_numpy(x: np.ndarray, val=True, shp=False):
    """Print the mean, min, max, median, std, and size of a numpy array

    Parameters:
        x (numpy array) -- input numpy array
        val (bool) -- if print the values of the numpy array
        shp (bool) -- if print the shape of the numpy array
    """
    try:
        x = x.astype(np.float64)
        if shp:
            print('shape,', x.shape)
        if val:
            x = x.flatten()
            print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
                np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))
    except Exception as e:
        print(f"Error in print_numpy: {str(e)}")


def tensor2im(image, imtype=np.uint16) -> np.ndarray:
    """Converts a Tensor array into a numpy image array.

    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    try:
        if not isinstance(image, np.ndarray):
            if isinstance(image, torch.Tensor):  # get the data from a variable
                image_tensor = image.data
            else:
                return image
            image_numpy = image_tensor.cpu().float().numpy()  # convert it into a numpy array
            image_numpy = np.rint((image_numpy + 1.0) / 2.0 * 255.0)
        else:  # if it is a numpy array, do nothing
            image_numpy = image
        return image_numpy.astype(imtype)
    except Exception as e:
        print(f"Error in tensor2im: {str(e)}")


def create_difference_map(img_1: np.ndarray, 
                          img_2: np.ndarray) -> np.ndarray:
    """
    Create a difference map between images.

    Parameters:
        img_1(numpy array) -- 3D NumPy array (L x H x W) representing the image.
        img_2(numpy array) -- 3D NumPy array (L x H x W) representing the image.
    """
    try:
        if img_1.shape != img_2.shape or img_1 is None or img_2 is None:
            raise ValueError("Images must have the same shape.")
        difference_map = img_1.astype(np.float32) - img_2.astype(np.float32)
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
    Args:
        img: 3D NumPy array (L x H x W) representing the image.
        line_coords: Coordinates of the start and end points of the line.
        layer_index: Index of the layer to extract the profile from.
    Returns:
        A normalized NumPy array containing the intensity profile along the line.
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
    Args:
        array_3d: 3D numpy array to extract from (shape: (layers, height, width))
        layer_index: The index of the layer to extract from
        x_start, x_end: Range of rows (height) to extract
        y_start, y_end: Range of columns (width) to extract
    Returns:
        segment: The extracted 2D numpy array (segment from the specified layer)
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
    Args:
        img (np.ndarray): 3D numpy array (shape: depth, height, width)
        remove_start (int): Number of frames to remove from the start (depth axis).
        remove_end (int): Number of frames to remove from the end (depth axis).
    Returns:
        Cropped image as a 3D numpy array.
    """
    try:
        if img.ndim != 3:
            raise IndexError("Layer index out of bounds.")
        return img[remove_start:img.shape[0] - remove_end, :, :]
    except (ValueError, Exception) as e:
        print(f"Error in remove_frames: {str(e)}")
        raise


def extract_plane(img: np.ndarray, index: int, axis: int) -> np.ndarray:
    """
    Extracts a specific plane from a 3D image based on the given axis.
    Args:
        img (np.ndarray): 3D input image of shape (layers, rows, cols).
        index (int): The row, column, or layer index to extract.
        axis (int): Extraction axis:
        - 0 -> Extracts row plane
        - 1 -> Extracts column plane
        - 2 -> Extracts layer plane

    Returns:
        np.ndarray: Extracted 2D plane.
    """
    if axis not in (0, 1, 2):
        raise ValueError("Invalid axis. Choose from 0 (row), 1 (column), or 2 (layer).")
    return np.copy(np.take(img, index, axis=axis))


def create_tiff_multistack(folder_path: str, 
                           output_path: str):
    """
    Creates a multi-stack TIFF file from a folder containing JPEG images.
    Args:
        folder_path (str): Path to the folder containing JPEG images.
        output_path (str): Path to save the output TIFF file.
    """
    images = sorted([f for f in os.listdir(folder_path) if f.lower().endswith('.jpg') or f.lower().endswith('.jpeg')])
    if not images:
        raise ValueError("No JPEG images found in the specified folder.")
    
    img_list = [cv2.imread(os.path.join(folder_path, img), cv2.IMREAD_COLOR)  for img in images]
    img_list[0].save(output_path, save_all=True, append_images=img_list[1:], format='TIFF')
    _noisy_img = load_image(output_path)
    controller = DenoiseController()
    controller.preprocess_image(_noisy_img)
    _noisy_img = controller.noisy_img
    print(f"Multi-stack TIFF saved to {output_path}")