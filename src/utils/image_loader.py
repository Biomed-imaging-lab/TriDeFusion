import os
from typing import List, Tuple, Union

import cv2
import numpy as np
from PIL import Image
from utils.exceptions import check_not_none


def load_images(image_input: Union[str, Tuple[str, ...]]) -> np.ndarray:
        """
        Unified method to load either a multi-stack image (3D) or multiple 2D images.
        
        Args:
            image_input (Union[str, Tuple[str, ...]]): Path to a multi-stack image (str) 
                                                 or list of paths to 2D images (Tuple[str, ...]).
            
        Returns:
            np.ndarray: Loaded image as a NumPy array.
                - For multi-stack image: Shape (N, H, W) or (N, H, W, C)
                - For merged 2D images: Shape (N, H, W, C)
        """
        try:
            if isinstance(image_input, str):
                print(f"Loading a single multi-stack image from: {image_input}")
                source_img = load_multistack_image(image_input)
            elif isinstance(image_input, tuple) and all(isinstance(path, str) for path in image_input):
                print(f"Loading and merging multiple 2D images from paths: {image_input}")
                source_img = load_multiple_images(image_input)
            else:
                raise ValueError(f"Invalid input type for image_input: {type(image_input)}. Must be str or List[str].")
            print(f"Image uploaded successfully. Loaded image shape: {source_img.shape}")
            return source_img
        except Exception as e:
            print(f"Error in load_images: {str(e)}")
            raise

def load_multistack_image(image_path: str) -> np.ndarray:
    """
    Load a multi-stack image (3D) from a given path.

    Args:
        image_path (str): Path to the image file.
    Returns:
        np.ndarray: Loaded 3D image as a NumPy array of shape (N, H, W), 
                    where N is the number of frames in the image stack.
    """
    try:
        check_not_none(value=image_path, name="Loadable image path", func_name="load_image")
        if not os.path.exists(image_path):
            raise ValueError("File not found: {image_path}")
        ret, images = cv2.imreadmulti(image_path, [], cv2.IMREAD_ANYCOLOR)
        if ret is None or len(images) <= 0:
            raise ValueError("Failed to read the source image from path: {image_path}")
        source_img = np.asarray(images)
        return source_img
    except (ValueError, FileNotFoundError, Exception) as e:
        print(f"Error in load_image: {str(e)}")
        raise

def load_multiple_images(image_paths: Tuple[str, ...]) -> np.ndarray:
    """
    Load multiple 2D images from a tuple of paths and merge them into a single 4D NumPy array.
    
    Args:
        image_paths (Tuple[str, ...]): Tuple of image paths.
        
    Returns:
        np.ndarray: Merged 4D NumPy array of shape (N, H, W, C), 
                    where N is the total number of images, H is height, W is width, and C is channels.
    """
    try:
        if not image_paths or len(image_paths) == 0:
            raise ValueError("Image paths tuple must not be empty")
        layers = []
        for path in image_paths:
            if not os.path.exists(path):
                raise FileNotFoundError(f"File not found: {path}")
            image = cv2.imread(path, cv2.IMREAD_COLOR) 
            if image is None:
                raise ValueError(f"Failed to read image from path: {path}")
            layers.append(image)
        layer_shapes = [layer.shape for layer in layers]
        if len(set(layer_shapes)) > 1:
            raise ValueError(f"All images must have the same shape, but got shapes: {layer_shapes}")
        merged_image = np.stack(layers, axis=0)  
        print(f"Image uploaded successfully. Loaded image shape: {merged_image.shape}")  
        return merged_image
    except Exception as e:
        print(f"Error in load_and_merge_images: {str(e)}")
        raise

def save_tiff(image_3d: np.ndarray, path: str, color_channel: int | None = None, dtype: str = "uint8") -> None:
    """
    Saves a 3D numpy array as a multi-page TIFF with optional color channel selection and data type conversion.

    Parameters:
    - image_3d (np.ndarray): 3D array of image layers.
    - path (str): File path to save the TIFF.
    - color_channel (int | None): Color channel to use (0 for red, 1 for green, 2 for blue). If None, save grayscale.
    - dtype (str): Data type for saving the images ('uint8', 'uint16', etc.).

    Returns:
    - None
    """
    images_list = []
    for layer in image_3d:
        if color_channel is not None:
            color_layer = np.zeros((layer.shape[0], layer.shape[1], 3), dtype=dtype)
            color_layer[..., color_channel] = layer.astype(dtype)
            layer = color_layer
        else:
            layer = layer.astype(dtype)
        images_list.append(Image.fromarray(layer))
    images_list[0].save(path, save_all=True, append_images=images_list[1:])
    print(f"TIFF saved to {path}")

def tiff_to_gif(input_tiff: str, output_gif: str, duration: int = 50, loop: int = 1):
    """
    Convert a multi-frame TIFF image to an animated GIF.
    
    :param input_tiff: Path to the input multi-frame TIFF file.
    :param output_gif: Path to save the output GIF file.
    :param duration: Duration of each frame in milliseconds.
    :param loop: Number of loops (0 for infinite looping).
    """
    with Image.open(input_tiff) as img:
        frames = []
        try:
            while True:
                frames.append(img.convert("RGB").copy()) 
                img.seek(img.tell() + 1)
        except EOFError:
            pass  
        
        if frames:
            frames[0].save(output_gif, save_all=True, append_images=frames[1:], duration=duration, loop=loop, disposal=2)
    
    print(f"GIF saved as {output_gif}")
