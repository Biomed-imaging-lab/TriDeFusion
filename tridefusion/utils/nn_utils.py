from typing import Any
import numpy as np
import torch
import tifffile as tiff
from .decorators import performance_monitor

def reformat_to_zchw(img: np.ndarray) -> np.ndarray:
    """
    Convert input image to (Z, C, H, W) format for processing.
    Supported input shapes:
        (H, W)
        (H, W, C)
        (Z, H, W)
        (Z, H, W, C)

    Args:
        img (np.ndarray): Input image.
    Returns:
        np.ndarray: Image in (Z, C, H, W) format.
    """
    original_shape = img.shape
    ndim = img.ndim

    if ndim == 2:  # (H, W)
        return img[np.newaxis, np.newaxis, ...]  # (1, 1, H, W)

    if ndim == 3:
        # Detect if last axis is channels (RGB, etc.)
        if img.shape[-1] <= 4:  # (H, W, C)
            return img.transpose(2, 0, 1)[np.newaxis, ...]  # (1, C, H, W)
        else:  # (Z, H, W)
            return img[:, np.newaxis, ...]  # (Z, 1, H, W)

    if ndim == 4:  # (Z, H, W, C)
        return img.transpose(0, 3, 1, 2)  # (Z, C, H, W)
    raise ValueError(f"Unsupported input shape: {original_shape}")

@performance_monitor  
def nn_denoise(img: np.ndarray, model: Any, save_path=None, device='cuda'):
    """
    Denoise a 3D image stack using a pretrained model.
    Preserves original dtype in output.
    Automatically handles FP16 models.
    """
    try:
        original_dtype = img.dtype
        
        img = reformat_to_zchw(img)

        # Convert RGB to grayscale if needed (expecting zchw)
        if img.shape[1] == 3:
            img = img.mean(axis=1, keepdims=True)

        # Normalize to [0,1]
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        img = img.astype(np.float32)

        denoised_stack = np.empty_like(img, dtype=np.float32)

        # Determine model dtype
        model_dtype = next(model.parameters()).dtype

        for z in range(img.shape[0]):
            print(f"Processing slice {z+1}/{img.shape[0]}...")
            tensor = torch.from_numpy(img[z:z+1]).to(device)

            # Convert input tensor to match model dtype
            tensor = tensor.to(dtype=model_dtype)

            with torch.no_grad():
                denoised_slice = model(tensor)

            # Detach, move to CPU, convert to float32
            denoised_stack[z:z+1] = denoised_slice.detach().cpu().float().numpy()

        # Convert back to original dtype
        if np.issubdtype(original_dtype, np.integer):
            max_val = np.iinfo(original_dtype).max
            denoised = (np.clip(denoised_stack, 0, 1) * max_val).astype(original_dtype)
        else:
            denoised = denoised_stack.astype(original_dtype)
        if save_path:
            tiff.imwrite(save_path, denoised)
        if denoised.ndim == 4 and denoised.shape[1] == 1:
            denoised = denoised[:, 0, :, :]
        return denoised
    except Exception as e:
        print(f"Error in nn_denoise: {str(e)}")
        raise