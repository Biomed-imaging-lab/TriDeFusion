import os
from typing import List
import numpy as np
from pathlib import Path
import tifffile as tiff


def add_poisson_noise(img: np.ndarray, scale: float = 1.0) -> np.ndarray:
    """Signal-dependent Poisson noise"""
    img = img.astype(np.float32)
    noisy = np.random.poisson(img * scale) / scale
    return noisy.astype(np.float32)

def add_gaussian_noise(img: np.ndarray, mu: float = 0.0, sigma: float = 0.1) -> np.ndarray:
    """Additive Gaussian noise"""
    img = img.astype(np.float32)
    noise = np.random.normal(mu, sigma, img.shape).astype(np.float32)
    return img + noise

def add_mpg_noise(img, gaussian_sigma: float = 0.1, poisson_scale: float = 1.0) -> np.ndarray:
    img = np.nan_to_num(img, nan=0.0)      
    img = np.clip(img, 0.0, 1.0)           
    img_poisson = add_poisson_noise(img, scale=poisson_scale)
    img_mpg = add_gaussian_noise(img_poisson, sigma=gaussian_sigma)
    return np.clip(img_mpg, 0.0, 1.0).astype(np.float32)

def apply_noise_to_folder(
    input_folder: str,
    output_folder: str,
    poisson_levels: List[int],
    gaussian_levels: List[float],
    file_ext: str = ".tif"
):
    """
    Apply combinations of Poisson and Gaussian noise to TIFF images in a folder.

    Args:
        input_folder (str): Path to folder with clean images.
        output_folder (str): Path to save noisy images.
        poisson_levels (List[int]): List of Poisson noise lambda values.
        gaussian_levels (List[float]): List of Gaussian noise sigmas.
        file_ext (str): Image file extension to process.
    """
    input_path = Path(input_folder)
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    image_files = list(input_path.glob(f"*{file_ext}"))
    if not image_files:
        print(f"No {file_ext} files found in {input_folder}")
        return
    for img_file in image_files:
        img = tiff.imread(img_file).astype(np.float32)
        if img.max() > 1.0:
            img = img / np.amax(img)
        for p_level in poisson_levels:
            for g_sigma in gaussian_levels:
                noisy_img = add_mpg_noise(img, gaussian_sigma=g_sigma, poisson_scale=p_level)
                combo_folder = output_path / f"poisson_{p_level}_gauss_{g_sigma}"
                combo_folder.mkdir(exist_ok=True, parents=True)
                out_file = combo_folder / img_file.name
                tiff.imwrite(out_file, noisy_img)
                print(f"Saved {out_file}")

