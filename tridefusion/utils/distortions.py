import os
from typing import List
import numpy as np
from pathlib import Path
import tifffile as tiff
from .exceptions import check_not_none, check_positive_integer


def add_poisson_noise(img : np.ndarray, lambda_val : int) -> np.ndarray:
    return (np.sum([
                    img.astype("float32"),
                    (
                        np.random.poisson(
                            lam=lambda_val,
                            size=np.prod(img.shape),
                        )
                        .reshape(img.shape)
                        .astype("float32")
                    ),
                ],
                axis=0,
            ).astype("float32")).astype("uint8")

def add_gaussian_noise(img: np.ndarray, mu: float = 0, sigma: float = 0.1):
    return (np.sum([
                    img.astype("float32"),
                    (
                        np.random.normal(
                            loc=mu,
                            scale=sigma,
                            size=np.prod(img.shape),
                        )
                        .reshape(img.shape)
                        .astype("float32")
                    ),
                ],
                axis=0,
            ).astype("float32")).astype("uint8")

def add_mpg_noise(img: np.ndarray, gaussian_sigma: int = 0.1, poisson_level: int = 8) -> np.ndarray:
    check_not_none(value=img, name="Clean image")
    return add_gaussian_noise(img=add_poisson_noise(img=img, lambda_val=poisson_level), sigma=gaussian_sigma)


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
        img = tiff.imread(img_file)
        for p_level in poisson_levels:
            for g_sigma in gaussian_levels:
                noisy_img = add_mpg_noise(img, gaussian_sigma=g_sigma, poisson_level=p_level)
                combo_folder = output_path / f"poisson_{p_level}_gauss_{g_sigma}"
                combo_folder.mkdir(exist_ok=True, parents=True)
                out_file = combo_folder / img_file.name
                tiff.imwrite(out_file, noisy_img)
                print(f"Saved {out_file}")

