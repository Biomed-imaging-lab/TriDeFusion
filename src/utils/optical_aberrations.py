from typing import List
import numpy as np
from scipy.ndimage.filters import gaussian_filter, median_filter
from utils.exceptions import check_not_none, check_positive_integer


def add_poisson_noise(img : np.ndarray, lambda_val : int) -> np.ndarray:
    return (np.sum([
                    img.astype("float32"),
                    (
                        np.random.poisson(
                            lambda_val,
                            np.prod(img.shape),
                        )
                        .reshape(img.shape)
                        .astype("float32")
                    ),
                ],
                axis=0,
            ).astype("float32")).astype("uint8")


def gauss_blur(_img: np.ndarray, gaussian_sigma: int = 3) -> np.ndarray:
    check_not_none(value=_img, name="Image")
    check_positive_integer(value=gaussian_sigma, name="Gaussian sigma")
    _img = gaussian_filter(_img, sigma=gaussian_sigma)
    return _img


def median_filter_img(_img: np.ndarray, _block_size: int = 3) -> np.ndarray:
    check_not_none(value=_img, name="Image")
    check_positive_integer(value=_block_size, name="Block size")
    _img = median_filter(_img, size=_block_size)
    return _img


def make_aberration_image(_clean_img: np.ndarray, gaussian_sigma: int = 3, noise_level: int = 8) -> np.ndarray:
    check_not_none(value=_clean_img, name="Clean image")
    _aberration_img = add_poisson_noise(img=gauss_blur(_img=_clean_img, gaussian_sigma=gaussian_sigma), lambda_val=noise_level)
    return _aberration_img


def make_noise_images(_clean_img: np.ndarray,
                      _noise_levels: np.ndarray,
                      _specific_noise_level: int) -> List[np.ndarray]:
    try:
        max_val = 255 if _clean_img.dtype == np.uint8 else 1.0 
        noisy_images = [
            add_poisson_noise(_img=_clean_img, lambda_val=noise_level, max_val=max_val)
            for noise_level in _noise_levels
        ]
        return noisy_images
    except (ValueError, Exception) as e:
        print(f"Error in add_poisson_noise: {str(e)}")
        raise