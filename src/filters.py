import cv2
import numpy as np
from scipy.ndimage import gaussian_filter, median_filter
from bm3d import bm3d
from exceptions import check_not_none, check_positive_integer


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


def bm3d_denoise(_img: np.ndarray, sigma_psd: float = 0.05) -> np.ndarray:
    """Apply BM3D denoising to the image."""
    if sigma_psd <= 0:
        raise ValueError("Sigma PSD must be a positive float")
    denoised_img = bm3d(_img, sigma_psd)
    return denoised_img


def nlm_denoise_layer(_noisy_layer: np.ndarray) -> np.ndarray:
    try:
        _noisy_layer = cv2.convertScaleAbs(_noisy_layer)
        _denoised_layer = cv2.fastNlMeansDenoising(_noisy_layer, None, 10, 7, 21)
        return _denoised_layer
    except (ValueError, Exception) as e:
        print(f"Error in nlm_denoise_layer: {str(e)}")
        raise
        
def nlm_denoise_image(_noisy_img: np.ndarray) -> None:
    try:
        nlm_denoised_layers = []
        for layer in _noisy_img:
            _denoised_layer = nlm_denoise_layer(_noisy_layer=layer)
            if _denoised_layer is not None:
                nlm_denoised_layers.append(_denoised_layer)
        if len(nlm_denoised_layers) != 0:
            return np.array(nlm_denoised_layers, dtype=float)
    except (ValueError, Exception) as e:
        print(f"Error in nlm_denoise_image: {str(e)}")
        raise

