import os
import sys
import cv2
import numpy as np
from PIL import Image

from utils import crop_image, load_image, normalize_image, save_tiff
from exceptions import check_not_none, valid_method_name
from src.decorators import memory_monitor
from models.filters import bm3d_denoise, gauss_blur, median_filter_img, nlm_denoise_image

DENOISE_METHODS = ['gaussian', 'median', 'nlm', 'bm3d', 'fluoro_msa', 'attention_unet', 'tridefusion']
GAUSSIAN_SIGMA = 3
MEDIAN_SIGMA = 3
SIGMA_PSD = 0.05
NOISE2NOISE_DIV = 32

class Denoiser:
    def __init__(self):
        self.__noisy_img = None

    @property
    def noisy_img(self) -> np.ndarray:
        return self.__noisy_img

    @memory_monitor
    def preprocess_image(self, _source_img: np.ndarray) -> None:
        try:
            preprocessed_img = normalize_image(
                np.stack(
                    [
                        cv2.cvtColor(layer, cv2.COLOR_BGR2GRAY)
                        if len(layer.shape) == 3
                        else layer
                        for layer in _source_img
                    ],
                    axis=0,
                ).astype(float),
                dtype=_source_img.dtype,
            )
            self.__noisy_img = preprocessed_img
            return preprocessed_img
        except Exception as e:
            print(f"Error in preprocess_image: {str(e)}")
            raise

    @memory_monitor
    def denoise_image(self, method: str = 'gaussian') -> None:
        try:
            valid_method_name(method=method, method_list=DENOISE_METHODS, method_type="denoise method")
            check_not_none(value=self.__noisy_img, name="Noisy image")

            if method == 'gaussian':
                denoised_img = gauss_blur(_img=self.__noisy_img, gaussian_sigma=GAUSSIAN_SIGMA)
            elif method == 'median':
                denoised_img = median_filter_img(_img=self.__noisy_img, _block_size=MEDIAN_SIGMA)
            elif method == 'nlm':
                denoised_img = nlm_denoise_image(_noisy_img=self.__noisy_img)
            elif method == 'bm3d':
                denoised_img = bm3d_denoise(_img=self.__noisy_img, sigma_psd=SIGMA_PSD)
            elif method == 'fluoro_msa':
                # TODO: Implement fluoro_msa denoising
                pass
            elif method == 'attention_unet':
                # TODO: Implement attention_unet denoising
                pass
            elif method == 'tridefusion-v1':
                # TODO: Implement tridefusion denoising
                pass
            elif method == 'tridefusion-v2':
                # TODO: Implement tridefusion denoising
                pass
            elif method == 'tridefusion-v2-mini':
                # TODO: Implement tridefusion denoising
                pass

            denoised_img = denoised_img.astype(self.__noisy_img.dtype)
            self.__denoised_img = denoised_img
            return denoised_img
        except Exception as e:
            print(f"Error in denoise_image: {str(e)}")
            raise
