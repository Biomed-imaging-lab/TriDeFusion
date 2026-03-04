import importlib
from pathlib import Path
from typing import Any, Callable, Tuple
import yaml

from .utils.decorators import performance_monitor
from .utils.logger import Logger
import os
import cv2
import asyncio
import numpy as np
import torch
import torch.nn as nn
from tifffile import imread, imwrite


SHOW_LOG = True

class Denoiser:
    def __init__(self, config_path = "config/method_params.yaml"):
        self.logger = Logger(SHOW_LOG)
        self.__noisy_data = None
        self.__denoised_data = None
        self.logger = Logger(show=SHOW_LOG).get_logger(__name__)
        self.logger.info("Denoiser created with config: %s", config_path)
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

    def _load_config(self, path: str, debug: bool = True) -> dict:
        """Load method configuration from YAML file."""
        config_file = Path(path)
        if not config_file.exists():
            self.logger.warning("Config file %s not found. Using defaults.", path)
            return {}
        with open(config_file, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        if debug:
            self.logger.info("Loaded config parameters:", config)
        self.logger.info("Loaded config from %s: %s", path, config)
        return config or {}

    @property
    def noisy_data(self) -> np.ndarray:
        return self.__noisy_data
    
    @property
    def denoised_data(self) -> np.ndarray:
        return self.__denoised_data
    
    @performance_monitor 
    def tridefusion_denoise(img: np.ndarray, 
                            nn_model: Any, 
                            filter_func: Tuple[str, Callable[[np.ndarray], np.ndarray]], 
                            save_path=None, 
                            device='cuda'):
        try:
            rauden_img = nn_denoise(img, nn_model, device=device)
            method_name, denoise_fn = filter_func
            print(f"Using {method_name}")
            denoised = denoise_fn(rauden_img)
            if save_path:
                tiff.imwrite(save_path, denoised)
            return denoised
        except Exception as e:
            print(f"Error in tridefusion_denoise: {str(e)}")
            raise

    @memory_monitor(print_params=None)
    def _preprocess_image(self, source_img: np.ndarray) -> np.ndarray:
        try:
            self.original_dtype = source_img.dtype
            self.original_shape = source_img.shape
            self.num_channels = (
                source_img.shape[-1] if source_img.ndim >= 3 else 1
            )
            preprocessed_img = normalize_image(
                np.stack(
                    [
                        cv2.cvtColor(layer, cv2.COLOR_BGR2GRAY)
                        if len(layer.shape) == 3
                        else layer
                        for layer in source_img
                    ],
                    axis=0,
                ).astype(float)
            )
            self.__noisy_data = preprocessed_img
            return preprocessed_img
        except Exception as e:
            self.log.error(f"Error in preprocess_image: {str(e)}")
            raise
    
    @performance_monitor
    def _apply_method(self, method: str) -> np.ndarray:
        try:
            check_not_none(self.__noisy_data, "Noisy image")
            method_cfg = self.config.get(method)
            if not method_cfg:
                raise ValueError(f"No config found for method '{method}'")
            func_name = method_cfg["function"]
            params = method_cfg.get("parameters", {})
            if func_name is None:
                raise NotImplementedError(f"Denoising method '{method}' is not implemented.")
            filters_module = importlib.import_module("src.filters")
            if not hasattr(filters_module, func_name):
                raise AttributeError(f"Function '{func_name}' not found in 'src.filters' module")
            func = getattr(filters_module, func_name)
            denoised = func(img=self.__noisy_data, **params)
            return denoised
        except (ValueError, Exception) as e:
                self.log.error(f"Error in _apply_method: {str(e)}")
                raise
    
    @memory_monitor(print_params=None)
    def build_network(self,
                      model_path: str,
                      device: str,
                      model_type: nn.Module):
        try:
            model = model_type(in_channels=1, out_channels=1)
            model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
            model.eval()
            self.log.info("Model preloaded successfully!")
        except (ValueError, Exception) as e:
                self.log.error(f"Error in build_network: {str(e)}")
                raise


    @performance_monitor
    def denoise_image(self, noisy_img: np.ndarray, 
                      method: str = "gaussian",
                      normalize_output: bool = True,
                      save_path: str = None) -> np.ndarray:
        try:
            self.validate_functions(method)
            self._preprocess_image(noisy_img)
            denoised = self._apply_method(method)
            denoised = np.clip(denoised, 0, 1)  
            if np.issubdtype(self.original_dtype, np.integer):
                max_val = np.iinfo(self.original_dtype).max
            else:
                max_val = 1.0 
            denoised = (denoised * max_val).astype(self.original_dtype)
            if self.num_channels == 3:
                denoised = np.stack([denoised] * 3, axis=-1) 
            elif self.original_shape and len(self.original_shape) == 4:
                denoised = denoised.reshape(self.original_shape)
            self.__denoised_data = denoised
            if save_path:
                imwrite(save_path, denoiser.denoised_data)
            return denoised
        except Exception as e:
            self.log.error(f"Error in denoise_image: {str(e)}")
            raise
    
    @memory_monitor(print_params=None)
    def denoise_folder(self, input_dir: str, output_dir: str):
        """
        Recursively applies denoise_image (callable) to .tiff/.tifff (.npy) files in input_dir and saves results to output_dir.
        Parameters:
        - input_dir (str): Path to the input directory containing noisy 3D images.
        - output_dir (str): Path to the output directory to save denoised images.
        """
        for root, _, files in os.walk(input_dir):
            for file in files:
                if file.endswith('.tiff'):
                    input_file_path = os.path.join(root, file)
                    relative_path = os.path.relpath(root, input_dir)
                    output_subdir = os.path.join(output_dir, relative_path)
                    os.makedirs(output_subdir, exist_ok=True)
                    noisy_image = imread(input_file_path)
                    print(noisy_image.dtype, noisy_image.max())
                    noisy_image = normalize_image(noisy_image)
                    denoised_image = self.denoise_image(noisy_image)
                    output_file_path = os.path.join(output_subdir, file)
                    imwrite(output_file_path, denoised_image)

    @memory_monitor(print_params=None)
    def denoise_video(self, video_path, output_frames_dir, fps):
        """
        Extracts frames from a video, applies denoising, and saves them.
        Parameters:
        - video_path (str): Path to the input video file.
        - output_frames_dir (str): Directory where denoised frames will be saved.
        - fps (int): Number of frames per second to extract and process.
        """
        os.makedirs(output_frames_dir, exist_ok=True)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Failed to open video: {video_path}")
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        if original_fps == 0:
            raise ValueError(f"Failed to get FPS from video: {video_path}")
        frame_interval = int(round(original_fps / fps))
        if frame_interval <= 0:
            frame_interval = 1
        count = 0
        saved_count = 0
        success, frame = cap.read()
        while success:
            if count % frame_interval == 0:
                denoised = self.denoise_image(frame)  
                frame_path = os.path.join(output_frames_dir, f"frame_{saved_count:06d}.png")
                cv2.imwrite(frame_path, denoised)
                saved_count += 1
            success, frame = cap.read()
            count += 1
        cap.release()


if __name__ == "__main__":
    denoiser = Denoiser()
    noisy_img = imread("./test_data/noisy.tif")
    denoiser.denoise_image(noisy_img)
    print(noisy_img.dtype, denoiser.denoised_data.dtype)
    print(denoiser.denoised_data.shape, denoiser.noisy_data.max(), denoiser.denoised_data.max())
    imwrite("./test_data/denoised.tif", denoiser.denoised_data)