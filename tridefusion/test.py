import argparse
from enum import Enum
from dataclasses import dataclass
import os

from tifffile import imread
# from .denoiser import Denoiser


class InputType(Enum):
    IMAGE = "image"
    FOLDER = "folder"
    VIDEO = "video"


@dataclass
class TestParams:
    input_type: InputType
    input_path: str
    output_path: str
    method: str = "gaussian"
    fps: int = 15  # only for video


def run_test(params: TestParams):
    pass
    # denoiser = Denoiser()
    # if params.input_type == InputType.IMAGE:
    #     noisy_img = imread(params.input_path)
    #     denoiser.denoise_image(noisy_img, method=params.method, save_path=params.output_path)
    # elif params.input_type == InputType.FOLDER:
    #     denoiser.denoise_folder(input_dir=params.input_path, output_dir=params.output_path)
    # elif params.input_type == InputType.VIDEO:
    #     denoiser.denoise_video(video_path=params.input_path,
    #                            output_frames_dir=params.output_path,
    #                            fps=params.fps)
    # else:
    #     raise ValueError(f"Unsupported input type: {params.input_type}")


def parse_args():
    parser = argparse.ArgumentParser(description="Run denoising inference on image, folder, or video")
    parser.add_argument("--input_type", type=str, choices=[t.value for t in InputType], required=True)
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--method", type=str, default="gaussian", help="Denoising method (from config)")
    parser.add_argument("--fps", type=int, default=1, help="FPS to sample from video (only for video input)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    params = TestParams(
        input_type=InputType(args.input_type),
        input_path=args.input_path,
        output_path=args.output_path,
        method=args.method,
        fps=args.fps
    )
    run_test(params)