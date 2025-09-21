import asyncio
from typing import List, Tuple
import numpy as np
from matplotlib import cm, pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from src.utils.image_utils import extract_segment_from_layer
from src.utils.plot import draw_histograms


async def analyze_segments(images: List[np.ndarray],
                     segment_coords: Tuple[Tuple[int, int], Tuple[int, int]],
                     info_methods: List[Tuple[str, str]],
                     save_paths: List[str],
                     hist_save_path: str) -> None:
    try:
        global_max = max(
            np.max(extract_segment_from_layer(img, img.shape[0] // 2, segment_coords)) for img in images)    
        async def process_image(i, img, save_path, info):
            crop_img = extract_segment_from_layer(img, img.shape[0] // 2, segment_coords)
            await asyncio.to_thread(draw_segment_rectangle, img, segment_coords, None, img_with_segment_path)
            await asyncio.to_thread(generate_2d_projections, crop_img, info[0], save_path, None, global_max, 30)
            return crop_img
        crop_images = await asyncio.gather(*[
            process_image(i, _img, save_paths[i], info_methods[i])
            for i, _img in enumerate(images)
        ])
        await asyncio.to_thread(
            draw_histograms, _images=crop_images, _info_methods=info_methods,
            _title="Intensity Values Distribution", save_path=hist_save_path, is_log=True
        )
    except Exception as e:
        print(f"Error in analyze_segments: {str(e)}")
        raise


def calc_metrics(clean_img: np.ndarray, 
                 denoised_img: np.ndarray):
    psnr_value = psnr(clean_img, denoised_img, data_range=255)
    ssim_value = ssim(clean_img, denoised_img, data_range=255)
    return psnr_value, ssim_value

def plot_metrics(methods_data: List[Tuple[str]], 
                         metric_name: str, 
                         save_path: str = None) -> None:
    names = []
    colors = []
    values = []
    for name, color, value in methods_data:
        names.append(name)
        colors.append(color)
        if isinstance(value, (list, np.ndarray)):
            value = np.mean(value)
        values.append(value)
    plt.figure(figsize=(8, 5))
    bars = plt.bar(names, values, color=colors)
    for bar, val in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                 f"{val:.3f}", ha="center", va="bottom")
    plt.ylabel(metric_name)
    plt.title(f"Comparison of {metric_name}")
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()