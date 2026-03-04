import json
import os
import re
from typing import Callable, List, Tuple
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt
import tifffile as tiff
from skimage import measure


def extract_noise_levels(folder_name: str) -> dict:
    """
    Extract noise level values from a folder name, e.g.:
    'poisson_16_gauss_0' → {'poisson': 16.0, 'gauss': 0.0}
    Supports both integer and float levels.
    """
    folder_name = folder_name.lower()
    match_poisson = re.search(r"poisson[_\-]?([0-9]*\.?[0-9]+)", folder_name)
    match_gauss = re.search(r"gauss[_\-]?([0-9]*\.?[0-9]+)", folder_name)
    poisson_level = float(match_poisson.group(1)) if match_poisson else 0.0
    gauss_level = float(match_gauss.group(1)) if match_gauss else 0.0
    return {"poisson": poisson_level, "gauss": gauss_level}

# def _calc_metric(gt_path: Path, file_path: Path) -> dict:
#     """Compute metrics between GT and denoised images."""
#     denoised_img = tiff.imread(file_path).astype(np.float32)
#     gt_img = tiff.imread(gt_path).astype(np.float32)
#     gt_norm = (gt_img - gt_img.min()) / (gt_img.max() - gt_img.min() + 1e-8)
#     denoised_norm = (denoised_img - denoised_img.min()) / (denoised_img.max() - denoised_img.min() + 1e-8)
#     psnr_val = peak_signal_noise_ratio(gt_norm, denoised_norm, data_range=1.0)
#     ssim_val = structural_similarity(gt_norm, denoised_norm, data_range=1.0)
#     gt_tensor = torch.tensor(gt_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
#     dn_tensor = torch.tensor(denoised_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
#     ms_ssim_val = ms_ssim(gt_tensor, dn_tensor, data_range=1.0).item()
#     mae_val = compute_mae(gt_norm, denoised_norm)
#     lpips_val = compute_lpips_stack(gt_norm, denoised_norm)
#     return {
#         "PSNR": psnr_val,
#         "SSIM": ssim_val,
#         "MS-SSIM": ms_ssim_val,
#         "MAE": mae_val,
#         "LPIPS": lpips_val
#     }

# def calc_metrics(gt_folder: str, denoised_root: str, csv_dir: str = "./results"):
#     """
#     Compute metrics for all methods and noise levels.
#     Saves:
#       - One CSV per method: <csv_dir>/<method>.csv
#       - One combined CSV: <csv_dir>/all_methods.csv
#     """
#     gt_folder = Path(gt_folder)
#     denoised_root = Path(denoised_root)
#     csv_dir = Path(csv_dir)
#     csv_dir.mkdir(parents=True, exist_ok=True)
#     all_results = []

#     for method_path in sorted(denoised_root.iterdir()):
#         if not method_path.is_dir():
#             continue
#         method_name = method_path.name
#         method_results = []
#         for noise_path in sorted(method_path.iterdir()):
#             if not noise_path.is_dir():
#                 continue

#             noise_levels = extract_noise_levels(noise_path.name)
#             for file_path in noise_path.rglob("*.tif*"):
#                 gt_path = gt_folder / file_path.name
#                 if not gt_path.exists():
#                     print(f"⚠️ Warning: GT file not found for {file_path}, skipping...")
#                     continue
#                 metrics = _calc_metric(gt_path, file_path)
#                 entry = {
#                     "Method": method_name,
#                     "NoiseFolder": noise_path.name,
#                     "Poisson": noise_levels["poisson"],
#                     "Gauss": noise_levels["gauss"],
#                     "File": file_path.name,
#                     **metrics,
#                 }
#                 method_results.append(entry)
#                 all_results.append(entry)
#         if method_results:
#             df_method = pd.DataFrame(method_results)
#             method_csv_path = csv_dir / f"{method_name}.csv"
#             df_method.to_csv(method_csv_path, index=False)
#             print(f"Saved metrics for {method_name} → {method_csv_path}")
#     if all_results:
#         df_all = pd.DataFrame(all_results)
#         all_csv_path = csv_dir / "all_methods.csv"
#         df_all.to_csv(all_csv_path, index=False)
#         print(f"Saved combined metrics → {all_csv_path}")
#     return pd.DataFrame(all_results)

def denoise_folder(
    input_folder: str,
    output_folder: str,
    methods: List[Tuple[str, Callable[[np.ndarray], np.ndarray]]]
):
    """
    Apply multiple denoising methods to all TIFF images in a folder
    and save the results. Metrics are NOT computed here.
    """
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)
    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.lower().endswith((".tif", ".tiff")):
                img_path = Path(root) / file
                relative_path = Path(root).relative_to(input_folder)
                img = tiff.imread(img_path).astype(np.float32)
                for method_name, denoise_fn in methods:
                    denoised_img = denoise_fn(img)
                    res_norm = (denoised_img - denoised_img.min()) / (denoised_img.max() - denoised_img.min() + 1e-8)
                    save_dir = output_folder / method_name / relative_path
                    save_dir.mkdir(parents=True, exist_ok=True)
                    save_path = save_dir / file
                    tiff.imwrite(save_path, (res_norm * 255).astype(np.uint8))
                    print(f"[{method_name}] Saved {save_path}")


def extract_segments_from_json(ref_segments_root: Path, target_folder: Path, out_root: Path):
    """
    Extracts 3D segments from volumes in target_folder using params.json files
    from ref_segments_root (which contain segment coordinates).

    Images are normalized to [0,1] and converted to uint8 [0,255].
    Saves segments in per-image folders (params_used.json per folder) and
    also creates root-level JSONs:
      - all_params_used.json
      - all_mean_intensities.json
    """
    exts = {".tif", ".tiff"}
    out_root.mkdir(parents=True, exist_ok=True)
    log = []
    all_params = []
    all_mean_intensities = {}

    for json_dir in sorted(ref_segments_root.iterdir()):
        if not json_dir.is_dir():
            continue
        json_path = json_dir / "params.json"
        if not json_path.exists():
            log.append(f"Skipping: no params.json in {json_dir}")
            continue
        image_name = json_dir.name
        candidates = [p for p in target_folder.iterdir()
                      if p.stem == image_name and p.suffix.lower() in exts]
        if not candidates:
            log.append(f"Skipping: no image found for {image_name}")
            continue
        img_path = candidates[0]
        try:
            vol = tiff.imread(str(img_path)).astype(np.float32)
            if vol.ndim == 4 and vol.shape[0] == 1:
                vol = vol[0]
            if vol.ndim != 3:
                log.append(f"Skipping {img_path.name}: unsupported shape {vol.shape}")
                continue
            vol_min, vol_max = vol.min(), vol.max()
            vol = (vol - vol_min) / (vol_max - vol_min + 1e-8)
            with open(json_path, "r", encoding="utf-8") as f:
                params = json.load(f)
            out_dir = out_root / image_name
            out_dir.mkdir(parents=True, exist_ok=True)
            per_image_params = []
            for p in params:
                z0, y0, x0 = p["z0"], p["y0"], p["x0"]
                size = p["size"]
                z1, y1, x1 = z0 + size, y0 + size, x0 + size
                if z1 > vol.shape[0] or y1 > vol.shape[1] or x1 > vol.shape[2]:
                    log.append(f"Skipping segment {p['index']} in {image_name}: out of bounds")
                    continue

                seg = vol[z0:z1, y0:y1, x0:x1]
                mean_int = float(np.mean(seg))
                seg_uint8 = (seg * 255.0).clip(0, 255).astype(np.uint8)
                out_fname = out_dir / f"segment_{p['index']:03d}.tif"
                tiff.imwrite(str(out_fname), seg_uint8)

                param_entry = {
                    **p,
                    "mean_intensity_target": mean_int,
                    "source_target": str(img_path.name),
                    "out_file_target": str(out_fname.name)
                }
                per_image_params.append(param_entry)
                all_params.append({
                    **p,
                    "image_name": image_name,
                    **param_entry
                })
                all_mean_intensities[str(out_fname.relative_to(out_root))] = mean_int
            with open(out_dir / "params.json", "w", encoding="utf-8") as f:
                json.dump(per_image_params, f, ensure_ascii=False, indent=2)
            log.append(f"Extracted {len(per_image_params)} segments from {img_path.name}")
        except Exception as e:
            log.append(f"Error processing {img_path.name}: {e}")
    with open(out_root / "all_params.json", "w", encoding="utf-8") as f:
        json.dump(all_params, f, ensure_ascii=False, indent=2)
    with open(out_root / "all_mean_intensities.json", "w", encoding="utf-8") as f:
        json.dump(all_mean_intensities, f, ensure_ascii=False, indent=2)
    log.append(f"Saved all JSONs in {out_root}")
    return log


def compute_diff_maps(gt, noisy, denoised_dict):
    """
    Compute difference maps for multiple methods.
    Returns a dict: {method: (D_gn, D_nd, D_gd)}
    """
    results = {}
    for method, den in denoised_dict.items():
        D_gn = gt - noisy
        D_nd = noisy - den
        D_gd = gt - den
        results[method] = (D_gn, D_nd, D_gd)
    return results


def save_diff_maps(gt, diff_dict, save_folder, slice_axis=0, slice_index=None, cmap="seismic"):
    """
    Save difference maps and RGB composites for each method,
    normalized to the range [-1, 1] across all maps for that method.
    """
    os.makedirs(save_folder, exist_ok=True)

    for method, (D_gn, D_nd, D_gd) in diff_dict.items():
        maps = [D_gn, D_nd, D_gd]
        names = ["GT-Noisy", "Noisy-Denoised", "GT-Denoised"]
        if slice_index is None:
            slice_index = gt.shape[slice_axis] // 2

        def slice_along(arr):
            if slice_axis == 0:
                return arr[slice_index, :, :]
            if slice_axis == 1:
                return arr[:, slice_index, :]
            return arr[:, :, slice_index]

        gt_slice = slice_along(gt)
        diff_slices = [slice_along(m) for m in maps]

        contours = measure.find_contours(gt_slice, 0.5)
        all_vals = np.concatenate([d.flatten() for d in diff_slices])
        global_abs_max = np.max(np.abs(all_vals)) + 1e-8
        diff_norms = [d / global_abs_max for d in diff_slices]
        vmin, vmax = -1.0, 1.0

        for diff_norm, name in zip(diff_norms, names):
            fig, ax = plt.subplots()
            im = ax.imshow(diff_norm, cmap=cmap, vmin=vmin, vmax=vmax)
            for c in contours:
                ax.plot(c[:, 1], c[:, 0], color='yellow', linewidth=0.8)
            ax.axis("off")
            fig.colorbar(im, ax=ax, shrink=0.6)
            fig.savefig(os.path.join(save_folder, f"{method}_{name}.png"),
                        dpi=200, bbox_inches='tight')
            plt.close(fig)
            alpha = np.clip(np.abs(diff_norm) * 3.0, 0, 1)

            rgb = np.zeros((*diff_norm.shape, 4))
            rgb[..., 0] = np.clip(diff_norm, 0, 1)      # positive → red
            rgb[..., 2] = np.clip(-diff_norm, 0, 1)     # negative → blue
            rgb[..., 3] = alpha
            plt.imsave(os.path.join(save_folder, f"{method}_{name}_RGB.png"), rgb)

# def draw_lines_profile(
#         _images: List[np.ndarray],
#         line_coords: Tuple[Tuple[int, int], Tuple[int, int]],
#         _info_methods: List[Tuple[str, str]],
#         save_path: str
# ) -> None:
#     try:
#         x_values = np.arange(line_coords[0][0], line_coords[1][0] + 1, dtype=int)
#         _images[1] = np.clip(_images[1].astype(np.float32) / 1.3, 0, 255).astype(np.uint8)
#         _images[3] = np.clip(_images[3].astype(np.float32) / 1.6, 0, 255).astype(np.uint8)

#         def process_image(_img, _state, color):
#             profile_values = line_profile(_img, line_coords, _img.shape[0] // 2)
#             return (profile_values, _state, color)

#         with ThreadPoolExecutor() as executor:
#             lines_data = list(executor.map(process_image, _images, [state for state, _ in _info_methods],
#                                            [color for _, color in _info_methods]))
#         draw_lines(x_values=x_values, y_values_with_details=lines_data,
#                    axis_labels=("Pixel Position", "Intensity Value"), title="Line Profiles", save_path=save_path)
#         print(f"Line profiles saved to {save_path}")
#     except (ValueError, Exception) as e:
#         print(f"Error in draw_lines_profile: {str(e)}")
#         raise

# async def analyze_segments(images: List[np.ndarray],
#                      segment_coords: Tuple[Tuple[int, int], Tuple[int, int]],
#                      info_methods: List[Tuple[str, str]],
#                      save_paths: List[str],
#                      hist_save_path: str) -> None:
#     try:
#         global_max = max(
#             np.max(extract_segment_from_layer(img, img.shape[0] // 2, segment_coords)) for img in images)    
#         async def process_image(i, img, save_path, info):
#             crop_img = extract_segment_from_layer(img, img.shape[0] // 2, segment_coords)
#             await asyncio.to_thread(draw_segment_rectangle, img, segment_coords, None, img_with_segment_path)
#             await asyncio.to_thread(generate_2d_projections, crop_img, info[0], save_path, None, global_max, 30)
#             return crop_img
#         crop_images = await asyncio.gather(*[
#             process_image(i, _img, save_paths[i], info_methods[i])
#             for i, _img in enumerate(images)
#         ])
#         await asyncio.to_thread(
#             draw_histograms, _images=crop_images, _info_methods=info_methods,
#             _title="Intensity Values Distribution", save_path=hist_save_path, is_log=True
#         )
#     except Exception as e:
#         print(f"Error in analyze_segments: {str(e)}")
#         raise