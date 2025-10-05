from typing import Optional, Tuple
import numpy as np
import pandas as pd
from pathlib import Path
import tifffile as tiff
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from scipy.stats import pearsonr
import torch
import lpips
from .image_utils import normalize_image

class Metric:
    def __init__(self, img1: np.ndarray, img2: Optional[np.ndarray] = None, name: Optional[str] = None):
        """
        Base class for image metrics.
        Args:
            img1: Ground truth or reference image.
            img2: Processed/denoised image (optional for single-image metrics).
            name: Optional metric name.
        """
        self.source_img = normalize_image(img1) 
        self.processed_img = normalize_image(img2)
        self.name = name or self.__class__.__name__

    def run(self) -> float:
        """
        Compute the metric.
        Returns:
            float: metric value
        """
        if self.processed_img is None:
            raise ValueError(f"Processed image is required for metric {self.name}")
        return self.compute()

    def compute(self) -> float:
        """
        Override this method in subclasses to implement the actual metric computation.
        """
        raise NotImplementedError(f"Compute method not implemented for metric {self.name}")
    
class PSNR(Metric):
    def compute(self) -> float:
        return psnr(self.source_img, self.processed_img, data_range=1.0)
    
class MAE(Metric):
    def compute(self) -> float:
        return float(np.mean(np.abs(self.source_img - self.processed_img)))
    
class SSIM(Metric):
    def compute(self) -> float:
        return ssim(self.source_img, self.processed_img, data_range=1.0)

class LPIPS(Metric):
    def __init__(self, img1: np.ndarray, img2: np.ndarray, net: str = 'alex'):
        super().__init__(img1, img2)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.lpips_model = lpips.LPIPS(net=net).to(self.device)

    def compute(self) -> float:
        lpips_scores = []
        for z in range(self.source_img.shape[0]):
            gt_slice = torch.tensor(self.source_img[z], dtype=torch.float32).unsqueeze(0).repeat(1,3,1,1)
            den_slice = torch.tensor(self.processed_img[z], dtype=torch.float32).unsqueeze(0).repeat(1,3,1,1)
            gt_slice = gt_slice.to(self.device)
            den_slice = den_slice.to(self.device)
            with torch.no_grad():
                score = self.lpips_model(gt_slice, den_slice)
            lpips_scores.append(score.item())
        return float(np.mean(lpips_scores))
    
class SNR(Metric):
    def __init__(self, img1):
        super().__init__(img1)
    def compute(self) -> float:
        """
        Signal-to-Noise Ratio (SNR).
        SNR = 10 * log10( signal_power / noise_power )
        """
        signal_power = np.mean(self.source_img ** 2)
        noise_power = np.mean((self.source_img - self.processed_img) ** 2)
        if noise_power == 0:
            return float('inf')
        return 10 * np.log10(signal_power / noise_power)

class PearsonCorrelation(Metric):
    def compute(self) -> float:
        """
        Pearson correlation coefficient between flattened arrays.
        """
        src = self.source_img.flatten()
        proc = self.processed_img.flatten()
        if np.std(src) == 0 or np.std(proc) == 0:
            return np.nan
        corr, _ = pearsonr(src, proc)
        return float(corr)
    

def calc_metrics(
    metrics: list[Metric],
    denoised_folder: str,
    gt_folder: str,
    csv_path: Optional[str] = None
)-> pd.DataFrame:
    """
    Compute metrics for all TIFF images in subfolders
    of denoised_folder against corresponding ground truth images.
    Each subfolder name is treated as the Method name.
    """
    denoised_folder = Path(denoised_folder)
    gt_folder = Path(gt_folder)

    for method_dir in denoised_folder.iterdir():
        if not method_dir.is_dir():
            continue
        method_name = method_dir.name
    
    
def compute_image_metrics(file_path: Path, gt_folder: Path, method_name: str) -> Optional[Dict]:
    gt_path = gt_folder / file_path.name
    if not gt_path.exists():
        print(f"Warning: GT file not found for {file_path}, skipping...")
        return None

    denoised_img = tiff.imread(file_path)
    gt_img = tiff.imread(gt_path)
    gt_norm = (gt_img - gt_img.min()) / (gt_img.max() - gt_img.min())
    denoised_norm = (denoised_img - denoised_img.min()) / (denoised_img.max() - denoised_img.min())
    
    psnr_val = peak_signal_noise_ratio(gt_norm, denoised_norm, data_range=1.0)
    ssim_val = structural_similarity(gt_norm, denoised_norm, data_range=1.0)
    mae_val = compute_mae(gt_norm, denoised_norm)
    lpips_val = compute_lpips_stack(gt_norm, denoised_norm)
    ms_ssim_val = compute_ms_ssim(gt_norm, denoised_norm)

    print(f"[{method_name}] {file_path.name}: PSNR={psnr_val:.2f}, SSIM={ssim_val:.4f}, MS_SSIM={ms_ssim_val:.4f}, MAE={mae_val:.4f}, LPIPS={lpips_val:.4f}")
    return {
        "Method": method_name,
        "Image path": str(file_path),
        "PSNR": psnr_val,
        "SSIM": ssim_val,
        "MS-SIM": ms_ssim_val,
        "MAE": mae_val,
        "LPIPS": lpips_val
    }

def calc_metrics(
    denoised_folder: str,
    gt_folder: str,
    csv_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Compute PSNR, SSIM, MS-SSIM, MAE, LPIPS for all TIFF images in subfolders
    of denoised_folder against corresponding ground truth images.
    Each subfolder name is treated as the Method name.
    """
    denoised_folder = Path(denoised_folder)
    gt_folder = Path(gt_folder)
    
    results = []

    # Loop over method subfolders
    for method_dir in denoised_folder.iterdir():
        if not method_dir.is_dir():
            continue
        method_name = method_dir.name

        for file_path in method_dir.rglob("*.tif*"):
            gt_path = gt_folder / file_path.name

            if not gt_path.exists():
                print(f"Warning: GT file not found for {file_path}, skipping...")
                continue

            denoised_img = tiff.imread(file_path)
            gt_img = tiff.imread(gt_path)

            # Normalize images to [0,1]
            gt_norm = (gt_img - gt_img.min()) / (gt_img.max() - gt_img.min())
            denoised_norm = (denoised_img - denoised_img.min()) / (denoised_img.max() - denoised_img.min())

            # Compute metrics
            psnr_val = peak_signal_noise_ratio(gt_norm, denoised_norm, data_range=1.0)
            ssim_val = structural_similarity(gt_norm, denoised_norm, data_range=1.0)
            mae_val = compute_mae(gt_norm, denoised_norm)
            lpips_val = compute_lpips_stack(gt_norm, denoised_norm)
            # ms_ssim_val = compute_ms_ssim(gt_norm, denoised_norm)

            results.append({
                "Method": method_name,
                "Image path": str(file_path),
                "PSNR": psnr_val,
                "SSIM": ssim_val,
                "MAE": mae_val,
                "LPIPS": lpips_val
            })

            print(f"[{method_name}] {file_path.name}: PSNR={psnr_val:.2f}, SSIM={ssim_val:.4f}, MAE={mae_val:.4f}, LPIPS={lpips_val:.4f}")

    df = pd.DataFrame(results)
    # Optionally save to CSV
    if csv_path:
        df.to_csv(csv_path, index=False)

    return df