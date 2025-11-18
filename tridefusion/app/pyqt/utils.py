import numpy as np
from pathlib import Path


def list_3d_tiff_files(folder: Path) -> list[Path]:
    exts = {".tif", ".tiff"}
    return [p for p in sorted(folder.iterdir()) if p.suffix.lower() in exts and p.is_file()]

def to_grayscale(img: np.ndarray) -> np.ndarray:
    """Convert RGB/RGBA or multichannel to grayscale."""
    if img.ndim == 3:
        if img.shape[-1] == 3:
            img = np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])
        elif img.shape[-1] == 4:
            img = np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])
        else:
            img = img.mean(axis=-1)

    img = img.astype(np.float32)
    min_val, max_val = np.min(img), np.max(img)
    if max_val > min_val:  # avoid divide by zero
        img = (img - min_val) / (max_val - min_val)
    else:
        img[:] = 0.0
    return img
