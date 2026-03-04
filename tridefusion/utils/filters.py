import cv2
import numpy as np
from scipy.ndimage import gaussian_filter, median_filter
from bm3d import bm3d
from .exceptions import check_not_none, check_positive_integer, check_input_image
import pywt
from bm3d import bm3d, BM3DStages
from joblib import Parallel, delayed
import tifffile as tiff

# @performance_decorator
def gaussian_denoise(img: np.ndarray, sigma: int = 3, save_path: str = None,
                     normalize: bool = True) -> np.ndarray:
    """
    Apply Gaussian denoising with universal support for different input types.

    Args:
        img: Input image (uint8, uint16, float16, float32, float64, etc.)
        sigma: Gaussian blur sigma
        save_path: Optional path to save the denoised image
        normalize: Whether to normalize output to [0,1] float32
    Returns:
        Denoised image as float32 (or same dtype as input if normalize=False)
    """
    try:
        check_positive_integer(sigma, "Gaussian sigma")
        check_input_image(img, "Input image", "gaussian_denoise")

        img_f = img.astype(np.float32)
        denoised = gaussian_filter(img_f, sigma=sigma)
        # if normalize:
        #     denoised = normalize_image(denoised)
        # if save_path:
        #     save_tiff(denoised, save_path)
        return denoised
    except Exception as e:
        print(f"Error in gaussian_denoise: {str(e)}")
        raise

# # @performance_decorator        
def median_denoise(img: np.ndarray, size: int = 3) -> np.ndarray:
    try:
        check_not_none(value=img, name="Image")
        check_positive_integer(value=size, name="Block size")
        img_f = img.astype(np.float32)
        denoised = median_filter(img_f, size=size)
        # if normalize:
        #     denoised = normalize_image(denoised)
        # if save_path:
        #     save_tiff(denoised, save_path)
        return denoised
    except Exception as e:
        print(f"Error in median_filter_img: {str(e)}")
        raise

# @performance_monitor
def bm3d_denoise(img: np.ndarray, sigma_psd: float = 0.05) -> np.ndarray:
    """Apply BM3D denoising to the image."""
    if sigma_psd <= 0:
        raise ValueError("Sigma PSD must be a positive float")
    denoised_img = bm3d(img, sigma_psd)
    return denoised_img

def nlm_denoise_layer(_noisy_layer: np.ndarray) -> np.ndarray:
    try:
        _noisy_layer = cv2.convertScaleAbs(_noisy_layer)
        _denoised_layer = cv2.fastNlMeansDenoising(_noisy_layer, None, 10, 7, 21)
        return _denoised_layer
    except (ValueError, Exception) as e:
        print(f"Error in nlm_denoise_layer: {str(e)}")
        raise
   
import numpy as np
import cv2

def _make_odd(x: int) -> int:
    return x if x % 2 == 1 else x - 1


def nlm_denoise(img: np.ndarray, 
                h: int = 10,
                temporal_window: int = 3,
                template_window: int = 7,
                search_window: int = 21) -> np.ndarray:
    """
    Apply fastNlMeansDenoisingMulti to a 3D volume (Z, H, W).
    Ensures all window sizes are odd, even at borders.
    """
    try:
        if img.ndim != 3:
            raise ValueError("Expected a 3D volume with shape (Z, H, W).")

        Z, H, W = img.shape

        # Convert each slice to uint8 properly
        noisy_img_uint8 = np.stack([cv2.convertScaleAbs(frame) for frame in img], axis=0)

        denoised = []

        # Ensure global parameters are odd
        template_window = _make_odd(template_window)
        search_window   = _make_odd(search_window)
        temporal_window = _make_odd(temporal_window)

        for i in range(Z):
            start = max(0, i - temporal_window // 2)
            end   = min(Z, i + temporal_window // 2 + 1)

            # local window size (may be smaller at borders)
            local_tw = end - start     # number of frames in slice
            local_tw = _make_odd(local_tw)

            if local_tw < 1:
                local_tw = 1

            ref_index = (i - start)

            denoised_frame = cv2.fastNlMeansDenoisingMulti(
                noisy_img_uint8[start:end],
                ref_index,
                temporalWindowSize=local_tw,
                h=h,
                templateWindowSize=template_window,
                searchWindowSize=search_window
            )

            denoised.append(denoised_frame)

        return np.array(denoised, dtype=float)

    except Exception as e:
        print(f"Error in nlm_denoise: {str(e)}")
        raise

# # @performance_decorator
def bm3d_denoise(img_stack: np.ndarray, sigma_psd: float = 0.05, 
                 n_jobs: int = -1, fast: bool = True, save_path=None) -> np.ndarray:
    """
    Apply BM3D denoising to each 2D slice of a 3D stack in parallel.
    Input: float32 normalized [0,1].
    Output: float32 normalized [0,1].

    Args:
        img_stack (np.ndarray): 3D stack (Z, H, W).
        sigma_psd (float): BM3D noise parameter (0.01–0.1 typical for normalized images).
        n_jobs (int): Number of parallel jobs (-1 = all cores).
        fast (bool): Use fast BM3D (HARD_THRESHOLDING) if True.
        save_path (str, optional): Path to save the output as TIFF.

    Returns:
        np.ndarray: Denoised 3D stack (float32 [0,1]).
    """
    try:
        check_input_image(img_stack, 'Noisy image', 'bm3d_denoise')
        img_stack = img_stack.astype(np.float32)
        if img_stack.max() > 1.0:
            img_stack /= 255.0
        denoised_slices = Parallel(n_jobs=n_jobs)(
            delayed(bm3d_denoise)(img_stack[z], sigma_psd, fast)
            for z in range(img_stack.shape[0])
        )

        denoised = np.stack(denoised_slices, axis=0).astype(np.float32)
        # denoised = normalize_image(denoised)
        if save_path:
            tiff.imwrite(save_path, (denoised*255).astype(np.uint8))
        return denoised

    except (ValueError, Exception) as e:
        print(f"Error in bm3d_denoise: {str(e)}")
        raise

def wavelet_denoise_layer(layer, wavelet, level):
    coeffs = pywt.wavedec2(layer, wavelet, level=level)
    coeffs_thresholded = [pywt.threshold(c, np.std(c), mode='soft') if isinstance(c, np.ndarray) else c for c in coeffs]
    return np.clip(pywt.waverec2(coeffs_thresholded, wavelet), 0, 255).astype(np.uint8)

def wavelet_denoise(noisy_img: np.ndarray, wavelet='db1', level=1, save_path=None) -> np.ndarray:
    dtype = noisy_img.dtype
    max_val = np.iinfo(dtype).max if np.issubdtype(dtype, np.integer) else 1.0
    denoised_layers = [wavelet_denoise_layer(layer, wavelet, level) for layer in noisy_img]
    denoised_img = np.array(denoised_layers)
    print("max", max_val, dtype, noisy_img.max())
    denoised_img = np.clip(denoised_img, 0, max_val)
    if np.issubdtype(dtype, np.integer):
        denoised_img = np.rint(denoised_img).astype(dtype)
    else:
        denoised_img = denoised_img.astype(dtype)
    return denoised_img

# @performance_monitor
def tv_denoise(image,
               weight=0.1,
               n_iter_max=200,
               eps=1e-3,
               isotropic=True,
               voxel_size=None,
               resample_order=1):
    """
    Total Variation (TV) denoising wrapper for 2D/3D images.

    Parameters
    ----------
    image : np.ndarray
        Input image. 2D (H,W) or 3D (Z,H,W) is supported.
    weight : float
        Denoising strength. Larger -> stronger smoothing.
        Typical 0.01..0.5 depending on noise level.
    n_iter_max : int
        Maximum iterations for TV solver (skimage default ~200).
    eps : float
        Tolerance for convergence (passed to skimage if supported).
    isotropic : bool
        If True use isotropic TV (skimage default). Kept for API clarity.
    voxel_size : None or tuple of floats
        Physical voxel sizes. If provided and not isotropic, function will
        resample to isotropic spacing before denoising and then resample back.
        Format: (z, y, x) or (y, x) for 2D. If scalar provided, treated as uniform.
    resample_order : int
        Interpolation order for resampling (0..5). 0=nearest, 1=linear.

    Returns
    -------
    denoised : np.ndarray
        Denoised image, same shape and dtype as input.
    """
    try:
        from skimage.restoration import denoise_tv_chambolle
        from scipy.ndimage import zoom
    except Exception as e:
        raise RuntimeError("tv_denoise requires scikit-image and scipy. "
                           "Install them with `pip install scikit-image scipy`.") from e

    orig_dtype = image.dtype
    img = image.astype(np.float32)

    # normalize to [0,1] for stable weight behavior, keep scale for restoring
    imin, imax = img.min(), img.max()
    if imax - imin > 0:
        img_norm = (img - imin) / (imax - imin)
    else:
        img_norm = img - imin  # constant image

    need_resample = False
    zoom_factors = None

    # handle voxel_size for anisotropic stacks (only relevant for 3D)
    if voxel_size is not None:
        if np.isscalar(voxel_size):
            vs = (voxel_size,) * img_norm.ndim
        else:
            vs = tuple(voxel_size)
        # if dims mismatch try to align (e.g., user passed (x,y,z))
        if len(vs) != img_norm.ndim:
            # try reversing if lengths match in reverse
            if len(vs) == img_norm.ndim:
                pass
            else:
                raise ValueError("voxel_size length must match image.ndim (or be scalar).")

        if img_norm.ndim == 3:
            # vs should be (z, y, x)
            # if these are not (1,1,1) and not equal, resample to isotropic spacing
            if not np.allclose(vs, vs[0]):
                need_resample = True
                # target voxel = min voxel (finest sampling) to avoid loss
                target = min(vs)
                zoom_factors = tuple(v / target for v in vs)  # how many voxels per dimension
                # but zoom in ndimage.zoom is output/input: new = input * zoom_factor
                # we want to make physical spacing equal -> zoom = vs / target
        else:
            # 2D: usually no need to resample; handle scalar vs
            need_resample = False

    if need_resample and zoom_factors is not None:
        # perform resampling to isotropic grid
        # note: zoom_factors computed as vs/target ( > =1 for coarser dims)
        img_for_tv = zoom(img_norm, zoom=zoom_factors, order=resample_order)
    else:
        img_for_tv = img_norm

    # skimage accepts nD arrays; it expects floating images in [0,1]
    # For nD, denoise_tv_chambolle signature: denoise_tv_chambolle(image, weight, eps, n_iter_max, multichannel)
    # multichannel should be False for grayscale volumes.
    denoised_norm = denoise_tv_chambolle(img_for_tv,
                                         weight=weight,
                                         eps=eps,
                                         max_num_iter=n_iter_max,
                                         channel_axis=None)  # newer versions use channel_axis

    # if older skimage where channel_axis not present, fall back
    if denoised_norm is None:
        # try older signature (scikit-image < 0.19)
        denoised_norm = denoise_tv_chambolle(img_for_tv,
                                             weight=weight,
                                             eps=eps,
                                             max_num_iter=n_iter_max,
                                             multichannel=False)

    # resample back to original shape if we changed it
    if need_resample and zoom_factors is not None:
        inv_zoom = tuple(1.0 / z for z in zoom_factors)
        denoised_norm = zoom(denoised_norm, zoom=inv_zoom, order=resample_order)

    # restore original scale and dtype
    if imax - imin > 0:
        denoised = denoised_norm * (imax - imin) + imin
    else:
        denoised = denoised_norm + imin
    if np.issubdtype(orig_dtype, np.integer):
        info = np.iinfo(orig_dtype)
        denoised = np.clip(np.rint(denoised), info.min, info.max).astype(orig_dtype)
    else:
        denoised = denoised.astype(orig_dtype)
    return denoised