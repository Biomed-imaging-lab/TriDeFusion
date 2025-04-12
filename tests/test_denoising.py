import os
import sys
import pytest
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from src.denoiser import Denoiser 
from src.utils import load_image, save_tiff  

# Test parameters
BASE_DIR = os.path.dirname(__file__)
TEST_IMAGE_PATH = os.path.abspath(os.path.join(BASE_DIR, "test_images/noisy_tubes.tif"))
TEST_PREPROCESSED_PATH = os.path.abspath(os.path.join(BASE_DIR, "test_images/preprocessed_tubes.tif"))
# TEST_GAUSSIAN_PATH = os.path.abspath(os.path.join(BASE_DIR, "test_images/gauss_filtered_tubes.tif"))
# TEST_MEDIAN_PATH = os.path.abspath(os.path.join(BASE_DIR, "test_images/median_filtered_tubes.tif"))
# TEST_NLM_PATH = os.path.abspath(os.path.join(BASE_DIR, "test_images/nlm_filtered_tubes.tif"))
# TEST_BM3D_PATH = os.path.abspath(os.path.join(BASE_DIR, "test_images/bm3d_filtered_tubes.tif"))
TEST_DENOISE_METHODS = ['gaussian', 'median', 'nlm', 'bm3d']

@pytest.fixture
def noisy_image():
    """Fixture to load the noisy image for testing."""
    return load_image(TEST_IMAGE_PATH)

@pytest.fixture
def denoiser():
    """Fixture to initialize the DenoiseController."""
    return Denoiser()

def test_preprocess_image(noisy_image, denoiser):
    """Test preprocessing of the noisy image."""
    try:
        denoiser.preprocess_image(noisy_image)
        assert denoiser._Denoiser__noisy_img is not None, "Preprocessed image is not set."
        assert isinstance(
            denoiser._Denoiser__noisy_img, np.ndarray
        ), "Preprocessed image is not a numpy array."
    except Exception as e:
        pytest.fail(f"Preprocessing failed with exception: {e}")

@pytest.mark.parametrize("method", TEST_DENOISE_METHODS)
def test_denoise_image(noisy_image, denoiser, method):
    """Test denoising of the preprocessed image with different methods."""
    try:
        preprocessed_img = denoiser.preprocess_image(noisy_image)
        denoised_img = denoiser.denoise_image(method=method)
        save_tiff(denoised_img, os.path.abspath(os.path.join(BASE_DIR, f"test_images/{method}_tubes.tif")), 1)
        assert denoised_img is not None, f"Denoising with {method} failed: Denoised image is None."
        assert isinstance(denoised_img, np.ndarray), f"Denoising with {method} failed: Denoised image is not a numpy array."
        assert denoised_img.shape == preprocessed_img.shape, f"Denoising with {method} failed: Shape mismatch between input and denoised images."
    except Exception as e:
        pytest.fail(f"Denoising with {method} failed with exception: {e}")
