from typing import Any, Optional
import numpy as np
import torch
import tifffile as tiff
from .decorators import performance_monitor

def reformat_to_zchw(img: np.ndarray) -> np.ndarray:
    """
    Convert input image to (Z, C, H, W) format for processing.
    Supported input shapes:
        (H, W)
        (H, W, C)
        (Z, H, W)
        (Z, H, W, C)

    Args:
        img (np.ndarray): Input image.
    Returns:
        np.ndarray: Image in (Z, C, H, W) format.
    """
    original_shape = img.shape
    ndim = img.ndim

    if ndim == 2:  # (H, W)
        return img[np.newaxis, np.newaxis, ...]  # (1, 1, H, W)

    if ndim == 3:
        # Detect if last axis is channels (RGB, etc.)
        if img.shape[-1] <= 4:  # (H, W, C)
            return img.transpose(2, 0, 1)[np.newaxis, ...]  # (1, C, H, W)
        else:  # (Z, H, W)
            return img[:, np.newaxis, ...]  # (Z, 1, H, W)

    if ndim == 4:  # (Z, H, W, C)
        return img.transpose(0, 3, 1, 2)  # (Z, C, H, W)
    raise ValueError(f"Unsupported input shape: {original_shape}")

@performance_monitor  
def nn_denoise(img: np.ndarray, 
               model: Any, 
               save_path=None, 
               device='cuda') -> np.ndarray:
    """
    Denoise a 3D image stack using a pretrained model.
    Preserves original dtype in output.
    Automatically handles FP16 models.
    """
    try:
        original_dtype = img.dtype
        img = reformat_to_zchw(img)

        if img.shape[1] == 3:
            img = img.mean(axis=1, keepdims=True)
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        img = img.astype(np.float32)
        denoised_stack = np.empty_like(img, dtype=np.float32)

        model_dtype = next(model.parameters()).dtype
        for z in range(img.shape[0]):
            print(f"Processing slice {z+1}/{img.shape[0]}...")
            tensor = torch.from_numpy(img[z:z+1]).to(device)
            tensor = tensor.to(dtype=model_dtype)

            with torch.no_grad():
                denoised_slice = model(tensor)
            denoised_stack[z:z+1] = denoised_slice.detach().cpu().float().numpy()
            del tensor, denoised_slice
            torch.cuda.empty_cache()
            
        if np.issubdtype(original_dtype, np.integer):
            max_val = np.iinfo(original_dtype).max
            denoised = (np.clip(denoised_stack, 0, 1) * max_val).astype(original_dtype)
        else:
            denoised = denoised_stack.astype(original_dtype)
        if save_path:
            tiff.imwrite(save_path, denoised)
        if denoised.ndim == 4 and denoised.shape[1] == 1:
            denoised = denoised[:, 0, :, :]
        return denoised
    except Exception as e:
        print(f"Error in nn_denoise: {str(e)}")
        raise

def onnx_export(Model: Any,
                chkpt_path: str,
                onnx_path: str,
                in_channels=1,
                out_channels=1,
                device='cuda'
                ) -> bool:
    try:
        model = Model(in_channels, out_channels).to(device)
        state_dict = torch.load(chkpt_path, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()
        model = model.half()
        dummy_input = torch.randn(1, in_channels, 256, 256).half().to(device)
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={
                "input": {0: "batch", 2: "height", 3: "width"},
                "output": {0: "batch", 2: "height", 3: "width"}
            }
        )
        print("ONNX export completed: dynamic batch, height, width enabled")
    except Exception as e:
        print(f"Error in onnx_export: {str(e)}")
        raise


def tile_inference_forward(model, tensor, tile_size=(256,256), overlap=16, amp=True):
    """
    Run model on large HxW image using overlapping tiles.
    tensor: [1,C,H,W] float32 CPU tensor moved to device inside.
    Returns CPU numpy array [1,C,H,W] float32.
    """
    _, C, H, W = tensor.shape
    device = next(model.parameters()).device
    out = np.zeros((1, C, H, W), dtype=np.float32)
    weight = np.zeros((1, C, H, W), dtype=np.float32)

    stride_h = tile_size[0] - overlap
    stride_w = tile_size[1] - overlap
    hs = list(range(0, H, stride_h))
    ws = list(range(0, W, stride_w))
    if hs[-1] + tile_size[0] < H:
        hs.append(max(0, H - tile_size[0]))
    if ws[-1] + tile_size[1] < W:
        ws.append(max(0, W - tile_size[1]))

    for y in hs:
        for x in ws:
            y1, y2 = y, min(y + tile_size[0], H)
            x1, x2 = x, min(x + tile_size[1], W)

            patch = tensor[:, :, y1:y2, x1:x2].to(device, non_blocking=True)
            with torch.no_grad():
                if amp:
                    with torch.amp.autocast():
                        out_patch = model(patch)
                else:
                    out_patch = model(patch)
            out_patch_np = out_patch.detach().cpu().float().numpy()
            out[:, :, y1:y2, x1:x2] += out_patch_np
            weight[:, :, y1:y2, x1:x2] += 1.0
            del patch, out_patch
            torch.cuda.empty_cache()
    weight[weight == 0] = 1.0
    out = out / weight
    return out

@performance_monitor
def rauden_denoise(img: np.ndarray,
               model: Any,
               save_path: Optional[str] = None,
               device: str = 'cuda',
               use_half: bool = True,
               use_amp: bool = True,
               tile_size: Optional[tuple] = None):
    """
    Memory-optimized neural-network denoising.
    - img: input image (Z,H,W) or (Z,C,H,W)
    - model: PyTorch model (already loaded)
    - device: 'cuda' or 'cpu'
    - use_half: convert model to half precision (if supported)
    - use_amp: use torch.cuda.amp.autocast() during inference
    - tile_size: (h,w) to enable tiled inference on each slice (optional). If None, whole-slice forward.
    """
    try:
        original_dtype = img.dtype
        zchw = reformat_to_zchw(img)  # (Z,C,H,W)
        zchw = zchw.astype(np.float32)
        zchw = (zchw - zchw.min()) / (zchw.max() - zchw.min() + 1e-8)
        model.to(device)
        model.eval()
        if use_half:
            try:
                model.half()
            except Exception:
                model = model.float()
        Z, C, H, W = zchw.shape
        denoised_stack = np.zeros((Z, C, H, W), dtype=np.float32)

        for z in range(Z):
            slice_np = zchw[z:z+1]  # shape (1,C,H,W)
            tensor = torch.from_numpy(slice_np).to(device, non_blocking=True)
            if next(model.parameters()).dtype == torch.float16:
                tensor = tensor.half()
            else:
                tensor = tensor.float()
            with torch.no_grad():
                if tile_size is None:
                    if use_amp and device.startswith('cuda'):
                        with torch.amp.autocast():
                            out_tensor = model(tensor)
                    else:
                        out_tensor = model(tensor)
                    out_np = out_tensor.detach().cpu().float().numpy()
                    del out_tensor
                    torch.cuda.empty_cache()
                else:
                    out_np = tile_inference_forward(model, tensor, tile_size=tile_size, overlap=16, amp=use_amp)
                denoised_stack[z:z+1] = out_np.astype(np.float32)
            del tensor
            torch.cuda.empty_cache()

        denoised_stack = np.clip(denoised_stack, 0.0, 1.0)
        if np.issubdtype(original_dtype, np.integer):
            max_val = np.iinfo(original_dtype).max
            denoised = (denoised_stack * max_val).astype(original_dtype)
        else:
            denoised = denoised_stack.astype(original_dtype)
        if save_path:
            tiff.imwrite(save_path, denoised)
        if denoised.ndim == 4 and denoised.shape[1] == 1:
            denoised = denoised[:, 0, :, :]
        return denoised
    except Exception as e:
        print(f"Error in nn_denoise: {e}")
        raise