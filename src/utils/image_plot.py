from typing import Optional, Tuple
from matplotlib import cm, pyplot as plt, patches
from mpl_toolkits.axes_grid1 import AxesGrid
import numpy as np
import imageio
from PIL import Image, ImageDraw, ImageFont

from utils.image_loader import normalize_image


def generate_image_slices(
        img: np.ndarray,
        cmap,
        xy_resolution: float,
        z_resolution: float,
        coords: Tuple[int, int, int] = (0, 0, 0),
        scale_borders: Optional[Tuple[int, int]] = None,
        is_show: bool = False
) -> plt.Figure:
    """
    Generates image slices from a 3D image and visualizes XZ, YZ, and XY projections.
    Parameters:
    - img (np.ndarray): 3D input image.
    - cmap: Colormap for visualization.
    - xy_resolution (float): Resolution in the XY plane.
    - z_resolution (float): Resolution in the Z direction.
    - coords (tuple[int, int, int]): Indices for slicing (layer, row, column).
    - scale_borders (tuple[int, int] or None): Min/max intensity values for normalization.
    - is_show (bool): Whether to display the figure.

    Returns:
    - plt.Figure: The generated figure.
    """
    plane_layer = img[coords[0], :, :]
    plane_row = img[:, coords[1], :]
    plane_col = img[:, :, coords[2]].T 
    if scale_borders is None:
        min_val = min(plane_row.min(), plane_col.min(), plane_layer.min())
        max_val = max(plane_row.max(), plane_col.max(), plane_layer.max())
        scale_borders = (min_val, max_val)
    aspect_xz = z_resolution / xy_resolution
    aspect_yz = 1 / aspect_xz
    img_shape = np.array(img.shape)
    plt_width = (img_shape[0] + img_shape[2]) / 25
    plt_height = (img_shape[0] + img_shape[1]) / 25
    fig = plt.figure(figsize=(plt_width, plt_height))
    grid = AxesGrid(fig, 111, nrows_ncols=(2, 2), axes_pad=0.05,
                    cbar_mode="single", cbar_location="right", cbar_pad=0.1)
    grid[0].axis("off")
    grid[1].axis("off")
    grid[1].set_title("X-Z projection")
    im_main = grid[1].imshow(plane_row, cmap=cmap, vmin=scale_borders[0], vmax=scale_borders[1], aspect=aspect_xz)
    grid[2].axis("off")
    grid[2].set_title("Y-Z projection")
    im = grid[2].imshow(plane_col, cmap=cmap, vmin=scale_borders[0], vmax=scale_borders[1], aspect=aspect_yz)
    grid[3].axis("off")
    grid[3].set_title("X-Y projection")
    grid[3].imshow(plane_layer, cmap=cmap, vmin=scale_borders[0], vmax=scale_borders[1])
    grid.cbar_axes[0].colorbar(im_main)
    if is_show:
        plt.show()
    return fig


def draw_segment_rectangle(img: np.ndarray, segment_coords: Tuple[Tuple[int, int], Tuple[int, int]], layer_idx: int = None, save_path: str = None):
    """
    Draw a rectangle around the specified segment in the 2D projection of the image.
    Parameters:
    - img: 2D or 3D numpy array (if 3D, a middle slice is chosen for 2D projection).
    - segment_coords: List of 4 integers [y1, y2, x1, x2], where (y1, x1) is the top-left corner
                      and (y2, x2) is the bottom-right corner of the segment.
    - save_path: Path to save the output image with the drawn rectangle.
    - show: Whether to display the image with the rectangle.
    """
    if img.ndim == 3:
        if layer_idx is None:
            img_2d = img[img.shape[0] // 2]
        else:
            img_2d = img[layer_idx]
    else:
        img_2d = img
    fig, ax = plt.subplots(1)
    ax.imshow(img_2d, cmap='gray')
    (x1, y1), (x2, y2) = segment_coords
    rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    if save_path:
        plt.savefig(save_path)
    plt.close(fig)

def draw_3d_figure(data: np.ndarray, output_path: str = None) -> None:
    """
    Draw a 3D figure of a NumPy array (e.g., a 3D surface plot) where each slice in the Z-axis is visualized
    as a separate surface in one single 3D plot, making the XY plane vertical.

    Args:
        data (np.ndarray): The 3D NumPy array to be visualized.
        output_path (str, optional): The path to save the output plot as PNG. If None, the plot will be shown instead.

    Returns:
        None
    """
    if data.ndim != 3:
        raise ValueError("Input data must be a 3D NumPy array.")

    normalized_data = normalize_image(data)

    z = np.arange(normalized_data.shape[0])  
    x = np.arange(normalized_data.shape[1])  
    y = np.arange(normalized_data.shape[2])  
    x, y = np.meshgrid(x, y)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    for i in range(normalized_data.shape[0]):
        z_layer = normalized_data[i, :, :]  
        ax.plot_surface(x, y, np.full_like(x, i), facecolors=cm.jet(z_layer), 
                        rstride=1, cstride=1, alpha=0.8, linewidth=0)
    ax.set_xlabel("X Axis", labelpad=15)
    ax.set_ylabel("Y Axis", labelpad=15)
    ax.set_zlabel("Layer (Z Axis)", labelpad=10)
    ax.zaxis.label.set_position((0, -0.1))  
    mappable = cm.ScalarMappable(cmap='jet')
    mappable.set_array(normalized_data)
    cbar = fig.colorbar(mappable, ax=ax, shrink=0.5, aspect=15, pad=0.1)  
    cbar.set_label("Normalized Intensity", labelpad=10)
    fig.patch.set_alpha(0)
    ax.view_init(elev=20, azim=120)  
    if output_path:
        plt.savefig(output_path, transparent=True, dpi=300, bbox_inches='tight', pad_inches=0)
        print(f"3D plot saved to {output_path}")
    else:
        plt.show()
    plt.close(fig)

def create_transition_gif(noisy_gif_path, denoised_gif_path, output_path, duration=0.1, width_ratio=0.005, loop: int = 0):
    noisy = Image.open(noisy_gif_path)
    processed = Image.open(denoised_gif_path)
    frames_noisy, frames_processed = [], []
    try:
        while True:
            frames_noisy.append(noisy.convert("RGB").copy())  
            frames_processed.append(processed.convert("RGB").copy())
            noisy.seek(noisy.tell() + 1)
            processed.seek(processed.tell() + 1)
    except EOFError:
        pass  

    num_frames = min(len(frames_noisy), len(frames_processed))
    width, height = frames_noisy[0].size
    bar_width = int(width * width_ratio)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", int(height * 0.05))
    except IOError:
        font = ImageFont.load_default(int(height * width_ratio * 10))
    new_frames = []
    for i in range(num_frames):
        noisy_frame = frames_noisy[i]
        denoised_frame = frames_processed[i]
        new_frame = Image.new("RGB", (width, height))
        bar_position = int(i / (num_frames - 1) * width)
        new_frame.paste(denoised_frame.crop((0, 0, bar_position, height)), (0, 0))
        new_frame.paste(noisy_frame.crop((bar_position, 0, width, height)), (bar_position, 0))
        draw = ImageDraw.Draw(new_frame)
        draw.rectangle([bar_position - bar_width // 2, 0, bar_position + bar_width // 2, height], fill="white")
        draw.rectangle([0, 0, width - 1, height - 1], outline="white", width=bar_width)
        draw.text((20, 20), "Denoised image", font=font, fill="white")
        right_label_x = width - 20 - draw.textlength("Noisy image", font=font)
        if bar_position < right_label_x:
            draw.text((right_label_x, 10), "Noisy image", font=font, fill="white")  
        new_frames.append(np.array(new_frame))
    imageio.mimsave(output_path, new_frames, duration=duration, loop=loop)