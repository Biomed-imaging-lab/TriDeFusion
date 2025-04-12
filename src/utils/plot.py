from typing import Dict, List, Optional, Tuple
import cv2
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np

def generate_state_colors(states: List[str]) -> List[Tuple[str, str]]:
    """
    Generate a unique color for each state.

    Args:
        states (List[str]): List of state names.

    Returns:
        List[Tuple[str, str]]: List of tuples with state names and assigned colors.
    """
    colors = sns.color_palette("hsv", len(states))
    return [(state, f"#{int(c[0]*255):02x}{int(c[1]*255):02x}{int(c[2]*255):02x}") for state, c in zip(states, colors)]


def draw_lines(
        x_values: np.ndarray,
        y_values_with_details: List[Tuple[np.ndarray, str, str]],
        axis_labels: Tuple[str, str],
        title: str,
        save_path: str,
        _marker: Optional[str] = None,
        _legend_loc: str = 'upper left',
        _legend_font_size: int = 7,
        highlight_indexes: Optional[List[int]] = None,
        x_unit: Optional[Tuple[float, str]] = None  
) -> None:
    """
    Draws vertical lines at the same x values for different y value.
    
    Parameters:
    - x_values: An array of x-coordinates.
    - y_values_with_details: A list of tuples, each containing a NumPy array of y values, a name, and a color.
    - axis_labels: A tuple containing x-label and y-label.
    - title: The title of the plot.
    - save_path: The path to save the figure. If None, the figure will not be saved.
    - _marker: The marker style for each point.
    - highlight_indexes: A list of indexes indicating which lines should be highlighted (thicker).
    - x_unit: A tuple (conversion factor, unit name) to convert x-values (e.g., (0.019, "\\mu m")).

    Returns:
    - None
    """
    fig, ax = plt.subplots()
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")

    if highlight_indexes is None:
        highlight_indexes = []

    if x_unit is not None:
        conversion_factor, unit_name = x_unit
        x_values = x_values * conversion_factor
        axis_labels = (f"{axis_labels[0]} ({unit_name})", axis_labels[1])

    for i, (y_values, name, color) in enumerate(y_values_with_details):
        line_width = 2 if i in highlight_indexes else 1
        ax.plot(x_values, y_values, color=color, marker=_marker, label=name, linewidth=line_width)

    ax.set_xlabel(f"${axis_labels[0]}$", fontweight='bold')  
    ax.set_ylabel(axis_labels[1], fontweight='bold')
    ax.set_title(title, fontweight='bold')

    legend = ax.legend(fontsize=_legend_font_size, loc=_legend_loc)
    for text in legend.get_texts():
        text.set_fontweight('bold')
    plt.setp(legend.get_title(), fontweight='bold')

    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_color('black')
    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)

    ax.tick_params(axis='x', colors='black')
    ax.tick_params(axis='y', colors='black')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    if save_path:
        print(f"Graphic {title} saved to {save_path}")
        plt.savefig(save_path, transparent=True, bbox_inches='tight', pad_inches=0)
    
    plt.close()


def draw_histogram(
    image: np.ndarray, color: str, label: str, global_max_intensity: int, is_log: bool = False, offset: float = 0
):
    """
    Draws a histogram for a single image with optional offset to avoid bar overlap.

    Args:
        image (np.ndarray): The image segment for which the histogram is drawn.
        color (str): Color for the histogram.
        label (str): Label for the image.
        global_max_intensity (int): Maximum intensity to clip the pixel values.
        is_log (bool): If True, uses logarithmic scale for the y-axis.
        offset (float): Offset for bar positions to avoid overlap.
    """
    if image.ndim == 3:  
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        image_gray = image

    pixel_values = image_gray.flatten()
    pixel_values = np.clip(pixel_values, 0, global_max_intensity)
    bins = 64 
    counts, bin_edges = np.histogram(pixel_values, bins=bins, range=(0, global_max_intensity))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bar_width = (bin_edges[1] - bin_edges[0]) * 0.9
    adjusted_bin_centers = bin_centers + offset
    plt.bar(
        adjusted_bin_centers,
        counts,
        width=bar_width,
        color=color,
        alpha=0.5,
        label=label,
        log=is_log,
        align="center",
        edgecolor="black",
    )


def draw_histograms(
    _images: List[np.ndarray],
    _info_methods: List[Tuple[str, str]],
    _title: str,
    save_path: str,
    is_log: bool = False,
):
    """
    Draw histograms for a list of images.

    Args:
        _images (List[np.ndarray]): List of images.
        _info_methods (List[Tuple[str, str]]): List of tuples containing state info (state name, color).
        _title (str): Title of the plot.
        save_path (str): Path to save the final plot.
        is_log (bool): If True, uses logarithmic scale for the y-axis.
    """
    global_max_intensity = max([np.max(image) for image in _images if np.any(image)])
    offsets = np.linspace(-2, 2, len(_images))  

    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    ax.set_facecolor("none")

    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_color('black')
    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    for image, (state, color), offset in zip(_images, _info_methods, offsets):
        draw_histogram(image, color, state, global_max_intensity, is_log=is_log, offset=offset)
    plt.title(_title, fontweight="bold")
    plt.xlabel("Intensity Value", fontweight="bold")
    plt.ylabel("Frequency", fontweight="bold")
    legend = plt.legend(title="Methods", title_fontproperties={"weight": "bold"})
    for text in legend.get_texts():
        text.set_fontweight("bold")
    plt.grid(False)
    plt.savefig(save_path, transparent=True, bbox_inches="tight", pad_inches=0)
    plt.close()
    print(f"Histograms saved to {save_path}")


def draw_box_plot(metric_data: Dict[str, List[float]], metric_name: str, title: str, save_path: str):
    """
    Draws a box plot for the given metric data.

    Args:
        metric_data (Dict[str, List[float]]): A dictionary with method names as keys and metric values as lists.
        metric_name (str): The name of the metric to display on the plot.
        save_path (str): Path to save the box plot.
    """
    plt.figure(figsize=(10, 6))
    plt.gca().set_facecolor((0, 0, 0, 0))
    plt.boxplot(
        metric_data.values(),
        patch_artist=True,
        boxprops=dict(facecolor="white", color="black", linewidth=1.5),
        whiskerprops=dict(color="black", linewidth=1.5),
        capprops=dict(color="black", linewidth=1.5),
        medianprops=dict(color="red", linewidth=1.5),
        flierprops=dict(marker="o", color="black", alpha=0.6),
    )
    plt.xticks(
        ticks=range(1, len(metric_data.keys()) + 1),
        labels=metric_data.keys(),
        rotation=30,
        ha="right",
        fontsize=10,
        fontweight="bold"
    )
    plt.title(f"{title}", fontweight="bold")
    plt.ylabel(metric_name, fontweight="bold")
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)
    plt.gca().spines["left"].set_color("black")
    plt.gca().spines["bottom"].set_color("black")
    plt.savefig(save_path, bbox_inches="tight", transparent=True)
    plt.close()
    print(f"Minimalistic box plot for {metric_name} saved to {save_path}")


def draw_box_plots(
    metric_values: Dict[str, Dict[str, List[float]]],
    metric_flags: Dict[str, bool],
    save_dir: str
):
    """
    Wrapper function to draw box plots for selected metrics.

    Args:
        metric_values (Dict[str, Dict[str, List[float]]]): A dictionary with method names as keys and metric values for each metric as nested dictionaries.
        metric_flags (Dict[str, bool]): Flags indicating which metrics to display.
        save_dir (str): Directory to save the plots.
    """
    for metric_name, flag in metric_flags.items():
        if flag:
            save_path = f"{save_dir}/{metric_name}_boxplot.png"
            # Fix: Pass all required arguments to draw_box_plot
            draw_box_plot(
                {method: values[metric_name] for method, values in metric_values.items()},
                metric_name,
                f"Box Plot for {metric_name}", 
                save_path
            )