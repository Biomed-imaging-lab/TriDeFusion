import numpy as np
import typing as tp

def generate_circle(rad_x: float, rad_y: float, rad_z: float,
                    center_layer: float, center_row: float, center_col: float,
                    shape: tp.Tuple[int], lower_intensity: int) -> np.ndarray:
    Z, Y, X = shape
    z = np.arange(Z).reshape(Z, 1, 1)
    y = np.arange(Y).reshape(1, Y, 1)
    x = np.arange(X).reshape(1, 1, X)
    dz = ((z - center_layer) ** 2) / (rad_z ** 2)
    dy = ((y - center_row) ** 2) / (rad_x ** 2)
    dx = ((x - center_col) ** 2) / (rad_y ** 2)
    ellipsoid = dz + dy + dx
    mask = ellipsoid <= 1
    circle = np.zeros((Z, Y, X), dtype=np.float32)
    circle[mask] = lower_intensity + (255 - lower_intensity) * (1 - ellipsoid[mask])
    return circle



def draw_mask_in_position(orig_img: np.ndarray, brush_mask: np.ndarray, coord: np.ndarray, brush_center: np.ndarray):
    start = np.array(coord) - brush_center
    l0, r0 = max(0, -start[0]), min(brush_mask.shape[0], orig_img.shape[0] - start[0])
    l1, r1 = max(0, -start[1]), min(brush_mask.shape[1], orig_img.shape[1] - start[1])
    l2, r2 = max(0, -start[2]), min(brush_mask.shape[2], orig_img.shape[2] - start[2])
    s0, e0 = start[0] + l0, start[0] + r0
    s1, e1 = start[1] + l1, start[1] + r1
    s2, e2 = start[2] + l2, start[2] + r2
    orig_img[s0:e0, s1:e1, s2:e2] += brush_mask[l0:r0, l1:r1, l2:r2]
    return orig_img
