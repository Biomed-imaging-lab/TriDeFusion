from typing import Any, Union, List
import numpy as np


def check_positive_integer(value: Union[int, List[int]], name: str):
    if isinstance(value, int) or isinstance(value, float):
        assert value > 0, f"{name} must be a positive integer."
    else:
        assert all(v > 0 for v in value), f"All elements in {name} must be positive integers."

def check_not_none(value: Any, name: str, func_name: str = None):
    func_desc = f" Run {func_name} first." if func_name else ""
    assert value is not None and (not isinstance(value, list) or len(value) > 0), (f"{name} is not available."
                                                                                   f"{func_desc}")
    
def valid_method_name(method: str, method_list: List[str], method_name: str):
    if not isinstance(method, str) or method is None or method not in method_list:
        raise ValueError(f"Incorrect {method_name} type not found in the cache.")

def check_input_image(image: Any, name: str, func_name: str = None):
    """Validate the input image format and dimensions.
    Args:
        image (Any): The input image.
        name (str): Name of the image variable (for error messages).
        func_name (str, optional): Function name where the check is performed.
    Raises:
        TypeError: If the image is not a NumPy array.
        ValueError: If the image does not have 2, 3 or 4 dimensions.
    """
    check_not_none(image, name, func_name) 
    if not isinstance(image, np.ndarray):
        raise TypeError(f"{'[' + func_name + '] ' if func_name else ''}{name} must be a NumPy array.")
    if image.ndim not in [2, 3, 4]:
        raise ValueError(f"{'[' + func_name + '] ' if func_name else ''}{name} must be a 2D, 3D or 4D NumPy array.")