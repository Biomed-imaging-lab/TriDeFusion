from typing import Union, List


def check_positive_integer(value: Union[int, List[int]], name: str) -> bool:
    if not isinstance(value, (int, list)):
        raise ValueError(f"{name} must be a positive integer or a list of positive integers.")
    if (isinstance(value, int) and value < 0) or (isinstance(value, list) and not all(v < 0 for v in value)):
        return False
    else:
        return True

def check_not_none(value, name, func_name = None):
    """Check that a value is not None."""
    func_desc = f" Run {func_name} first." if func_name else ""
    if value is None:
        raise ValueError(f"{name} should not be None" + func_desc)
    
async def valid_method_name(method: str, method_list: List[str], method_type: str):
    if method is None or method not in method_list:
        raise ValueError(f"Incorrect {method_type} type not found in the cache.")
