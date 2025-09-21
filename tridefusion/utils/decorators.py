import logging
import psutil
import functools
import time
import torch
import inspect

def get_logger(func, args):
    """Retrieve logger from class instance or module."""
    if args and hasattr(args[0], "log"):
        return getattr(args[0], "log")
    logger = logging.getLogger(func.__module__)
    if not logger.handlers:
        logging.basicConfig(level=logging.INFO)
    return logger

def get_gpu_events():
    if torch.cuda.is_available():
        return torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    return None, None

def record_gpu_start(start_event):
    if torch.cuda.is_available() and start_event:
        torch.cuda.synchronize()
        start_event.record()

def record_gpu_end(end_event):
    if torch.cuda.is_available() and end_event:
        end_event.record()
        torch.cuda.synchronize()

def measure_gpu_time(start_event, end_event):
    if torch.cuda.is_available() and start_event and end_event:
        return start_event.elapsed_time(end_event) / 1000  # ms to s
    return None

def measure_resources_before():
    process = psutil.Process()
    return {
        "cpu_mem": process.memory_info().rss / (1024 ** 2),
        "cpu_time": process.cpu_times(),
        "gpu_mem": torch.cuda.memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else None,
        "time": time.time()
    }

def measure_resources_after(start_info, gpu_start, gpu_end):
    process = psutil.Process()
    cpu_mem_after = process.memory_info().rss / (1024 ** 2)
    cpu_time_after = process.cpu_times()
    time_after = time.time()

    cpu_mem_used = cpu_mem_after - start_info["cpu_mem"]
    cpu_time = (
        (cpu_time_after.user - start_info["cpu_time"].user) +
        (cpu_time_after.system - start_info["cpu_time"].system)
    )
    wall_time = time_after - start_info["time"]

    gpu_time = measure_gpu_time(gpu_start, gpu_end)
    gpu_mem_after = torch.cuda.memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else None
    gpu_mem_used = gpu_mem_after - start_info["gpu_mem"] if gpu_mem_after is not None else None

    return wall_time, cpu_time, cpu_mem_used, gpu_time, gpu_mem_used

def log_metrics(logger, func_name, wall_time, cpu_time, cpu_mem_used, gpu_time=None, gpu_mem_used=None):
    logger.info(f"Finished {func_name} | Wall time: {wall_time:.4f}s | CPU time: {cpu_time:.4f}s | CPU mem: {cpu_mem_used:.2f}MB")
    if gpu_time is not None and gpu_mem_used is not None:
        logger.info(f"{func_name} | GPU time: {gpu_time:.4f}s | GPU mem: {gpu_mem_used:.2f}MB")

def format_selected_params(func, args, kwargs, selected):
    bound_args = inspect.signature(func).bind(*args, **kwargs)
    bound_args.apply_defaults()
    return {k: v for k, v in bound_args.arguments.items() if k in selected}

def performance_monitor(print_params=None):
    """
    Decorator factory to monitor CPU/GPU memory, wall time, CPU time, GPU time.
    Args:
        print_params (list[str] or None): list of parameter names to print, or None to skip printing.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            logger = get_logger(func, args)
            start_info = measure_resources_before()
            gpu_start, gpu_end = get_gpu_events()
            logger.info(f"Starting {func.__name__}")
            if print_params:
                selected_values = format_selected_params(func, args, kwargs, print_params)
                logger.info(f"Selected parameters: {selected_values}")
                print(f">>> {func.__name__} called with selected params: {selected_values}")
            record_gpu_start(gpu_start)

            try:
                result = func(*args, **kwargs)
            except Exception as e:
                logger.error(f"Error in {func.__name__}: {str(e)}")
                raise

            record_gpu_end(gpu_end)
            wall_time, cpu_time, cpu_mem_used, gpu_time, gpu_mem_used = measure_resources_after(
                start_info, gpu_start, gpu_end
            )
            log_metrics(logger, func.__name__, wall_time, cpu_time, cpu_mem_used, gpu_time, gpu_mem_used)
            return result
        return wrapper
    return decorator
