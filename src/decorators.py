import logging
import psutil
import functools
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def memory_monitor(func):
    """
    Decorator to monitor memory usage before and after a function call.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        process = psutil.Process()
        mem_before = process.memory_info().rss / (1024 ** 2) 
        start_time = time.time()
        logging.info(f"Starting {func.__name__} | Memory before: {mem_before:.2f} MB")
        try:
            result = func(*args, **kwargs)
        except Exception as e:
            logging.error(f"Error in {func.__name__}: {str(e)}")
            raise
        mem_after = process.memory_info().rss / (1024 ** 2) 
        mem_used = mem_after - mem_before 
        elapsed_time = time.time() - start_time
        logging.info(
            f"Finished {func.__name__} | Memory after: {mem_after:.2f} MB | "
            f"Memory used: {mem_used:.2f} MB | Elapsed time: {elapsed_time:.2f} seconds"
        )
        return result
    return wrapper
