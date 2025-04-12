from prometheus_client import Summary, Counter, Gauge
import psutil  
import gpustat  


image_processing_time = Summary("image_processing_time_seconds", "Time spent processing images")
image_processing_errors = Counter("image_processing_errors_total", "Total failed image processing attempts")
queue_size_gauge = Gauge("celery_queue_size", "Current number of tasks in Celery queue")
cpu_usage_gauge = Gauge("cpu_usage_percent", "Current CPU usage percentage")
memory_usage_gauge = Gauge("memory_usage_percent", "Current system memory usage percentage")
gpu_usage_gauge = Gauge("gpu_usage_percent", "Current GPU usage percentage")

def track_processing_time(func):
    """Decorator to track processing time using Prometheus."""
    def wrapper(*args, **kwargs):
        with image_processing_time.time():
            return func(*args, **kwargs)
    return wrapper

def update_system_metrics():
    """Update system metrics such as CPU, memory, and GPU usage."""
    cpu_usage = psutil.cpu_percent(interval=1)
    cpu_usage_gauge.set(cpu_usage)
    memory_usage = psutil.virtual_memory().percent
    memory_usage_gauge.set(memory_usage)
    try:
        gpu_stats = gpustat.GPUStatCollection.new_query()
        gpu_usage = gpu_stats[0].utilization  
        gpu_usage_gauge.set(gpu_usage)
    except Exception as e:
        print(f"Error while fetching GPU stats: {e}")

# Example usage: call `update_system_metrics()` periodically to update the metrics

