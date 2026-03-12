import time
import psutil
import tiktoken

try:
    import pynvml
    pynvml.nvmlInit()
    GPU_ENABLED = True
except Exception:
    GPU_ENABLED = False


class Timer:
    def __init__(self):
        self.start_time = None
        self.elapsed = 0

    def start(self):
        self.start_time = time.time()

    def stop(self):
        self.elapsed = time.time() - self.start_time


def count_tokens(text: str) -> int:
    if not text:
        return 0
    enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))


def get_full_usage_snapshot():
    process = psutil.Process()

    snapshot = {
        "system_cpu_percent": psutil.cpu_percent(interval=None),
        "system_ram_percent": psutil.virtual_memory().percent,
        "process_cpu_percent": process.cpu_percent(interval=None),
        "process_ram_mb": process.memory_info().rss / (1024 * 1024),
    }

    if GPU_ENABLED:
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)

            snapshot["gpu_util_percent"] = util.gpu
            snapshot["gpu_mem_used_mb"] = mem.used / (1024 * 1024)
            snapshot["gpu_mem_total_mb"] = mem.total / (1024 * 1024)
        except Exception:
            snapshot["gpu_util_percent"] = None
            snapshot["gpu_mem_used_mb"] = None
            snapshot["gpu_mem_total_mb"] = None
    else:
        snapshot["gpu_util_percent"] = None
        snapshot["gpu_mem_used_mb"] = None
        snapshot["gpu_mem_total_mb"] = None

    return snapshot