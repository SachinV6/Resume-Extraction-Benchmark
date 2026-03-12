import time
import psutil

try:
    import pynvml
    pynvml.nvmlInit()
    GPU_AVAILABLE = True
except Exception:
    GPU_AVAILABLE = False


# -------------------------
# Token Estimation
# -------------------------
def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return int(len(text.split()) * 1.3)


# -------------------------
# System Usage Snapshot
# -------------------------
def capture_usage():
    usage = {
        "system": {
            "system_cpu_percent": psutil.cpu_percent(interval=None),
            "system_ram_percent": psutil.virtual_memory().percent
        },
        "python_process": {
            "process_cpu_percent": psutil.Process().cpu_percent(interval=None),
            "process_ram_mb": round(psutil.Process().memory_info().rss / 1024 / 1024, 2)
        }
    }

    if GPU_AVAILABLE:
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        usage["gpu"] = {
            "gpu_memory_used_mb": round(mem.used / 1024 / 1024, 2),
            "gpu_memory_total_mb": round(mem.total / 1024 / 1024, 2),
            "gpu_utilization_percent": pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
        }

    return usage


# -------------------------
# Compute Metrics
# -------------------------
def compute_metrics(prompt, output, start_time, end_time, usage_before, usage_after):
    time_seconds = round(end_time - start_time, 3)

    prompt_tokens = estimate_tokens(prompt)
    output_tokens = estimate_tokens(output)
    total_tokens = prompt_tokens + output_tokens

    return {
        "prompt_tokens_estimate": prompt_tokens,
        "output_tokens_estimate": output_tokens,
        "total_tokens_estimate": total_tokens,
        "time_seconds": time_seconds,
        "tokens_per_second_estimate": round(total_tokens / time_seconds, 2) if time_seconds > 0 else 0,
        "usage_before": usage_before,
        "usage_after": usage_after,
    }