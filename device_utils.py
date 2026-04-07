import platform
import torch


def get_best_device() -> str:
    """Prefer CUDA on Linux/NVIDIA, then Apple MPS, then CPU."""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def describe_device() -> str:
    device = get_best_device()
    system = platform.system()

    if device == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        return f"{device} ({gpu_name} on {system})"
    if device == "mps":
        return f"{device} (Apple Silicon on {system})"
    return f"{device} ({system})"
