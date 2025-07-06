import torch
import platform
import sys

def get_device():
    """
    デバイスを自動選択
    優先順位: CUDA (Windows RTX) > MPS (Mac) > CPU
    """
    device = "cpu"
    device_name = "CPU"
    
    if torch.cuda.is_available():
        device = "cuda"
        device_name = f"CUDA ({torch.cuda.get_device_name(0)})"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = "mps"
        device_name = "Apple Metal Performance Shaders"
    
    return device, device_name

def get_system_info():
    """システム情報を取得"""
    info = {
        "platform": platform.system(),
        "platform_version": platform.version(),
        "python_version": sys.version,
        "torch_version": torch.__version__,
    }
    
    device, device_name = get_device()
    info["device"] = device
    info["device_name"] = device_name
    
    if device == "cuda":
        info["cuda_version"] = torch.version.cuda
        info["gpu_count"] = torch.cuda.device_count()
    
    return info