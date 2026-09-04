import platform
import os
import sys
import psutil

print("="*40)
print("SYSTEM PRE-FLIGHT")
print("="*40)

print(f"OS: {platform.system()} {platform.release()}")
print(f"CPU: {platform.processor()}")
print(f"Architecture: {platform.machine()}")
print(f"Python: {sys.version.split(' ')[0]}")

ram = psutil.virtual_memory()
print(f"Total RAM: {ram.total / (1024**3):.2f} GB")
print(f"Available RAM: {ram.available / (1024**3):.2f} GB")

try:
    import torch
    print(f"PyTorch: {torch.__version__}")
    print(f"Intel XPU: {'AVAILABLE' if hasattr(torch, 'xpu') and torch.xpu.is_available() else 'UNAVAILABLE'}")
except ImportError:
    print("PyTorch: NOT INSTALLED")

try:
    from openvino import Core
    core = Core()
    print(f"OpenVINO: AVAILABLE")
    print(f"OpenVINO Devices: {core.available_devices}")
except ImportError:
    print("OpenVINO: NOT INSTALLED")

print("="*40)
