"""
=============================================================================
 HARDWARE SNAPSHOT - the workshop warm-up
=============================================================================
 Run this first, on the projector, before demo 1:

     python inspect_system.py

 It exists to make one point before anything else happens: everything you are
 about to see runs on THIS machine. No cloud, no API key, no GPU, no NPU.
 Just the CPU already in the laptop.

 Note what it reports about accelerators. On almost every machine in the room
 the answer will be "none available" - and the workshop works anyway. That is
 the entire premise of edge AI.

 (For a full check of packages, models and databases, run verify_setup.py
 instead. This file only looks at hardware.)
=============================================================================
"""
import os
import sys
import platform

import psutil


def main():
    print("=" * 60)
    print("HARDWARE SNAPSHOT")
    print("=" * 60)

    # --- Operating system and Python ---------------------------------------
    print("\nSYSTEM")
    print("-" * 60)
    print(f"  OS           : {platform.system()} {platform.release()}")
    print(f"  Architecture : {platform.machine()}")
    print(f"  Python       : {sys.version.split()[0]}")

    # --- Processor ----------------------------------------------------------
    # This is the only compute the workshop uses.
    print("\nPROCESSOR  (this is what runs the model)")
    print("-" * 60)
    physical = psutil.cpu_count(logical=False)
    logical = psutil.cpu_count(logical=True)
    print(f"  Name         : {platform.processor() or 'unknown'}")
    print(f"  Cores        : {physical} physical / {logical} logical")

    freq = psutil.cpu_freq()
    if freq and freq.max:
        print(f"  Max clock    : {freq.max / 1000:.2f} GHz")

    # --- Memory -------------------------------------------------------------
    # The unquantized model needs ~3.7 GB; the quantized one needs ~0.9 GB.
    print("\nMEMORY")
    print("-" * 60)
    ram = psutil.virtual_memory()
    print(f"  Total RAM    : {ram.total / (1024 ** 3):.2f} GB")
    print(f"  Available    : {ram.available / (1024 ** 3):.2f} GB")
    print(f"  Base model needs ~3.7 GB | quantized model needs ~0.9 GB")

    # --- Disk ---------------------------------------------------------------
    print("\nDISK")
    print("-" * 60)
    try:
        usage = psutil.disk_usage(os.getcwd())
        print(f"  Free space   : {usage.free / (1024 ** 3):.1f} GB "
              f"(the workshop needs about 10 GB)")
    except Exception:
        print("  Free space   : could not determine")

    # --- Accelerators -------------------------------------------------------
    # Deliberately last, and deliberately expected to be empty.
    print("\nACCELERATORS  (we use none of these)")
    print("-" * 60)
    found = []

    try:
        import torch
        print(f"  PyTorch      : {torch.__version__}")
        if torch.cuda.is_available():
            found.append(f"CUDA GPU ({torch.cuda.get_device_name(0)})")
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            found.append("Intel XPU")
    except ImportError:
        print("  PyTorch      : not installed")

    if found:
        for f in found:
            print(f"  Detected     : {f}")
        print("\n  An accelerator is present, but this workshop does not use it.")
        print("  Every demo is pinned to CPU on purpose.")
    else:
        print("  Detected     : none - CPU only")
        print("\n  Good. This is the normal case, and everything still works.")

    print("\n" + "=" * 60)
    print("Ready. Next: python tests/test_00_base_slm.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
