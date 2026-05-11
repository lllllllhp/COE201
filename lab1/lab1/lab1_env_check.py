import sys
import time


def check_environment():
    print(f"Python 版本: {sys.version}")

    # 1. 检查 NumPy
    try:
        import numpy as np

        print(f"NumPy 版本: {np.__version__}")
    except ImportError:
        print("Error: 未找到 NumPy")

    # 2. 检查 PyTorch
    try:
        import torch

        print(f"PyTorch 版本: {torch.__version__}")
        print(f"CUDA 是否可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"当前 GPU 设备: {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("Error: 未找到 PyTorch")


if __name__ == "__main__":
    check_environment()
