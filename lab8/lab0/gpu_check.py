import torch
import sys

def main():
    print("=== COE201 GPU Connection Check ===")
    print(f"Python Version: {sys.version}")
    
    # Check PyTorch version
    print(f"PyTorch Version: {torch.__version__}")
    
    # Check if CUDA is available
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_available}")
    
    if cuda_available:
        # Get count of GPUs
        gpu_count = torch.cuda.device_count()
        print(f"Number of GPUs: {gpu_count}")
        
        for i in range(gpu_count):
            properties = torch.cuda.get_device_properties(i)
            print(f"\n--- GPU {i} ---")
            print(f"Name: {properties.name}")
            print(f"Compute Capability: {properties.major}.{properties.minor}")
            print(f"Total Memory: {properties.total_memory / (1024**3):.2f} GB")
            
            # Additional check: can we actually move a tensor to GPU?
            try:
                x = torch.randn(1, device=f'cuda:{i}')
                print(f"Tensor creation on GPU {i}: SUCCESS")
            except Exception as e:
                print(f"Tensor creation on GPU {i}: FAILED ({e})")
    else:
        print("\n[WARNING] CUDA is not available. Check your NVIDIA drivers and PyTorch installation.")
        if sys.platform == 'linux':
             print("Try running 'nvidia-smi' in your terminal to check driver status.")

if __name__ == "__main__":
    main()
