"""Check GPU and PyTorch CUDA support"""
import torch

print("\n[GPU CHECK]")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU device: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f}GB")
else:
    print("\n[WARNING] CUDA not available!")
    print("PyTorch is installed without GPU support.")
    print("\nTo fix:")
    print("  1. Uninstall current PyTorch: py -m pip uninstall torch torchvision torchaudio")
    print("  2. Install CUDA version: py -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")


