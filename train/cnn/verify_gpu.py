"""
Fail-fast GPU check. Exits non-zero (aborting the pipeline) if CUDA is not
available or a real forward pass does not land on the GPU. Run before the
overnight chain so we never silently train the CNN on CPU.
"""
import sys
import torch
from cnn_forecaster import SpatioTemporalCNN, K, BASE_CH

if not torch.cuda.is_available():
    print("FAIL: torch.cuda.is_available() == False — no GPU visible.")
    sys.exit(1)

dev = torch.device("cuda")
name = torch.cuda.get_device_name(0)
print(f"CUDA OK — device: {name}")

# real forward pass on GPU
model = SpatioTemporalCNN(ny=50, nx=50, k=K, base_ch=BASE_CH).to(dev)
x = torch.randn(2, 1, K, 50, 50, device=dev)
y = model(x)
assert y.device.type == "cuda", f"FAIL: output on {y.device}, expected cuda"
mem = torch.cuda.max_memory_allocated() / 1e6
print(f"Forward pass on GPU OK — output {tuple(y.shape)} on {y.device}, "
      f"peak {mem:.0f} MB")
print("GPU verification passed.")
