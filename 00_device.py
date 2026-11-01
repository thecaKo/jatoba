import torch
import time

matrizA = torch.randn(4096, 4096)
matrizB = torch.randn(4096, 4096)

matrizA_gpu = matrizA.to("mps")
matrizB_gpu = matrizB.to("mps")

torch.mps.synchronize()
start = time.time()
result_gpu = matrizA_gpu @ matrizB_gpu
torch.mps.synchronize()
end = time.time()

print("GPU: ", end - start, "segundos")
