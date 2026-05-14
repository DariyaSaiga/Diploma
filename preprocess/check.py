
import torch

checkpoint = torch.load("EMA_grad_4702.pt", map_location="cpu")

print(checkpoint.keys())
print(checkpoint.get("iteration"))
print(checkpoint.get("epoch"))