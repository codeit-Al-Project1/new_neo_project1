# 로컬 GPU 구동 확인
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Now using device: {device}")  # cuda
