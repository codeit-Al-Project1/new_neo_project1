from ultralytics import YOLO
from pathlib import Path
import torch

num_classes = 75

model_path = 'yolov5s.pt'  # 모델 경로

model_path = Path(model_path)
model = YOLO(model_path)  # pretrained load

if num_classes:
    # 클래스 수가 지정되면 head 수정
    model.model.nc = num_classes  
    model.model.names = [f'class_{i}' for i in range(num_classes)]  # 이름 초기화 (원하면 커스텀 가능)

print(model)

model.eval()

with torch.no_grad():
    img = torch.randn(1, 3, 640, 640)  # 랜덤 이미지 생성
    pred = model(img)  # 예측
    print(pred)  # 예측 결과 출력