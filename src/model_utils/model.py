import os
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights

def get_fast_rcnn_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)

    # 기존 분류 헤드 수정
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model

def save_model(model, save_dir="../models", base_name="model", ext=".pth"):
    """
    모델을 저장할 때, 기존 파일 개수를 기준으로 자동 넘버링하여 저장하는 함수.
    
    Args:
        model (torch.nn.Module): 저장할 PyTorch 모델
        save_dir (str): 모델을 저장할 폴더 (기본값: "models")
        base_name (str): 저장할 파일의 기본 이름 (기본값: "model")
        ext (str): 저장할 파일 확장자 (기본값: ".pth")

    Returns:
        str: 저장된 파일의 전체 경로
    """

    # 저장할 디렉토리 생성 (없으면 생성)
    os.makedirs(save_dir, exist_ok=True)

    # 기존 모델 파일 목록 가져오기
    existing_models = [f for f in os.listdir(save_dir) if f.startswith(base_name) and f.endswith(ext)]

    # 저장할 모델 번호 계산
    model_count = len(existing_models) + 1

    # 저장할 파일 경로 생성
    model_save_path = os.path.join(save_dir, f"{base_name}_{model_count}{ext}")

    # 모델 저장
    torch.save(model.state_dict(), model_save_path)

    print(f"Model saved to {model_save_path}")
    return model_save_path

def load_model():
    pass