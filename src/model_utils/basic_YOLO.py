import os
import torch


def get_yolov5_model(model_name='yolov5s', num_classes=None, pretrained=True):
    """
    YOLOv5 모델을 로드하는 함수

    Args:
        model_name (str): 사용할 YOLOv5 모델 이름 (예: yolov5s, yolov5m, yolov5l, yolov5x)
        num_classes (int, optional): 클래스 수 변경이 필요할 때 입력 (기본 None이면 COCO 80 클래스)
        pretrained (bool): pretrained weight 여부

    Returns:
        model: 로드된 YOLOv5 모델
    """
    # YOLOv5 모델을 PyTorch Hub에서 로드
    model = torch.hub.load('ultralytics/yolov5', model_name, pretrained=pretrained)

    # 클래스 수 변경이 필요한 경우
    if num_classes is not None:
        model.model[-1].nc = num_classes
        model.model[-1].reset_parameters()

    return model


def save_model(model, save_dir="../models", base_name="yolov5", ext=".pt"):
    """
    YOLOv5 모델을 저장할 때, 자동 넘버링하여 저장하는 함수

    Args:
        model (torch.nn.Module): 저장할 PyTorch YOLOv5 모델
        save_dir (str): 모델을 저장할 폴더 (기본값: "../models")
        base_name (str): 저장할 파일의 기본 이름 (기본값: "yolov5")
        ext (str): 저장할 파일 확장자 (기본값: ".pt")

    Returns:
        str: 저장된 파일의 전체 경로
    """
    # 저장 디렉토리 생성
    os.makedirs(save_dir, exist_ok=True)

    # 기존 모델 파일 목록 가져오기
    existing_models = [f for f in os.listdir(save_dir) if f.startswith(base_name) and f.endswith(ext)]

    # 저장할 모델 번호 계산
    model_count = len(existing_models) + 1

    # 저장할 파일 경로 생성
    model_save_path = os.path.join(save_dir, f"{base_name}_{model_count}{ext}")

    # 모델 저장 (전체 모델)
    torch.save(model.state_dict(), model_save_path)

    print(f"Model saved to {model_save_path}")
    return model_save_path


def load_model(model_path, model_name='yolov5s', num_classes=None):
    """
    저장된 YOLOv5 모델 로드

    Args:
        model_path (str): 저장된 모델의 경로
        model_name (str): 기본 YOLOv5 모델 구조 (기본: yolov5s)
        num_classes (int, optional): 클래스 수 (기본 None)

    Returns:
        model: 로드된 YOLOv5 모델
    """
    # 모델 구조 초기화
    model = get_yolov5_model(model_name=model_name, num_classes=num_classes, pretrained=False)

    # weights 로드
    model.load_state_dict(torch.load(model_path))

    print(f"Model loaded from {model_path}")
    return model
