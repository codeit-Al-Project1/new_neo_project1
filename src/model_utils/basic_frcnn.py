import os
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import EfficientNetB3_FPN_Weights

def get_fast_rcnn_model(num_classes, backbone="resnet50"):
    # 사용할 백본 선택
    if backbone == "resnet50":
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
    
    elif backbone == "mobilenet_v3_large":
        backbone_model = torchvision.models.mobilenet_v3_large(weights="IMAGENET1K_V1").features
        backbone_model.out_channels = 960  # MobileNetV3의 출력 채널
        anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),), aspect_ratios=((0.5, 1.0, 2.0),) * 5)
        model = FasterRCNN(backbone_model, num_classes=num_classes, rpn_anchor_generator=anchor_generator)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
    
    elif backbone == "resnext101":
        backbone_model = torchvision.models.resnext101_32x8d(weights="IMAGENET1K_V1")
        backbone_model = torchvision.models.detection.backbone_utils.resnet_fpn_backbone("resnext101_32x8d", pretrained=True)
        model = FasterRCNN(backbone_model, num_classes=num_classes)
        in_features = model.roi_heads.box_predictor.cls_score.in_features

    elif backbone == "efficientnet_b3":
        backbone_model = torchvision.models.efficientnet_b3(weights="IMAGENET1K_V1").features
        backbone_model.out_channels = 1536  # EfficientNet-B3의 출력 채널
        anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),), aspect_ratios=((0.5, 1.0, 2.0),) * 5)
        model = FasterRCNN(backbone_model, num_classes=num_classes, rpn_anchor_generator=anchor_generator)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
    
    else:
        raise ValueError(f"Unsupported backbone: {backbone}")

    # 분류 헤드 수정
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


def get_new_session_folder(save_dir="./models", session_prefix="frcnn_session_"):
    """
    새로운 학습 세션을 위한 폴더명을 생성하는 함수.

    Args:
        save_dir (str): 세션 폴더를 저장할 디렉토리 (기본값: "models")
        session_prefix (str): 세션 폴더의 기본 이름 (기본값: "frcnn_session_")

    Returns:
        str: 생성된 세션 폴더 경로
    """
    os.makedirs(save_dir, exist_ok=True)

    # 기존 세션 폴더 개수를 기준으로 새로운 세션 번호 부여
    existing_sessions = [d for d in os.listdir(save_dir) if d.startswith(session_prefix) and os.path.isdir(os.path.join(save_dir, d))]
    new_session_id = len(existing_sessions) + 1
    session_folder = os.path.join(save_dir, f"{session_prefix}{new_session_id}")
    
    os.makedirs(session_folder, exist_ok=True)
    return session_folder

def save_model(model, session_folder, base_name="best_model", ext=".pth", lr=None, epoch=None, batch_size=None, optimizer=None, scheduler=None, weight_decay=None):
    """
    특정 세션 폴더에 최고 성능 모델을 저장하는 함수.

    Args:
        model (torch.nn.Module): 저장할 PyTorch 모델
        session_folder (str): 세션 폴더 경로
        base_name (str): 저장할 파일의 기본 이름 (기본값: "best_model")
        ext (str): 저장할 파일 확장자 (기본값: ".pth")
        lr (float, optional): 학습률
        epoch (int, optional): 현재 에포크
        batch_size (int, optional): 배치 크기
        optimizer (str, optional): 옵티마이저 이름
        scheduler (str, optional): 스케줄러 이름
        weight_decay (float, optional): 가중치 감소 (L2 정규화)

    Returns:
        str: 저장된 파일의 전체 경로
    """
    # None 값을 빈 문자열로 변환
    def safe_str(value):
        return str(value) if value is not None else "NA"

    # 파일명에 들어갈 하이퍼파라미터 문자열 구성
    param_str = f"lr={safe_str(lr)}_ep={safe_str(epoch)}_bs={safe_str(batch_size)}_opt={safe_str(optimizer)}_scd={safe_str(scheduler)}_wd={safe_str(weight_decay)}"
    
    # 파일명에서 문제될 수 있는 문자 제거 (특히 `/`, `\`, `:` 등)
    param_str = param_str.replace("/", "_").replace("\\", "_").replace(":", "_")

    model_save_path = os.path.join(session_folder, f"{base_name}_{param_str}{ext}")
    torch.save(model.state_dict(), model_save_path)

    print(f"Model saved to {model_save_path}")
    return model_save_path