# import torch
# from tqdm import tqdm
# import os
# from src.data_utils.data_loader import get_loader, get_category_mapping
# from src.utils import get_optimizer, get_scheduler  # utils.py에서 가져오기
# # from src.model_utils.basic_YOLO import get_yolov5  # YOLO 모델
# # from ultralytics.yolo.utils.loss import ComputeLoss - 이 놈이 너무 문제여서 git clone으로 가져옴
# ############################################# 추가
# import sys
# sys.path.append("yolov5")  # YOLOv5 폴더 경로 추가
# from yolov5.utils.loss import ComputeLoss  # YOLOv5 공식 코드에서 직접 가져오기
# from yolov5.models.yolo import Model  # YOLO 모델 로드
# from src.model_utils.basic_YOLO import get_yolov5
# import yaml

# def train_YOLO(img_dir, ann_dir, batch_size=8, num_epochs=5, lr=0.001, weight_decay=0.005, optimizer_name="sgd", scheduler_name="step", device="cpu", debug=False):
#     # 데이터 로더 
#     train_loader = get_loader("data/train_images/train", "data/train_labels/train", batch_size, mode="train", debug=debug)
#     val_loader = get_loader("data/train_images/val", "data/train_labels/val", batch_size, mode="val", debug=debug)

#     # 어노테이션 디렉토리를 기준으로 카테고리 매핑 가져오기
#     name_to_idx, idx_to_name = get_category_mapping(ann_dir)

#     # 클래스 개수는 카테고리 길이로 설정
#     num_classes = len(name_to_idx)

#     # YOLO 모델 정의
#     model = get_yolov5(model_path="yolov5s.pt", num_classes=num_classes).to(device)
#     # model = Model("yolov5/models/yolov5s.yaml")  # YOLOv5s 모델 YAML 사용

#     # 모델에서의 yaml 파일은 모델의 구조를 정의하는 설정 파일
#     model.nc = num_classes  # 클래스 수 설정
#     model.to(device)

#     # 옵티마이저, 스케쥴러, 로스 정의
#     optimizer = get_optimizer(optimizer_name, model, lr, weight_decay)
#     scheduler = get_scheduler(scheduler_name, optimizer, T_max=100)

#     compute_loss = ComputeLoss(model.model)

#     # 학습률 스케줄러 정의
#     scheduler = get_scheduler(scheduler_name, optimizer)

#     best_val_loss = float("inf")

#     # 학습 루프
#     for epoch in range(num_epochs):
#         print("학습 시작")
#         model.train()
#         total_loss = 0

#         train_bar = tqdm(train_loader, total=len(train_loader), desc=f"🟢 Training {epoch+1}/{num_epochs}")
#         # 학습 단계
#         for batch_i, (imgs, targets) in tqdm(enumerate(train_loader), total=len(train_loader), desc="Training"):
#             imgs = imgs.to(device)
#             targets = targets.to(device)

#             # 모델 학습
#             optimizer.zero_grad()
#             loss.backward()
#             torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
#             optimizer.step()
#             total_loss += loss.item()
#             train_bar.set_postfix(loss=loss.item())  # 진행 상태 표시
        
#         # 학습 후 스케줄러 업데이트
#         if scheduler_name == "plateau":
#             scheduler.step(total_loss)  # ReduceLROnPlateau는 loss를 인자로 받음
#         else:
#             scheduler.step()

#         # 2. 검증 단계
#         model.eval()
#         val_loss = 0
#         with torch.no_grad():
#             val_bar = tqdm(val_loader, total=len(val_loader), desc=f"🔵 Validation {epoch+1}/{num_epochs}")
#             for imgs, targets in val_bar:
#                 imgs = imgs.to(device)
#                 targets = targets.to(device)

#                 preds = model(imgs)
#                 loss, _ = compute_loss(preds, targets)
#                 val_loss += loss.item()
#                 val_bar.set_postfix(val_loss=loss.item())

#         print(f"Epoch {epoch+1} - Train Loss: {total_loss:.4f}, Val Loss: {val_loss:.4f}")
        
#         if scheduler_name == "plateau":
#             scheduler(val_loss)
#         else:
#             scheduler()

#         # ✅ 최적 모델 저장
#         if val_loss < best_val_loss:
#             best_val_loss = val_loss
#             print(f"검증 손실 개선됨.. 모델 저장 중... (Best Val Loss: {best_val_loss:.4f})")
#             save_model(model, epoch, best_val_loss)  # 모델 저장

# def save_model(model, epoch, val_loss):
#     """ 모델 가중치 저장 함수 """
#     save_dir = "models/weights"
#     os.makedirs(save_dir, exist_ok=True)
#     save_path = os.path.join(save_dir, f"yolov5_epoch_{epoch}_val_{val_loss:.4f}.pt")
#     torch.save(model.state_dict(), save_path)
#     print(f"모델 저장 완료: {save_path}")



## TODOLIST
## - 상대경로 입력해도 절대경로로 바꿀 수 있게
###################################################################################################################################################
# 모델 선택을 위한 실행 코드 예제
# python train.py --model_variant s  # YOLOv8s 사용
# python train.py --model_variant l  # YOLOv8l 사용
# 자동 최적화 실행 예제
# python train.py --model_variant n --batch_size 16 --num_epochs 10
# - lr과 weight_decay를 입력하지 않으면 자동 최적화 진행
# 최종 실행 예시 !!
# python train.py --model_variant s --batch_size 16 --num_epochs 20 --lr 0.01 --weight_decay 0.0001
###################################################################################################################################################
import os
import numpy as np
import torch
from tqdm import tqdm
import argparse
from ultralytics import YOLO
import matplotlib.pyplot as plt


def train_YOLO(img_dir: str,
               ann_dir: str,
               model_variant: str = 'n',
               batch_size: int = 8,
               num_epochs: int = 5,
               lr: float = 0.001,
               weight_decay: float = 0.0005,
               debug: bool = False
               ):
    """
    YOLO 모델을 학습하는 함수
    """

    # 입력값 검증
    # 매개변수 올바른 형태인지 확인
    valid_variants = ['n', 's', 'm', 'l']
    yaml_path = r"C:\Users\user\Desktop\PythonWorkspace\new_neo_project1\data\train_labels\data.yaml"
    assert model_variant in valid_variants, f"model_variant must be one of {valid_variants}"
    assert isinstance(img_dir, str), "img_dir must be a string"
    assert isinstance(ann_dir, str), "ann_dir must be a string"
    assert isinstance(yaml_path, str), "yaml_path must be a string"
    assert isinstance(batch_size, int) and batch_size > 0, "batch_size must be a positive integer"
    assert isinstance(num_epochs, int) and num_epochs > 0, "num_epochs must be a positive integer"
    assert isinstance(lr, float) and lr > 0, "lr must be a positive float"
    assert isinstance(weight_decay, float) and weight_decay >= 0, "weight_decay must be a non-negative float"
    assert isinstance(debug, bool), "debug must be a boolean"

    # YOLO 모델 로드
    model_path = f'yolov8{model_variant}.pt'
    model = YOLO(model_path)

    # 모델 학습
    results = model.train(
        data=yaml_path,          # YAML 파일 경로
        epochs=num_epochs,        # 학습 에폭 수
        batch=batch_size,         # 배치 크기
        lr0=lr if lr else None,  # 값이 없으면 YOLO가 자동으로 최적값 찾음
        weight_decay=weight_decay if weight_decay else None,  # 자동 최적화 가중치 감쇠
        verbose=not debug,        # 디버그 모드에 따른 출력 설정
        save_period=1,            # 모델 저장 주기
        project='runs/detect',     # 결과 저장 프로젝트 디렉토리
        name=f'yolov8{model_variant}_custom'  # 실행 이름 바꾸기 
    )

    return model, results
def main():
    parser = argparse.ArgumentParser(description='YOLO Model Training Script')
    parser.add_argument("--img_dir", type=str, required=False, default=r"C:\Users\nihao\Desktop\new_neo\new_neo_project1\data\train_labels\train")
    parser.add_argument("--ann_dir", type=str, required=False, default=r"C:\Users\nihao\Desktop\new_neo\new_neo_project1\data\train_labels\train")
    parser.add_argument("--yaml_path", type=str, required=False, default=r"C:\Users\nihao\Desktop\new_neo\new_neo_project1\data\train_labels\data.yaml")
    parser.add_argument('--model_variant', type=str, default='n', choices=['n', 's', 'm', 'l'], help='YOLO model variant')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.0005, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0005, help='Weight decay')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')

    args = parser.parse_args()

    model, results = train_YOLO(
        img_dir=args.img_dir,
        ann_dir=args.ann_dir,
        model_variant=args.model_variant,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        debug=args.debug
    )


def enable_weights_only_false():
    """
    PyTorch의 torch.load 함수의 기본 동작을 monkey-patch 하여 
    weights_only=False 옵션을 강제로 적용하는 함수.

    이 함수는 신뢰할 수 있는 소스의 YOLO 가중치를 로드할 때 
    Unpickling 오류를 방지하기 위해 사용됩니다.
    주의: 외부에서 받은 불확실한 가중치 파일에는 보안 위험이 있을 수 있습니다.
    """

    original_load = torch.load
    def custom_torch_load(*args, **kwargs):
        kwargs['weights_only'] = False
        return original_load(*args, **kwargs)
    torch.load = custom_torch_load
    print("[INFO] torch.load monkey-patched: weights_only=False")


if __name__ == "__main__":
    enable_weights_only_false()
    main()
