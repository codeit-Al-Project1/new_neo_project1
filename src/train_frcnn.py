"""
이 스크립트는 PyTorch를 사용하여 Faster R-CNN 객체 탐지 모델을 학습합니다.
데이터를 로드하고, 모델을 학습하며, 성능을 평가하고, 가장 좋은 성능을 보인 모델을 저장하는 기능을 포함합니다.

주요 단계:
1. 데이터셋을 로드하고 어노테이션을 전처리
2. 주어진 하이퍼파라미터로 모델 학습
3. 검증을 수행하고 mAP(mean Average Precision) 평가
4. 검증 성능이 가장 좋은 모델을 저장

필수 라이브러리:
- numpy
- torch
- tqdm
- src.utils, src.data_utils, src.model_utils의 커스텀 모듈
"""
# 외부 모듈듈
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils

# 내부 모듈듈
from src.utils import get_optimizer
from src.utils import get_scheduler
from src.utils import compute_iou, compute_precision_recall, compute_ap
from src.data_utils.data_loader import get_loader
from src.data_utils.data_loader import get_category_mapping
from src.model_utils.basic_frcnn import get_new_session_folder, save_model
from src.model_utils.basic_frcnn import get_fast_rcnn_model

# 텐서보드 객체 생성
writer = SummaryWriter("tensorboard_log_dir")


def train(img_dir: str, json_dir: str, backbone: str = "resnet50", batch_size: int = 8, num_epochs: int = 5, optimizer_name: str = "sgd", 
          scheduler_name: str = "plateau", lr: float = 0.001, weight_decay: float = 0.0005, 
          device: str = "cpu", debug: bool = False):
    """
    Faster R-CNN 모델을 학습하는 함수
    
    파라미터:
    - img_dir (str): 학습 이미지가 저장된 디렉토리 경로
    - json_dir (str): 어노테이션 JSON 파일이 저장된 디렉토리 경로
    - batch_size (int): 미니배치 크기 (기본값: 16)
    - num_epochs (int): 학습 에폭 수 (기본값: 5)
    - optimizer_name (str): 옵티마이저 종류 (기본값: 'sgd')
    - scheduler_name (str): 스케줄러 종류 (기본값: 'plateau')
    - lr (float): 학습률 (기본값: 0.001)
    - weight_decay (float): 가중치 감쇠 (기본값: 0.0005)
    - device (str): 학습을 수행할 디바이스 ('cpu' 또는 'cuda')
    - debug (bool): 디버깅 모드 활성화 여부
    """
    
    # 입력값 검증
    assert isinstance(img_dir, str), "img_dir must be a string"
    assert isinstance(json_dir, str), "json_dir must be a string"
    assert backbone in ["resnet50", "mobilenet_v3_large", "resnext101"], "backbone must be one of ['resnet50', 'mobilenet_v3_large', 'resnext101']"
    assert isinstance(batch_size, int) and batch_size > 0, "batch_size must be a positive integer"
    assert isinstance(num_epochs, int) and num_epochs > 0, "num_epochs must be a positive integer"
    assert isinstance(optimizer_name, str), "optimizer_name must be a string"
    assert isinstance(scheduler_name, str), "scheduler_name must be a string"
    assert isinstance(lr, float) and lr > 0, "lr must be a positive float"
    assert isinstance(weight_decay, float) and weight_decay >= 0, "weight_decay must be a non-negative float"
    assert isinstance(device, str), "device must be a string"
    assert isinstance(debug, bool), "debug must be a boolean"

    # 데이터 로드 및 클래스 매핑
    name_to_idx, idx_to_name = get_category_mapping(json_dir, add_more=True, debug=debug, return_types=['name_to_idx', 'idx_to_name'])
    num_classes = len(name_to_idx)
    train_loader, val_loader = get_loader(img_dir, json_dir, batch_size, mode="train", val_ratio=0.2, bbox_format="XYXY", debug=debug)

    # 모델 및 학습 설정
    model = get_fast_rcnn_model(num_classes, backbone=backbone).to(device)
    optimizer = get_optimizer(optimizer_name, model, lr, weight_decay)
    scheduler = get_scheduler(scheduler_name, optimizer, step_size=5, gamma=0.1, T_max=100) # T_max는 CosineAnnealingLR에서만 사용

    # 새로운 세션 폴더 생성
    session_folder = get_new_session_folder(save_dir="./models", session_prefix="frcnn_session_")

    # 최고 성능 모델 저장을 위한 변수
    best_map_score = 0
    
    # 학습 루프
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        epoch_loss_details = {}

        progress_bar = tqdm(train_loader, total=len(train_loader), desc="Train", dynamic_ncols=True)

        # images 포맷 맞추기
        for images, targets in progress_bar:
            images = [img.to(device) for img in images] 
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())


            losses.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += losses.item()

            for k, v in loss_dict.items():
                if k not in epoch_loss_details:
                    epoch_loss_details[k] = 0
                epoch_loss_details[k] += v.item()

            avg_loss_details = ", ".join([f"{k}: {v / len(train_loader):.4f}" for k, v in epoch_loss_details.items()])
            progress_bar.set_postfix(Avg_Loss=avg_loss_details)
        
        # 텐서보드 기록 추가
        writer.add_scalar("Loss/Total", total_loss, epoch)
        for k, v in epoch_loss_details.items():
            writer.add_scalar(f"Loss/{k}", v / len(train_loader), epoch)


        print(f"Epoch {epoch+1} Complete - Total Loss: {total_loss:.4f}, Avg Loss Per Component: {avg_loss_details}")

        # 검증
        with torch.no_grad():
            model.eval()
            precision_list = []
            recall_list = []
            ap_list = []

            progress_bar = tqdm(val_loader, total=len(val_loader), desc='Validation', dynamic_ncols=True)
            for images, targets in progress_bar:
                images = [img.to(device) for img in images]
                outputs = model(images) # boxes, labels, scores

                precision, recall, tp, fp, fn = compute_precision_recall(targets, outputs)
                precision_list.append(precision)
                recall_list.append(recall)

                ap = compute_ap(precision_list, recall_list)
                ap_list.append(ap)

                # 텐서보드에 기록
                writer.add_scalar("ap", ap, epoch)
                writer.add_scalar("Precision", precision, epoch)
                writer.add_scalar("Recall", recall, epoch)

                progress_bar.set_postfix(Precision=precision, Recall=recall, AP=ap)
            
            mean_precision = np.mean(precision_list)
            mean_recall = np.mean(recall_list)
            map_score = np.mean(ap_list)

        print(f"Validation Complete - mAP: {map_score:.4f}, Mean Precision: {mean_precision:.4f}, Mean Recall: {mean_recall:.4f}")

        # 모델 저장
        if map_score > best_map_score:
            best_map_score = map_score
            save_model(model, session_folder, lr=lr, epoch=num_epochs, batch_size=batch_size, optimizer=optimizer_name, scheduler=scheduler_name, weight_decay=weight_decay)
            print(f"Model saved with mAP score: {best_map_score:.4f}")

        # 텐서보드 학습률 기록
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar("Learning Rate", current_lr, epoch)
        
        # 학습률 스케줄러 업데이트
        if scheduler_name == "plateau":
            scheduler.step(map_score) # ReduceLROnPlateau의 경우 mode='max'로 설정 (성능이 좋아지면 학습률 감소)
        else:
            scheduler.step()

    # 텐서보드 종료        
    writer.close()
    print(f"Training complete. Best mAP: {best_map_score:.4f}")
