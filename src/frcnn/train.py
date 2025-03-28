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

# 서드파티 라이브러리 (외부 모듈)
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils

# 내부 모듈
from src.frcnn.utils import get_optimizer, get_scheduler, compute_iou, compute_ap
from src.data_utils.data_loader import get_loader
from src.utils import get_category_mapping
from src.model_utils.basic_frcnn import (
    get_new_session_folder,
    save_model,
    get_fast_rcnn_model,
)

# 텐서보드 객체 생성
writer = SummaryWriter("tensorboard_log_dir")


def train(img_dir: str,
          json_dir: str,
          backbone: str = "resnet50",
          batch_size: int = 8,
          num_epochs: int = 5,
          optimizer_name: str = "sgd", 
          scheduler_name: str = "plateau",
          lr: float = 0.001,
          weight_decay: float = 0.0005,
          iou_threshold: float = 0.5,
          conf_threshold: float = 0.5,
          device: str = "cpu",
          debug: bool = False):
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
    assert backbone in ["resnet50", "mobilenet_v3_large", "resnext101", "efficientnet_b3"], "backbone must be one of ['resnet50', 'mobilenet_v3_large', 'resnext101', 'efficientnet_b3']"
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
    model = get_fast_rcnn_model(num_classes, backbone=backbone, focal=True).to(device)
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

            loss_weights = {
                "loss_classifier": 3.0,
                "loss_box_reg": 1.0,
                "loss_objectness": 1.0,
                "loss_rpn_box_reg": 1.0
            }
            # 예: box_reg 줄이고 classifier 가중치 높이기
            # loss_weights["loss_classifier"] = 1.5
            # loss_weights["loss_box_reg"] = 0.5

            # 가중 손실 총합 계산
            loss_dict = model(images, targets)
            losses = sum(loss_weights[k] * loss_dict[k] for k in loss_dict)


            losses.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # SGD를 사용하는 모델의 그래디언트 값이 너무 커지는 것을 방지하기 위해 그래디언트 클리핑
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
        model.eval()
        ap_list, precision_list, recall_list = [], [], []
        with torch.no_grad():

            progress_bar = tqdm(val_loader, total=len(val_loader), desc='Validation', dynamic_ncols=True)

            for images, targets in progress_bar:
                images = [img.to(device) for img in images]
                outputs = model(images) # boxes, labels, scores

                filtered_outputs = []
                for output in outputs:
                    keep = output["scores"] > conf_threshold  # 특정 임계값 이상인 것만 선택
                    filtered_outputs.append({
                        "boxes": output["boxes"][keep],
                        "labels": output["labels"][keep],
                        "scores": output["scores"][keep]
                    })

                for i, output in enumerate(filtered_outputs):  
                    pred_boxes = output['boxes'].cpu().numpy()  # 예측 박스
                    pred_scores = output['scores'].cpu().numpy()  # 예측 신뢰도
                    pred_labels = output['labels'].cpu().numpy()  # 예측 클래스

                    gt_boxes = targets[i]['boxes'].cpu().numpy()  # 정답 박스
                    gt_labels = targets[i]['labels'].cpu().numpy()  # 정답 클래스
                    
                    all_ap, all_precision, all_recall = [], [], []

                    # 클래스별 AP, Precision, Recall 계산
                    for cls in np.unique(np.concatenate((pred_labels, gt_labels))):
                        pred_mask = pred_labels == cls
                        gt_mask = gt_labels == cls

                        pred_cls_boxes = pred_boxes[pred_mask]
                        pred_cls_scores = pred_scores[pred_mask]
                        gt_cls_boxes = gt_boxes[gt_mask]

                        # Confidence Score 순으로 정렬
                        sorted_indices = np.argsort(-pred_cls_scores)
                        pred_cls_boxes = pred_cls_boxes[sorted_indices]
                        pred_cls_scores = pred_cls_scores[sorted_indices]

                        tp = np.zeros(len(pred_cls_boxes))
                        fp = np.zeros(len(pred_cls_boxes))
                        matched = np.zeros(len(gt_cls_boxes))

                        for pred_idx, pred_box in enumerate(pred_cls_boxes):
                            best_iou = 0
                            best_gt_idx = -1

                            for gt_idx, gt_box in enumerate(gt_cls_boxes):
                                iou = compute_iou(pred_box, gt_box)
                                if iou > best_iou:
                                    best_iou = iou
                                    best_gt_idx = gt_idx

                            if best_iou > iou_threshold and best_gt_idx != -1 and matched[best_gt_idx] == 0:
                                tp[pred_idx] = 1  # TP (True Positive)
                                matched[best_gt_idx] = 1
                            else:
                                fp[pred_idx] = 1  # FP (False Positive)

                        # Precision, Recall 계산
                        tp_cumsum = np.cumsum(tp)
                        fp_cumsum = np.cumsum(fp)
                        recall = tp_cumsum / len(gt_cls_boxes) if len(gt_cls_boxes) > 0 else np.zeros(len(tp_cumsum))
                        precision = tp_cumsum / (tp_cumsum + fp_cumsum)

                        # AP, Precision, Recall 저장
                        ap = compute_ap(precision, recall)
                        all_ap.append(ap)
                        all_precision.append(precision[-1] if len(precision) > 0 else 0)
                        all_recall.append(recall[-1] if len(recall) > 0 else 0)

                    # 한 이미지에서 모든 클래스에 대한 평균값
                    image_ap = np.mean(all_ap) if len(all_ap) > 0 else 0
                    image_precision = np.mean(all_precision) if len(all_precision) > 0 else 0
                    image_recall = np.mean(all_recall) if len(all_recall) > 0 else 0
                    
                    ap_list.append(image_ap)
                    precision_list.append(image_precision)
                    recall_list.append(image_recall)

        mAP = np.mean(ap_list)  # 전체 이미지에서 AP 평균
        mean_precision = np.mean(precision_list)
        mean_recall = np.mean(recall_list)
        

        # 텐서보드에 기록
        writer.add_scalar("mAP", mAP, epoch)
        writer.add_scalar("Precision", mean_precision, epoch)
        writer.add_scalar("Recall", mean_recall, epoch)

        progress_bar.set_postfix(Precision=mean_precision, Recall=mean_recall, mAP=mAP)

        print(f"Validation Complete - mAP: {mAP:.4f}, Mean Precision: {mean_precision:.4f}, Mean Recall: {mean_recall:.4f}")

        # 모델 저장
        if mAP > best_map_score:
            best_map_score = mAP
            save_model(model, session_folder, lr=lr, epoch=num_epochs, batch_size=batch_size, optimizer=optimizer_name, scheduler=scheduler_name, weight_decay=weight_decay)
            print(f"Model saved with mAP score: {best_map_score:.4f}")

        # 텐서보드 학습률 기록
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar("Learning Rate", current_lr, epoch)
        
        # 학습률 스케줄러 업데이트
        if scheduler_name == "plateau":
            scheduler.step(mAP) # ReduceLROnPlateau의 경우 mode='max'로 설정 (성능이 좋아지면 학습률 감소)
        else:
            scheduler.step()

    # 텐서보드 종료        
    writer.close()
    print(f"Training complete. Best mAP: {best_map_score:.4f}")
