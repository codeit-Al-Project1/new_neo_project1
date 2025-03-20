from matplotlib import pyplot as plt
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch
from torchvision.ops import box_iou
from sklearn.metrics import average_precision_score


def get_optimizer(name, model, lr=1e-3, weight_decay=0):
    """
    주어진 이름(name)에 해당하는 옵티마이저를 생성하여 반환합니다.

    :param name: 사용할 옵티마이저 이름 (예: 'adam', 'sgd', 'adamw')
    :param model: 학습할 모델의 파라미터
    :param lr: 학습률
    :param weight_decay: 가중치 감쇠 (L2 정규화)
    :return: 선택된 옵티마이저 인스턴스
    """
    optimizers = {
        "sgd": optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay),
        "adam": optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay),
        "adamw": optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay),
        "rmsprop": optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay),
    }

    if name.lower() not in optimizers:
        raise ValueError(f"지원되지 않는 옵티마이저: {name}. 사용 가능한 옵션: {list(optimizers.keys())}")

    return optimizers[name.lower()]


import torch.optim.lr_scheduler as lr_scheduler
def get_scheduler(name, optimizer, step_size=10, gamma=0.1, T_max=50):
    """
    주어진 이름(name)에 해당하는 스케줄러를 생성하여 반환합니다.

    :param name: 사용할 스케줄러 이름 (예: 'step', 'cosine', 'plateau')
    :param optimizer: 옵티마이저 인스턴스
    :param step_size: StepLR에서 사용하는 스텝 크기
    :param gamma: StepLR, ExponentialLR에서 사용되는 감쇠 계수
    :param T_max: CosineAnnealingLR에서 사용하는 주기 길이
    :return: 선택된 스케줄러 인스턴스
    """
    schedulers = {
        "step": lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma),
        "cosine": lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max),
        "plateau": lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=gamma, patience=step_size),
        "exponential": lr_scheduler.ExponentialLR(optimizer, gamma=gamma),
    }

    if name.lower() not in schedulers:
        raise ValueError(f"지원되지 않는 스케줄러: {name}. 사용 가능한 옵션: {list(schedulers.keys())}")

    return schedulers[name.lower()]


# 평가 함수 정의 (IoU 기반)
def calculate_iou(pred_boxes, true_boxes):
    """
    IoU (Intersection over Union) 계산
    pred_boxes: 예측된 바운딩 박스
    true_boxes: 실제 바운딩 박스
    """
    return box_iou(pred_boxes, true_boxes)

import numpy as np
from sklearn.metrics import precision_recall_curve


def calculate_ap(pred_boxes, true_boxes, pred_scores, iou_threshold=0.5):
    """
    주어진 예측 박스와 실제 박스를 비교하여 Average Precision (AP)를 계산합니다.
    pred_boxes: 예측된 바운딩 박스 (xyxy 포맷)
    true_boxes: 실제 바운딩 박스 (xyxy 포맷)
    pred_scores: 예측된 점수 (confidence score)
    iou_threshold: IoU 임계값 (기본값: 0.5)
    """
    # 각 예측에 대해 IoU를 계산하고, 적절한 precision, recall을 구합니다.
    ious = calculate_iou(pred_boxes, true_boxes)
    
    # 각 예측에 대해 IoU가 threshold를 넘는지 확인
    tp = (ious > iou_threshold).sum(dim=1).bool()  # True Positive (IoU > threshold)
    fp = (~tp).bool()  # False Positive (IoU < threshold)
    
    # Precision과 Recall 계산
    tp_cumsum = torch.cumsum(tp.int(), dim=0)
    fp_cumsum = torch.cumsum(fp.int(), dim=0)
    
    precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)  # 작은 값 더해주기
    recall = tp_cumsum / (len(true_boxes) + 1e-6)
    
    # AP 계산 (Precision-Recall 곡선 아래의 면적을 구합니다.)
    ap = np.trapz(precision.cpu().numpy(), recall.cpu().numpy())  # Precision-Recall 곡선 아래 면적 계산
    return ap

def evaluate_predictions_direct(predictions, targets, num_classes):
    """
    예측과 실제값을 비교하여 mAP를 계산하는 함수입니다.
    predictions: 모델 예측 결과 (box, score, label 포함)
    targets: 실제 값 (box, label 포함)
    num_classes: 클래스 수
    """
    total_ap_score = 0
    num_predictions = 0

    # 각 클래스별로 AP 계산
    for i in range(1, num_classes):  # 클래스 0은 배경이므로 제외
        pred_boxes = predictions['boxes'][predictions['labels'] == i]
        pred_scores = predictions['scores'][predictions['labels'] == i]
        true_boxes = targets['boxes'][targets['labels'] == i]
        
        if len(pred_boxes) == 0 or len(true_boxes) == 0:
            continue
        
        ap = calculate_ap(pred_boxes, true_boxes, pred_scores)
        total_ap_score += ap
        num_predictions += 1
    
    avg_ap_score = total_ap_score / num_predictions if num_predictions > 0 else 0
    return avg_ap_score


# def draw_bbox(ax, box, text, color):
#     """
#     - ax: matplotlib Axes 객체
#     - box: 바운딩 박스 좌표 (x_min, y_min, x_max, y_max)
#     - text: 바운딩 박스 위에 표시할 텍스트
#     - color: 바운딩 박스와 텍스트의 색상
#     """
#     ax.add_patch(
#         plt.Rectangle(
#             xy=(box[0], box[1]),
#             width=box[2] - box[0],
#             height=box[3] - box[1],
#             fill=False,
#             edgecolor=color,
#             linewidth=2,
#         )
#     )
#     ax.annotate(
#         text=text,
#         xy=(box[0] - 5, box[1] - 5),
#         color=color,
#         weight="bold",
#         fontsize=13,
#     )