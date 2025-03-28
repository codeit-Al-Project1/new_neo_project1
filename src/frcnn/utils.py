# 표준 라이브러리
import os

# 서드파티 라이브러리
import numpy as np
import cv2
from tqdm import tqdm
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torchvision.transforms.v2 as T
from matplotlib import pyplot as plt

# 옵티마이저 생성 함수
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

# 스케줄러 생성 함수
def get_scheduler(name, optimizer, step_size=5, gamma=0.1, T_max=50):
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
        "plateau": lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=gamma, patience=step_size),
        "exponential": lr_scheduler.ExponentialLR(optimizer, gamma=gamma),
    }

    if name.lower() not in schedulers:
        raise ValueError(f"지원되지 않는 스케줄러: {name}. 사용 가능한 옵션: {list(schedulers.keys())}")

    return schedulers[name.lower()]

# IoU 계산 함수
def compute_iou(box1, box2):
    """ 두 개의 바운딩 박스(box1, box2) 간 IoU(Intersection over Union) 계산 """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # 교집합 영역 계산
    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    # 각 박스의 면적
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # IoU = 교집합 영역 / 합집합 영역
    union = box1_area + box2_area - intersection
    return intersection / union if union > 0 else 0

def compute_ap(precision, recall):
    """ Precision-Recall 곡선을 기반으로 AP(Average Precision) 계산 """
    recall = np.concatenate(([0.], recall, [1.]))
    precision = np.concatenate(([0.], precision, [0.]))

    # Precision 값이 단조 감소하도록 정리
    for i in range(len(precision) - 1, 0, -1):
        precision[i - 1] = max(precision[i - 1], precision[i])

    # Recall 구간별 차이 계산 및 AP 적분
    indices = np.where(recall[1:] != recall[:-1])[0]
    ap = np.sum((recall[indices + 1] - recall[indices]) * precision[indices + 1])
    return ap


def evaluate_faster_rcnn(model, val_loader, device, iou_threshold=0.1):
    """ Faster R-CNN의 mAP, mean Precision, mean Recall 계산 """
    model.eval()
    ap_list, precision_list, recall_list = [], [], []

    with torch.no_grad():
        progress_bar = tqdm(val_loader, total=len(val_loader), desc='Validation', dynamic_ncols=True)
        
        for images, targets in progress_bar:
            images = [img.to(device) for img in images]
            outputs = model(images)

            for i, output in enumerate(outputs):  
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
    
    return mAP, mean_precision, mean_recall


# 시각화 함수
def draw_bbox(ax, box, text, color):
    """
    - ax: matplotlib Axes 객체
    - box: 바운딩 박스 좌표 (x_min, y_min, x_max, y_max)
    - text: 바운딩 박스 위에 표시할 텍스트
    - color: 바운딩 박스와 텍스트의 색상
    """
    ax.add_patch(
        plt.Rectangle(
            xy=(box[0], box[1]),
            width=box[2] - box[0],
            height=box[3] - box[1],
            fill=False,
            edgecolor=color,
            linewidth=2,
        )
    )
    # 텍스트 위치가 이미지 밖으로 나가지 않도록 보정
    text_x = max(box[0], 5)
    text_y = max(box[1] - 10, 5)

    ax.annotate(
        text=text,
        xy=(text_x, text_y),
        color='blue',
        weight="bold",
        fontsize=8,
        bbox=dict(facecolor='white', edgecolor='none', alpha=0.7)
    )

# f1 스코어 계산 함수 
def f1_score(precision, recall):
    score = 2 * ((precision * recall)/ (precision + recall))

    return score


def visualization(results:list, page_size:int=20, page_lim:int=None, debug:bool=True):
    """
    모델 예측값 시각화
    산발적인 출력 대신, data/results/frcnn폴더에 산출된 결과를 토대로 그려진 이미지 페이지를 저장.
    - results: test 모듈에서 반환환된 예측 결과 list
    - idx_to_name: index와 약재의 이름이 매핑된 결과 dict
    - page_size: 한 페이지에 들어갈 이미지의 수
    - page_lim: 페이지 제한 (예: total_page = 40 일 경우, 모든 페이지를 받는 대신, 제한을 둬서 페이지를 샘플링)
    - debug: 디버그 여부.
    """

    total_num = len(results)
    total_pages = np.ceil(total_num / page_size).astype(int)

    if page_lim is not None:
        if page_lim <= 0:
            raise ValueError("page_lim은 양의 정수여야 합니다.")
        total_pages = min(total_pages, page_lim)

    print(f"전체 {total_num}개의 이미지, {total_pages} 페이지로 분할 저장합니다.")

    save_dir = './data/results/frcnn'
    os.makedirs(save_dir, exist_ok=True)

    for page in range(total_pages):    
        start_idx = page * page_size
        end_idx = min(start_idx + page_size, total_num)

        print(f"[페이지 {page + 1} / {total_pages}] | {start_idx} - {end_idx}번째 이미지 표시")

        num_images = end_idx - start_idx
        col_img = min(4, num_images)
        row_img = (num_images + col_img - 1) // col_img if num_images > 0 else 1
        figsize = (5 * col_img, 5 * row_img)

        print(f"페이지 당 {row_img} 행, {col_img} 열 형태의 이미지 플롯")

        fig, ax = plt.subplots(row_img, col_img, figsize=figsize)

        if row_img == 1: 
            ax = np.expand_dims(ax, axis=0)
        if col_img == 1:
            ax = np.expand_dims(ax, axis=1)

        for i in range(start_idx, end_idx):
            file_name = results[i]['file_name']
            drug_id = results[i]['category_id']
            drug_names = results[i]['category_name']
            boxes = results[i]['boxes']
            scores = results[i]['scores']
            bbox_num = len(boxes)
            path = os.path.join('./data/test_images', file_name)
            

            if debug:
                print(f"[{i + 1}] Visualize Image: {file_name}, DRUG ID: {drug_id}, BBox Num: {bbox_num}")
                print(f"Scores: {scores}")
                print(f"Drug Names: {drug_names}")

            if not os.path.exists(path):
                print(f"[Error] 이미지 경로를 찾을 수 없습니다: {path}")
                continue 

            image = cv2.imread(path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            ax_idx = i - start_idx
            ax_row = ax_idx // col_img
            ax_col = ax_idx % col_img

            ax[ax_row, ax_col].imshow(image)
            assert len(boxes) == len(scores), "Bounding Box와 점수의 개수가 맞지 않습니다."
            for name, box, score in zip(drug_names, boxes, scores):
                draw_bbox(ax[ax_row, ax_col], box, f'{name}: {score:.2f}', color='red')

            ax[ax_row, ax_col].axis("off")
            ax[ax_row, ax_col].set_title(f"{file_name}")

        page_file_name = f"page_{page + 1}_{start_idx + 1}_{end_idx}.png"
        page_save_path = os.path.join(save_dir, page_file_name)

        plt.tight_layout()
        plt.savefig(page_save_path, bbox_inches='tight')
        plt.close()

        print(f"페이지 {page + 1} 이미지가 저장되었습니다: {page_save_path}")

    print(f"총 {total_pages}페이지 저장 완료!")

def resize_bbox_to_original(boxes, orig_size, resized_size=(640, 640)):
    if isinstance(boxes, list):
        boxes = np.array(boxes)
    W_orig, H_orig = orig_size
    W_resized, H_resized = resized_size

    # Scaling factors
    scale_x = W_orig / W_resized
    scale_y = H_orig / H_resized

    # 원래 사이즈로 변환
    if isinstance(boxes, np.ndarray):
        bbox_orig = boxes.copy()  # numpy 배열의 경우 copy() 사용
    else:
        bbox_orig = boxes.clone()  # PyTorch Tensor의 경우 clone() 사용
    bbox_orig[:, [0, 2]] *= scale_x  # x 좌표
    bbox_orig[:, [1, 3]] *= scale_y  # y 좌표

    return bbox_orig.tolist()
# ============================한글 폰트=================================================
# Colab 환경에서 실행 중인지 
import platform
import warnings

def is_colab():
    try:
        # google.colab이 존재하는지 확인하여 Colab 환경을 판별
        import google.colab
        return True
    except ImportError:
        return False
    
def is_lightning_ai():
    # lightning.ai 환경 여부를 확인 (예시: 환경 변수나 설치된 라이브러리 체크)
    try:
        import lightning
        return True
    except ImportError:
        return False

# Colab 환경일 경우와 로컬 환경일 경우 폰트 설정 분리
if is_colab():
    # Colab 환경에서 사용할 한글 폰트 설정 (예: NanumGothic)
    plt.rc('font', family='NanumBarunGothic')
    plt.rcParams['axes.unicode_minus'] = False
    print("Colab 환경에서 실행 중입니다.")
elif is_lightning_ai():
    # lightning.ai 환경에서 사용할 폰트 설정 (예: NanumGothic)
    plt.rc('font', family='NanumBarunGothic')
    plt.rcParams['axes.unicode_minus'] = False
    print("lightning.ai 환경에서 실행 중입니다.")
else:
    # 로컬 환경에서 사용할 폰트 설정
    plt.rc('font', family='Malgun Gothic')  # Windows의 경우 'Malgun Gothic' 사용
    plt.rcParams['axes.unicode_minus'] = False
    print("로컬 환경에서 실행 중입니다.")

# ▶ Warnings 제거
warnings.filterwarnings('ignore')
# ====================================================================================