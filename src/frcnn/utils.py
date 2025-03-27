# 표준 라이브러리
import os

# 서드파티 라이브러리
import numpy as np
import cv2
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


def compute_iou(box1, box2):
    """Intersection over Union (IoU) 계산"""
    x1, y1, x2, y2 = box1
    x1_pred, y1_pred, x2_pred, y2_pred = box2

    inter_x1 = max(x1, x1_pred)
    inter_y1 = max(y1, y1_pred)
    inter_x2 = min(x2, x2_pred)
    inter_y2 = min(y2, y2_pred)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2_pred - x1_pred) * (y2_pred - y1_pred)

    union_area = box1_area + box2_area - inter_area
    iou = inter_area / union_area if union_area > 0 else 0
    return iou

def compute_precision_recall(targets, predictions, iou_threshold=0.5):
    """Precision과 Recall 계산"""
    tp = 0
    fp = 0
    fn = 0

    # 대상 박스와 예측 박스를 비교하여 TP, FP, FN 계산
    for target, prediction in zip(targets, predictions):
        target_labels = target['labels']
        target_boxes = target['boxes']
        
        pred_labels = prediction['labels']
        pred_boxes = prediction['boxes']
        pred_scores = prediction['scores']

        for t_label, t_box in zip(target_labels, target_boxes):
            found_match = False
            for p_label, p_box, p_score in zip(pred_labels, pred_boxes, pred_scores):
                if compute_iou(t_box, p_box) >= iou_threshold and t_label == p_label:
                    tp += 1
                    found_match = True
                    break
            if not found_match:
                fn += 1

        fp += len(pred_labels) - tp  # False positives are the predictions without matches
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    return precision, recall, tp, fp, fn

def compute_ap(precision, recall):
    """Average Precision 계산 (Interpolated)"""
    # Interpolated AP를 구하려면 precision과 recall의 값들을 보간하여 계산합니다.
    recall = np.concatenate(([0], recall, [1]))
    precision = np.concatenate(([0], precision, [0]))
    
    for i in range(len(precision) - 2, -1, -1):
        precision[i] = max(precision[i], precision[i + 1])
    
    # AP는 Precision-Recall Curve 아래의 면적 (곡선의 누적합)으로 구합니다.
    ap = np.sum((recall[1:] - recall[:-1]) * precision[1:])
    
    return ap

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