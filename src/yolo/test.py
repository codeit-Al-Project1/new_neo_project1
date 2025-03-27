"""
====================================================================================
[ 실행 방법 및 설명 ]

이 스크립트는 학습된 YOLO 모델을 불러와 테스트 이미지 디렉토리에 대한 
객체 탐지를 실행하고, 결과를 submission.csv 형식으로 저장합니다. 
필요 시 이미지 결과 파일도 함께 저장 가능합니다.

■ 터미널 실행 예시:
-----------------------------------------------
python src/test_YOLO.py --model_path runs/detect/yolov8n_custom/weights --image_dir ./data/test_images --conf_threshold 0.5 --iou_threshold 0.7 --save_images --verbose
-----------------------------------------------

■ 각 옵션 설명:
--model_path          : 학습된 YOLO 가중치 경로 (필수)
--image_dir           : 테스트 이미지 디렉토리 (필수)
--conf_threshold      : Confidence threshold (기본값=0.5)
--iou_threshold       : NMS IoU threshold (기본값=0.7)
--save_csv_path       : CSV 저장 경로 (생략 시 자동 지정)
--device              : 강제 디바이스 선택 (예: 'cpu' 또는 'cuda')
--verbose             : YOLO 내부 출력 로그 활성화 여부
--save_images         : 예측 결과 이미지 저장 여부
--force_load          : torch.load monkey-patch 활성화 (Unpickling 오류 방지)

■ 출력:
- 지정한 경로에 submission.csv 파일 생성
- (옵션 시) 예측 이미지 시각화 파일 저장
====================================================================================
"""


from ultralytics import YOLO
import torch
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import argparse
from src.utils import get_category_mapping

ANN_DIR = "data/train_annots_modify"
idx_to_id = get_category_mapping(ann_dir=ANN_DIR, debug=True, return_types=['idx_to_id'])


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


def predict_and_get_csv(model_path, image_dir, conf_threshold=0.5, iou_threshold=0.7,
                                save_csv_path=None, device=None, verbose=False, save_images=False):
    """
    YOLO 모델을 테스트 이미지에 적용하고 예측 결과를 주어진 형식의 CSV로 저장하는 함수.

    Args:
        model_path (str): 학습된 YOLO 모델 가중치 경로 (예: 'best.pt')
        image_dir (str): 테스트 이미지 디렉토리 경로
        conf_threshold (float): confidence threshold (기본 0.5)
        iou_threshold (float): NMS IOU threshold (기본 0.7)
        save_csv_path (str or None): 결과 CSV 저장 경로 (None이면 자동 지정)
        device (str or None): 강제 디바이스 선택 ('cpu' 또는 'cuda')
        verbose (bool): YOLO 내부 출력 여부
        save_images (bool): 예측 결과 이미지 파일을 저장할지 여부 (기본 False)

    Returns:
        pd.DataFrame: 예측 결과 DataFrame (annotation_id, image_id, category_id, bbox_x, bbox_y, bbox_w, bbox_h, score)
    """

    # 인자 타입 검사
    if not isinstance(model_path, str):
        raise TypeError(f"[ERROR] model_path는 str 타입이어야 합니다. 현재 타입: {type(model_path)}")
    if not isinstance(image_dir, str):
        raise TypeError(f"[ERROR] image_dir는 str 타입이어야 합니다. 현재 타입: {type(image_dir)}")
    if not isinstance(conf_threshold, (float, int)) or not (0.0 <= conf_threshold <= 1.0):
        raise ValueError(f"[ERROR] conf_threshold는 0~1 사이의 float/int 여야 합니다. 현재 값: {conf_threshold}")
    if not isinstance(iou_threshold, (float, int)) or not (0.0 <= iou_threshold <= 1.0):
        raise ValueError(f"[ERROR] iou_threshold는 0~1 사이의 float/int 여야 합니다. 현재 값: {iou_threshold}")
    if save_csv_path is not None and not isinstance(save_csv_path, str):
        raise TypeError(f"[ERROR] save_csv_path는 str 또는 None 이어야 합니다. 현재 타입: {type(save_csv_path)}")
    if device is not None and not isinstance(device, str):
        raise TypeError(f"[ERROR] device는 str (cpu/cuda) 또는 None 이어야 합니다. 현재 타입: {type(device)}")
    if not isinstance(verbose, bool):
        raise TypeError(f"[ERROR] verbose는 bool 타입이어야 합니다. 현재 타입: {type(verbose)}")
    if not isinstance(save_images, bool):
        raise TypeError(f"[ERROR] save_images는 bool 타입이어야 합니다. 현재 타입: {type(verbose)}")

    model = YOLO(model_path)
    results = model.predict(
        source=image_dir, 
        save=save_images, 
        conf=conf_threshold,
        iou=iou_threshold,
        device=device,
        verbose=verbose
        )
    if not isinstance(results, list):
        raise TypeError(f"[ERROR] model.predict() 반환 타입 오류: list가 아님 ({type(results)})")
    if not results or not hasattr(results[0], 'boxes'):
        raise RuntimeError("[ERROR] 결과가 비어있거나 boxes 속성이 없습니다.")
    
    # 처음 5개 결과만 시각화
    for res in results[:5]:
        image = cv2.imread(res.path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(image)

        boxes = res.boxes.xywh.cpu().numpy()
        class_ids = res.boxes.cls.cpu().numpy().astype(int)
        scores = res.boxes.conf.cpu().numpy()

        for cls, box, score in zip(class_ids, boxes, scores):
            x1 = box[0] - (box[2] / 2)
            y1 = box[1] - (box[3] / 2)
            w = box[2]
            h = box[3]

            rect = patches.Rectangle((x1, y1), w, h, linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            ax.text(x1, y1 - 5, f"{res.names[cls]} ({score:.2f})", color="white", fontsize=8, backgroundcolor="red")

        plt.axis('off')
        plt.show()
    
    # csv파일 패스 처리리
    if save_csv_path is None:
        save_dir = getattr(results[0], 'save_dir', None) or os.path.dirname(model_path) or os.getcwd()
        save_csv_path = os.path.join(save_dir, "submission.csv")

    submission = []
    annotation_id = 1
    for res in results:
        if not hasattr(res, 'boxes'):
            continue

        image_name = os.path.basename(res.path)
        image_id = int(os.path.splitext(image_name)[0])
        boxes = res.boxes.xywh.cpu().numpy()        # [x_center, y_center, width, height]
        class_ids = res.boxes.cls.cpu().numpy().astype(int)
        scores = res.boxes.conf.cpu().numpy()

        # 각 값의 타입 검증
        if not (isinstance(boxes, np.ndarray) and isinstance(class_ids, np.ndarray) and isinstance(scores, np.ndarray)):
            print(f"[WARNING] 잘못된 형식 감지 (image: {image_name}). 해당 항목 건너뜁니다.")
            continue

        for cls, box, score in zip(class_ids, boxes, scores):
            submission.append({
                "annotation_id": annotation_id,
                "image_id": image_id,
                "category_id": idx_to_id[cls],
                # x_center, y_center -> top-left x, top-left y
                "bbox_x": box[0] - (box[2] / 2),
                "bbox_y": box[1] - (box[3] / 2),
                "bbox_w": box[2],
                "bbox_h": box[3],
                "score": score
            })
            annotation_id += 1

    df = pd.DataFrame(submission)
    df.to_csv(save_csv_path, index=False)
    print(f"[INFO] 결과가 저장되었습니다: {save_csv_path}")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="YOLO 가중치 파일 경로")
    parser.add_argument("--image_dir", type=str, required=True, help="테스트 이미지 디렉토리")
    parser.add_argument("--conf_threshold", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--save_csv_path", type=str, default=None, help="CSV 저장 경로 (생략 시 자동 경로)")
    parser.add_argument("--device", type=str, default=None, help="inference device (cpu 또는 cuda)")
    parser.add_argument("--iou_threshold", type=float, default=0.7, help="NMS IOU threshold")
    parser.add_argument("--verbose", action="store_true", help="YOLO 내부 출력 활성화 여부")
    parser.add_argument("--save_images", action="store_true", help="예측 시 result 이미지를 저장할지 여부")
    parser.add_argument("--force_load", action="store_true", help="weights_only=False 강제 적용")

    args = parser.parse_args()

    if args.force_load:
        enable_weights_only_false()

    predict_and_get_csv(
        model_path=args.model_path,
        image_dir=args.image_dir,
        conf_threshold=args.conf_threshold,
        iou_threshold=args.iou_threshold,
        save_csv_path=args.save_csv_path,
        device=args.device,
        verbose=args.verbose,
        save_images=args.save_images
    )