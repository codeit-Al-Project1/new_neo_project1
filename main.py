import argparse
import os
import torch

# FRCNN
from src.frcnn.train import train as train_frcnn
from src.frcnn.test import test as test_frcnn
from src.frcnn.utils import visualization
from src.frcnn.make_csv import submission_csv

# YOLO
from src.yolo.train import train as train_yolo
from src.yolo.train import validate as val_yolo
from src.yolo.train import visualize as visualize_yolo
from src.yolo.test import predict_and_get_csv, enable_weights_only_false

"""
====================================================================================
🔹 Object Detection Main Entry (Faster R-CNN + YOLOv8 통합 스크립트)

이 스크립트는 FRCNN과 YOLOv8 객체 탐지 모델의 학습, 테스트, 검증을 통합 제공합니다.
`--model` 인자를 통해 사용할 모델을 선택하고, `--mode`로 동작 모드를 지정하세요.

------------------------------------------------------------------------------------
🔹 실행 예제

# ▶ [1] Faster R-CNN 학습
python main.py --model frcnn --mode train --img_dir data/train_images --json_path data/train_annots_modify --backbone resnet50 --batch_size 4 --epochs 30 --optimizer_name sgd --scheduler_name plateau --lr 0.001 --weight_decay 0.0005

# ▶ [2] Faster R-CNN 테스트 (시각화 및 CSV 저장)
python main.py --model frcnn --mode test --img_dir data/test_images --model_path models/frcnn_session_2/model_31.pth --threshold 0.5 --visualization --page_size 20 --page_lim 5

# ▶ [3] YOLOv8 학습
python main.py --model yolo --mode train --img_dir data/train_labels/train --yaml_path data/train_labels/data.yaml --model_variant n --batch_size 8 --epochs 100 --lr 0.001 --weight_decay 0.0005

# ▶ [4] YOLOv8 검증 (validation)
python main.py --model yolo --mode val --val_model_path runs/detect/yolov8n_custom/weights/best.pt

# ▶ [5] YOLOv8 테스트 (결과 이미지 저장 및 CSV 저장)
python main.py --model yolo --mode test --model_path runs/detect/yolov8n_custom/weights/best.pt --img_dir data/test_images --save_images --save_csv_path submission_yolo.csv --conf_threshold 0.5 --iou_threshold 0.7

------------------------------------------------------------------------------------
🔸 공통 옵션
--model               : 사용할 모델 선택 ['frcnn', 'yolo'] (필수)
--mode                : 실행 모드 선택 ['train', 'test', 'val'] (필수)
--img_dir             : 입력 이미지가 있는 디렉토리 (train/test 공통)
--model_path          : 사전 학습된 모델 경로 (.pth or .pt)
--device              : 사용할 디바이스 ['cuda', 'cpu'], 기본값: 자동 선택
--debug               : 디버그 출력 활성화
--force_load          : YOLO 가중치 pickle 오류 대응 (weights_only=False로 로딩 강제)

------------------------------------------------------------------------------------
🔸 FRCNN 전용 옵션
--json_path           : 어노테이션 JSON 디렉토리 (default: data/train_annots_modify)
--backbone            : 백본 모델 선택 ['resnet50', 'mobilenet_v3_large', 'resnext101']
--batch_size          : 학습 배치 크기
--epochs              : 학습 반복 횟수
--optimizer_name      : 옵티마이저 종류 ['sgd', 'adam', 'adamw', 'rmsprop']
--scheduler_name      : 러닝레이트 스케줄러 ['step', 'cosine', 'plateau', 'exponential']
--lr                  : 초기 학습률
--weight_decay        : L2 정규화 (weight decay)
--test_batch_size     : 테스트용 배치 크기 (default: 4)
--threshold           : confidence 임계값 (default: 0.5)
--visualization       : 시각화 이미지 저장 여부
--page_size           : 시각화 시 한 페이지당 이미지 수 (default: 20)
--page_lim            : 시각화 페이지 수 제한 (default: None, 전체 시각화)

------------------------------------------------------------------------------------
🔸 YOLO 전용 옵션
--yaml_path           : YOLO 학습 시 사용할 data.yaml 경로
--model_variant       : YOLOv8 크기 선택 ['n', 's', 'm', 'l']
--patience            : 조기 종료 patience (default: 100)
--optimizer           : YOLO 전용 옵티마이저 ['auto', 'SGD', 'Adam' 등]
--resume              : 학습 재시작 여부
--val_model_path      : 검증 시 사용할 .pt 파일 경로
--conf_threshold      : confidence threshold (default: 0.5)
--iou_threshold       : NMS IoU threshold (default: 0.7)
--save_images         : 예측 이미지 저장 여부 (YOLO test)
--save_csv_path       : YOLO 테스트 결과를 저장할 CSV 경로

====================================================================================
"""

def main():
    parser = argparse.ArgumentParser(description="Unified Object Detection Entry Point")

    # ───────────── 공통 인자 ─────────────
    parser.add_argument("--model", type=str, choices=["frcnn", "yolo"], required=True, help="사용할 모델 선택")
    parser.add_argument("--mode", type=str, choices=["train", "test", "val"], required=True, help="모드 선택: train/test/val")
    parser.add_argument("--img_dir", type=str, required=True, help="이미지 디렉토리 경로")
    parser.add_argument("--model_path", type=str, help="모델 경로")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="학습/추론 디바이스")
    parser.add_argument("--debug", action="store_true", help="디버그 모드")
    parser.add_argument("--force_load", action="store_true", help="YOLO 가중치 로딩 강제 옵션 (pickle 오류 대응)")

    # ───────────── FRCNN 인자 ─────────────
    # 학습
    parser.add_argument("--json_path", type=str, default="data/train_annots_modify", help="어노테이션 JSON 파일 경로")
    parser.add_argument("--backbone", type=str, choices=["resnet50", "mobilenet_v3_large", "resnext101"], help="백본 모델")
    parser.add_argument("--batch_size", type=int, default=16, help="학습 배치 사이즈")
    parser.add_argument("--epochs", type=int, default=5, help="에폭 수")
    parser.add_argument("--optimizer_name", type=str, choices=["sgd", "adam", "adamw", "rmsprop"], default="sgd", help="FRCNN 옵티마이저")
    parser.add_argument("--scheduler_name", type=str, choices=["step", "cosine", "plateau", "exponential"], default="plateau", help="FRCNN 스케줄러")
    parser.add_argument("--lr", type=float, default=0.001, help="학습률")
    parser.add_argument("--weight_decay", type=float, default=0.0005, help="L2 정규화")

    # 테스트
    parser.add_argument("--test_batch_size", type=int, default=4, help="테스트 배치 사이즈")
    parser.add_argument("--threshold", type=float, default=0.5, help="예측 임계값")
    parser.add_argument("--visualization", action="store_true", help="시각화 여부")
    parser.add_argument("--page_size", type=int, default=20, help="시각화 한 페이지당 이미지 수")
    parser.add_argument("--page_lim", type=int, default=None, help="시각화 페이지 수 제한")

    # ───────────── YOLO 인자 ─────────────
    parser.add_argument("--yaml_path", type=str, help="YOLO 학습용 데이터 설정 YAML 경로")
    parser.add_argument("--model_variant", type=str, default='n', choices=['n', 's', 'm', 'l'], help="YOLO 크기 설정")
    parser.add_argument("--patience", type=int, default=100, help="YOLO 학습 조기종료 patience")
    parser.add_argument("--optimizer", type=str, default='auto', help="YOLO 옵티마이저 (ultralytics 정의)")
    parser.add_argument("--resume", action="store_true", help="YOLO 학습 재시작 여부")

    parser.add_argument("--val_model_path", type=str, help="YOLO 검증할 모델 경로")
    parser.add_argument("--save_images", action="store_true", help="YOLO 예측 이미지 저장 여부")
    parser.add_argument("--save_csv_path", type=str, help="YOLO 예측 결과 CSV 저장 경로")
    parser.add_argument("--conf_threshold", type=float, default=0.5, help="YOLO confidence threshold")
    parser.add_argument("--iou_threshold", type=float, default=0.7, help="YOLO NMS IoU threshold")

    args = parser.parse_args()

    # monkey-patch if requested
    if args.force_load:
        enable_weights_only_false()

    # ───────────── 실행 분기 ─────────────
    if args.model == "frcnn":
        if args.mode == "train":
            print("[FRCNN] 학습 시작")
            train_frcnn(
                img_dir=args.img_dir,
                json_dir=args.json_path,
                backbone=args.backbone,
                batch_size=args.batch_size,
                num_epochs=args.epochs,
                optimizer_name=args.optimizer_name,
                scheduler_name=args.scheduler_name,
                lr=args.lr,
                weight_decay=args.weight_decay,
                device=args.device,
                debug=args.debug,
            )

        elif args.mode == "test":
            print("[FRCNN] 테스트 시작")
            results = test_frcnn(
                img_dir=args.img_dir,
                device=args.device,
                model_path=args.model_path,
                batch_size=args.test_batch_size,
                threshold=args.threshold,
                debug=args.debug,
            )
            if args.visualization:
                visualization(results, args.page_size, args.page_lim, args.debug)
                submission_csv(results, submission_file_path="./submission_frcnn.csv", debug=args.debug)

    elif args.model == "yolo":
        if args.mode == "train":
            print("[YOLO] 학습 시작")
            train_yolo(
                img_dir=args.img_dir,
                yaml_path=args.yaml_path,
                model_variant=args.model_variant,
                batch_size=args.batch_size,
                num_epochs=args.epochs,
                lr=args.lr,
                weight_decay=args.weight_decay,
                patience=args.patience,
                device=args.device,
                optimizer=args.optimizer,
                seed=42,
                resume=args.resume,
                debug=args.debug,
            )
        elif args.mode == "val":
            if not args.val_model_path:
                raise ValueError("YOLO 검증 모드에서는 --val_model_path 인자가 필요합니다.")
            save_dir = val_yolo(args.val_model_path)
            print(f"[YOLO] 검증 완료. 결과 저장 위치: {save_dir}")
            visualize_yolo(save_dir)
        elif args.mode == "test":
            print("[YOLO] 테스트 시작")
            predict_and_get_csv(
                model_path=args.model_path,
                image_dir=args.img_dir,
                conf_threshold=args.conf_threshold,
                iou_threshold=args.iou_threshold,
                save_csv_path=args.save_csv_path,
                device=args.device,
                verbose=args.debug,
                save_images=args.save_images,
            )

if __name__ == "__main__":
    main()