###################################################################################################################################################
# 모델 선택을 위한 실행 코드 예제
# python src/train_YOLO.py --img_dir data\train_labels\train --yaml_path data\train_labels\data.yaml --model_variant n
###################################################################################################################################################
import os
import numpy as np
import torch
from tqdm import tqdm
import argparse
from ultralytics import YOLO
import matplotlib.pyplot as plt
import glob


def train_YOLO(img_dir, yaml_path, model_variant='n', batch_size=8, num_epochs=100, lr=0.001, weight_decay=0.0005, 
               patience=100, device='cpu', optimizer='auto', seed=42, resume=False, debug=False):
    # 절대 경로로 변환
    if not os.path.isabs(img_dir):
        img_dir = os.path.abspath(img_dir)
    if not os.path.isabs(yaml_path):
        yaml_path = os.path.abspath(yaml_path)

    valid_variants = ['n', 's', 'm', 'l']

    if not isinstance(img_dir, str):
        raise TypeError("img_dir must be a string")
    if not isinstance(yaml_path, str):
        raise TypeError("yaml_path must be a string")
    if not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    if not isinstance(num_epochs, int) or num_epochs <= 0:
        raise ValueError("num_epochs must be a positive integer")
    if not isinstance(lr, float) or lr <= 0:
        raise ValueError("lr must be a positive float")
    if not isinstance(weight_decay, float) or weight_decay < 0:
        raise ValueError("weight_decay must be a non-negative float")
    if not isinstance(debug, bool):
        raise TypeError("debug must be a boolean")
    if model_variant not in valid_variants:
        raise ValueError(f"model_variant must be one of {valid_variants}")
    if resume is True:
        last_run_dir = sorted(glob.glob('runs/detect/yolov8*'), key=os.path.getmtime)[-1]
        resume = os.path.join(last_run_dir, 'weights', 'last.pt')
    elif isinstance(resume, str):
        resume = os.path.abspath(resume)

    model_path = f'yolov8{model_variant}.pt'
    model = YOLO(model_path)

    results = model.train(
        data=yaml_path,
        epochs=num_epochs,
        batch=batch_size,
        lr0=lr,
        weight_decay=weight_decay,
        device=device,
        optimizer=optimizer,
        seed=seed,
        resume=resume,
        verbose=not debug,
        project='runs/detect',
        name=f'yolov8{model_variant}_custom'
    )

    return model, results


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


def main():
    parser = argparse.ArgumentParser(description='YOLO Model Training Script')
    parser.add_argument("--img_dir", type=str, required=True, help="Training image directory")
    parser.add_argument("--yaml_path", type=str, required=True, help="YAML config file path")
    parser.add_argument('--model_variant', type=str, default='n', choices=['n', 's', 'm', 'l'], help='YOLO model variant')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.0005, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0005, help='Weight decay')
    parser.add_argument('--patience', type=int, default=100, help='Early stopping patience')
    parser.add_argument('--device', type=str, default='cpu', help='Device to train on (cpu or cuda)')
    parser.add_argument('--optimizer', type=str, default='auto', help='Optimizer (if supported)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--resume', action='store_true', help='Resume training from last checkpoint')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--force_load', action='store_true', help='Force loading with weights_only=False')

    args = parser.parse_args()

    if args.force_load:
        enable_weights_only_false()

    model, results = train_YOLO(
        img_dir=args.img_dir,
        yaml_path=args.yaml_path,
        model_variant=args.model_variant,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        device=args.device,
        optimizer=args.optimizer,
        seed=args.seed,
        resume=args.resume,
        debug=args.debug
    )


if __name__ == "__main__":
    main()
