#####################################################################################################
# [모델 학습 실행 방법 예제]
# python src/train.py --mode train --img_dir data/train_labels/train --yaml_path data/train_labels/data.yaml --model_variant n --batch_size 8 --num_epochs 100 --device cpu
#
# [모델 검증 실행 방법 예제]
# python src/train.py --mode val --val_model_path runs/detect/yolov8n_custom/weights/best.pt
#
# 각 인자 설명:
# --mode             : 실행 모드 (train 또는 val)
# --img_dir          : 학습 이미지 경로 (train 모드에서 필요)
# --yaml_path        : 데이터 yaml 경로 (train 모드에서 필요)
# --val_model_path   : 검증 시 사용할 best.pt 경로 (val 모드에서 필요)
# --model_variant    : 사용할 YOLOv8 모델 크기 (n, s, m, l 중 선택)
# --batch_size       : 배치 크기
# --num_epochs       : 학습 에폭 수
# --device           : 학습 및 검증 디바이스 ('cpu' 또는 'cuda')
# --force_load       : pickle 오류 발생 시 강제 weights_only=False 로 로딩
#####################################################################################################

import os
import numpy as np
import torch
from tqdm import tqdm
import argparse
from ultralytics import YOLO
import matplotlib.pyplot as plt
import matplotlib.image as mpimg   # <-- [추가] 이미지 시각화를 위한 모듈
import glob
import platform
import matplotlib.font_manager as fm

# 운영체제별 한글 폰트 설정
os_name = platform.system()
if os_name == "Darwin":
    font_path = "/Library/Fonts/Arial Unicode.ttf"
elif os_name == "Windows":
    font_path = "C:/Windows/Fonts/malgun.ttf"
elif os_name == "Linux":
    font_path = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"
else:
    font_path = None

if font_path:
    font_prop = fm.FontProperties(fname=font_path)
    plt.rcParams["font.family"] = font_prop.get_name()
    print(f"폰트 설정 완료: {font_prop.get_name()}")
else:
    print("지원하지 않는 운영체제입니다.")


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


def train(img_dir, yaml_path, model_variant='n', batch_size=8, num_epochs=100, lr=0.001, weight_decay=0.0005, 
               patience=100, device='cpu', optimizer='auto', seed=42, resume=False, debug=False):
    # 절대 경로로 변환
    if not os.path.isabs(img_dir):
        img_dir = os.path.abspath(img_dir)
    if not os.path.isabs(yaml_path):
        yaml_path = os.path.abspath(yaml_path)

    valid_variants = ['n', 's', 'm', 'l']

    if not img_dir or not os.path.exists(img_dir):
        raise ValueError(f"[ERROR] img_dir 경로가 존재하지 않습니다: {img_dir}")
    if not yaml_path or not os.path.exists(yaml_path):
        raise ValueError(f"[ERROR] yaml_path 경로가 존재하지 않습니다: {yaml_path}")
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

    model.train(
        data=yaml_path,
        epochs=num_epochs,
        batch=batch_size,
        lr0=lr,
        weight_decay=weight_decay,
        patience=patience,
        device=device,
        optimizer=optimizer,
        seed=seed,
        resume=resume,
        verbose=debug,
        project='runs/detect',
        name=f'yolov8{model_variant}_custom'
    )


def validate(model_path):
    # 절대 경로 변환 (안전하게)
    if not os.path.isabs(model_path):
        model_path = os.path.abspath(model_path)
        
    model = YOLO(model_path)
    val_results = model.val()
    print(f"[INFO] 검증 완료: {model_path}")
    return val_results.save_dir


def visualize(result_dir):
    if not result_dir or not os.path.exists(result_dir):
        raise ValueError(f"시각화 경로가 존재하지 않습니다: {result_dir}")

    image_files = glob.glob(os.path.join(result_dir, '*.jpg')) + glob.glob(os.path.join(result_dir, '*.png'))
    image_files = [f for f in image_files if "confusion_matrix" not in f]

    for image_file in image_files:
        img = mpimg.imread(image_file)
        plt.imshow(img)
        plt.axis('off')
        plt.title(os.path.basename(image_file))
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='YOLO Model Training Script')
    parser.add_argument('--mode', type=str, choices=['train', 'val'], required=True, help='실행 모드 (train or val)')
    parser.add_argument("--img_dir", type=str, help="Training image directory")
    parser.add_argument("--yaml_path", type=str, help="YAML config file path")
    parser.add_argument('--val_model_path', type=str, help='검증 시 사용할 best.pt 경로')
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
    
    if args.mode == 'train':
        if not args.img_dir:
            raise ValueError("[ERROR] 학습 모드에서는 --img_dir 인자가 필요합니다.")
        if not args.yaml_path:
            raise ValueError("[ERROR] 학습 모드에서는 --yaml_path 인자가 필요합니다.")
        train(
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
    elif args.mode == 'val':
        if not args.val_model_path:
            raise ValueError("[ERROR] 검증 모드에서는 --val_model_path 인자를 지정해야 합니다.")
        save_dir = validate(args.val_model_path)
        visualize(save_dir)

if __name__ == "__main__":
    main()