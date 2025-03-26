# 실행방법 :
# python src/train_YOLO.py --yaml_dir "yaml파일 절대경로" --batch_size 8 --num_epochs 5 --iou_threshold 0.5 --imgsz 640 --device "cpu" --debug False
# 
import torch
from ultralytics.nn.tasks import DetectionModel
from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.modules.block import Bottleneck, C3, SPPF
from torch.nn import Sequential
from ultralytics import YOLO
import os
import argparse
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# 운영체제별 한글 폰트 설정 ==========================================================
import platform
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

os_name = platform.system()

if os_name == "Darwin":  # macOS
    font_path = "/Library/Fonts/Arial Unicode.ttf"  # 예시 경로
elif os_name == "Windows":  # Windows
    font_path = "C:/Windows/Fonts/malgun.ttf"  # 예시 경로
elif os_name == "Linux":  # Linux
    font_path = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"  # 예시 경로
else:
    font_path = None

if font_path:
    font_prop = fm.FontProperties(fname=font_path)
    plt.rcParams["font.family"] = font_prop.get_name()
    print(f"폰트 설정 완료: {font_prop.get_name()}")
else:
    print("지원하지 않는 운영체제입니다.")
# ===============================================================================


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


def train_yolo(img_dir, yaml_path, model_variant='n', batch_size=8, num_epochs=100, lr=0.001, weight_decay=0.0005, 
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
    if not isinstance(debug, bool):  # verbose 인자 확인
        raise TypeError("verbose must be a boolean")
    if model_variant not in valid_variants:
        raise ValueError(f"model_variant must be one of {valid_variants}")

    # resume 처리 (True일 경우 자동으로 마지막 체크포인트 검색)
    if resume is True:
        last_run_dir = sorted(glob.glob('runs/detect/yolov8*'), key=os.path.getmtime)[-1]  # 최근 실행 디렉토리 탐색
        resume = os.path.join(last_run_dir, 'weights', 'last.pt')  # 해당 경로의 last.pt로 resume
    elif isinstance(resume, str):
        resume = os.path.abspath(resume)  # 경로 문자열이면 절대 경로 변환

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
        verbose=debug,
        project='runs/detect',
        name=f'yolov8{model_variant}_custom'
    )

    return model, results


def visualize(result_dir):
    '''
    학습된 모델 결과 폴더를 입력받아서 시각화된 이미지를 출력하는 함수
    '''
    # 폴더경로에 대한 에러처리
    if result_dir is None:
        user_input = input("폴더 경로를 입력하세요: ")
        result_dir = user_input

    if not os.path.exists(result_dir):
        raise ValueError(f"폴더 경로 '{result_dir}'를 찾을 수 없습니다.")

    if os.path.isdir(result_dir):
        if not os.listdir(result_dir): # 폴더가 비어있으면
            raise ValueError(f"폴더 경로 '{result_dir}'가 비어있습니다.")

    # 모델 폴더 내의 이미지 파일
    image_files = glob.glob(os.path.join(result_dir, '*.jpg')) + glob.glob(os.path.join(result_dir, '*.png')) # jpg와 png 모두 찾기
    
    # confusion_matrix 이미지 제외
    image_files = [f for f in image_files if "confusion_matrix" not in f]  

    # 이미지 출력
    for image_file in image_files:
        img = mpimg.imread(image_file)
        plt.imshow(img)
        plt.axis('off') 
        plt.grid(False)
        plt.title(os.path.basename(image_file))
        plt.show()
        

def val_yolo(model_path):
    if not os.path.isabs(model_path):
        model_path = os.path.abspath(model_path)

    model = YOLO(model_path)
    val_results = model.val()
    print("[INFO] 검증 완료.")

    return val_results.save_dir


def main():
    parser = argparse.ArgumentParser(description='YOLO Model Training Script')
    parser.add_argument('--mode', type=str, choices=['train', 'val'], required=True, help='train 또는 val 선택')
    parser.add_argument("--img_dir", type=str, help="학습 이미지 경로 (train 모드일 때 필요)")
    parser.add_argument("--yaml_path", type=str, help="YAML 경로 (train 모드일 때 필요)")
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
        if not args.img_dir or not args.yaml_path:
            raise ValueError("train 모드에서는 --img_dir 및 --yaml_path가 필요합니다.")
        train_yolo(
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
        save_dir = val_yolo(args.val_model_path, verbose=True)
        visualize(save_dir)



# TODO: 시각화 글자 겹침문제해결 (나중에 여유되면 해결하기로 미루기로 함 3/26 아침회의)
# TODO: 경로받아서 시각화 

"""
python src/train_YOLO.py --imgsz 32 --num_epochs 1 --yaml_dir /Users/apple/Documents/codeit-Al-Project1/new_neo_project1/data/train_labels/data.yaml
python -m src.train_YOLO --imgsz 32 --num_epochs 1 --yaml_dir /Users/apple/Documents/codeit-Al-Project1/new_neo_project1/data/train_labels/data.yaml
"""
