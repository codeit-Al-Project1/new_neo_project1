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

# PyTorch 2.6 이후는 반드시 안전 글로벌 등록!
torch.serialization.add_safe_globals([
    DetectionModel, 
    Conv, 
    Bottleneck, 
    C3, 
    SPPF, 
    Sequential
])

def train(yaml_dir, batch_size=8, num_epochs=5, iou_threshold=0.5, imgsz=640, device="cpu", debug=False):
    '''
    YOLO 모델을 학습하는 함수
    Args:
    yaml_dir (str): 데이터셋 설정이 저장된 YAML 파일 경로(**절대경로**)
    iou_threshold (float): IoU 임계값 (0.5, 0.75, 0.95 지원, 기본값 0.5)
    batch_size (int): 미니배치 크기 (기본값 8)
    num_epochs (int): 학습 에폭 수 (기본값 5)
    '''

    model = YOLO('yolov8n.pt').to(device) 

    model.train(
        data=yaml_dir,
        epochs=num_epochs,
        imgsz=imgsz, 
        batch=batch_size,
        patience=10,
        save=True,  
        verbose = debug,
    ).to(device)

    # 검증 수행
    model.val()


def visualize(model_path=None):
    '''
    학습된 모델 결과 폴더를 입력받아서 시각화된 이미지를 출력하는 함수
    '''
    # 폴더경로에 대한 에러처리
    if model_path is None:
        user_input = input("폴더 경로를 입력하세요: ")
        model_path = user_input

    if not os.path.exists(model_path):
        raise ValueError(f"폴더 경로 '{model_path}'를 찾을 수 없습니다.")

    if os.path.isdir(model_path):
        if not os.listdir(model_path): # 폴더가 비어있으면
            raise ValueError(f"폴더 경로 '{model_path}'가 비어있습니다.")

    # 모델 폴더 내의 이미지 파일
    image_files = glob.glob(os.path.join(model_path, '*.jpg')) + glob.glob(os.path.join(model_path, '*.png')) # jpg와 png 모두 찾기
    
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


if __name__ == "__main__":
    # 데이터셋 설정 파일 경로(절대경로)
    parser = argparse.ArgumentParser(description="YOLO 모델 학습 스크립트")
    parser.add_argument("--yaml_dir", type=str, required=True, help="데이터셋 설정 YAML 파일 (절대)경로")
    parser.add_argument("--batch_size", type=int, default=8, help="미니배치 크기")
    parser.add_argument("--num_epochs", type=int, default=5, help="학습 에폭 수")
    parser.add_argument("--iou_threshold", type=float, default=0.5, help="IoU 임계값")
    parser.add_argument("--imgsz", type=int, default=640, help="이미지 크기")
    parser.add_argument("--device", type=str, default="cpu", help="학습 장치 (cpu 또는 cuda)")
    parser.add_argument("--debug", type=bool, default=False, help="디버깅 모드")
    parser.add_argument("--model_path", type=str, default=None, help="학습된 모델 이름")

    args = parser.parse_args()

    train(
        yaml_dir=args.yaml_dir,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        iou_threshold=args.iou_threshold,
        device=args.device,
        debug=args.debug,
        imgsz=args.imgsz
    )

    visualize(args.model_path)


# TODO: 시각화 글자 겹침문제해결 (나중에 여유되면 해결하기로 미루기로 함 3/26 아침회의)
# TODO: 경로받아서 시각화 

"""
python src/train_YOLO.py --imgsz 32 --num_epochs 1 --yaml_dir /Users/apple/Documents/codeit-Al-Project1/new_neo_project1/data/train_labels/data.yaml
python -m src.train_YOLO --imgsz 32 --num_epochs 1 --yaml_dir /Users/apple/Documents/codeit-Al-Project1/new_neo_project1/data/train_labels/data.yaml
"""