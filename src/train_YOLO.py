import torch
from ultralytics.nn.tasks import DetectionModel
from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.modules.block import Bottleneck, C3, SPPF
from torch.nn import Sequential
from ultralytics import YOLO
import os
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


# 운영체제별 한글 폰트 설정 #########################################################
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

################################################################################
# PyTorch 2.6 이후는 반드시 안전 글로벌 등록!
torch.serialization.add_safe_globals([
    DetectionModel, 
    Conv, 
    Bottleneck, 
    C3, 
    SPPF, 
    Sequential
])

# TODO: 

def train(yaml_dir, batch_size=8, num_epochs=5, iou_threshold=0.5, device="cpu", debug=False):
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
        imgsz=640,
        batch=batch_size,
        patience=10,
        save=True,  
        verbose = debug,   
    )

    # 검증 수행
    metrics = model.val()

    # 검증 결과 출력
    if iou_threshold == 0.5:
        map_score = metrics.box.map50  
    elif iou_threshold == 0.75:
        map_score = metrics.box.map75  
    elif iou_threshold == 0.95:
        map_score = metrics.box.map95
    else:
        map_score = metrics.box.map
    precision = metrics.box.precision.mean()
    recall = metrics.box.recall.mean()

    print(f"mAP: {map_score:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")


    # 훈련중 시각화 
    train_folders = glob.glob(os.path.join("runs/detect", 'train*'))
    latest_folder = max(train_folders, key=os.path.getmtime)           # 가장 최근에 수정된 폴더 찾기
    image_files = glob.glob(os.path.join(latest_folder, '*.jpg'))

    # 이미지 시각화


if __name__ == "__main__":
    # 데이터셋 설정 파일 경로(절대경로)
    yaml_dir = "/Users/apple/Documents/codeit-Al-Project1/new_neo_project1/data/train_labels/data.yaml"

    # 학습 수행
    train(yaml_dir, batch_size=8, num_epochs=1, iou_threshold=0.5, device="cpu", debug=False)


    # 훈련중 시각화 
    train_folders = glob.glob(os.path.join("runs/detect", 'train*'))
    latest_folder = max(train_folders, key=os.path.getmtime)           # 가장 최근에 수정된 폴더 찾기
    image_files = glob.glob(os.path.join(latest_folder, '*.jpg')) + glob.glob(os.path.join(latest_folder, '*.png')) # jpg와 png 모두 찾기

    # 이미지 시각화
    for image_file in image_files:
        img = mpimg.imread(image_file)
        plt.imshow(img)
        plt.axis('off') 
        plt.grid(False)
        plt.title(os.path.basename(image_file))
        plt.show()