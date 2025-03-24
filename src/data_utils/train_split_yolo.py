################################################################################################
# 데이터 다운 -> data loader 실행 -> json modify 실행 -> coco to yolo -> train split yolo 순서
# 실행 코드 python train_split_yolo.py
# train_labels → train/val 분할 및 train.txt, val.txt 생성
# train, val.txt는 train과 val에 맞는 img 파일 path 정보가 들어있는 파일
# ###############################################################################################

import os
import shutil
from sklearn.model_selection import train_test_split

# 원본 라벨 경로
label_dir = "data/train_labels"

# 이미지 파일이 있는 곳
image_dir = "data/train_images"

# 분할된 라벨 저장할 경로
train_label_dir = "data/train_labels/train"
val_label_dir = "data/train_labels/val"

# train.txt / val.txt 저장 경로
train_txt_path = "data/train.txt"
val_txt_path = "data/val.txt"

# 폴더 생성
os.makedirs(train_label_dir, exist_ok=True)
os.makedirs(val_label_dir, exist_ok=True)

# 원본 라벨 리스트 수집
label_files = [f for f in os.listdir(label_dir) if f.endswith(".txt")]
image_files = [f.replace(".txt", ".png") for f in label_files]

# 분할
train_labels, val_labels = train_test_split(label_files, test_size=0.2, random_state=42)

# .txt 파일 초기화
open(train_txt_path, "w").close()
open(val_txt_path, "w").close()

# 복사 및 경로 저장
for label_file in train_labels:
    shutil.move(os.path.join(label_dir, label_file), os.path.join(train_label_dir, label_file))
    img_path = os.path.join(image_dir, label_file.replace(".txt", ".png"))
    with open(train_txt_path, "a") as f:
        f.write(f"{img_path}\n")

for label_file in val_labels:
    shutil.move(os.path.join(label_dir, label_file), os.path.join(val_label_dir, label_file))
    img_path = os.path.join(image_dir, label_file.replace(".txt", ".png"))
    with open(val_txt_path, "a") as f:
        f.write(f"{img_path}\n")

print("train_labels → train_labels/train & val 분할 완료")
print("train.txt / val.txt 파일 생성 완료")
