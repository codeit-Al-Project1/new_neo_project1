################################################################################################
# 데이터 다운 -> data loader 실행 -> json modify 실행 -> coco to yolo 순서
# 실행 코드
# python coco_to_yolo.py --json_folder data/train_annots_modify --output_dir data/train_labels
# category_id x_center y_center width height + 좌표 정규화
# 이렇게 바꿔놓아야 YOLO에서 돌아간다고 합니다
# ###############################################################################################

import json
import os
import argparse
from src.data_utils.data_loader import get_category_mapping

def convert_coco_to_yolo(json_file, output_dir):
    """
    COCO JSON 형식의 어노테이션 데이터를 YOLO 형식으로 변환하는 함수.

    Args:
        json_file (str): COCO 형식의 JSON 어노테이션 파일 경로
        output_dir (str): YOLO 형식의 라벨 파일이 저장될 디렉토리

    Returns:
        None
    """
    os.makedirs(output_dir, exist_ok=True)
    ANN_DIR = "data/train_annots_modify"
    name_to_idx, _ = get_category_mapping(ANN_DIR)

    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for img in data["images"]:
        img_id = img["id"]
        img_w, img_h = img["width"], img["height"]
        label_path = os.path.join(output_dir, f"{img['file_name'].replace('.png', '.txt')}")

        with open(label_path, "w", encoding="utf-8") as f:
            for ann in data["annotations"]:
                if ann["image_id"] == img_id:
                    x, y, w, h = ann["bbox"]
                    x_center, y_center = (x + w / 2) / img_w, (y + h / 2) / img_h
                    w, h = w / img_w, h / img_h

                    for category in data['categories']:
                        if ann["category_id"] == category["id"]:
                            category_id = name_to_idx[category['name']]
                            f.write(f"{category_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")

def process_all_json(json_folder, output_dir):
    """
    폴더 내 모든 COCO JSON 파일을 YOLO 형식으로 변환하는 함수.

    Args:
        json_folder (str): 변환할 COCO JSON 파일이 저장된 폴더
        output_dir (str): YOLO 라벨을 저장할 폴더
    """
    if not os.path.exists(json_folder):
        print(f"JSON 폴더가 존재하지 않습니다: {json_folder}")
        return

    json_files = [f for f in os.listdir(json_folder) if f.endswith('.json')]

    if len(json_files) == 0:
        print("변환할 JSON 파일이 없습니다.")
        return

    os.makedirs(output_dir, exist_ok=True)

    # 변환 진행
    for i, json_file in enumerate(json_files, start=1):
        json_path = os.path.join(json_folder, json_file)
        convert_coco_to_yolo(json_path, output_dir)


    print("모든 JSON 파일 변환 완료")



################ YOLO 모델 학습을 위한 data.yaml 파일을 생성하는 함수. ################
# txt 파일생성  (이미지 경로랑 어떤이미지인지 파일이름들이 들어가있음)
from sklearn.model_selection import train_test_split
import os
def create_txt_file(txt_name, image_folder='data/train_images', output_folder='data/val_labels_YOLO', val_ratio=0.2, seed=42):

    # 이미지 파일 목록 가져오기
    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.png'))]

    # 학습 및 검증 데이터셋 분할
    train_files, val_files = train_test_split(image_files, test_size=val_ratio, random_state=42)

    # 출력 폴더 생성
    os.makedirs(output_folder, exist_ok=True)

    # 학습 데이터셋 파일 경로 저장
    with open(f"data/val_labels_YOLO/{txt_name}.txt", "w") as f:
        for file in train_files:
            f.write(file + '\n')

    # 검증 데이터셋 파일 경로 저장
    with open(f"data/val_labels_YOLO/{txt_name}.txt", "w") as f:
        for file in val_files:
            f.write(file + '\n')



import yaml
from src.data_utils.data_loader import get_category_mapping

def make_yaml_file(YOLO_dataset_name='yolo_dataset_1', output_dir=os.path.join("data"), ):
    """
    YOLO 모델 학습을 위한 data.yaml 파일을 생성하는 함수.
    args:
        YOLO_dataset_name (str): 생성할 데이터셋 이름 (yaml 파일이름으로 만들기 위한 정보)
        output_path (str): 생성될 YAML 파일 경로 (기본값: dataset.yaml)
    """
    # YAML 파일 경로 설정
    if not os.path.exists(output_dir):  # val_labels_YOLO 폴더가 없는 경우 생성
        os.makedirs(output_dir)
    yaml_file_dir = os.path.join(output_dir, f"{YOLO_dataset_name}.yaml")
    

    # 클래스와 클래스 수 설정
    _, idx_to_name = get_category_mapping("data/train_annots_modify")  
    class_names = [name for name in idx_to_name.values()]

    # train:과 val:이 같은 폴더를 가리키므로, YOLO가 _val.txt, _train.txt을 참조해 자동으로 이미지를 찾을 수 있습니다. 
    data = {
        "train": f"train.txt",  
        "val": f"val.txt", 
        "train_labels":f"train_labels_YOLO/{YOLO_dataset_name}.txt",  
        "val_labels":f"train_labels_YOLO/{YOLO_dataset_name}.txt",   
        "nc": len(class_names),
        "names": {i: name for i, name in enumerate(class_names)}
    }


    try:
        with open(output_dir, "w") as f:
            yaml.dump(data, f, default_flow_style=False)
        print(f"YAML 파일이 '{output_dir}'에 성공적으로 생성되었습니다.")
    except Exception as e:
        print(f"YAML 파일 생성 중 오류가 발생했습니다: {e}")

    print(f"✅ {yaml_file_dir} 파일이 생성되었습니다.")

###########################################################################################



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert COCO JSON annotations to YOLO format")
    parser.add_argument("--json_folder", type=str, default="data/train_annots_modify", help="Folder containing COCO JSON files")
    parser.add_argument("--output_dir", type=str, default="data/train_labels", help="Output directory for YOLO label files")
    args = parser.parse_args()

    process_all_json(args.json_folder, args.output_dir)