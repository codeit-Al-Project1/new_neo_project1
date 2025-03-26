# ================================================
# COCO JSON -> YOLO TXT 변환 및 데이터셋 분리 스크립트
# 실행 방법:
# python -m src.data_utils.coco_to_yolo \
#   --json_folder data/train_annots_modify \
#   --output_dir data/train_labels \
#   --image_dir data/train_images \
#   --test_size 0.2 \
#   --debug
#
# 인자 설명:
# --json_folder   : 변환할 COCO JSON 파일이 들어 있는 폴더 (상대경로 가능)
# --output_dir    : YOLO TXT 라벨이 저장될 출력 폴더
# --image_dir     : 학습 이미지가 들어 있는 경로
# --test_size     : 검증 데이터셋 분할 비율 (기본 0.2)
# --debug         : 디버깅 메시지 출력 (True 시 진행상황 및 파일 개수 출력)
# ================================================

import json
import os
import argparse
import shutil
import yaml
from tqdm import tqdm
from src.data_utils.data_loader import get_category_mapping
from sklearn.model_selection import train_test_split

ANN_DIR = "data/train_annots_modify"
name_to_idx, idx_to_name = get_category_mapping(ann_dir=ANN_DIR, debug=True, add_more=False, return_types=['name_to_idx', 'idx_to_name'])

# COCO JSON -> YOLO TXT 변환 함수
def convert_json_to_txt(json_file, output_dir):
    """
    COCO JSON 형식의 어노테이션 데이터를 YOLO 형식 텍스트 파일로 변환하는 함수.

    Args:
        json_file (str): 변환할 JSON 파일 경로
        output_dir (str): 출력 YOLO 라벨 디렉토리 경로

    동작:
    - 각 이미지 ID별로 관련 어노테이션을 찾아서
    - YOLO 형식 (class_id x_center y_center width height) 으로 변환
    - 좌표는 이미지 크기로 정규화
    - background(0) 라벨은 -1 처리로 제거
    - 디버깅 시 각 변환 완료 파일 이름을 출력
    """
    if not isinstance(json_file, str):
        raise TypeError(f"json_file 인자는 str이어야 합니다. (받은 타입: {type(json_file)})")
    if not isinstance(output_dir, str):
        raise TypeError(f"output_dir 인자는 str이어야 합니다. (받은 타입: {type(output_dir)})")

    os.makedirs(output_dir, exist_ok=True)

    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 각 이미지에 대해
    for img in tqdm(data["images"], desc=f"Converting {os.path.basename(json_file)}", leave=False, dynamic_ncols=True):
        img_id = img["id"]
        img_w, img_h = img["width"], img["height"]
        # .png -> .txt
        label_path = os.path.join(output_dir, f"{img['file_name'].replace('.png', '.txt')}")

        # 해당 이미지의 annotation을 찾아 YOLO format으로 저장
        with open(label_path, "w", encoding="utf-8") as f:
            for ann in data["annotations"]:
                if ann["image_id"] == img_id:
                    x, y, w, h = ann["bbox"]
                    x_center, y_center = (x + w / 2) / img_w, (y + h / 2) / img_h
                    w, h = w / img_w, h / img_h

                    # category_id 매칭 및 YOLO 형식 라벨 작성
                    for category in data['categories']:
                        if ann["category_id"] == category["id"]:
                            category_id = name_to_idx[category['name']]
                            if category_id >= 0:
                                f.write(f"{category_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")


# 폴더 내 모든 JSON을 변환
def process_all_json(json_folder, output_dir, debug=False):
    """
    폴더 내 모든 COCO JSON 파일을 YOLO TXT 형식으로 변환하는 함수.

    Args:
        json_folder (str): COCO JSON 파일들이 들어있는 폴더 경로
        output_dir (str): 변환된 YOLO TXT 파일을 저장할 경로
        debug (bool): 디버깅 모드 활성화 여부. True 시 중간 과정 및 파일 경로 출력

    동작:
    - json_folder 내의 모든 .json 파일을 순회
    - 각 JSON 파일을 YOLO TXT 형식으로 변환
    - 디버깅 모드에서는 변환 시작 및 전체 진행 상태 출력
    """
    if not isinstance(json_folder, str):
        raise TypeError(f"json_folder는 str 타입이어야 합니다. (현재: {type(json_folder)})")
    if not isinstance(output_dir, str):
        raise TypeError(f"output_dir는 str 타입이어야 합니다. (현재: {type(output_dir)})")
    if not isinstance(debug, bool):
        raise TypeError(f"debug는 bool 타입이어야 합니다. (현재: {type(debug)})")
    if not os.path.exists(json_folder):
        print(f"JSON 폴더가 존재하지 않습니다: {json_folder}")
        return
    
    # output_dir 폴더가 없으면 생성 (있으면 무시)
    os.makedirs(output_dir, exist_ok=True)

    json_files = [f for f in os.listdir(json_folder) if f.endswith('.json')]

    if len(json_files) == 0:
        print("변환할 JSON 파일이 없습니다.")
        return

    os.makedirs(output_dir, exist_ok=True)

    if debug:
        print(f"[DEBUG] 변환 대상 JSON 파일 개수: {len(json_files)}")

    # 각 JSON 파일 변환 호출
    for json_file in tqdm(json_files, desc="전체 JSON 변환 진행", leave=False, dynamic_ncols=True):
        json_path = os.path.join(json_folder, json_file)
        convert_json_to_txt(json_path, output_dir)

    if debug:
        converted_files = [f for f in os.listdir(output_dir) if f.endswith('.txt')]
        print(f"[DEBUG] 변환 완료 - 총 {len(converted_files)}개 YOLO TXT 파일 생성됨.")
        
    print("모든 JSON 파일 변환 완료")



# 라벨과 이미지 학습-검증 분할 및 복사 함수
def split_labels_and_images(label_dir, image_dir, output_train, output_val, test_size=0.2, random_state=42, debug=False):
    """
    YOLO 라벨(.txt)과 이미지(.png)를 학습(train)과 검증(val) 세트로 분리 및 복사하는 함수.

    Args:
        label_dir (str): 라벨 파일이 위치한 폴더 경로
        image_dir (str): 이미지 파일이 위치한 폴더 경로
        output_train (str): 학습용 라벨 및 이미지 저장 경로
        output_val (str): 검증용 라벨 및 이미지 저장 경로
        test_size (float): 검증 세트 비율 (default=0.2)
        random_state (int): 랜덤 시드 값
        debug (bool): True로 설정 시 진행 정보 및 요약 출력

    동작:
    - 라벨 목록 가져오기 및 stratify 정보 준비
    - stratify 기반 train/val 분할 (실패 시 랜덤 분할)
    - 각 라벨과 이미지 파일을 각각 학습/검증 폴더로 이동 및 복사
    - tqdm으로 복사 진행 상황 표시
    - debug 모드 시 분할 정보 및 파일 개수 출력
    """

    # 인자 타입 검증
    for var, name in zip([label_dir, image_dir, output_train, output_val], ["label_dir", "image_dir", "output_train", "output_val"]):
        if not isinstance(var, str):
            raise TypeError(f"{name}는 str 타입이어야 합니다. (현재: {type(var)})")
    if not isinstance(test_size, float):
        raise TypeError(f"test_size는 float 타입이어야 합니다. (현재: {type(test_size)})")
    if not isinstance(random_state, int):
        raise TypeError(f"random_state는 int 타입이어야 합니다. (현재: {type(random_state)})")
    if not isinstance(debug, bool):
        raise TypeError(f"debug는 bool 타입이어야 합니다. (현재: {type(debug)})")

    os.makedirs(output_train, exist_ok=True)
    os.makedirs(output_val, exist_ok=True)

    # 라벨 파일 목록 가져오기
    label_files = [f for f in os.listdir(label_dir) if f.endswith(".txt")]
    if debug:
        print(f"[DEBUG] 총 라벨 파일 개수: {len(label_files)}")
    if len(label_files) == 0:
        print(f"경로 {label_dir}에 .txt 라벨 파일이 없습니다. 경로를 다시 확인하세요.")
        return

    # stratify용 class id 추출
    class_ids = []
    for filename in label_files:
        file_path = os.path.join(label_dir, filename)
        with open(file_path, 'r') as f:
            lines = f.readlines()
            # 마지막 줄 사용
            if len(lines) == 4:
                class_id = int(lines[-1].split()[0])
            # 비어 있는 파일은 0 처리
            else:
                class_id = 0
        class_ids.append(class_id)

    # train/val 분할 (stratify 시도 후 실패 시 랜덤 분할)
    try:
        train_labels, val_labels = train_test_split(
            label_files, 
            test_size=test_size, 
            random_state=random_state, 
            stratify=class_ids
        )
        if debug:
            print(f"Stratify 성공, 마지막 class id를 기준으로 분할합니다.")
    except Exception as e:
        print(f"Stratify 실패, 랜덤 분할로 대체합니다. 오류: {e}")
        train_labels, val_labels = train_test_split(
            label_files, 
            test_size=test_size, 
            random_state=random_state
        )


    # train 라벨 및 이미지 복사
    for label_file in tqdm(train_labels, desc="Train 데이터 복사", leave=False, dynamic_ncols=True):
        shutil.move(os.path.join(label_dir, label_file), os.path.join(output_train, label_file))
        img_name = label_file.replace('.txt', '.png')
        src_img_path = os.path.join(image_dir, img_name)
        dst_img_path = os.path.join(output_train, img_name)
        shutil.copy(src_img_path, dst_img_path)

    # val 라벨 및 이미지 복사
    for label_file in tqdm(val_labels, desc="Val 데이터 복사", leave=False, dynamic_ncols=True):
        shutil.move(os.path.join(label_dir, label_file), os.path.join(output_val, label_file))
        img_name = label_file.replace('.txt', '.png')
        src_img_path = os.path.join(image_dir, img_name)
        dst_img_path = os.path.join(output_val, img_name)
        shutil.copy(src_img_path, dst_img_path)
    
    if debug:
        print(f"[DEBUG] Train 라벨 수: {len(train_labels)}, Val 라벨 수: {len(val_labels)}")

    print("라벨 및 이미지 분할 완료.")


def verify_label_image_pairs(directory, debug=False):
    """
    주어진 디렉토리 내에 라벨(.txt) 파일과 동일 이름의 이미지(.png) 파일이 모두 존재하는지 검증합니다.

    Args:
        directory (str): 검사할 경로
        debug (bool): True 시 누락된 짝 목록 출력

    동작:
    - 디렉토리 내에서 라벨과 이미지가 모두 존재하는지 확인
    - 누락 시 경고 메시지 출력 및 디버깅 모드에서 상세 리스트 출력
    """
    txt_files = [f for f in os.listdir(directory) if f.endswith('.txt')]
    png_files = [f for f in os.listdir(directory) if f.endswith('.png')]

    png_set = set([os.path.splitext(f)[0] for f in png_files])
    txt_set = set([os.path.splitext(f)[0] for f in txt_files])

    missing_images = txt_set - png_set
    missing_labels = png_set - txt_set

    if missing_images:
        print(f"[WARNING] 다음 라벨에 해당하는 이미지가 없습니다: {missing_images}")
    if missing_labels:
        print(f"[WARNING] 다음 이미지에 해당하는 라벨이 없습니다: {missing_labels}")

    if debug:
        print(f"[DEBUG] {directory} 내 검증 완료. 총 라벨 {len(txt_set)}개, 이미지 {len(png_set)}개.")
        if not missing_images and not missing_labels:
            print("[DEBUG] 모든 라벨과 이미지가 잘 매칭되어 있습니다.")



# data.yaml 파일 생성 함수
def make_yaml(train_dir, val_dir, output_dir, debug=False):
    """
    YOLO 학습을 위한 data.yaml 파일을 생성하는 함수.

    Args:
        train_dir (str): 학습 데이터 라벨 및 이미지 경로
        val_dir (str): 검증 데이터 라벨 및 이미지 경로
        output_dir (str): YAML 파일을 저장할 디렉토리 경로
        debug (bool): True 시 생성된 yaml 파일 내용 일부를 출력

    동작:
    - 클래스 이름 목록을 가져와 background, no_class 제거
    - YAML 형식에 맞는 텍스트 생성
    - train, val 경로, 클래스 개수, names를 yaml 파일로 작성
    - 디버깅 모드 시 클래스 개수 및 yaml 내용 일부 출력
    """

    # 인자 타입 체크
    for var, name in zip([train_dir, val_dir, output_dir], ["train_dir", "val_dir", "output_dir"]):
        if not isinstance(var, str):
            raise TypeError(f"{name}는 str 타입이어야 합니다. (현재: {type(var)})")
    if not isinstance(debug, bool):
        raise TypeError(f"debug는 bool 타입이어야 합니다. (현재: {type(debug)})")
    
    # 경로 준비
    os.makedirs(output_dir, exist_ok=True)

    # YAML 파일 경로 설정
    yaml_dir = os.path.join(output_dir, "data.yaml")
    
    # 클래스 이름 목록 가져오기
    keys = list(idx_to_name.keys())  
    class_names = [name for name in idx_to_name.values()]

    # names 항목을 한 줄 문자열 포맷으로 변환
    formatted_names = "[" + ", \n".join([f' \"{name}\"' for name in class_names]) + "]"

    # YAML 내용 문자열 직접 작성 (따옴표 및 포맷 유지)
    yaml_content = f"""
train: {train_dir}
val: {val_dir}
nc: {len(class_names)}
names: {formatted_names}
"""

    try:
        with open(yaml_dir, "w", encoding='utf-8') as f:
            f.write(yaml_content.strip())
        print(f"YAML 파일이 '{yaml_dir}'에 성공적으로 생성되었습니다.")

        if debug:
            print(f"[DEBUG] 생성된 data.yaml 내용 예시:\n{yaml_content[:300]}...")

    except Exception as e:
        print(f"YAML 파일 생성 중 오류가 발생했습니다: {e}")

    print(f"{yaml_dir} 파일이 생성되었습니다.")

###########################################################################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert COCO JSON annotations to YOLO format and split datasets")
    parser.add_argument("--json_folder", type=str, default="data/train_annots_modify", help="Folder containing COCO JSON files (상대경로 입력 시 절대경로로 변환됨)")
    parser.add_argument("--output_dir", type=str, default="data/train_labels", help="Output directory for YOLO label files (상대경로 입력 시 절대경로로 변환됨)")
    parser.add_argument("--image_dir", type=str, default="data/train_images", help="Image folder path (상대경로 입력 시 절대경로 변환)")
    parser.add_argument("--test_size", type=float, default=0.2, help="Validation set ratio")
    parser.add_argument("--debug", action="store_true", help="Enable debug messages")

    args = parser.parse_args()

    # 상대경로 → 절대경로 변환
    json_folder = os.path.abspath(args.json_folder)
    output_dir = os.path.abspath(args.output_dir)
    image_dir = os.path.abspath(args.image_dir)

    
    if args.debug:
        print(f"[DEBUG] JSON 폴더: {json_folder}")
        print(f"[DEBUG] 라벨 저장 폴더: {output_dir}")
        print(f"[DEBUG] 이미지 폴더: {image_dir}")

    process_all_json(args.json_folder, args.output_dir)

    label_dir = output_dir
    output_train = os.path.join(label_dir, "train")
    output_val = os.path.join(label_dir, "val")

    split_labels_and_images(label_dir, image_dir, output_train, output_val, test_size=args.test_size, random_state=42, debug=args.debug)

    verify_label_image_pairs(output_train)
    verify_label_image_pairs(output_val)

    make_yaml(output_train, output_val, label_dir, debug=args.debug)
