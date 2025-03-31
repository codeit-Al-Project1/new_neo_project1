# 표준 라이브러리
import os
import json
import shutil
import zipfile
import glob
import argparse

# 서드파티 라이브러리
import gdown
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from kaggle.api.kaggle_api_extended import KaggleApi

# 내부 모듈
from src.utils import get_category_mapping

# 전체 실행 (다운로드 → json 변환 → yolo 변환 → 분할 → yaml 생성)
# python -m src.data_utils.data_pipeline --path ./data --output_dir ./data/train_annots_modify --json_folder ./data/train_annotations --download --extract --image_dir ./data/train_images --label_output_dir ./data/train_labels --test_size 0.2 --debug --step all

# 1) 다운로드 및 json 변환만 실행
# python -m src.data_utils.data_pipeline --path ./data --output_dir ./data/train_annots_modify --json_folder ./data/train_annotations --download --extract --step download

# 2) json 변환만 실행 (이미 다운로드된 경우)
# python -m src.data_utils.data_pipeline --output_dir ./data/train_annots_modify --json_folder ./data/train_annotations --step json_modify

# 3) YOLO 변환만 실행
# python -m src.data_utils.data_pipeline --output_dir ./data/train_annots_modify --label_output_dir ./data/train_labels --debug --step convert_yolo

# 4) split (train/val) 및 검증만 실행
# python -m src.data_utils.data_pipeline --label_output_dir ./data/train_labels --image_dir ./data/train_images --test_size 0.2 --debug --step split

# 5) yaml 파일만 생성
# python -m src.data_utils.data_pipeline --output_dir ./data/train_annots_modify --label_output_dir ./data/train_labels --debug --step make_yaml

### 다운로드 및 압축 해제
def download_data(path = './data', download=True, extract=True):
    """
    Kaggle API를 이용해 데이터를 다운로드하고, 압축을 해제하는 메소드, 
    메소드의 마지막 과정에서 압축파일을 제거한다.

    ***Note: Kaggle API를 사용하기 위해서는 Kaggle 계정이 있어야 하며,
    Kaggle API Token이 필요하다.***
    ***Note: 다운로드 위치를 주의하도록 하자.***
    
    :param path: str: 다운로드 경로
    :param download: bool: 데이터 다운로드 여부
    :param extract: bool: 압축 해제 여부
    """
    if os.path.exists(path):
        print("data폴더가 존재합니다.")
    elif not os.path.exists(path):
        os.mkdir(path)
        print("지정한 위치에 새로운 폴더를 생성합니다.")

    competition_name = 'ai01-level1-project'

    if download:
        api = KaggleApi()
        api.authenticate()

        if f"{competition_name}.zip" in os.listdir(path):
            print(f"{competition_name}가 {path}에 이미 존재합니다.")
        else:
            print(f"{competition_name} 다운로드 시작...")
            try:
                api.competition_download_files(competition_name, path=path, quiet=False)
                print(f"다운로드 완료! 저장위치: {path}")
            except Exception as e:
                print(f"Error: {e}")
                return           
            
    if extract:
        zip_file = f'{competition_name}.zip'
        if zip_file not in os.listdir(path):
            print(f"{zip_file}를 {path}에서 찾을 수 없습니다.")
            return
        
        zip_path = os.path.join(path, zip_file)

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(path)
            print(f"{zip_file} 압축 해제 완료!")
        
        os.remove(zip_path)
        print(f"{zip_file} 파일 삭제 완료!")
    
    return


### 구글 드라이브로 annotation 교체
def wrap_annotation(path = './data'):
    """
    수정된 train_annotations 압축 파일을 내려받아,
    기존 train_annotations 폴더를 대체 하는 과정.
    """

    if os.path.exists(path):
        path = './data'
    else:
        path = './'

    zip_name = "train_annotations"
    file_id = "1dCMvEIqIhbJKa8G5poO5MPdeDHTiEcQ8"
    url = f'https://drive.google.com/uc?export=download&id={file_id}'

    print("Train_annotations를 덮어씌웁니다.")
    gdown.download(url, os.path.join(path, f'{zip_name}.zip'), quiet=False)

    if os.path.exists(os.path.join(path, zip_name)):
        shutil.rmtree(os.path.join(path, zip_name))
        print("기존 데이터 삭제 후 대체합니다.")

    zip_path = os.path.join(path, f"{zip_name}.zip")
    # 압축 해제
    if os.path.exists(zip_path):
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(path)
        print("Train_annotations 압축 해제 완료!")
        os.remove(zip_path)  # 다운로드한 zip 파일 삭제
        print(f"{zip_name}.zip 파일 삭제 완료!")
    else:
        print(f"{zip_path} 파일을 찾을 수 없습니다.")


### folder_check
def folder_check():
    if os.path.exists('./data'):
        path = './data'
    else:
        path = './'

    annot_folder = "train_annotations"

    annot_path = os.path.join(path, annot_folder)
    
    count = 0

    # 모든 파일을 먼저 수집하고 그 후 tqdm으로 진행 표시
    all_files = []
    for root, _, files in os.walk(annot_path):
        for file in files:
            if file.endswith('.json'):
                all_files.append(os.path.join(root, file))

    # 파일이 존재하면 tqdm으로 진행 상태 표시
    if all_files:
        for json in tqdm(all_files, desc="Counting JSON files", unit="file", mininterval=0.5, maxinterval=2):
            count += 1
        print(f"{count}개가 폴더 내부에 있습니다.")
    else:
        print("폴더 내부에 파일이 없습니다.")
        
    return

# json file의 목표 형태 초기화 (category의 형태는 원본에서 바뀌지 않는다)
img = {
    "file_name": "",
    "id": 0,
    "drug_id": [],
    "width": 0,
    "height": 0
}

annot = {
    "area": 0,
    "bbox": [0,0,0,0],
    "category_id": 0,
    "image_id": 0,
    "annotation_id": 0
}


### JSON Modify (형식 변환)
def json_modify(output_dir, json_folder, img=img, annot=annot):
    """
    json file 데이터를 검토 했을 경우, 한 annotation 파일에 종합되어 있지 않고,
    분할 되어, 각 bounding box, 라벨 데이터가 각각의 파일에 있는 것을 확인. 이를
    모델에 학습시키기에 적합한 형태로 바꾸기 위한 모듈. 한 json 파일은 이미지 정보,
    이미지에 내포된 알약들에 관한 bbox를 포함한 라벨 데이터. 그리고 카테고리의 정보를 
    포함하고있다.
    Input:
    output_dir = 최종 json file들의 저장 장소의 위치
    json_folder = json 파일들이 저장되어있는 위치 os.walk로 들어가 폴더 내부를 
                  탐사해 리스트 형태로 저장.
    """
    print("json파일을 학습에 적합한 형태로 변환합니다.")
    # 원하는 위치에 폴더 생성
    os.makedirs(output_dir, exist_ok=True)

    # 복잡하게 얽혀있는 데이터들을 열어 리스트로 저장
    json_list = []
    for root, _, files in os.walk(json_folder):
        for file in files:
            if file.endswith('.json'):
                json_file_path = os.path.join(root, file)

                # JSON 파일 로드
                with open(json_file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    json_list.append(data)


    # 전체 json파일에서 images, annotations, categories로 분리
    images = []
    annotations = []
    categories = []
    for json_file in json_list:
        images.extend(json_file["images"])
        annotations.extend(json_file["annotations"])
        categories.extend(json_file["categories"])

    # json 파일 전처리
    for i in tqdm(range(len(images))):
        temp_img = img.copy()

        temp_img["file_name"] = images[i]["file_name"]
        temp_img["id"] = images[i]["id"]
        temp_img["width"] = images[i]["width"]
        temp_img["height"] = images[i]["height"]

        # annotaion을 image_id 추적 후 저장
        temp_annotations = []
        drug_ids = set()
        for j in range(len(annotations)):
            if annotations[j]["image_id"] ==  temp_img["id"] and annotations[j]["category_id"] not in drug_ids:
                temp_annot = annot.copy()
                temp_annot["area"] = annotations[j]["area"]
                temp_annot["bbox"] = annotations[j]["bbox"]
                if annotations[j]["image_id"] == 772 and annotations[j]["bbox"][0] == 1771:
                    temp_annot["bbox"][0] = 167
                temp_annot["category_id"] = annotations[j]["category_id"]
                temp_annot["image_id"] = annotations[j]["image_id"]
                temp_annot["annotation_id"] = annotations[j]["id"]
                drug_ids.add(annotations[j]["category_id"])
                
                temp_annotations.append(temp_annot)

        # 알약 정보를 리스트로 저장 (단일 알약에 대해서만 적혀있었다면, 현재는 annotation이 포함된 알약의 id를 포함한 리스트)
        temp_img["drug_id"] = list(drug_ids)
        
        # 카테고리정보를 알약 정보로 추적
        temp_categories = []
        cat_ids = set()
        for n in range(len(categories)):
            cat_id = categories[n]["id"]
            if cat_id in temp_img["drug_id"] and cat_id not in cat_ids:
                temp_categories.append(categories[n])
                cat_ids.add(categories[n]["id"])

        # coco dataset에 맞는 형식의 Dictionary 형태의 저장
        json_data = {
            "images": [temp_img],
            "annotations": temp_annotations,
            "categories": temp_categories
        }

        # json file 저장
        file_name = temp_img["file_name"]
        json_file_name = f"{output_dir}/{file_name}.json"

        with open(json_file_name, "w", encoding="utf-8") as f:
            json.dump(json_data, f, indent=4, ensure_ascii=False)

    print(f"JSON 변환 파일 저장 완료!")


### COCO to YOLO 변환 함수
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


### 폴더 내 모든 JSON을 변환
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
            if len(lines) >= 1:
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

    # 절대경로 및 슬래시 통일
    train_dir = os.path.abspath(train_dir).replace("\\", "/")
    val_dir = os.path.abspath(val_dir).replace("\\", "/")
    output_dir = os.path.abspath(output_dir).replace("\\", "/")

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


def run_download_and_json(args):
    download_data(args.path, args.download, args.extract)
    wrap_annotation(args.path)
    json_modify(args.output_dir, args.json_folder)

def run_json_modify_only(args):
    json_modify(args.output_dir, args.json_folder)

def run_yolo_conversion(args):
    global name_to_idx, idx_to_name
    name_to_idx, idx_to_name = get_category_mapping(
        ann_dir=args.output_dir, debug=args.debug, add_more=False, return_types=['name_to_idx', 'idx_to_name']
    )
    process_all_json(args.output_dir, args.label_output_dir, debug=args.debug)

def run_split_and_verify(args):
    train_dir = os.path.join(args.label_output_dir, "train")
    val_dir = os.path.join(args.label_output_dir, "val")
    split_labels_and_images(args.label_output_dir, args.image_dir, train_dir, val_dir, test_size=args.test_size, debug=args.debug)
    verify_label_image_pairs(train_dir, debug=args.debug)
    verify_label_image_pairs(val_dir, debug=args.debug)

def run_make_yaml_only(args):
    train_dir = os.path.join(args.label_output_dir, "train")
    val_dir = os.path.join(args.label_output_dir, "val")
    global name_to_idx, idx_to_name
    name_to_idx, idx_to_name = get_category_mapping(
        ann_dir=args.output_dir, debug=args.debug, add_more=False, return_types=['name_to_idx', 'idx_to_name']
    )
    make_yaml(train_dir, val_dir, args.label_output_dir, debug=args.debug)

def main():
    parser = argparse.ArgumentParser(description="End-to-end data preparation pipeline")

    # 공통 인자
    parser.add_argument('--path', type=str, default="./data")
    parser.add_argument('--output_dir', type=str, default="./data/train_annots_modify")
    parser.add_argument('--json_folder', type=str, default='./data/train_annotations')
    parser.add_argument('--download', action="store_true")
    parser.add_argument('--extract', action="store_true")
    parser.add_argument('--json_modify', action="store_true")
    parser.add_argument('--image_dir', type=str, default="./data/train_images")
    parser.add_argument('--label_output_dir', type=str, default="./data/train_labels")
    parser.add_argument('--test_size', type=float, default=0.2)
    parser.add_argument('--debug', action="store_true")
    parser.add_argument('--step', type=str, choices=['download', 'json_modify', 'convert_yolo', 'split', 'make_yaml', 'all'], default='all', help="개별 스텝 실행")

    args = parser.parse_args()

    # 개별 실행 분기
    if args.step == 'download':
        run_download_and_json(args)
    elif args.step == 'json_modify':
        run_json_modify_only(args)
    elif args.step == 'convert_yolo':
        run_yolo_conversion(args)
    elif args.step == 'split':
        run_split_and_verify(args)
    elif args.step == 'make_yaml':
        run_make_yaml_only(args)
    else:
        # 전체 파이프라인 실행
        run_download_and_json(args)
        run_yolo_conversion(args)
        run_split_and_verify(args)
        run_make_yaml_only(args)

if __name__ == "__main__":
    main()

# 전체 실행 (다운로드 → json 변환 → yolo 변환 → 분할 → yaml 생성)
# python -m src.data_utils.data_pipeline --path ./data --output_dir ./data/train_annots_modify --json_folder ./data/train_annotations --download --extract --image_dir ./data/train_images --label_output_dir ./data/train_labels --test_size 0.2 --debug --step all

# 1) 다운로드 및 json 변환만 실행
# python -m src.data_utils.data_pipeline --path ./data --output_dir ./data/train_annots_modify --json_folder ./data/train_annotations --download --extract --step download

# 2) json 변환만 실행 (이미 다운로드된 경우)
# python -m src.data_utils.data_pipeline --output_dir ./data/train_annots_modify --json_folder ./data/train_annotations --step json_modify

# 3) YOLO 변환만 실행
# python -m src.data_utils.data_pipeline --output_dir ./data/train_annots_modify --label_output_dir ./data/train_labels --debug --step convert_yolo

# 4) split (train/val) 및 검증만 실행
# python -m src.data_utils.data_pipeline --label_output_dir ./data/train_labels --image_dir ./data/train_images --test_size 0.2 --debug --step split

# 5) yaml 파일만 생성
# python -m src.data_utils.data_pipeline --output_dir ./data/train_annots_modify --label_output_dir ./data/train_labels --debug --step make_yaml