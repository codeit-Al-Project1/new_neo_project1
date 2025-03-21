import os
from kaggle.api.kaggle_api_extended import KaggleApi
import zipfile
import gdown
import shutil
from tqdm import tqdm
import argparse
import json

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
        path = './data'
        print("data폴더가 존재합니다.")
    elif not os.path.exists(path):
        os.mkdir(path)
        print("지정한 위치에 새로운 폴더를 생성합니다.")
    else:
        path = './'
        print("Download Path를 지정하지 않았습니다. 현재위치에서 진행합니다.")

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
    zip_path = os.path.join(path, f"{zip_name}.zip")

    file_id = "1dCMvEIqIhbJKa8G5poO5MPdeDHTiEcQ8"

    url = f'https://drive.google.com/uc?export=download&id={file_id}'

    print("Train_annotations를 덮어씌웁니다.")
    gdown.download(url, os.path.join(path, f'{zip_name}.zip'), quiet=False)

    if os.path.exists(os.path.join(path, zip_name)):
        shutil.rmtree(os.path.join(path, zip_name))
        print("기존 데이터 삭제 후 대체합니다.")

    # 압축 해제
    if os.path.exists(zip_path):
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(path)
        print("Train_annotations 압축 해제 완료!")
        os.remove(zip_path)  # 다운로드한 zip 파일 삭제
        print(f"{zip_name}.zip 파일 삭제 완료!")
    else:
        print(f"{zip_path} 파일을 찾을 수 없습니다.")
    
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
    for root, dirs, files in os.walk(annot_path):
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

    # 원하는 위치에 폴더 생성
    os.makedirs(output_dir, exist_ok=True)

    # 복잡하게 얽혀있는 데이터들을 열어 리스트로 저장
    json_list = []
    for root, dirs, files in os.walk(json_folder):
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
    for i in range(len(images)):
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

    print(f"JSON 파일 저장 완료!")

    return

def data_setup(path = './data', 
               output_dir='./data/train_annots_modify', 
               json_folder='./data/train_annotations', 
               download=False, 
               extract=False, 
               img=img, 
               annot=annot):
    
    download_data(path, download, extract)
    folder_check()
    wrap_annotation()
    folder_check()
    json_modify(output_dir, json_folder)
    return

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Data download and annotation wrap")
    # default 저장 루트를 확인
    parser.add_argument('--path', type=str, default="./data", help='Download Path(default: ./data)')
    parser.add_argument('--output_dir', type=str, default="./data/train_annots_modify", help='Destination of json file save')
    parser.add_argument('--json_folder', type=str, default='./data/train_annotations', help='Location of original json files')
    parser.add_argument('--download', type=bool, default=False, help='Download Preference')
    parser.add_argument('--extract', type=bool, default=False, help='Extract Preference')
    args = parser.parse_args()

    data_setup(args.path, args.output_dir, args.json_folder, args.download, args.extract)