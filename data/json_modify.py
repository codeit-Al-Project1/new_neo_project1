import argparse
import os
import json

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



# 실행 
# output_dir = "data/train_annots_modify"
# json_folder = "/content/Project1/data/train_annotations"

# json_modify(output_dir, json_folder)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Json modification")
    # default 저장 루트를 확인
    parser.add_argument('--output_dir', type=str, default="data/train_annots_modify", help='Destination of json file save')
    parser.add_argument('--json_folder', type=str, default='data/train_annotations', help='Location of original json files')
    args = parser.parse_args()
    json_modify(args.output_dir, args.json_folder)