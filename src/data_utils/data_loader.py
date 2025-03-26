########################################
# 실행 방법 및 인자 설명 (터미널 기준)
#
# 사용법:
#   python src/data_utils/data_loader.py --mode <모드> --batch_size <배치 크기> [--debug] [--val_ratio <검증 비율>] [--seed <랜덤 시드>
#
# 파싱 인자 설명:
# --mode (필수)  
#   - 선택 가능 값: 'train', 'test'  
#   - train  : 학습용/검증용 데이터셋 로드 및 디버깅 실행  
#   - test   : 테스트용 데이터셋 로드 (어노테이션 없이 이미지와 파일명 반환)  
#
# --batch_size (선택, default=4)  
#   - DataLoader에서 사용하는 배치 크기  
#   - ex) --batch_size 8 → 8개 샘플 단위로 배치 로딩  
#
# --debug (옵션)  
#   - 해당 플래그 추가 시 디버깅 모드 활성화  
#   - 데이터셋 및 배치 관련 상세 정보를 출력  
#   - 출력 예: 카테고리 매핑 정보, 총 데이터 수, Split 비율,  
#              각 배치별 이미지 크기, 박스 정보, 라벨 등장 분포 등  
#
# --val_ratio (선택, default=0.2)  
#   - 학습 및 검증 데이터셋 분할 시 검증 비율 설정 (0 ~ 1 사이 실수)  
#   - ex) --val_ratio 0.25 → 전체 데이터 중 25%를 검증 데이터로 분할  
#
# --seed (선택, default=42)  
#   - 무작위 split 및 DataLoader 생성 시 랜덤 시드 고정 값  
#   - 디버깅이나 재현성 테스트 시 유용  
#
# 실행 예시 (터미널):
# 1) 학습 데이터셋 로더 테스트
#   python src/data_utils/data_loader.py --mode train --batch_size 4 --debug --val_ratio 0.2 --seed 42
#
# 2) 검증 데이터셋 로더 테스트
#   python src/data_utils/data_loader.py --mode val --batch_size 8 --debug --val_ratio 0.2 --seed 42
#
# 3) 테스트 데이터셋 로더 테스트
#   python src/data_utils/data_loader.py --mode test --batch_size 16 --debug
#
# 프로젝트 폴더 예시:
# data/
# ├─ train_images/           (훈련 이미지)
# ├─ train_annots_modify/    (훈련 어노테이션 JSON)
# └─ test_images/            (테스트 이미지)
#
# ⚠ 디버깅 모드를 활성화하면, 매핑 테이블, 데이터 수,  
#    각 배치별 샘플 크기, 바운딩 박스 통계, 라벨별 출현 횟수,  
#    pill_names 목록 등을 모두 확인 가능합니다!
########################################

import os
import json
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.v2 as T
from torchvision.tv_tensors import BoundingBoxes, Image as TVImage
import argparse 
import sys

####################################################################################################
# 1. 데이터 증강 및 전처리 Transform 정의 (학습/검증/테스트 분기)
def get_transforms(mode='train'):
    """
    데이터 증강 및 전처리 함수를 반환합니다.

    Args:
        mode (str): 'train', 'val', 'test' 중 하나

    Returns:
        torchvision.transforms.v2.Compose: 변환 함수
    """

    # TODO: 리사이즈 크기 지정 필요 여부 검토 (현재 자동 처리)

    if mode == 'train':
        return T.Compose([
            T.ToImage(),
            T.RandomRotation(degrees=(-30, 30)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.ColorJitter(brightness=(0.9, 1.1), contrast=(0.9, 1.1), saturation=(0.9, 1.1), hue=(-0.1, 0.1)),
            T.RandomGrayscale(p=0.1),
            T.RandomPerspective(distortion_scale=0.2, p=0.5),
            T.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0)),
            T.RandomResizedCrop(size=(640, 640), scale=(0.8, 1.2)),
            T.ToDtype(dtype=torch.float32, scale=True)
        ])
    elif mode == "val" or mode == "test":
        return T.Compose([
            T.ToImage(),
            T.Resize((640, 640)),
            T.ToDtype(torch.float32, scale=True)
        ])
    else:
        raise ValueError(f"Invalid mode: {mode}. Choose either 'train', 'val', or 'test'.")

####################################################################################################
# 2. json파일에서 카테고리 매핑을 만드는 함수
def get_category_mapping(ann_dir, debug=False, add_more=False, return_types=None):
    """
    어노테이션 디렉토리 내 JSON 파일들을 탐색하여
    ID, 이름, 인덱스 간 매핑을 생성하고 요청된 매핑을 반환하는 함수입니다.

    Args:
        ann_dir (str): 어노테이션 JSON 파일들이 저장된 디렉토리 경로
        debug (bool): 디버깅 출력 여부
        add_more (bool): True인 경우 'Background'와 'No Class'를 포함하여 인덱싱  
                        False인 경우 0부터 시작하는 인덱스만 부여
        return_types (Optional[List[str]]):  
            반환할 매핑 키 리스트 (선택 가능한 매핑: 'id_to_name', 'name_to_id',  
            'name_to_idx', 'idx_to_name', 'id_to_idx', 'idx_to_id')

    Returns:
        Dict[str, Any]: 요청된 매핑 딕셔너리

    Note:
        - 중복된 카테고리 이름은 제거 후 정렬합니다.
        - add_more=True인 경우: 0은 'Background', 마지막 인덱스는 'No Class'로 지정
        - add_more=False인 경우: 인덱스는 0부터 순차 부여
        - 반환 매핑은 학습 및 시각화 등에 활용됩니다.
    """
    # 디버깅 메시지 출력
    if not isinstance(ann_dir, str):
        raise TypeError(f"ann_dir는 문자열(str)이어야 합니다. 현재 타입: {type(ann_dir)}")
    if not os.path.exists(ann_dir):
        raise FileNotFoundError(f"ann_dir 경로가 존재하지 않습니다: {ann_dir}")
    if not isinstance(add_more, bool):
        raise TypeError(f"add_more는 불리언 값이어야 합니다. 현재 타입: {type(add_more)}")
    
    id_to_name = {}
    # 어노테이션 폴더 내 파일 순회
    for file in os.listdir(ann_dir):
        file_path = os.path.join(ann_dir, file)

        if file.endswith(".json"):
            # JSON 파일 처리
            with open(file_path, 'r', encoding='utf-8') as f:
                ann = json.load(f)
                categories = ann.get('categories', [])

                for cat in categories:
                    id_to_name[cat['id']] = cat['name']

    # 이름 기준 정렬
    sorted_names = sorted(set(id_to_name.values())) # 중복 제거 후 정렬

    name_to_idx = {}
    if add_more:
        # 1부터 인덱싱, 0은 배경, 마지막 숫자는 No Class
        name_to_idx = {'Background': 0}
        for idx, name in enumerate(sorted_names, start=1):
            name_to_idx[name] = idx
        name_to_idx['No Class'] = len(name_to_idx)
    else:
        for idx, name in enumerate(sorted_names):
            name_to_idx[name] = idx

    # 역 매핑
    idx_to_name = {idx: name for name, idx in name_to_idx.items()}
    name_to_id = {name: id for id, name in id_to_name.items()}

    # 아이디 <-> 인덱스
    id_to_idx = {id_: name_to_idx[name] for id_, name in id_to_name.items()}
    idx_to_id = {idx: name_to_id[name] for idx, name in idx_to_name.items() if name in name_to_id}
    
    # 모든 매핑 저장
    all_mappings = {
        'id_to_name': id_to_name,
        'name_to_id': name_to_id,
        'name_to_idx': name_to_idx,
        'idx_to_name': idx_to_name,
        'id_to_idx': id_to_idx,
        'idx_to_id': idx_to_id
    }

    if return_types is None:
        result = all_mappings
    elif len(return_types) == 1:
        result = all_mappings[return_types[0]]
    else:
        result = tuple(all_mappings[k] for k in return_types)
    
    if debug:
        print(f"[DEBUG] 반환 직전 매핑 결과:")
        if isinstance(result, dict):
            for key, value in result.items():
                print(f"  {key}: {value}")
        elif isinstance(result, tuple):
            for idx, item in enumerate(result):
                print(f"  반환 항목 {idx+1}: {item}")
        else:
            print(f"  반환 값: {result}")

    return result


####################################################################################################
# 3-1. 데이터셋(FRCNN)
class PillDataset(Dataset):
    def __init__(self, image_dir, ann_dir=None, mode='train', category_mapping=None, transform=None, bbox_format="XYXY", debug=False):
        """
        PillDataset 클래스

        알약 이미지와 어노테이션(json) 파일을 로드하여  
        모델 학습 및 검증을 위한 데이터셋 형태로 제공하는 PyTorch Dataset 클래스입니다.  

        Attributes:
            image_dir (str): 이미지 파일이 저장된 디렉토리 경로  
            ann_dir (str, optional): 어노테이션 JSON 파일들이 저장된 디렉토리 경로 (train/val 모드에서 필요)  
            mode (str): 'train', 'val', 'test' 중 하나로 데이터셋의 동작 모드를 결정  `
            category_mapping (dict): 카테고리 이름과 인덱스 매핑 정보  
            transform (callable, optional): 이미지 및 bounding box에 적용할 변환 함수  
            debug (bool): True일 경우 불일치 및 파일 정보 등을 출력하여 디버깅 용도로 활용

        Notes:
            - train/val 모드에서는 이미지-어노테이션 쌍을 필터링 후 로드합니다.  
            - test 모드에서는 어노테이션 없이 이미지 파일 경로만 반환합니다.  
        """
        # 디버깅 메시지 출력 
        if not isinstance(image_dir, str):
            raise TypeError(f"image_dir는 문자열(str)이어야 합니다. 현재 타입: {type(image_dir)}")
        if ann_dir is not None and not isinstance(ann_dir, str):
            raise TypeError(f"ann_dir는 문자열(str)이거나 None이어야 합니다. 현재 타입: {type(ann_dir)}")
        if mode not in ['train', 'val', 'test']:
            raise ValueError(f"mode는 'train', 'test' 중 하나여야 합니다. 현재 입력: {mode}")
        if category_mapping is not None and not isinstance(category_mapping, dict):
            raise TypeError(f"category_mapping은 dict 타입이어야 합니다. 현재 타입: {type(category_mapping)}")
        if transform is not None and not callable(transform):
            raise TypeError(f"transform은 호출 가능 객체이어야 합니다. 현재 타입: {type(transform)}")
        if not isinstance(bbox_format, str):
            raise TypeError(f"bbox_format는 문자열(str)이어야 합니다. 현재 타입: {type(bbox_format)}")
        if not isinstance(debug, bool):
            raise TypeError(f"debug는 bool 타입이어야 합니다. 현재 타입: {type(debug)}")
        
        # 인자 받기기
        self.img_dir = image_dir
        self.ann_dir = ann_dir
        self.mode = mode
        self.transform = transform
        self.category_mapping = category_mapping    # 카테고리 이름 <-> 아이디 매핑
        self.bbox_format = bbox_format

        # 이미지
        self.images = sorted(os.listdir(image_dir))

        # train, val/ test 분기
        if self.mode in ['train', 'val']:
            assert ann_dir is not None, "Train/Val 모드에서는 ann_dir가 필요합니다."
            self.annots = sorted(os.listdir(ann_dir))

            # 이미지-어노테이션 불일치 필터링
            # 예시: K-001900-010224-016551-031705_0_2_0_2_70_000_200.png
            img_basename = set(os.path.splitext(f)[0] for f in self.images)
            # 예시: K-001900-010224-016551-031705_0_2_0_2_70_000_200.png.json
            ann_basename = set(os.path.splitext(os.path.splitext(f)[0])[0] for f in self.annots)    # 두 번 적용
            
            # 공통이름 및 차이점
            common_name = img_basename & ann_basename 
            missing_img = ann_basename - img_basename
            missing_ann = img_basename - ann_basename

            # 디버깅(차이점 출력)
            if debug:
                print(f"\n[DEBUG] 총 {len(common_name)}개의 이미지-어노테이션 파일 처리")
                if missing_img:
                    print(f"[WARNING] 어노테이션은 있지만 이미지가 없는 파일 목록(총 {len(missing_img)}개):")
                    for name in sorted(missing_img):
                        print(f" - {name}")
                if missing_ann:
                    print(f"[WARNING] 이미지는 있지만 어노테이션이 없는 파일 목록(총 {len(missing_ann)}개):")
                    for name in sorted(missing_ann):
                        print(f" - {name}")
            
            # 공통 파일만 필터링
            self.images = [f"{name}.png" for name in common_name]
            self.annots = [f"{name}.png.json" for name in common_name]

        else:
            self.annots = None

    def __getitem__(self, idx):
        """
        __getitem__ 함수

        인덱스를 입력 받아 해당 인덱스의 이미지 및 어노테이션 데이터를 반환합니다.

        Args:
            idx (int): 호출할 데이터의 인덱스  

        Returns:
            - train/val 모드: 이미지 텐서와 COCO-style targets 딕셔너리 반환 
                img (TVImage): 이미지 텐서  
                targets (dict):  
                    - boxes (BoundingBoxes): 바운딩 박스 좌표  
                    - labels (torch.Tensor): 클래스 인덱스 라벨  
                    - image_id (torch.Tensor): 이미지 고유 식별자  
                    - area (torch.Tensor): 객체 영역  
                    - is_crowd (torch.Tensor): crowd 플래그 (0으로 고정)  
                    - orig_size (torch.Tensor): 원본 이미지 크기  

            - test 모드: 이미지 텐서 및 파일 이름 반환 (추론 시 사용) 
                img (TVImage): 이미지 텐서  
                img_file (str): 이미지 파일명 (추론 시 사용)  
        """
        # 디버깅 메시지 출력
        if not isinstance(idx, int):
            raise TypeError(f"인덱스는 int 타입이어야 합니다. 현재 타입: {type(idx)}")
        if idx >= len(self.images):
            raise IndexError(f"인덱스 {idx}가 데이터셋 크기 {len(self.images)}보다 큽니다.")

        # 이미지 인덱싱
        img_file = self.images[idx]
        img_path = os.path.join(self.img_dir, img_file)

        # 이미지 파일 확인
        try:
            img = Image.open(img_path).convert("RGB")
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Error loading image file: {img_path}, {e}")

        # 학습과 검증 분기
        if self.mode in ['train', 'val']:
            # 어노테이션 파일 인덱싱
            ann_file = self.annots[idx]
            ann_path = os.path.join(self.ann_dir, ann_file)

            # 어노테이션 파일 확인
            try:
                with open(ann_path, 'r', encoding='utf-8') as f:
                    ann = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError) as e:
                raise RuntimeError(f"Error loading annotation file: {ann_path}, {e}")
            
            # bbox와 labels 추출
            bboxes = [obj["bbox"] for obj in ann["annotations"]]
            pill_names = [obj["name"] for obj in ann["categories"]]
            labels = [self.category_mapping[cat_name] for cat_name in pill_names]  # 카테고리 매핑으로 변환
            labels_tensor = torch.tensor(labels, dtype=torch.int64)
            areas = [obj["area"] for obj in ann["annotations"]]
            image_id = ann["images"][0]["id"]

            # Faster RCNN을 위한 bbox 형식 변환
            if self.bbox_format == "XYXY":
                bboxes_xyxy = []
                for bbox in bboxes:
                    xmin, ymin, width, height = bbox
                    xmax = xmin + width
                    ymax = ymin + height
                    bboxes_xyxy.append([xmin, ymin, xmax, ymax])

                # 텐서로 변환 tv_tensor
                bboxes_tensor = BoundingBoxes(
                    torch.tensor(bboxes_xyxy, dtype=torch.float32),
                    format=self.bbox_format,
                    canvas_size=(img.height, img.width)
                )
            # YOLO를 위한 bbox 형식 변환
            elif self.bbox_format == "XYWH":
                bboxes_tensor = BoundingBoxes(
                    torch.tensor(bboxes, dtype=torch.float32),
                    format=self.bbox_format,
                    canvas_size=(img.height, img.width)
                )
            
            labels_tensor = torch.tensor(labels, dtype=torch.int64)
            areas_tensor = torch.tensor(areas, dtype=torch.float32)
            image_id_tensor = torch.tensor(image_id, dtype=torch.int64)
            iscrowd_tensor = torch.zeros((len(bboxes_tensor),), dtype=torch.int64)
            orig_size_tensor = torch.tensor([img.height, img.width], dtype=torch.int64)

            # 트랜스폼 적용
            if self.transform:
                sample = {"image": img, "bbox": bboxes_tensor}
                sample = self.transform(sample)
                img = sample["image"]
                bboxes_tensor = sample["bbox"]

            # COCODateset 기준 + 알약 이름 추가
            targets = {
                'boxes': bboxes_tensor,
                'labels': labels_tensor,
                'image_id': image_id_tensor,
                'area': areas_tensor,       # 없는 경우가 존재함
                'is_crowd': iscrowd_tensor,
                'orig_size': orig_size_tensor,
            }
            

            # 이미지와 타겟
            return img, targets

        # 시험 분기 
        else:
            # 이미지는 트랜스폼에서 자동적으로 증강됨
            if self.transform:
                img = self.transform(img)

            return img, img_file


    def __len__(self):
        return len(self.images)

    def get_img_info(self, idx):
        """
        get_img_info 함수

        특정 인덱스에 해당하는 이미지의 파일명 및 크기 정보를 반환합니다.

        Args:
            idx (int): 가져올 이미지의 인덱스  

        Returns:
            dict:  
                - file_name (str): 이미지 파일 이름  
                - height (int): 이미지 세로 크기  
                - width (int): 이미지 가로 크기  

        Note:
            어노테이션 파일을 로드하여 이미지 메타데이터 정보를 가져옵니다.  
            사용 예시: info = dataset.get_img_info(0)
                       print(info)
        """
        ann_file = self.annots[idx]
        ann_path = os.path.join(self.ann_dir, ann_file)
        try:
            with open(ann_path, 'r', encoding='utf-8') as f:
                ann = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error loading annotation file: {ann_path}, {e}")
            return None
        return {"file_name": ann['images'][0]['file_name'], "height": ann['images'][0]['height'], "width": ann['images'][0]['width']}

    def get_ann_info(self, idx):
        """
        get_ann_info 함수

        특정 인덱스에 해당하는 어노테이션 JSON 파일의 내용을 반환합니다.

        Args:
            idx (int): 가져올 어노테이션 인덱스  

        Returns:
            dict: 해당 어노테이션 JSON의 전체 내용 (파싱된 JSON)  

        Note:
            어노테이션 파일이 잘못되었거나 JSON 디코딩 실패 시 None 반환  
            사용예시: ann = dataset.get_ann_info(0)
                      print(json.dumps(ann, indent=2, ensure_ascii=False))
        """
        ann_file = self.annots[idx]
        ann_path = os.path.join(self.ann_dir, ann_file)
        try:
            with open(ann_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error loading annotation file: {ann_path}, {e}")
            return None
        
####################################################################################################
# 3-2. 데이터셋(YOLO)
# class YOLODataset(Dataset):
#     def __ini__(self, img_dir, txt_dir, mode='train', category_mapping=None, transform=None, bbox_format="XYWH", debug=False):
#         self.img_dir = img_dir
#         self.txt_dir = txt_dir
#         self.transform = transform
#         self.category_mapping = category_mapping
#         self.bbox_format = bbox_format
#         self.debug = debug

#         self.txt_files = sorted([f for f in os.listdir(txt_dir) if f.endswith('.txt')])
#         self.image_files = sorted([f for f in os.listdir(img_dir) if f.endswith(('.png'))])

#         if self.mode in ['train', 'val']:
#             assert txt_dir is not None, "Train/Val 모드에서는 txt_dir가 필요합니다."
#             # 이름 비교
#             # txt_basename = 
#             pass
        

####################################################################################################
# 4. 데이터 로더 함수
def get_loader(img_dir, ann_dir=None, batch_size=8, mode="train", val_ratio=0.2, bbox_format="XYXY", debug=False, seed=42):
    """
    데이터 로더를 반환하는 함수

    Args:
        img_dir (str): 이미지 폴더 경로
        ann_dir (str, optional): 어노테이션 폴더 경로 (train/val 모드에서 필수)
        batch_size (int): 배치 크기
        mode (str): 'train' → (train_loader, val_loader) 반환 / 'test' → test_loader 반환
        val_ratio (float): 검증 데이터셋 비율 (0 ~ 1)
        debug (bool): True일 경우 디버깅 정보 출력
        seed (int): 랜덤 시드 (재현성 보장)

    Returns:
        - mode='train': (train_loader, val_loader) 반환
        - mode='test': test_loader 반환
    """
    # 디버깅 메시지 출력
    if not isinstance(img_dir, str):
        raise TypeError(f"img_dir는 문자열(str)이어야 합니다. 현재 타입: {type(img_dir)}")
    if ann_dir is not None and not isinstance(ann_dir, str):
        raise TypeError(f"ann_dir는 문자열(str)이거나 None이어야 합니다. 현재 타입: {type(ann_dir)}")
    if not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError(f"batch_size는 양의 정수여야 합니다. 현재 입력: {batch_size}")
    if mode not in ['train', 'test']:
        raise ValueError(f"mode는 'train', 'test' 중 하나여야 합니다. 현재 입력: {mode}")
    if not (0 < val_ratio < 1):
        raise ValueError(f"val_ratio는 0과 1 사이의 실수여야 합니다. 현재 입력: {val_ratio}")
    if not isinstance(debug, bool):
        raise TypeError(f"debug는 bool 타입이어야 합니다. 현재 타입: {type(debug)}")
    
    # 트랜스폼
    if mode == 'train':
        transforms_train = get_transforms(mode='train')
        transforms_val = get_transforms(mode='val')
    else:
        transforms_test = get_transforms(mode='test')

    # 카테고리 매핑 (train에서만 필요)
    if mode == 'train':
        name_to_idx, idx_to_name = get_category_mapping(ann_dir=ann_dir, add_more=True, return_types=['name_to_idx', 'idx_to_name'])
        # 매핑 보여주기
        if debug:
            print("\n[DEBUG] 카테고리 매핑 정보:")
            print(f"- 총 클래스 수: {len(name_to_idx)}")
            print(f"- No Class 인덱스: {name_to_idx['No Class']}")
            # name_to_idx 딕셔너리
            print(f"- 매핑 테이블 (name_to_idx):\n{json.dumps(name_to_idx, indent=2, ensure_ascii=False)}")
            # idx_to_name 딕셔너리
            print("\n[DEBUG] idx_to_name 매핑 (정렬 출력):")
            max_idx_len = len(str(max(idx_to_name.keys())))  # 인덱스 최대 길이 구하기
            for idx in sorted(idx_to_name.keys()):
                print(f"  {idx:>{max_idx_len}}: {idx_to_name[idx]}")
    else:
        name_to_idx, idx_to_name = None, None

    # collator 정의
    def collator(batch):
        batch = [b for b in batch if b is not None] # None 제거
        return tuple(zip(*batch)) if batch else None

    # 훈련/검증의 경우
    if mode == 'train' or mode == 'val':
        # 데이터셋
        base_dataset = PillDataset(image_dir=img_dir, ann_dir=ann_dir, mode=mode, category_mapping=name_to_idx, transform=None, bbox_format=bbox_format, debug=debug)

        # [DEBUG 추가]
        if debug:
            print(f"\n[DEBUG] 전체 데이터셋 크기: {len(base_dataset)}개")

        # 훈련/ 검증 분리하기
        from sklearn.model_selection import train_test_split
        
        # 라벨 추출
        labels_for_stratify = []
        # 마지막 카테고리 아이디를 기준으로 분포를 측정(왜곡이 될 수 있지만 감당 가능함)
        for idx in range(len(base_dataset)):
            ann = base_dataset.get_ann_info(idx)
            if ann is not None and ann["annotations"]:
                rep_label = ann["annotations"][-1]["category_id"]
            else:
                rep_label = 0
            labels_for_stratify.append(rep_label)
        
        train_idx, val_idx = train_test_split(
            range(len(base_dataset)), test_size=val_ratio, random_state=seed, stratify=labels_for_stratify
        )

        # Stratify 후 디버깅 출력
        if debug:
            print(f"[DEBUG] Stratify 분리 완료: Train {len(train_idx)}개 / Val {len(val_idx)}개")

            from collections import Counter
            label_counts = Counter(labels_for_stratify)
            print(f"[DEBUG] Stratify용 레이블 전체 분포 (총 {len(label_counts)}개):")
            print("[DEBUG] 현재 카테고리 아이디 변환 이전")
            for cat_id, count in sorted(label_counts.items(), key=lambda x: x[1], reverse=True):
                print(f"  - Category ID {cat_id}: {count}개")

        
        # Custom subset 클래스
        class TransformedSubset(Dataset):
            def __init__(self, dataset, indices, transform):
                self.dataset = dataset
                self.indices = indices
                self.transform = transform
            def __getitem__(self, idx):
                img, targets = self.dataset[self.indices[idx]]
                width, height = img.size
                if self.transform:
                    img, boxes = self.transform(img, targets['boxes'])
                    boxes[:, [0, 2]] = torch.clamp(boxes[:, [0, 2]], min=0, max=width)
                    boxes[:, [1, 3]] = torch.clamp(boxes[:, [1, 3]], min=0, max=height)
                targets['boxes'] = boxes
                return img, targets
            def __len__(self):
                return len(self.indices)
        
        train_dataset = TransformedSubset(base_dataset, train_idx, transforms_train)
        val_dataset = TransformedSubset(base_dataset, val_idx, transforms_val)

        # [DEBUG 추가]
        if debug:
            print(f"[DEBUG] 랜덤시드 고정: {seed}")
            print(f"[DEBUG] Train/Val 분리: Train size = {len(train_dataset)}, Val size = {len(val_dataset)}")

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, collate_fn=collator)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False, collate_fn=collator)

        if debug:
            print(f"\n[DEBUG] train loader 배치 수: {len(train_loader)}")
            for batch_idx, batch in enumerate(train_loader):
                if batch is not None:
                    images, targets = batch
                    print(f"[DEBUG] 현재 배치 index: {batch_idx}")
                    print(f"[DEBUG] Batch size: {len(images)}")
                    print(f"[DEBUG] 첫 이미지 크기: {images[0].shape}")

                    sample_target = targets[0]
                    boxes = sample_target['boxes']
                    print(f"[DEBUG] 첫 샘플 image_id: {sample_target['image_id'].item()}")
                    print(f"[DEBUG] 박스 개수: {boxes.shape[0]}")
                    print(f"[DEBUG] 박스 크기 (W,H) 최대/최소: {boxes[:, 2:].max().item()}, {boxes[:, 2:].min().item()}")
                    print("[DEBUG] 라벨별 출현 횟수:")
                    label_counts = torch.bincount(sample_target['labels'])
                    for idx, count in enumerate(label_counts):
                        if count > 0:
                            label_name = name_to_idx and idx_to_name.get(idx, "Unknown")
                            print(f"  - {idx}: {label_name} → {count}회")
                    break
            
            print(f"\n[DEBUG] val loader 배치 수: {len(val_loader)}")
            for batch_idx, batch in enumerate(val_loader):
                if batch is not None:
                    images, targets = batch
                    print(f"[DEBUG] 현재 배치 index: {batch_idx}")
                    print(f"[DEBUG] Val Batch size: {len(images)}")
                    print(f"[DEBUG] Val 첫 이미지 크기: {images[0].shape}")
                    break


        return train_loader, val_loader
    
    # 시험의 경우
    elif mode == 'test':
        test_dataset = PillDataset(image_dir=img_dir, ann_dir=ann_dir, mode='test', category_mapping=name_to_idx, transform=transforms_test, bbox_format=bbox_format, debug=debug)

        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False, collate_fn=collator)

        # 배치 사이즈 예시
        if debug:
            print(f"\n[DEBUG] 테스트 데이터셋 배치 수: {len(test_loader)}") 
            for batch in test_loader:
                if batch is not None:
                    images, img_name = batch
                    print(f"[DEBUG] Batch size: {len(images)}")
                    print(f"[DEBUG] 첫 이미지 shape: {images[0].shape}")
                    print(f"[DEBUG] 배치 이미지 파일명 목록: {img_name}")
                break  

        return test_loader
    
    else:
        raise ValueError(f"Invalid mode: {mode}. Choose either 'train' or 'test'.")

####################################################################################################
# 5. 메인 시작
if __name__ == "__main__":
    # argparse 시작
    parser = argparse.ArgumentParser(description="PillDataset DataLoader Debug Runner")
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'val', 'test'], help="운영 모드 (train/val/test)")
    parser.add_argument('--batch_size', type=int, default=8, help="배치 크기")
    parser.add_argument('--debug', action='store_true', help="디버깅 모드 여부")
    parser.add_argument('--val_ratio', type=float, default=0.2, help="검증 데이터셋 비율 (0 ~ 1)")
    parser.add_argument('--seed', type=int, default=42, help="랜덤 시드 (재현성 보장)")

    args = parser.parse_args()

    TRAIN_ROOT = "data/train_images"
    TRAIN_ANN_DIR = "data/train_annots_modify"
    TEST_ROOT = "data/test_images"

    # 선택한 모드에 맞춰 로더 실행 및 디버깅 테스트
    if args.mode == 'train':
        train_loader, val_loader = get_loader(TRAIN_ROOT, TRAIN_ANN_DIR, batch_size=args.batch_size, mode='train', val_ratio=args.val_ratio, debug=args.debug, seed=args.seed)
        print("train loader & val loader 생성 완료.")

    elif args.mode == 'val':
        _, val_loader = get_loader(TRAIN_ROOT, TRAIN_ANN_DIR, batch_size=args.batch_size, mode='train', val_ratio=args.val_ratio, debug=args.debug, seed=args.seed)
        print("val loader 생성 완료 (train 모드 내부 val 분리 사용).")

    elif args.mode == 'test':
        test_loader = get_loader(TEST_ROOT, None, batch_size=args.batch_size, mode='test', debug=args.debug)
        print("test loader 생성 완료.")

    else:
        raise ValueError("잘못된 mode 값입니다. 'train', 'val', 'test' 중 하나를 입력하세요.")

    # 공통 출력
    print("\n========== [로더 생성 완료] ==========")
    print(f"모드: {args.mode} | 배치 사이즈: {args.batch_size} | val_ratio: {args.val_ratio}")
    if args.debug:
        print("⚠ 디버깅 모드 활성화: 데이터 통계 및 배치 정보 출력")
    print("======================================")
