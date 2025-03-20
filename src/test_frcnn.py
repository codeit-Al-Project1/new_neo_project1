import torch
import torchvision
from tqdm import tqdm
import json
import os
import argparse
import pandas
from src.data_utils.data_loader import get_loader, get_category_mapping
from src.model_utils.basic_frcnn import get_fast_rcnn_model

#############################################################################################
# XYXY -> XYWH 형식 포맷
def xyxy_to_xywh(boxes):
    """
    바운딩 박스 좌표를 XYXY 형식에서 XYWH 형식으로 변환합니다.

    Args:
        boxes (list): 각 요소가 [x_min, y_min, x_max, y_max] 형태의 박스 좌표 리스트

    Returns:
        list: 변환된 [x, y, width, height] 형태의 박스 좌표 리스트
    """
    return [[x1, y1, x2 - x1, y2 - y1] for x1, y1, x2, y2 in boxes]



#############################################################################################
# 테스트 함수
def test(img_dir, device='cpu', model_path=None, batch_size=8, threshold=0.5, debug=False):
    """
    테스트 함수 (test)

    지정한 이미지 디렉토리에 대해 학습된 Fast R-CNN 모델을 사용하여 추론을 수행하고  
    이미지별 예측 결과(바운딩 박스 좌표, 클래스 레이블, confidence score)를 반환합니다.

    Args:
        img_dir (str): 테스트 이미지가 저장된 디렉토리 경로  
        device (str): 추론에 사용할 디바이스 ('cuda' 또는 'cpu')  
        model_path (str, optional): 학습된 모델의 가중치 파일 경로 (지정하지 않으면 랜덤 초기화된 모델 사용)  
        batch_size (int, optional): 배치 크기 (기본값: 8)  
        threshold (float, optional): confidence score 필터링 기준 값 (기본값: 0.5 이상만 포함)  
        debug (bool, optional): 디버깅 모드 활성화 여부 (기본값: False)  

    Returns:
        list: 각 이미지별 추론 결과 리스트.  
            각 요소는 딕셔너리 형태로 다음 정보를 포함:  
            - 'image_id': 이미지 파일명에서 확장자를 제거한 ID  
            - 'boxes': 예측된 바운딩 박스 좌표 목록 (형식: [x, y, w, h])  
            - 'category_id': 예측된 클래스 인덱스 목록  
            - 'category_name': 클래스 인덱스를 카테고리 이름으로 변환한 리스트  
            - 'scores': 각 예측에 대한 confidence score 목록  
            - 'file_name': 원본 파일명 (확장자 포함)  
    """
    # 입력값 타입 및 유효성 검사
    if not isinstance(img_dir, str):
        raise TypeError(f"img_dir는 str이어야 합니다. 현재 타입: {type(img_dir)}")
    if not os.path.exists(img_dir):
        raise FileNotFoundError(f"img_dir 경로가 존재하지 않습니다: {img_dir}")

    if not isinstance(device, str):
        raise TypeError(f"device는 str이어야 합니다. 현재 타입: {type(device)}")
    if device not in ["cuda", "cpu"]:
        raise ValueError(f"device는 'cuda' 또는 'cpu'이어야 합니다. 입력값: {device}")

    if model_path is not None and not isinstance(model_path, str):
        raise TypeError(f"model_path는 str 또는 None이어야 합니다. 현재 타입: {type(model_path)}")
    if model_path and not os.path.exists(model_path):
        raise FileNotFoundError(f"model_path 경로가 존재하지 않습니다: {model_path}")

    if not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError(f"batch_size는 양의 정수여야 합니다. 입력값: {batch_size}")

    if not isinstance(threshold, float) and not isinstance(threshold, int):
        raise TypeError(f"threshold는 float 또는 int여야 합니다. 현재 타입: {type(threshold)}")
    if not (0 <= threshold <= 1):
        raise ValueError(f"threshold는 0과 1 사이의 값이어야 합니다. 입력값: {threshold}")

    if not isinstance(debug, bool):
        raise TypeError(f"debug는 bool 타입이어야 합니다. 현재 타입: {type(debug)}")


    # 어노테이션 디렉토리를 기준으로 카테고리 매핑 가져오기
    ANN_DIR = "data/train_annots_modify"
    name_to_idx, idx_to_name = get_category_mapping(ANN_DIR)

    # 클래스 개수는 카테고리 길이로 설정
    num_classes = len(name_to_idx)

    # 모델 가져오기
    model = get_fast_rcnn_model(num_classes=num_classes)

    # 모델 패스 가져오기
    if model_path:
        model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # 테스트 로더
    test_loader = get_loader(img_dir=img_dir, batch_size=batch_size, mode='test', debug=debug)

    # 결과물 리스트
    results = []

    with torch.no_grad():
        progress_bar = tqdm(test_loader, total=len(test_loader), desc='Testing', dynamic_ncols=True)
        for images, file_names in progress_bar:
            images = [img.to(device) for img in images]
            outputs = model(images)     # boxes, labels, scores

            for file_name, output in zip(file_names, outputs):
                # cpu로 결과값 저장
                boxes = output['boxes'].cpu().tolist()
                labels = output['labels'].cpu().tolist()
                scores = output['scores'].cpu().tolist()

                # 스코어 필터링(confidence score >= threshold)
                filtered = [(b, l, s) for b, l, s in zip(boxes, labels, scores) if s >= threshold]
                boxes, labels, scores = zip(*filtered) if filtered else ([], [], [])

                # xyxy -> xywh 변환
                boxes = xyxy_to_xywh(boxes) if boxes else []

                # 파일네임에서 .png 제거
                image_id = os.path.splitext(file_name)[0]   # 확장자 제거

                # 카테고리 이름 매핑
                category_names = [idx_to_name[label] for label in labels]

                # 결과값 저장
                results.append(
                    {
                        'image_id': image_id,
                        'boxes': boxes,
                        'category_id': labels,
                        'category_name': category_names,
                        'scores': scores,
                        'file_name': file_name
                    }
                )
    
    return results


#############################################################################################
# main 실행
if __name__ == '__main__':
    
    TEST_DIR = "data/test_images"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    results = test(img_dir=TEST_DIR, device=device, threshold=0.5)
    print(results)