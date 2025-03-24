from ultralytics import YOLO
import cv2
import os
import glob
import torch
from src.data_utils import get_category_mapping

# 카테고리 매핑 정보 로드
ann_dir = './data/train_annots_modify'
name_to_idx, idx_to_name = get_category_mapping()


def test_yolo_model(model, test_image_path, model_path=None, categories=None, debug=False, save_results=False):
    """
    YOLO 모델 테스트 함수
    :param model: 학습된 YOLO 모델
    :param test_image_path: 테스트 이미지 경로
    :param model_path: 모델 가중치 경로 (선택)
    :param categories: 클래스 정보 (dict)
    :param debug: 디버그 모드 활성화 (True/False)
    :param save_results: 예측된 이미지를 저장할지 여부 (True/False)
    :return: 예측 결과 리스트
    """
    # 모델을 평가 모드로 전환
    model.eval()
    total_predictions = []

    # 클래스 수가 지정되면 모델의 nc와 names 속성 수정
    if categories:
        model.model.nc = len(categories)
        model.model.names = list(categories.values())  # 클래스 이름 초기화

    # 이미지 경로 탐색 (다양한 포맷 대응)
    image_paths = glob.glob(os.path.join(test_image_path, '*.[pjPj][pnPN][gG]'))

    if not image_paths:
        raise FileNotFoundError(f"이미지를 찾을 수 없습니다: {test_image_path}")

    # 결과 저장 디렉토리 설정 (선택적)
    result_dir = './data/test_results'
    if save_results and not os.path.exists(result_dir):
        os.makedirs(result_dir)

    with torch.no_grad():
        for idx, image_path in enumerate(image_paths):
            img = cv2.imread(image_path)
            if img is None:
                print(f"이미지를 로드하지 못했습니다: {image_path}")
                continue

            # YOLO 모델 예측
            results = model(img, verbose=debug)  # 각 이미지 예측마다 출력되는 정보를 보여줄지 여부
            total_predictions.append(results)

            # 디버그 모드에서 예측 결과 출력 및 시각화
            if debug:
                print(f"[{idx + 1}/{len(image_paths)}] {image_path} 예측 결과:")

                if isinstance(results, list):
                    for res in results:
                        if hasattr(res, 'pandas'):
                            try:
                                boxes = res.pandas().xyxy[0]  # pandas DataFrame으로 변환
                                print(boxes[['xmin', 'ymin', 'xmax', 'ymax', 'confidence', 'class', 'name']])
                            except Exception as e:
                                print(f"결과 출력 오류: {e}")
                else:
                    try:
                        boxes = results.pandas().xyxy[0]  # pandas DataFrame으로 변환
                        print(boxes[['xmin', 'ymin', 'xmax', 'ymax', 'confidence', 'class', 'name']])
                    except Exception as e:
                        print(f"결과 출력 오류: {e}")

                # 이미지 시각화
                if idx % 100 == 0:
                    if isinstance(results, list):
                        for res in results:
                            if hasattr(res, 'show'):
                                res.show()
                    else:
                        if hasattr(results, 'show'):
                            results.show()

            # 결과 이미지 저장 (선택적)
            if save_results:
                save_path = os.path.join(result_dir, f"result_{idx + 1}.jpg")
                if isinstance(results, list):
                    for res in results:
                        if hasattr(res, 'save'):
                            res.save(save_path)
                else:
                    if hasattr(results, 'save'):
                        results.save(save_path)
                print(f"결과 이미지 저장 완료: {save_path}")

    return total_predictions

# 모델과 테스트할 이미지 경로 설정
image_paths = "./data/test_images"  # 테스트할 이미지 경로
model_path = "yolov5su.pt"  # 모델 파일 경로


categories = {0: '보령부스파정 5mg', 1: '뮤테란캡슐 100mg', 2: '일양하이트린정 2mg', 3: '기넥신에프정(은행엽엑스)(수출용)', 4: '무코스타정(레바미피드)(비매품)', 5: '알드린정', 
              6: '뉴로메드정(옥시라세탐)', 7: '타이레놀정500mg', 8: '에어탈정(아세클로페낙)', 9: '삼남건조수산화알루미늄겔정', 10: '타이레놀이알서방정(아세트아미노펜)(수출용)', 
              11: '삐콤씨에프정 618.6mg/병', 12: '조인스정 200mg', 13: '쎄로켈정 100mg', 14: '넥시움정 40mg', 15: '리렉스펜정 300mg/PTP', 16: '아빌리파이정 10mg', 17: '자이프렉사정 2.5mg', 
              18: '다보타민큐정 10mg/병', 19: '써스펜8시간이알서방정 650mg', 20: '에빅사정(메만틴염산염)(비매품)', 21: '리피토정 20mg', 22: '크레스토정 20mg', 23: '가바토파정 100mg', 
              24: '동아가바펜 틴정 800mg', 25: '오마코연질캡슐(오메가-3-산에틸에스테르90)', 26: '란스톤엘에프디티정 30mg', 27: '리리카캡슐 150mg', 28: '종근당글리아티린연질캡슐(콜린알포세레이트)\xa0', 
              29: '콜리네이트연질캡슐 400mg', 30: '트루비타정 60mg/병', 31: '스토가정 10mg', 32: '노바스크정 5mg', 33: '마도파정', 34: '플라빅스정 75mg', 35: '엑스포지정 5/160mg', 
              36: '펠루비정(펠루비프로펜)', 37: '아토르바정 10mg', 38: '라비에트정 20mg', 39: '리피로우정 20mg', 40: '자누비아정 50mg', 41: '맥시부펜이알정 300mg', 42: '메가파워정 90mg/병', 
              43: '쿠에타핀정 25mg', 44: '비타비백정 100mg/병', 45: '놀텍정 10mg', 46: '자누메트정 50/850mg', 47: '큐시드정 31.5mg/PTP', 48: '아모잘탄정 5/100mg', 49: '세비카정 10/40mg', 
              50: '트윈스타정 40/5mg', 51: '카나브정 60mg', 52: '울트라셋이알서방정', 53: '졸로푸트정 100mg', 54: '트라젠타정(리나글립틴)', 55: '비모보정 500/20mg', 56: '레일라정', 57: '리바로정 4mg', 
              58: '렉사프로정 15mg', 59: '트라젠타듀오정 2.5/850mg', 60: '낙소졸정 500/20mg', 61: '아질렉트정(라사길린메실산염)', 62: '자누메트엑스알서방정 100/1000mg', 63: '글리아타민연질캡슐', 
              64: '신바로정', 65: '에스원엠프정 20mg', 66: '브린텔릭스정 20mg', 67: '글리틴정(콜린알포세 레이트)', 68: '제미메트서방정 50/1000mg', 69: '아토젯정 10/40mg', 70: '로수젯정10/5밀리그램', 
              71: '로수바미브정 10/20mg', 72: '카발린캡슐 25mg', 73: '케이캡정 50mg'}


# YOLO 모델 로드
device = "cuda" if torch.cuda.is_available() else "cpu"
model = YOLO(model_path)
model.to(device)

# YOLO 모델 테스트 실행
results = test_yolo_model(model, test_image_path=image_paths, model_path=model_path, categories=categories, debug=True, save_results=True)

# 결과 저장 리스트 초기화
submission = []

# 결과 파싱
for res in results:
    names = res.names if hasattr(res, 'names') else []
    
    # Check if res is a list
    if isinstance(res, list):
        # If res is a list, iterate over each item and extract boxes and confidences
        for item in res:
            if hasattr(item, 'boxes'):
                boxes = item.boxes.xywh.cpu().numpy() if hasattr(item.boxes, 'xywh') else []
                confs = item.boxes.conf.cpu().numpy() if hasattr(item.boxes, 'conf') else []
                names = item.names if hasattr(item, 'names') else []
                
                # Add the parsed data to submission
                for name, box, conf in zip(names, boxes, confs):
                    submission.append({
                        "name": name,
                        "box": box.tolist(),  # numpy 배열을 리스트로 변환
                        "conf": float(conf)  # numpy 값을 float으로 변환
                    })
    else:
        # If res is a single prediction object, process it directly
        if hasattr(res, 'boxes'):
            boxes = res.boxes.xywh.cpu().numpy() if hasattr(res.boxes, 'xywh') else []
            confs = res.boxes.conf.cpu().numpy() if hasattr(res.boxes, 'conf') else []
            names = res.names if hasattr(res, 'names') else []

            # Add the parsed data to submission
            for name, box, conf in zip(names, boxes, confs):
                submission.append({
                    "name": name,
                    "box": box.tolist(),  # numpy 배열을 리스트로 변환
                    "conf": float(conf)  # numpy 값을 float으로 변환
                })

# Print the first submission entry to check
print(submission[0])