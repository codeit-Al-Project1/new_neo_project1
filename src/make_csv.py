import csv
import numpy as np
import torch
import os
import argparse

test_dir = './data/test_images'

def submission_csv(predictions, test_dir, submission_file_path=None, YOLO=True, debug=False):
    # 테스트 이미지 파일 경로 가져오기 및 정렬
    test_dir = sorted(
        [os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.endswith('.png')],
        key=lambda x: int(os.path.splitext(os.path.basename(x))[0])  # 파일명에서 숫자만 추출 후 정렬
    )

    if debug:
        print("테스트 이미지 경로 리스트:", test_dir)

    submission_data = []
    annotation_id = 1
    num_files = len(test_dir)
    num_preds = len(predictions)

    if num_files != num_preds:
        print(f"Warning: 테스트 이미지 수({num_files})와 예측 결과 수({num_preds})가 일치하지 않습니다.")
        
    for i, file_path in enumerate(test_dir):
        image_id = os.path.splitext(os.path.basename(file_path))[0]  # 파일명에서 확장자 제거

        if predictions is None or i >= num_preds:
            print(f"Prediction이 없습니다: {image_id}")
            continue

        pred = predictions[i]

        # 예측 결과가 딕셔너리인 경우 처리
        if isinstance(pred['category_id'], dict):
            pred = {
                'boxes': np.array(list(pred.get('boxes', {}).values()), dtype=np.float32),
                'labels': np.array(list(pred.get('category_id', {}).values()), dtype=np.int32),
                'scores': np.array(list(pred.get('scores', {}).values()), dtype=np.float32)
            }
        else:
            pred = {
                'boxes': np.array(pred.get('boxes', []), dtype=np.float32),
                'labels': np.array(pred.get('category_id', []), dtype=np.int32),
                'scores': np.array(pred.get('scores', []), dtype=np.float32)
            }

        if debug:
            print(f"[{image_id}] 예측 결과 - Boxes: {pred['boxes']}, Labels: {pred['labels']}, Scores: {pred['scores']}")

        bbox = pred['boxes']
        labels = pred['labels']
        scores = pred['scores']

        # YOLO 형식의 바운딩 박스를 COCO 형식으로 변환: (x_center, y_center, w, h) -> (x, y, w, h)
        if YOLO and len(bbox) > 0:
            bbox[:, 0] = bbox[:, 0] - (bbox[:, 2] / 2)  # x_center -> x
            bbox[:, 1] = bbox[:, 1] - (bbox[:, 3] / 2)  # y_center -> y

        for j in range(len(bbox)):
            submission_data.append([
                annotation_id,                # annotation_id (순차적인 인덱스 넘버)
                image_id,                     # image_id (이미지 파일명)
                labels[j],                    # category_id (예측한 클래스)
                bbox[j][0],                   # bbox_x
                bbox[j][1],                   # bbox_y
                bbox[j][2],                   # bbox_w
                bbox[j][3],                   # bbox_h
                scores[j]                     # score (신뢰도)
            ])
            annotation_id += 1

    # CSV 파일 저장
    if submission_file_path:
        with open(submission_file_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['annotation_id', 'image_id', 'category_id', 'bbox_x', 'bbox_y', 'bbox_w', 'bbox_h', 'score'])
            writer.writerows(submission_data)

    return submission_data

# submission_csv(predictions, args.test_dir, args.submission_file_path, args.debug)
