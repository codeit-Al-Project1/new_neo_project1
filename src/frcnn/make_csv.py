import csv
import numpy as np
from src.utils import get_category_mapping

def submission_csv(predictions, submission_file_path=None, debug=False):
    submission_data = []
    annotation_id = 1

    idx_to_id = get_category_mapping(ann_dir="data/train_annots_modify", add_more=True, debug=False, return_types=['idx_to_id'])

    for i in range(len(predictions)):
        pred = predictions[i]

        # 예측 결과가 딕셔너리인 경우 처리
        if isinstance(pred['category_id'], dict):
            pred = {
                'id': np.array(list(pred.get('image_id', {}).values()), dtype=np.int32),
                'boxes': np.array(list(pred.get('boxes', {}).values()), dtype=np.float32),
                'labels': np.array(list(pred.get('category_id', {}).values()), dtype=np.int32),
                'scores': np.array(list(pred.get('scores', {}).values()), dtype=np.float32)
            }
        else:
            pred = {
                'id': np.array(pred.get('image_id', []), dtype=np.int32),
                'boxes': np.array(pred.get('boxes', []), dtype=np.float32),
                'labels': np.array(pred.get('category_id', []), dtype=np.int32),
                'scores': np.array(pred.get('scores', []), dtype=np.float32)
            }

        if debug:
            print(f"[{pred['id']}] 예측 결과 - Boxes: {pred['boxes']}, Labels: {pred['labels']}, Scores: {pred['scores']}")
        image_id = pred['id']
        bbox = pred['boxes']
        labels = pred['labels']
        scores = pred['scores']

        for j in range(len(bbox)):
            submission_data.append([
                annotation_id,                # annotation_id (순차적인 인덱스 넘버)
                image_id,                     # image_id (이미지 파일명)
                idx_to_id[labels[j]],                    # category_id (예측한 클래스)
                bbox[j][0],                   # x_min
                bbox[j][1],                   # y_min
                bbox[j][2] - bbox[j][0],                   # w = x_max - x_min
                bbox[j][3] - bbox[j][1],                   # h = y_max - y_min
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