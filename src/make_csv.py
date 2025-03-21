import csv
import numpy as np
import torch
import os
import argparse

test_dir = './data/test_images'

def submission_csv(predictions, test_dir, submission_file_path=None, debug=False):
    test_dir = sorted(
    [os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.endswith('.png')],
    key=lambda x: int(os.path.splitext(os.path.basename(x))[0])
    )
    print(test_dir)
    
    submission_data = []
    annotation_id = 1
    num_files = len(test_dir)
    num_preds = len(predictions)

    print(num_files, num_preds)

    if num_files != num_preds:
        print("Test 이미지와 Prediction의 크기가 맞지 않습니다.")
    
    for i, (file_path) in enumerate(test_dir):
        image_id = file_path.split('/')[-1].split('.png')[0]
        

        if predictions is not None:
            try:
                idx = int(image_id) - 1
            except ValueError:
                print(f"Invalid image_id: {image_id}")
                continue

            if idx >= num_preds or idx < 0:  # 인덱스가 유효한지 체크
                print(f"Prediction does not exist for image_id {image_id}")
                continue
            pred = predictions[idx]

            if i == 0:
                print(pred)

            # pred['boxes'], pred['labels'], pred['scores']가 tuple로 되어 있으므로 이를 numpy 배열로 변환
            pred = {
                'boxes': np.array(pred['boxes'], dtype=np.int32),  # tuple을 numpy 배열로 변환
                'labels': np.array(pred['category_id'], dtype=np.int32),  # tuple을 numpy 배열로 변환
                'scores': np.array(pred['scores'], dtype=np.float32)  # tuple을 numpy 배열로 변환
            }


            bbox = pred['boxes']
            score = pred['scores']
            labels = pred['labels']

            if len(bbox) != 1:
                for j in range(len(bbox)):
                    submission_data.append([annotation_id, image_id, labels[j], bbox[j][0], bbox[j][1], bbox[j][2], bbox[j][3], score[j]])
                    annotation_id += 1
            
            else:
                submission_data.append([annotation_id, image_id, labels[0], bbox[0][0], bbox[0][1], bbox[0][2], bbox[0][3], score[0]])
                annotation_id += 1

    if submission_file_path:
        with open(submission_file_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['annotation_id', 'image_id', 'category_id', 'bbox_x', 'bbox_y', 'bbox_w', 'bbox_h', 'score'])
            for data in submission_data:
                writer.writerow(data)

    return submission_data

# submission_csv(predictions, args.test_dir, args.submission_file_path, args.debug)
