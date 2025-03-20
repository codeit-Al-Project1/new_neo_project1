import csv
import numpy as np
import pandas as pd
import torch
import os

test_dir = './data/test_images'

test_file_paths = sorted(
    [os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.endswith('.png')],
    key=lambda x: int(os.path.splitext(os.path.basename(x))[0])
)

pred_example = [
    {
        'boxes': torch.tensor([[0, 1, 2, 3], [10, 20, 30, 40], [50, 60, 70, 80]], dtype=torch.float32),
        'labels': torch.tensor([1, 3, 6], dtype=torch.int64),
        'scores': torch.tensor([0.987, 0.932, 0.543], dtype=torch.float32)
    },
    {
        'boxes': torch.tensor([[5, 10, 15, 20], [25, 30, 35, 40], [45, 50, 55, 60]], dtype=torch.float32),
        'labels': torch.tensor([2, 4, 7], dtype=torch.int64),
        'scores': torch.tensor([0.876, 0.834, 0.622], dtype=torch.float32)
    },
    {
        'boxes': torch.tensor([[8, 16, 24, 32], [12, 24, 36, 48], [20, 30, 40, 50]], dtype=torch.float32),
        'labels': torch.tensor([5, 8, 9], dtype=torch.int64),
        'scores': torch.tensor([0.941, 0.789, 0.558], dtype=torch.float32)
    }
]

def submission_csv(test_file_path, predictions=None, submission_file_path=None, verbose=False):
    submission_data = []
    annotation_id = 1
    num_files = len(test_file_path)
    num_preds = len(predictions)
    
    for i, (file_path) in enumerate(test_file_path):
        image_id = file_path.split('\\')[-1].split('.png')[0]
        
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

            pred = {
                'boxes': pred['boxes'].numpy().astype(np.int32),
                'labels': pred['labels'].numpy(),
                'scores': pred['scores'].numpy()
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

            if verbose:
                print("submission first line: ", submission_data[0])
                print(bbox)
                print(score)
                print(labels)

    if submission_file_path:
        with open(submission_file_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['annotation_id', 'image_id', 'category_id', 'bbox_x', 'bbox_y', 'bbox_w', 'bbox_h', 'score'])
            for data in submission_data:
                writer.writerow(data)

    return submission_data


submission_csv(predictions=pred_example, test_file_path=test_file_paths, submission_file_path="./submission.csv", verbose=True)
