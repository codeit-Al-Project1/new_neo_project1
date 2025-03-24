import src.data_utils.coco_to_yolo as cocoyo
from src.data_utils.coco_to_yolo import make_yaml_file
import os
json_dir = "data/train_annots_modify"
json_files =  [f for f in os.listdir(json_dir) if f.endswith('.json')]
for json_file_name in json_files:
    json_file_path = os.path.join(json_dir, json_file_name)
    cocoyo.convert_coco_to_yolo(json_file_path, "data/train_labels_YOLO")


# make_yaml_file("dataset", "yaml_dataset") 
# from src.train_YOLO import train_YOLO
# train_YOLO("data/train_images/train", "data/train_labels/train", batch_size=8, num_epochs=5, lr=0.001, weight_decay=0.005, optimizer_name="sgd", scheduler_name="step", device="cpu", debug=False)


