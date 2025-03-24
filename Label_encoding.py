import os
import json

image_folder = './data/train_images'  
json_folder = './data/train_annotations/'

json_list = []
for root, dirs, files in os.walk(json_folder):
    for file in files:
        if file.endswith('.json'):
            json_file_path = os.path.join(root, file)

            # JSON 파일 로드
            with open(json_file_path, 'r', encoding = 'utf-8') as f:
                data = json.load(f)
                json_list.append(data)


categories = []
for json_file in json_list:
    categories.extend(json_file["categories"])

name = set()
cat = set()
cat_dict = {}
for i in range(len(categories)):
    if categories[i]['id'] not in cat:
        cat.add(categories[i]['id'])
        name.add(categories[i]['name'])

for i in range(len(categories)):
    cat_dict[categories[i]['id']] = categories[i]['name']

sorted_dict = dict(sorted(cat_dict.items()))

label_encoding = {idx: value for idx, (key, value) in enumerate(sorted_dict.items())}
print(label_encoding)