# Data_download 수정 사항

 기본 실행 방법(다운로드 및 annotation wrapping)
 - python src/data_utils/data_download.py --download --extract

 다운로드 제외실행(annotation wrapping 및 json modify)
 - python src/data_utils/data_download.py

 data는 이미 최신화 된 상태에서 json_modify 만을 실행하려는 경우.
 - python src/data_utils/data_download.py --json_modify

# Frcnn Test 실행 방법

 python main.py --mode test --img_dir "data/test_images"  --> 기본 실행

 python main.py --mode test --img_dir "data/test_images" --debug --visualization --> 디버그 + 시각화 추가

 python main.py --mode test --img_dir "data/test_images" --test_batch_size 4 --threshold 0.5 --debug --visualization --> 배치 조정, 임계값 조정

 python main.py --mode test --img_dir "data/test_images" --test_batch_size 4 --threshold 0.5 --debug --visualization --page_size 20 --page_lim None(int) --> 시각화 조정

 - model_path: weight & bias 정보가 담긴 .pth 파일이 존재할 경우 경로 지정.
 - test_batch_size: (default) 4
 - threshold: (default) 0.5
 - debug: 입력시 True, 아니면 False
 - visualization: 입력시 True, 아니면 False
 - page_size: 저장될 이미지 하나에 포함될 이미지의 개수 (default) 20
 - page_lim:  샘플링 여부 (default) None --> int 입력값 수정으로 샘플링의 양을 설정

