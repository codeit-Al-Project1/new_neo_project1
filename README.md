# 객체 탐지 실험 (알약)
 개요: 이미지에 포함된 알약을 탐지/ 분류하는 딥러닝 모델 실험.

 ## 진행 순서:

 1. 데이터 다운로드 및 전처리
    - 데이터 다운로드
    - Annotation 최신화
    - Annotation 형태 변환
    ***Note: 캐글 API를 이용한 다운로드 입니다. kaggle.json 파일을 케글에서 내려받아 .kaggle/폴더에 위치시켜야만 다운로드가 진행됩니다.***

 2. Faster R-CNN 실험:
    - 실행 코드:
    학습:
    python main.py --model frcnn --mode train --img_dir data/train_images --json_path data/train_annots_modify --backbone mobilenet_v3_large --batch_size 8 --epochs 25 --optimizer_name adamw --scheduler_name plateau --lr 1e-4 --weight_decay 5e-4 --iou_threshold 0.3 --conf_threshold 0.7

    결과 확인:
    python main.py --model frcnn --mode test --img_dir data/test_images --model_path models/frcnn_session_1/best_model_lr=0.0001_ep=1_bs=8_opt=adamw_scd=plateau_wd=0.0005.pth --backbone mobilenet_v3_large --threshold 0.5 --visualization --page_size 12 --page_lim 5
    Arguments Description:
    🔸 공통 옵션
        --model               : 사용할 모델 선택 ['frcnn', 'yolo'] (필수)
        --mode                : 실행 모드 선택 ['train', 'test', 'val'] (필수)
        --img_dir             : 입력 이미지가 있는 디렉토리 (train/test 공통)
        --model_path          : 사전 학습된 모델 경로 (.pth or .pt)
        --batch_size          : 학습 배치 크기
        --epochs              : 학습 반복 횟수
        --lr                  : 초기 학습률
        --weight_decay        : L2 정규화 (weight decay)
        --device              : 사용할 디바이스 ['cuda', 'cpu'], 기본값: 자동 선택
        --debug               : 디버그 출력 활성화

    ------------------------------------------------------------------------------------
    🔸 FRCNN 전용 옵션
        --json_path           : 어노테이션 JSON 디렉토리 (default: data/train_annots_modify)
        --backbone            : 백본 모델 선택 ['resnet50', 'mobilenet_v3_large', 'resnext101']
        --optimizer_name      : 옵티마이저 종류 ['sgd', 'adam', 'adamw', 'rmsprop']
        --scheduler_name      : 러닝레이트 스케줄러 ['step', 'cosine', 'plateau', 'exponential']
        --test_batch_size     : 테스트용 배치 크기 (default: 4)
        --threshold           : confidence 임계값 (default: 0.5)
        --visualization       : 시각화 이미지 및 CSV 파일 저장 여부
        --page_size           : 시각화 시 한 페이지당 이미지 수 (default: 12) --> 가독성 상향을 위한 조정 20 -> 12
        --page_lim            : 시각화 페이지 수 제한 (default: None, 전체 시각화)
        --iou_threshold       : IoU 임계값 (default: 0.3) --> 배경인식을 위한 조정 0.5 -> 0.3
        --conf_threshold      : confidence 임계값 (default: 0.5)  -->

 3. YOLO 모델 실험:
    - 실행 코드:
    # ▶ YOLOv8 학습
    python main.py --model yolo --mode train --img_dir data/train_labels/train --yaml_path data/train_labels/data.yaml --model_variant n --batch_size 8 --epochs 100 --lr 0.001 --weight_decay 0.0005

    # ▶ YOLOv8 검증 (validation)
    python main.py --model yolo --mode val --val_model_path runs/detect/yolov8n_custom/weights/best.pt

    # ▶ YOLOv8 테스트 (결과 이미지 저장 및 CSV 저장)
    python main.py --model yolo --mode test --model_path runs/detect/yolov8n_custom/weights/best.pt --img_dir data/test_images --save_images --save_csv_path submission_yolo.csv --conf_threshold 0.5 --iou_threshold 0.7

    🔸 YOLO 전용 옵션
        --yaml_path           : YOLO 학습 시 사용할 data.yaml 경로
        --model_variant       : YOLOv8 크기 선택 ['n', 's', 'm', 'l']
        --patience            : 조기 종료 patience (default: 100)
        --optimizer           : YOLO 전용 옵티마이저 ['auto', 'SGD', 'Adam', 'AdamW', 'RMSProp' 등]
        --resume              : 학습 재시작 여부
        --val_model_path      : 검증 시 사용할 .pt 파일 경로
        --conf_threshold      : confidence threshold (default: 0.5)
        --iou_threshold       : NMS IoU threshold (default: 0.7)
        --save_images         : 예측 이미지 저장 여부 (YOLO test)
        --save_csv_path       : YOLO 테스트 결과를 저장할 CSV 경로