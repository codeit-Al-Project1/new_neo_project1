import argparse
import torch
from src.train_frcnn import train
from src.test_frcnn import test
from src.utils import visualization
from src.make_csv import submission_csv

"""
학습 실행

gpt 추천
SGD (lr=0.02, momentum=0.9, weight_decay=0.0001) + MultiStepLR (milestones=[8, 11], gamma=0.1) 🚀

1번 실험 mAP = 0.9080 (데이터 증강 전)
python main.py --mode train --backbone resnet50 --batch_size 4 --epochs 30 --optimizer sgd --scheduler plateau --lr 0.001 --weight_decay 0.0005

2번 실험
python main.py --mode train --backbone resnet50 --batch_size 4 --epochs 15 --optimizer adamw --scheduler cosine --lr 0.0001 --weight_decay 0.0001


tensorboard --logdir=tensorboard_log_dir
"models/frcnn_session_2/model_epoch=30 batch_size=4 opt=sgd sch=plateau lr=0_001 wd=0_0005.pth"

예측 실행
python main.py --mode test --img_dir "data/test_images"  --> 기본 실행
python main.py --mode test --img_dir "data/test_images" --debug --visualization --> 디버그 + 시각화 추가
python main.py --mode test --img_dir "data/test_images" --test_batch_size 4 --threshold 0.5 --debug --visualization --> 배치 조정, 임계값 조정
python main.py --mode test --img_dir "data/test_images" --test_batch_size 4 --threshold 0.5 --debug --visualization --page_size --page_lim --> 시각화 조정

python main.py --mode test --img_dir "data/test_images" --threshold 0.5 --visualization --> 추천 실행(임계값 임의 조정 필요)


- model_path: weight & bias 정보가 담긴 .pth 파일이 존재할 경우 경로 지정.
- test_batch_size: (default) 4
- threshold: (default) 0.5
- debug: 입력시 True, 아니면 False
- visualization: 입력시 True, 아니면 False
- page_size: 저장될 이미지 하나에 포함될 이미지의 개수 (default) 20
- page_lim:  샘플링 여부 (default) None --> int 입력값 수정으로 샘플링의 양을 설정
"""


def main():
    parser = argparse.ArgumentParser(description="Fast R-CNN Object Detection")
    
    # 공통 인자
    parser.add_argument("--mode", type=str, choices=["train", "test"], required=True, help="모드를 선택하세요: train 또는 predict")

    # Train 모드 인자
    parser.add_argument("--img_dir", type=str, default="data/train_images", help="이미지 폴더 경로")
    parser.add_argument("--json_path", type=str, default="data/train_annots_modify", help="어노테이션 JSON 파일 경로")
    parser.add_argument("--backbone", type=str, choices=["resnet50", "mobilenet_v3_large", "resnext101"], help="백본 모델 선택")
    parser.add_argument("--batch_size", type=int, default=16, help="배치 사이즈")
    parser.add_argument("--epochs", type=int, default=5, help="학습할 에폭 수")
    parser.add_argument("--optimizer", type=str, choices=["sgd", "adam", "adamw", "rmsprop"], default="sgd", help="옵티마이저 선택")
    parser.add_argument("--scheduler", type=str, choices=["step", "cosine", "plateau", "exponential"], default="plateau", help="스케줄러 선택")
    parser.add_argument("--lr", type=float, default=0.001, help="학습률")
    parser.add_argument("--weight_decay", type=float, default=0.0005, help="L2 정규화")
    parser.add_argument("--debug", action="store_true", help="디버그 모드 활성화")

    # Test 모드 인자
    parser.add_argument("--model_path", type=str, required=False, help="테스트할 모델 경로")
    parser.add_argument("--test_batch_size", type=int, default=4, help="테스트 배치 사이즈")
    parser.add_argument("--threshold", type=float, default=0.5, help="예측 임계값")

    # 시각화
    parser.add_argument("--visualization", action="store_true", help="시각화 여부")
    parser.add_argument("--page_size", type=int, default=20, help="한 페이지에 저장될 이미지의 수")
    parser.add_argument("--page_lim", type=int, default=None, help="저장할 페이지의 제한")

    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.mode == "train":
        print("학습을 시작합니다.")
        train(
            img_dir=args.img_dir,
            json_dir=args.json_path,
            backbone=args.backbone,
            batch_size=args.batch_size,
            num_epochs=args.epochs,
            optimizer_name=args.optimizer,
            scheduler_name=args.scheduler,
            lr=args.lr,
            weight_decay=args.weight_decay,
            device=device,
            debug=args.debug
        )

    elif args.mode == "test":
        print("이미지 예측을 시작합니다.")
        results = test(
            img_dir=args.img_dir,
            device=device,
            model_path=args.model_path,
            batch_size=args.test_batch_size,
            threshold=args.threshold,
            debug=args.debug
        )

        if args.visualization:
            print("=========" * 5)
            print("시각화 이미지를 페이지 형태로 ./data/results에 저장합니다.")
            visualization(results, 
                          page_size=args.page_size, 
                          page_lim=args.page_lim, 
                          debug=args.debug)
            # DEBUG = False(default)
            print("=========" * 5)
            print("실험 결과를 csv형식으로 저장합니다.")
            submission_csv(results, submission_file_path='./submission_frcnn.csv', debug=args.debug)

if __name__ == "__main__":
    main()