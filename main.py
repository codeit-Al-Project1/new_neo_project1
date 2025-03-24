import argparse
import torch
from src.train_frcnn import train
from src.test_frcnn import test
from src.utils import visualization

"""
학습 실행
python main.py --mode train --batch_size 5 --epochs 30 --optimizer sgd --scheduler plateau --lr 0.001 --weight_decay 0.0005

예측 실행
python main.py --mode test --img_dir "data/test_images"  --> 기본 실행
python main.py --mode test --img_dir "data/test_images" --debug --visualization --> 디버그 + 시각화 추가
python main.py --mode test --img_dir "data/test_images" --test_batch_size 4 --threshold 0.5 --debug --visualization --> 배치 조정, 임계값 조정

- model_path: weight & bias 정보가 담긴 .pth 파일이 존재할 경우 경로 지정.
- test_batch_size: (default) 4
- threshold: (default) 0.5
- debug: 입력시 True, 아니면 False
- visualization: 입력시 True, 아니면 False
"""


def main():
    parser = argparse.ArgumentParser(description="Fast R-CNN Object Detection")
    
    # 공통 인자
    parser.add_argument("--mode", type=str, choices=["train", "test"], required=True, help="모드를 선택하세요: train 또는 predict")

    # Train 모드 인자
    parser.add_argument("--img_dir", type=str, default="data/train_images", help="이미지 폴더 경로")
    parser.add_argument("--json_path", type=str, default="data/train_annots_modify", help="어노테이션 JSON 파일 경로")
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
    parser.add_argument("--visualization", action="store_true", help="시각화 여부")

    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.mode == "train":
        train(
            img_dir=args.img_dir,
            json_dir=args.json_path,
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
        results, idx_to_name = test(
            img_dir=args.img_dir,
            device=device,
            model_path=args.model_path,
            batch_size=args.test_batch_size,
            threshold=args.threshold,
            debug=args.debug
        )

        if args.visualization:
            visualization(results, idx_to_name, page_size=20, debug=args.debug)

if __name__ == "__main__":
    main()


