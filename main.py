import argparse
import torch
from src.train_frcnn import train
from src.predict_frcnn import predict

"""
학습 실행
python main.py --mode train --batch_size 16 --epochs 3 --optimizer sgd --scheduler plateau --lr 0.001 --weight_decay 0.0005

예측 실행
python main.py --mode eval --image_path data/sample.jpg

"""


def main():
    parser = argparse.ArgumentParser(description="Fast R-CNN Object Detection")
    
    # 공통 인자
    parser.add_argument("--mode", type=str, choices=["train", "predict"], required=True, help="모드를 선택하세요: train 또는 predict")

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

    # Predict 모드 인자
    parser.add_argument("--model_path", type=str, default="models/fast_rcnn.pth", help="저장된 모델 경로")
    parser.add_argument("--image_path", type=str, help="예측할 이미지 경로 (predict 모드에서 필요)")

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

    elif args.mode == "predict":
        if not args.image_path:
            print("predict 모드에서는 --image_path를 지정해야 합니다.")
            return
        if not args.model_path:
            print("predict 모드에서는 --model_path를 지정해야 합니다.")
            return

        predict(
            image_path=args.image_path,
            model_path=args.model_path,
            device=args.device
        )

if __name__ == "__main__":
    main()

