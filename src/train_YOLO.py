import torch
from tqdm import tqdm
from utils import get_optimizer, get_scheduler  # utils.py에서 가져오기
from datasetYOLO import YOLODataset  # YOLO 데이터셋 로드
from models.yolo import Model  # YOLO 모델
from utils.torch_utils import select_device
from utils.general import increment_path
from utils.loss import ComputeLoss  # 손실 함수

def train_YOLO(img_dir, ann_dir, batch_size=16, num_classes=80, num_epochs=5, lr=0.001, optimizer_name="sgd", scheduler_name="step", device="cuda"):
    # 디바이스 설정 (GPU 또는 CPU)
    device = select_device(device)

    # 데이터셋 로딩
    dataset = YOLODataset(img_dir=img_dir, ann_dir=ann_dir, mode='train')
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # YOLO 모델 정의
    model = Model(cfg='yolov5s.yaml', ch=3, nc=num_classes).to(device)  # YOLOv5 모델 로드

    # 옵티마이저 정의
    optimizer = get_optimizer(optimizer_name, model, lr=lr, weight_decay=0.0005)

    # 손실 함수 정의
    compute_loss = ComputeLoss(model)

    # 학습률 스케줄러 정의
    scheduler = get_scheduler(scheduler_name, optimizer)

    best_val_loss = float("inf")

    # 학습 루프
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        # 학습 단계
        for batch_i, (imgs, targets) in tqdm(enumerate(train_loader), total=len(train_loader), desc="Training"):
            imgs = imgs.to(device)
            targets = targets.to(device)

            # 모델 학습
            optimizer.zero_grad()
            loss_items = model(imgs, targets)  # 모델을 통해 학습
            loss = loss_items[0]  # 첫 번째 항목이 손실
            loss.backward()

            optimizer.step()
            total_loss += loss.item()

        # 학습 후 스케줄러 업데이트
        scheduler.step()

        # 2. 검증 단계
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for imgs, targets in tqdm(train_loader, total=len(train_loader), desc="Validation"):
                imgs = imgs.to(device)
                targets = targets.to(device)

                loss_items = model(imgs, targets)
                loss = loss_items[0]
                val_loss += loss.item()

        # 검증 손실 개선 시 모델 저장
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(f"Validation Loss decreased to {best_val_loss:.4f}. Saving model...")
            save_model(model, epoch)  # 모델 저장 함수 호출

def save_model(model, epoch):
    save_path = increment_path('models/weights', exist_ok=True)  # 경로 생성
    torch.save(model.state_dict(), f"{save_path}/yolov5_epoch_{epoch}.pt")
    print(f"Model saved to {save_path}/yolov5_epoch_{epoch}.pt")

if __name__ == "__main__":
    train_YOLO(img_dir="data/train_images", ann_dir="data/train_annots_modify")
