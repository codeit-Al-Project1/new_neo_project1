import torch
from tqdm import tqdm
from src.data_utils.data_loader import get_loader, get_category_mapping
from utils import get_optimizer, get_scheduler  # utils.py에서 가져오기
from src.model_utils.basic_YOLO import get_yolov5  # YOLO 모델
from ultralytics.utils.loss import ComputeLoss


def train_YOLO(img_dir, ann_dir, batch_size=8, num_epochs=5, lr=0.001, weight_decay=0.005, optimizer_name="sgd", scheduler_name="step", device="cpu", debug=False):
    # 데이터 로더 
    train_loader = get_loader(img_dir, ann_dir, batch_size, mode="train", val_ratio=0.2, debug=debug)
    val_loader = get_loader(img_dir, ann_dir, batch_size, mode="val", val_ratio=0.2, debug=debug)

    # 어노테이션 디렉토리를 기준으로 카테고리 매핑 가져오기
    name_to_idx, idx_to_name = get_category_mapping(ann_dir)

    # 클래스 개수는 카테고리 길이로 설정
    num_classes = len(name_to_idx)

    # YOLO 모델 정의
    model = get_yolov5(model_path="yolov5s.pt", num_classes=num_classes).to(device)

    # 옵티마이저, 스케쥴러, 로스 정의
    optimizer = get_optimizer(optimizer_name, model, lr, weight_decay)
    scheduler = get_scheduler(scheduler_name, optimizer, T_max=100)
    compute_loss = ComputeLoss(model)

    best_val_loss = float("inf")

    # 학습 루프
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        train_bar = tqdm(train_loader, total=len(train_loader), desc="Training")
        # 학습 단계
        for imgs, targets in train_bar:
            imgs = imgs.to(device)

#############################################################################################
            # 데이터셋 타겟은
            # targets = {
            #     'boxes': bboxes_tensor,
            #     'labels': labels_tensor,
            #     'image_id': image_id_tensor,
            #     'area': areas_tensor,      \
            #     'is_crowd': iscrowd_tensor,
            #     'orig_size': orig_size_tensor,
            #     'pill_names': pill_names
            # }
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
#############################################################################################
            # 모델 타겟과 어느정도 일치하는지 확인 필요
            preds = model(imgs)
            loss, loss_items = compute_loss(preds, targets)

            # 모델 학습
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            loss.backward()

            optimizer.step()
            total_loss += loss.item()

        # 학습 후 스케줄러 업데이트
        scheduler.step()

        # 2. 검증 단계
        model.eval()
        val_loss = 0
        with torch.no_grad():
            val_bar = tqdm(train_loader, total=len(train_loader), desc="Validation")
            for imgs, targets in val_bar:
                imgs = imgs.to(device)
#############################################################################################
                # 데이터셋 타겟 확인 필요
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                preds = model(imgs, targets)
                loss, _ = compute_loss(preds, targets)
                val_loss += loss.item()

        print(f"{total_loss}, {val_loss}")
        
        if scheduler_name == "plateau":
            scheduler(val_loss)
        else:
            scheduler()

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
