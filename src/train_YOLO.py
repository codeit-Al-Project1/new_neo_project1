#########################################################################
# git clone https://github.com/ultralytics/yolov5.git
# cd yolov5
# pip install -r requirements.txt
#########################################################################

import torch
from tqdm import tqdm
import os
from src.data_utils.data_loader import get_loader, get_category_mapping
from src.utils import get_optimizer, get_scheduler  # utils.py에서 가져오기
# from src.model_utils.basic_YOLO import get_yolov5  # YOLO 모델
# from ultralytics.yolo.utils.loss import ComputeLoss - 이 놈이 너무 문제여서 git clone으로 가져옴
############################################# 추가
import sys
sys.path.append("yolov5")  # YOLOv5 폴더 경로 추가
from yolov5.utils.loss import ComputeLoss  # YOLOv5 공식 코드에서 직접 가져오기
from yolov5.models.yolo import Model  # YOLO 모델 로드
from src.model_utils.basic_YOLO import get_yolov5
import yaml

def train_YOLO(img_dir, ann_dir, batch_size=8, num_epochs=5, lr=0.001, weight_decay=0.005, optimizer_name="sgd", scheduler_name="step", device="cpu", debug=False):
    # 데이터 로더 
    train_loader = get_loader("data/train_images/train", "data/train_labels/train", batch_size, mode="train", debug=debug)
    val_loader = get_loader("data/train_images/val", "data/train_labels/val", batch_size, mode="val", debug=debug)

    # 어노테이션 디렉토리를 기준으로 카테고리 매핑 가져오기
    name_to_idx, idx_to_name = get_category_mapping(ann_dir)

    # 클래스 개수는 카테고리 길이로 설정
    num_classes = len(name_to_idx)

    # YOLO 모델 정의
    model = get_yolov5(model_path="yolov5s.pt", num_classes=num_classes).to(device)
    # model = Model("yolov5/models/yolov5s.yaml")  # YOLOv5s 모델 YAML 사용

    # 모델에서의 yaml 파일은 모델의 구조를 정의하는 설정 파일
    model.nc = num_classes  # 클래스 수 설정
    model.to(device)

    # 옵티마이저, 스케쥴러, 로스 정의
    optimizer = get_optimizer(optimizer_name, model, lr, weight_decay)
    scheduler = get_scheduler(scheduler_name, optimizer, T_max=100)

    compute_loss = ComputeLoss(model.model)

    best_val_loss = float("inf")

    # 학습 루프
    for epoch in range(num_epochs):
        print("학습 시작")
        model.train()
        total_loss = 0

        train_bar = tqdm(train_loader, total=len(train_loader), desc=f"🟢 Training {epoch+1}/{num_epochs}")
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
            optimizer.step()
            total_loss += loss.item()
            train_bar.set_postfix(loss=loss.item())  # 진행 상태 표시
        
        # 학습 후 스케줄러 업데이트
        if scheduler_name == "plateau":
            scheduler.step(total_loss)  # ReduceLROnPlateau는 loss를 인자로 받음
        else:
            scheduler.step()

        # 2. 검증 단계
        model.eval()
        val_loss = 0
        with torch.no_grad():
            val_bar = tqdm(val_loader, total=len(val_loader), desc=f"🔵 Validation {epoch+1}/{num_epochs}")
            for imgs, targets in val_bar:
                imgs = imgs.to(device)
#############################################################################################
                # 데이터셋 타겟 확인 필요
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                preds = model(imgs)
                loss, _ = compute_loss(preds, targets)
                val_loss += loss.item()
                val_bar.set_postfix(val_loss=loss.item())

        print(f"Epoch {epoch+1} - Train Loss: {total_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        if scheduler_name == "plateau":
            scheduler(val_loss)
        else:
            scheduler()

        # ✅ 최적 모델 저장
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(f"검증 손실 개선됨.. 모델 저장 중... (Best Val Loss: {best_val_loss:.4f})")
            save_model(model, epoch, best_val_loss)  # 모델 저장

def save_model(model, epoch, val_loss):
    """ 모델 가중치 저장 함수 """
    save_dir = "models/weights"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"yolov5_epoch_{epoch}_val_{val_loss:.4f}.pt")
    torch.save(model.state_dict(), save_path)
    print(f"모델 저장 완료: {save_path}")


if __name__ == "__main__":
    # train_YOLO(img_dir="data/train_images", ann_dir="data/train_labels", device="cuda" if torch.cuda.is_available() else "cpu")
    from ultralytics import YOLO
    model = YOLO('yolov5s.pt')
    model.train(
        data='data.yaml',
        epochs=5,
        imgsz=640,
        batch=8,
        patience=10,
        save=True,
    )
