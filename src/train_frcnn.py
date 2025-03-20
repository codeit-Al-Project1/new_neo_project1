import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from src.utils import get_optimizer
from src.utils import get_scheduler
from src.utils import calculate_iou
from src.utils import calculate_ap
from src.utils import evaluate_predictions_direct
from src.data_utils.data_loader import get_loader
from src.data_utils.data_loader import get_category_mapping
from src.model_utils.basic_frcnn import save_model
from src.model_utils.basic_frcnn import get_fast_rcnn_model

# 학습 함수 정의

def train(img_dir, json_dir, batch_size=16, num_epochs=5, optimizer_name="sgd", scheduler_name ="plateau", lr=0.001, weight_decay=0.0005, device="cpu", debug=False):

    # bbox 모델 맞추는 함수
    # def format_bbox(targets, device):
    #     formatted_targets = []
    #     for t in targets:
    #         boxes = t["boxes"]  # BoundingBox (XYWH format)
    #         if len(boxes) == 0:  # 빈 박스를 처리 (이 경우는 넘어가거나 에러 처리 가능)
    #             continue
    #         # XYWH -> XYXY로 변환 (모델에 맞는 형식으로 변경)
    #         xyxy_boxes = []
    #         for box in boxes:
    #             x_min = box[0] - box[2] / 2  # 좌상단 x
    #             y_min = box[1] - box[3] / 2  # 좌상단 y
    #             x_max = box[0] + box[2] / 2  # 우하단 x
    #             y_max = box[1] + box[3] / 2  # 우하단 y
    #             xyxy_boxes.append([x_min, y_min, x_max, y_max])

    #         xyxy_boxes = torch.tensor(xyxy_boxes, dtype=torch.float32).to(device)  # 텐서를 GPU로 이동

    #         formatted_targets.append({
    #             "boxes": xyxy_boxes,  # BoundingBox를 텐서로 변환
    #             "labels": t["labels"].to(device),  # 클래스 레이블
    #         })

    #     return formatted_targets
    # bbox 모델 맞추는 함수 (벡터 연산으로 최적화)
    def format_bbox(targets, device):
        formatted_targets = []
        for t in targets:
            if len(t["boxes"]) == 0:  # 빈 박스 처리
                continue
            
            boxes = np.array(t["boxes"])  # 리스트 → NumPy 배열 변환
            x_min = boxes[:, 0] - boxes[:, 2] / 2
            y_min = boxes[:, 1] - boxes[:, 3] / 2
            x_max = boxes[:, 0] + boxes[:, 2] / 2
            y_max = boxes[:, 1] + boxes[:, 3] / 2

            xyxy_boxes = np.stack([x_min, y_min, x_max, y_max], axis=1)
            xyxy_boxes = torch.tensor(xyxy_boxes, dtype=torch.float32).to(device)

            formatted_targets.append({
                "boxes": xyxy_boxes,
                "labels": t["labels"].to(device),
            })
        return formatted_targets

    name_to_idx, idx_to_name = get_category_mapping(json_dir)  # 카테고리 매핑 정보를 불러옵니다.
    num_classes = len(name_to_idx) + 1  # 클래스 수 : 74

    # 학습용 데이터 로더, 분할
    train_loader = get_loader(img_dir, json_dir, batch_size, mode="train", val_ratio=0.2, debug=debug)
    val_loader = get_loader(img_dir, json_dir, batch_size, mode="val", val_ratio=0.2, debug=debug)

    # 모델, 옵티마이저, 스케쥴러 정의
    model = get_fast_rcnn_model(num_classes).to(device)
    optimizer = get_optimizer(optimizer_name, model, lr, weight_decay)
    scheduler = get_scheduler(scheduler_name, optimizer, T_max=100)  # T_max는 cosine만 적용되는 파라미터

    best_map_score = 0
    
    # 학습 루프
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        epoch_loss_details = {}

        # 1. 학습 단계
        progress_bar = tqdm(train_loader, total=len(train_loader), desc="Train", dynamic_ncols=True)

        # images 포맷 맞추기
        for images, targets in progress_bar:
            images = [img.to(device) for img in images]

            # target 포맷 맞추기
            formatted_targets = format_bbox(targets, device)    

            # 모델 학습
            optimizer.zero_grad()  # gradient 초기화
            loss_dict = model(images, formatted_targets)
            losses = sum(loss for loss in loss_dict.values())

            losses.backward()

            # Gradient Clipping 추가
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += losses.item()

            # 개별 손실 요소 누적
            for k, v in loss_dict.items():
                if k not in epoch_loss_details:
                    epoch_loss_details[k] = 0
                epoch_loss_details[k] += v.item()

            # tqdm 진행 바의 postfix 업데이트
            avg_loss_details = ", ".join([f"{k}: {v / len(train_loader):.4f}" for k, v in epoch_loss_details.items()])
            progress_bar.set_postfix(Avg_Loss=avg_loss_details)

        avg_loss_details = ", ".join([f"{k}: {v / len(train_loader):.4f}" for k, v in epoch_loss_details.items()])
        print(f"Epoch {epoch+1} Complete - Total Loss: {total_loss:.4f}, Avg Loss Per Component: {avg_loss_details}")


        # 2. 검증 단계
        model.eval()
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for images, targets in tqdm(val_loader, desc="Validation", dynamic_ncols=True):
                images = [img.to(device) for img in images]
                formatted_targets = format_bbox(targets, device)

                # 모델 예측
                predictions = model(images)

                # 예측 결과를 리스트에 저장
                all_predictions.extend(predictions)
                all_targets.extend(formatted_targets)

        # mAP 계산
        total_ap_score = 0
        for i in range(len(all_predictions)):
            ap_score = evaluate_predictions_direct(all_predictions[i], all_targets[i], num_classes)
            total_ap_score += ap_score

        avg_map_score = total_ap_score / len(all_predictions) if len(all_predictions) > 0 else 0
        print(f"Average Precision (mAP) for the validation set: {avg_map_score:.4f}")

        # 3. 모델 저장
        if avg_map_score > best_map_score:
            best_map_score = avg_map_score
            save_model(model, save_dir="./models", base_name="model", ext=".pth")
            print(f"Model saved with mAP score: {best_map_score:.4f}")
        
        # 학습률 스케줄러 업데이트
        scheduler.step(avg_map_score)

    print(f"Training complete. Best mAP: {best_map_score:.4f}")


if __name__ == "__main__":
    train(img_dir="data/train_images", json_dir="data/train_annots_modify", batch_size=16, num_epochs=5, lr=0.001, weight_decay=0.0005, device="cuda", debug=False)



