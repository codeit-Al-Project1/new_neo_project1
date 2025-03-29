"""
[CSV 기반 로그 시각화 + 저장]
python visualize_log.py --source csv --csv_path yolologs/results.csv --all --save_dir results/yolo_logs

[TensorBoard 로그 시각화 (화면 출력)]
python visualize_log.py --source tensorboard --log_dir runs/detect/yolov8n_custom --metrics metrics/mAP50(B)

[특정 메트릭만 시각화 + 최고 epoch 출력]
python visualize_log.py --source csv --csv_path yolologs/results.csv --metrics metrics/recall(B) metrics/mAP50(B) --best_metric metrics/recall(B)
"""

import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def load_yolo_csv(csv_path):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV 파일이 존재하지 않습니다: {csv_path}")
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()  # 🔥 공백 제거
    return df


def load_tensorboard_log(log_dir, scalar_tags=None):
    event_acc = EventAccumulator(log_dir)
    event_acc.Reload()

    if scalar_tags is None:
        scalar_tags = event_acc.Tags().get("scalars", [])

    data = {}
    for tag in scalar_tags:
        events = event_acc.Scalars(tag)
        steps = [e.step for e in events]
        values = [e.value for e in events]
        data[tag] = pd.Series(data=values, index=steps)

    return pd.DataFrame(data)


def plot_metrics(df, metrics, save_dir=None):
    for metric in metrics:
        if metric not in df.columns:
            print(f"[경고] '{metric}'은 데이터프레임에 없습니다.")
            continue

        plt.figure(figsize=(8, 5))
        plt.plot(df.index if df.index.name else df['epoch'], df[metric], marker='o')
        plt.title(f"{metric} over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel(metric)
        plt.grid(True)

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            file_path = os.path.join(save_dir, f"{metric.replace('/', '_')}.png")
            plt.savefig(file_path)
            print(f"[저장됨] {file_path}")
        else:
            plt.show()

        plt.close()


def get_best_epoch(df, metric="metrics/mAP50(B)", mode="max"):
    if metric not in df.columns:
        raise ValueError(f"{metric} is not in dataframe")

    if mode == "max":
        idx = df[metric].idxmax()
    elif mode == "min":
        idx = df[metric].idxmin()
    else:
        raise ValueError("mode must be 'max' or 'min'")

    best_row = df.loc[idx]
    print(f"[Best Epoch] {metric}: {best_row[metric]:.4f} at epoch {int(best_row.name)}")
    return best_row


def main():
    parser = argparse.ArgumentParser(description="YOLO 학습 로그 시각화 도구 (CSV + TensorBoard 지원)")
    parser.add_argument("--csv_path", type=str, help="YOLO 학습 CSV 로그 파일 경로")
    parser.add_argument("--log_dir", type=str, help="TensorBoard 로그 디렉토리 경로")
    parser.add_argument("--metrics", nargs="+", help="시각화할 메트릭 이름들")
    parser.add_argument("--save_dir", type=str, default=None, help="시각화 저장 디렉토리")
    parser.add_argument("--all", action="store_true", help="주요 메트릭 전체 시각화")
    parser.add_argument("--best_metric", type=str, default="metrics/mAP50(B)", help="최고 epoch 기준 메트릭")
    parser.add_argument("--source", type=str, choices=["csv", "tensorboard"], default="csv", help="불러올 로그 형식")

    args = parser.parse_args()

    default_metrics = [
        "train/box_loss", "train/cls_loss", "metrics/precision(B)",
        "metrics/recall(B)", "metrics/mAP50(B)", "metrics/mAP50-95(B)"
    ]

    metrics_to_plot = args.metrics or (default_metrics if args.all else [])

    if args.source == "csv":
        if not args.csv_path:
            raise ValueError("--csv_path가 필요합니다.")
        df = load_yolo_csv(args.csv_path)
        if "epoch" in df.columns:
            df.set_index("epoch", inplace=True)
    else:
        if not args.log_dir:
            raise ValueError("--log_dir가 필요합니다.")
        df = load_tensorboard_log(args.log_dir, scalar_tags=metrics_to_plot if metrics_to_plot else None)

    if not metrics_to_plot:
        metrics_to_plot = list(df.columns)

    plot_metrics(df, metrics_to_plot, args.save_dir)
    get_best_epoch(df, metric=args.best_metric)


if __name__ == "__main__":
    main()