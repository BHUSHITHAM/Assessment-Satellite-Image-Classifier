"""
Evaluate any checkpoint (small_cnn or resnet18_finetune) against the held-out
eval_set/ + eval_labels.csv. Prints accuracy, per-class precision/recall/F1,
a confusion matrix, and writes per-tile predictions to a CSV for closer
inspection (e.g. "which specific tiles did it get wrong / was it unsure about").

Usage:
    python eval.py --checkpoint checkpoints/small_cnn.pt \
        --eval-dir /path/to/eval_set --eval-labels /path/to/eval_labels.csv \
        --out-csv results/small_cnn_predictions.csv
"""

import argparse
import json

import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

from data import EvalTileDataset, idx_to_class
from models import load_checkpoint


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--eval-dir", required=True, help="Path to eval_set/")
    p.add_argument("--eval-labels", required=True, help="Path to eval_labels.csv")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--out-csv", default=None, help="Optional path to write per-tile predictions")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, class_to_idx, input_size, model_name = load_checkpoint(args.checkpoint, device)
    idx2class = idx_to_class(class_to_idx)
    print(f"Loaded {model_name} (input_size={input_size}) from {args.checkpoint}")

    ds = EvalTileDataset(args.eval_dir, args.eval_labels, input_size, class_to_idx)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    all_true, all_pred, all_conf, all_files, all_probs = [], [], [], [], []

    with torch.no_grad():
        for images, labels, filenames in loader:
            images = images.to(device)
            logits = model(images)
            probs = torch.softmax(logits, dim=1)
            confs, preds = probs.max(dim=1)

            all_true.extend(labels.tolist())
            all_pred.extend(preds.tolist())
            all_conf.extend(confs.tolist())
            all_files.extend(filenames)
            all_probs.extend(probs.tolist())

    class_names = [idx2class[i] for i in range(len(idx2class))]
    acc = sum(t == p for t, p in zip(all_true, all_pred)) / len(all_true)

    print(f"\n=== {model_name} on eval_set ({len(all_true)} tiles) ===")
    print(f"Accuracy: {acc:.3f}\n")
    print(classification_report(all_true, all_pred, target_names=class_names, digits=3))

    cm = confusion_matrix(all_true, all_pred)
    print("Confusion matrix (rows=true, cols=predicted):")
    header = "          " + " ".join(f"{c[:8]:>8}" for c in class_names)
    print(header)
    for name, row in zip(class_names, cm):
        print(f"{name[:10]:>10} " + " ".join(f"{v:>8}" for v in row))

    if args.out_csv:
        import csv
        import os

        os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
        with open(args.out_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["filename", "true_label", "predicted_label", "confidence", "class_probabilities"])
            for fname, t, p, conf, probs in zip(all_files, all_true, all_pred, all_conf, all_probs):
                prob_map = {idx2class[i]: round(v, 4) for i, v in enumerate(probs)}
                writer.writerow([fname, idx2class[t], idx2class[p], round(conf, 4), json.dumps(prob_map)])
        print(f"\nPer-tile predictions written to {args.out_csv}")


if __name__ == "__main__":
    main()
