"""
Fine-tune a pretrained ResNet18 on candidate_tiles/ -- the backup/comparison
model alongside the from-scratch SmallCNN.

Requires internet access on first run to download ImageNet weights (one-time,
not needed again once cached / not needed at all at serving time). Works out
of the box on Colab or any machine with normal internet access.

Usage:
    python train_finetune.py --data-dir /path/to/candidate_tiles --out checkpoints/resnet18_finetune.pt
"""

import argparse
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data import load_candidate_tiles
from models import INPUT_SIZE, build_finetuned_resnet18


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True, help="Path to candidate_tiles/")
    p.add_argument("--out", default="checkpoints/resnet18_finetune.pt")
    p.add_argument("--epochs", type=int, default=8)  # transfer learning converges fast
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--val-split", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--unfreeze-backbone",
        action="store_true",
        help="Also fine-tune layer4, not just the classifier head",
    )
    return p.parse_args()


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train() if train else model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.set_grad_enabled(train):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += images.size(0)

    return total_loss / total, correct / total


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    input_size = INPUT_SIZE["resnet18_finetune"]
    train_ds, val_ds, class_to_idx = load_candidate_tiles(
        args.data_dir, input_size, val_split=args.val_split, seed=args.seed
    )
    print(f"Train: {len(train_ds)} images, Val: {len(val_ds)} images")
    print(f"Classes: {class_to_idx}")
    print(
        "NOTE: tiles are 64x64 natively and are upsampled to 224x224 for this "
        "backbone -- expected to be blurry; this is the normal trade-off for "
        "reusing an ImageNet-pretrained network."
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = build_finetuned_resnet18(
        num_classes=len(class_to_idx),
        pretrained=True,
        freeze_backbone=not args.unfreeze_backbone,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"Training {sum(p.numel() for p in trainable_params):,} of {sum(p.numel() for p in model.parameters()):,} params")
    optimizer = torch.optim.Adam(trainable_params, lr=args.lr, weight_decay=1e-4)

    best_val_acc = 0.0
    start = time.time()

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, train=False)

        print(
            f"Epoch {epoch:2d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.3f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.3f}"
        )

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
            torch.save(
                {
                    "model_name": "resnet18_finetune",
                    "state_dict": model.state_dict(),
                    "class_to_idx": class_to_idx,
                    "input_size": input_size,
                    "val_acc": val_acc,
                },
                args.out,
            )

    elapsed = time.time() - start
    print(f"\nDone in {elapsed:.1f}s. Best val_acc={best_val_acc:.3f}. Checkpoint saved to {args.out}")


if __name__ == "__main__":
    main()
