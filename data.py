"""
Dataset loading shared by both training scripts and eval.py.

candidate_tiles/<ClassName>/<ClassName>_<n>.png -- loaded via ImageFolder,
then split into train/val (stratified, so each class is represented
proportionally in both splits despite the small dataset size).

eval_set/tile_NNN.png + eval_labels.csv (filename,true_label) -- loaded via
a small custom Dataset since it's a flat folder with labels in a sidecar CSV,
not ImageFolder's class-per-folder layout.
"""

import csv
import os
from typing import Dict, List, Tuple

import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, Subset
from torchvision import datasets, transforms

from models import CLASS_NAMES


def get_transforms(input_size: int, train: bool) -> transforms.Compose:
    if train:
        return transforms.Compose(
            [
                transforms.Resize((input_size, input_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),  # satellite tiles have no "up" -- a
                # valid augmentation here that would be wrong for, say, photos of faces
                transforms.RandomRotation(15),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def load_candidate_tiles(
    data_dir: str, input_size: int, val_split: float = 0.15, seed: int = 42
) -> Tuple[Subset, Subset, Dict[str, int]]:
    """
    Returns (train_subset, val_subset, class_to_idx).

    Two ImageFolder instances are created over the same directory -- one with
    train-time augmentation, one without -- and split with the *same* indices,
    so the validation subset never sees augmented images even though it
    shares the underlying files.
    """
    base_train = datasets.ImageFolder(data_dir, transform=get_transforms(input_size, train=True))
    base_val = datasets.ImageFolder(data_dir, transform=get_transforms(input_size, train=False))
    assert base_train.class_to_idx == base_val.class_to_idx

    labels = [s[1] for s in base_train.samples]
    train_idx, val_idx = train_test_split(
        range(len(base_train)), test_size=val_split, stratify=labels, random_state=seed
    )

    train_subset = Subset(base_train, train_idx)
    val_subset = Subset(base_val, val_idx)
    return train_subset, val_subset, base_train.class_to_idx


class EvalTileDataset(Dataset):
    """
    Loads eval_set/ tiles using eval_labels.csv for ground truth.
    Returns (image_tensor, true_label_idx, filename) per item.
    """

    def __init__(self, eval_dir: str, eval_labels_csv: str, input_size: int, class_to_idx: Dict[str, int]):
        self.eval_dir = eval_dir
        self.class_to_idx = class_to_idx
        self.transform = get_transforms(input_size, train=False)
        self.items: List[Tuple[str, int]] = []

        with open(eval_labels_csv, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                filename = row["filename"].strip()
                label_name = row["true_label"].strip()
                if label_name not in class_to_idx:
                    raise ValueError(
                        f"Label '{label_name}' in {eval_labels_csv} not in known classes {list(class_to_idx)}"
                    )
                self.items.append((filename, class_to_idx[label_name]))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        filename, label_idx = self.items[idx]
        img_path = os.path.join(self.eval_dir, filename)
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        return image, label_idx, filename


def idx_to_class(class_to_idx: Dict[str, int]) -> Dict[int, str]:
    return {v: k for k, v in class_to_idx.items()}
