"""
Model definitions shared by training, evaluation, and the serving API.

Two models on the table, per the design note:
  - SmallCNN: trained from scratch on candidate_tiles/, our primary bet.
  - build_finetuned_resnet18: pretrained ImageNet backbone, last block +
    classifier head fine-tuned on the same data, kept as a backup/comparison.

CLASS_NAMES is fixed (alphabetical, matching torchvision.datasets.ImageFolder's
default ordering) so checkpoints and inference code always agree on what
index 0..6 means.
"""

import torch
import torch.nn as nn
from torchvision import models

CLASS_NAMES = [
    "AnnualCrop",
    "Forest",
    "Highway",
    "Industrial",
    "Residential",
    "River",
    "SeaLake",
]
NUM_CLASSES = len(CLASS_NAMES)


class SmallCNN(nn.Module):
    """
    Small conv net for 64x64 RGB tiles. Kept deliberately shallow:
    the dataset is small (1,050 images), so a deep net would just overfit,
    and the whole point is something that trains in a couple of minutes
    on CPU on isolated hardware.
    """

    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 64 -> 32
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 32 -> 16
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 16 -> 8
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


def build_finetuned_resnet18(
    num_classes: int = NUM_CLASSES, pretrained: bool = True, freeze_backbone: bool = True
) -> nn.Module:
    """
    ResNet18 with ImageNet weights, classifier head replaced for our 7 classes.

    freeze_backbone=True only trains the final fc layer (fast, less prone to
    overfitting on 1,050 images). Set False to also unfreeze layer4 for a bit
    more capacity if you have time/data to spare.

    Requires internet access the first time it runs (to download ImageNet
    weights) -- fine on Colab or your own machine, not available in fully
    air-gapped environments. That's a one-time setup step, not a runtime
    dependency of the deployed service.
    """
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    net = models.resnet18(weights=weights)

    if freeze_backbone:
        for param in net.parameters():
            param.requires_grad = False
        # unfreeze the last residual block for a little adaptation capacity
        for param in net.layer4.parameters():
            param.requires_grad = True

    net.fc = nn.Linear(net.fc.in_features, num_classes)
    return net


# Input size expected by each model -- used by training/eval/serving to pick
# the right transform.
INPUT_SIZE = {
    "small_cnn": 64,
    "resnet18_finetune": 224,  # resnet expects the standard ImageNet input size
}


def load_checkpoint(path: str, device: torch.device):
    """
    Loads a checkpoint saved by train_cnn.py or train_finetune.py and
    reconstructs the right architecture. Used by eval.py and the serving API
    so there's exactly one place that knows how to go from a checkpoint file
    to a ready-to-use model.

    Returns (model, class_to_idx, input_size, model_name).
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model_name = ckpt["model_name"]
    class_to_idx = ckpt["class_to_idx"]
    input_size = ckpt["input_size"]
    num_classes = len(class_to_idx)

    if model_name == "small_cnn":
        model = SmallCNN(num_classes=num_classes)
    elif model_name == "resnet18_finetune":
        # pretrained=False: we're loading fine-tuned weights from the
        # checkpoint, not re-downloading ImageNet weights, so no internet
        # access is needed at inference/serving time -- only training needed it.
        model = build_finetuned_resnet18(num_classes=num_classes, pretrained=False)
    else:
        raise ValueError(f"Unknown model_name in checkpoint: {model_name}")

    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()
    return model, class_to_idx, input_size, model_name


def discover_checkpoints(checkpoints_dir: str, device: torch.device):
    """
    Scans checkpoints_dir for *.pt files and loads every one that parses.
    Keyed by the model_name field stored *inside* the checkpoint (not the
    filename) so the registry is correct even if someone renames a file.

    Returns a dict: {model_name: {"model": ..., "class_to_idx": ...,
    "input_size": ..., "transform": ..., "val_acc": ..., "checkpoint_path": ...}}

    A checkpoint that fails to load (corrupt file, unknown architecture) is
    skipped with a warning rather than crashing the whole service -- one bad
    file shouldn't take down every other model.
    """
    import glob
    import os

    registry = {}
    for path in sorted(glob.glob(os.path.join(checkpoints_dir, "*.pt"))):
        try:
            model, class_to_idx, input_size, model_name = load_checkpoint(path, device)
        except Exception as e:  # noqa: BLE001 -- deliberately broad, see docstring
            print(f"WARNING: failed to load checkpoint {path}: {e}")
            continue

        ckpt_raw = torch.load(path, map_location=device, weights_only=False)
        registry[model_name] = {
            "model": model,
            "class_to_idx": class_to_idx,
            "input_size": input_size,
            "val_acc": ckpt_raw.get("val_acc"),
            "checkpoint_path": path,
        }
    return registry
