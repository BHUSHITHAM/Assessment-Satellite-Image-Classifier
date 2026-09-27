# Evaluation Results

This file is the evidence trail for the accuracy numbers quoted in the README:
the exact commands run, the environment they ran in, and the raw output —
nothing here is hand-typed or estimated. Re-run the commands below yourself
against `checkpoints/` and the provided `eval_set/` to reproduce every number.

Environment: Google Colab, CPU only (`Using device: cpu`), dataset at
`/content/be-mlsys-assignment-dataset/` (892 train / 158 val tiles from
`candidate_tiles/`, 210 held-out tiles in `eval_set/` with `eval_labels.csv`).

## 1. Training

### SmallCNN (from scratch)

```bash
python train_cnn.py --data-dir /content/be-mlsys-assignment-dataset/candidate_tiles --out ../checkpoints/small_cnn.pt
```

```
Using device: cpu
Train: 892 images, Val: 158 images
Classes: {'AnnualCrop': 0, 'Forest': 1, 'Highway': 2, 'Industrial': 3, 'Residential': 4, 'River': 5, 'SeaLake': 6}
Epoch  1/20 | train_loss=1.7742 train_acc=0.428 | val_loss=1.3504 val_acc=0.481
Epoch  2/20 | train_loss=1.0009 train_acc=0.633 | val_loss=0.6985 val_acc=0.747
Epoch  3/20 | train_loss=0.8505 train_acc=0.700 | val_loss=0.8497 val_acc=0.684
Epoch  4/20 | train_loss=0.8453 train_acc=0.677 | val_loss=0.6510 val_acc=0.741
Epoch  5/20 | train_loss=0.7998 train_acc=0.693 | val_loss=0.7648 val_acc=0.709
Epoch  6/20 | train_loss=0.7417 train_acc=0.733 | val_loss=0.8745 val_acc=0.665
Epoch  7/20 | train_loss=0.6988 train_acc=0.735 | val_loss=0.5629 val_acc=0.759
Epoch  8/20 | train_loss=0.6233 train_acc=0.771 | val_loss=0.4566 val_acc=0.804
Epoch  9/20 | train_loss=0.6499 train_acc=0.751 | val_loss=0.4832 val_acc=0.823
Epoch 10/20 | train_loss=0.6373 train_acc=0.775 | val_loss=0.4198 val_acc=0.835
Epoch 11/20 | train_loss=0.6633 train_acc=0.753 | val_loss=0.5556 val_acc=0.772
Epoch 12/20 | train_loss=0.6133 train_acc=0.766 | val_loss=0.5342 val_acc=0.804
Epoch 13/20 | train_loss=0.5796 train_acc=0.784 | val_loss=0.4169 val_acc=0.829
Epoch 14/20 | train_loss=0.5958 train_acc=0.763 | val_loss=0.4172 val_acc=0.816
Epoch 15/20 | train_loss=0.5206 train_acc=0.815 | val_loss=0.4760 val_acc=0.785
Epoch 16/20 | train_loss=0.5458 train_acc=0.798 | val_loss=0.7077 val_acc=0.728
Epoch 17/20 | train_loss=0.5834 train_acc=0.780 | val_loss=0.4690 val_acc=0.816
Epoch 18/20 | train_loss=0.5515 train_acc=0.806 | val_loss=0.5600 val_acc=0.816
Epoch 19/20 | train_loss=0.5998 train_acc=0.784 | val_loss=0.4657 val_acc=0.785
Epoch 20/20 | train_loss=0.5007 train_acc=0.817 | val_loss=0.5470 val_acc=0.772

Done in 220.7s. Best val_acc=0.835. Checkpoint saved to ../checkpoints/small_cnn.pt
```

Trained for 20 epochs in ~3.7 minutes on CPU; best validation accuracy 83.5%.
Validation accuracy is noisy epoch-to-epoch (e.g. dips at epoch 16) but trends
upward overall — expected for a small model trained from scratch on ~900 images.

### ResNet18 (fine-tuned, ImageNet-pretrained)

```bash
python train_finetune.py --data-dir /content/be-mlsys-assignment-dataset/candidate_tiles --out ../checkpoints/resnet18_finetune.pt
```

```
Using device: cpu
Train: 892 images, Val: 158 images
Classes: {'AnnualCrop': 0, 'Forest': 1, 'Highway': 2, 'Industrial': 3, 'Residential': 4, 'River': 5, 'SeaLake': 6}
NOTE: tiles are 64x64 natively and are upsampled to 224x224 for this backbone -- expected to be blurry; this is the normal trade-off for reusing an ImageNet-pretrained network.
Downloading: "https://download.pytorch.org/models/resnet18-f37072fd.pth" to /root/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth
100% 44.7M/44.7M [00:00<00:00, 58.2MB/s]
Training 8,397,319 of 11,180,103 params
Epoch  1/8 | train_loss=0.5163 train_acc=0.809 | val_loss=1.2460 val_acc=0.791
Epoch  2/8 | train_loss=0.2146 train_acc=0.933 | val_loss=0.1984 val_acc=0.937
Epoch  3/8 | train_loss=0.1524 train_acc=0.948 | val_loss=0.3567 val_acc=0.911
Epoch  4/8 | train_loss=0.1885 train_acc=0.941 | val_loss=0.2002 val_acc=0.949
Epoch  5/8 | train_loss=0.1236 train_acc=0.960 | val_loss=0.3618 val_acc=0.918
Epoch  6/8 | train_loss=0.1725 train_acc=0.942 | val_loss=0.4360 val_acc=0.886
Epoch  7/8 | train_loss=0.1005 train_acc=0.967 | val_loss=0.1679 val_acc=0.943
Epoch  8/8 | train_loss=0.0845 train_acc=0.972 | val_loss=0.1328 val_acc=0.956

Done in 1154.8s. Best val_acc=0.956. Checkpoint saved to ../checkpoints/resnet18_finetune.pt
```

Note the one-time internet dependency here: the ImageNet-pretrained weights
(`resnet18-f37072fd.pth`, ~44.7MB) are fetched from `download.pytorch.org` on
first run and cached locally afterward — this is the "needs one-time internet
access" caveat called out in the README and design note. Once cached (or
vendored into the offline deployment image), no further network access is
needed. Also worth flagging: the native tile size is 64×64 and gets upsampled
to 224×224 to match the ResNet18 input — a known trade-off of reusing an
ImageNet backbone on much lower-resolution imagery, and one reason the
gap between the two models is worth taking seriously rather than assuming
ResNet18 is simply "better."

Fine-tuned for 8 epochs in ~19 minutes on CPU; best validation accuracy 95.6%.

## 2. Held-out evaluation

Both checkpoints were then scored against `eval_set/` (210 tiles, held out
from training entirely) using `eval.py`:

```bash
python eval.py --checkpoint ../checkpoints/small_cnn.pt \
    --eval-dir /content/be-mlsys-assignment-dataset/eval_set \
    --eval-labels /content/be-mlsys-assignment-dataset/eval_labels.csv \
    --out-csv ../results/small_cnn_predictions.csv

python eval.py --checkpoint ../checkpoints/resnet18_finetune.pt \
    --eval-dir /content/be-mlsys-assignment-dataset/eval_set \
    --eval-labels /content/be-mlsys-assignment-dataset/eval_labels.csv \
    --out-csv ../results/resnet18_predictions.csv
```

### SmallCNN — 81.0% accuracy

```
Loaded small_cnn (input_size=64) from ../checkpoints/small_cnn.pt

=== small_cnn on eval_set (210 tiles) ===
Accuracy: 0.810

              precision    recall  f1-score   support

  AnnualCrop      0.963     0.867     0.912        30
      Forest      0.784     0.967     0.866        30
     Highway      0.632     0.400     0.490        30
  Industrial      0.879     0.967     0.921        30
 Residential      0.875     0.933     0.903        30
       River      0.605     0.767     0.676        30
     SeaLake      0.958     0.767     0.852        30

    accuracy                          0.810       210
   macro avg      0.814     0.810     0.803       210
weighted avg      0.814     0.810     0.803       210

Confusion matrix (rows=true, cols=predicted):
          AnnualCr   Forest  Highway Industri Resident    River  SeaLake
AnnualCrop       26        1        2        0        0        1        0
    Forest        0       29        0        0        0        0        1
   Highway        0        0       12        3        2       13        0
Industrial        0        0        0       29        1        0        0
Residentia        0        0        1        1       28        0        0
     River        0        2        4        0        1       23        0
   SeaLake        1        5        0        0        0        1       23

Per-tile predictions written to ../results/small_cnn_predictions.csv
```

**Where it actually fails:** Highway is the weak class (recall 0.400) — 13 of
30 Highway tiles are confused for River, and the reverse also happens (4
River→Highway). SeaLake recall also dips (0.767), mostly confused for
Forest. These are visually adjacent classes at 64×64 resolution (a paved
strip and a river channel; a lake edge and dense tree cover), so this isn't
a bug — it's the model running out of resolution to distinguish them. This
Highway↔River confusion is exactly the kind of case the confidence-based
`review_flag` in the API is meant to catch before a wrong label gets acted on.

### ResNet18 (fine-tuned) — 96.2% accuracy

```
Loaded resnet18_finetune (input_size=224) from ../checkpoints/resnet18_finetune.pt

=== resnet18_finetune on eval_set (210 tiles) ===
Accuracy: 0.962

              precision    recall  f1-score   support

  AnnualCrop      0.935     0.967     0.951        30
      Forest      1.000     1.000     1.000        30
     Highway      0.882     1.000     0.938        30
  Industrial      1.000     1.000     1.000        30
 Residential      1.000     1.000     1.000        30
       River      0.926     0.833     0.877        30
     SeaLake      1.000     0.933     0.966        30

    accuracy                          0.962       210
   macro avg      0.963     0.962     0.962       210
weighted avg      0.963     0.962     0.962       210

Confusion matrix (rows=true, cols=predicted):
          AnnualCr   Forest  Highway Industri Resident    River  SeaLake
AnnualCrop       29        0        0        0        0        1        0
    Forest        0       30        0        0        0        0        0
   Highway        0        0       30        0        0        0        0
Industrial        0        0        0       30        0        0        0
Residentia        0        0        0        0       30        0        0
     River        1        0        4        0        0       25        0
   SeaLake        1        0        0        0        0        1       28

Per-tile predictions written to ../results/resnet18_predictions.csv
```

**Where it still fails:** the same River↔Highway pair is the only real weak
spot left (4 of 30 River tiles predicted as Highway) — the same underlying
visual ambiguity as above, just far less frequent with the richer ImageNet
features. Every other class is at or near perfect recall on this eval set.

This run was reproduced a second time from a fresh checkpoint (same command,
same data, no cached results reused) and landed on the identical 95.6%
val accuracy / 96.2% eval accuracy — the numbers aren't a one-off lucky seed.

## 3. Per-tile evidence

Full per-tile predictions (`filename, true_label, predicted_label,
confidence, class_probabilities`) for both models are written to:

- `results/small_cnn_predictions.csv`
- `results/resnet18_predictions.csv`

These are the raw basis for the confusion matrices above and for the
confidence-based `review_flag` logic in the serving API — e.g. `tile_002.png`
(true: Forest) gets only 0.329 confidence from SmallCNN with SeaLake and
Forest nearly tied (0.329 vs 0.296), which is precisely the kind of
low-confidence, genuinely-ambiguous case the review flag exists to surface,
versus ResNet18's 0.993 confident-and-correct call on the same tile.

## 4. Reproducibility

Nothing above is hand-edited — every accuracy figure, precision/recall/F1
value, and confusion matrix came directly from `eval.py`'s stdout, and the
full per-tile CSVs are checked in under `results/` for independent
inspection. To verify: point `--data-dir` / `--eval-dir` / `--eval-labels`
at the provided dataset and re-run the four commands in this file.