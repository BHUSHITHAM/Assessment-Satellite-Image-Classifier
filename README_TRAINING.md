# Training setup

## Layout expected

```
tile_classifier/
  models.py           # SmallCNN + fine-tuned ResNet18 architectures
  data.py             # dataset loading for candidate_tiles/ and eval_set/
  train_cnn.py        # trains SmallCNN from scratch
  train_finetune.py   # fine-tunes pretrained ResNet18 (needs internet once)
  eval.py             # evaluates either checkpoint against eval_set + eval_labels.csv
  requirements.txt
checkpoints/          # created automatically; holds .pt checkpoint files
results/              # created automatically; holds per-tile prediction CSVs
```

Point `--data-dir` at wherever you unzipped
`Galaxeye-BE_MLSys-TakeHome_Assignment-Tiles.zip`'s `candidate_tiles/` folder,
and `--eval-dir` / `--eval-labels` at `eval_set/` and `eval_labels.csv`.

## Option A: Google Colab (recommended for the fine-tune model)

1. Upload `tile_classifier/` (all 5 files above) and the dataset zip to your
   Colab session (drag into the file browser, or mount Drive).
2. Runtime -> Change runtime type -> GPU (T4 is plenty; not required, just
   faster) -- or leave as CPU, both scripts run fine either way.
3. In a cell:
   ```
   !unzip -q Galaxeye-BE_MLSys-TakeHome_Assignment-Tiles.zip
   !pip install -q torch torchvision scikit-learn  # Colab already has these
   %cd tile_classifier
   !python train_cnn.py --data-dir ../be-mlsys-assignment-dataset/candidate_tiles --out ../checkpoints/small_cnn.pt
   !python train_finetune.py --data-dir ../be-mlsys-assignment-dataset/candidate_tiles --out ../checkpoints/resnet18_finetune.pt
   !python eval.py --checkpoint ../checkpoints/small_cnn.pt --eval-dir ../be-mlsys-assignment-dataset/eval_set --eval-labels ../be-mlsys-assignment-dataset/eval_labels.csv --out-csv ../results/small_cnn_predictions.csv
   !python eval.py --checkpoint ../checkpoints/resnet18_finetune.pt --eval-dir ../be-mlsys-assignment-dataset/eval_set --eval-labels ../be-mlsys-assignment-dataset/eval_labels.csv --out-csv ../results/resnet18_predictions.csv
   ```
4. Download `checkpoints/*.pt` and `results/*.csv` back to your machine when
   done (Files panel -> right-click -> Download) -- the serving API only
   needs the checkpoint files, not Colab itself.

## Option B: your own machine (CPU only, no GPU)

```bash
cd tile_classifier
pip install torch --no-deps
pip install torchvision --no-deps
pip install scikit-learn pillow numpy

python train_cnn.py --data-dir /path/to/candidate_tiles --out ../checkpoints/small_cnn.pt

# needs one-time internet access to fetch ImageNet weights:
python train_finetune.py --data-dir /path/to/candidate_tiles --out ../checkpoints/resnet18_finetune.pt

python eval.py --checkpoint ../checkpoints/small_cnn.pt \
    --eval-dir /path/to/eval_set --eval-labels /path/to/eval_labels.csv \
    --out-csv ../results/small_cnn_predictions.csv

python eval.py --checkpoint ../checkpoints/resnet18_finetune.pt \
    --eval-dir /path/to/eval_set --eval-labels /path/to/eval_labels.csv \
    --out-csv ../results/resnet18_predictions.csv
```

The `--no-deps` install matters if you're short on disk: a plain
`pip install torch` on Linux drags in several GB of nvidia-cu* packages you
don't need for CPU-only work.

## Already validated (in this environment, real numbers)

`train_cnn.py` was run end-to-end here on the actual `candidate_tiles/`
data: trained in ~110s on CPU, best validation accuracy 84.8%, then
evaluated against the real held-out `eval_set/` (210 tiles, untouched during
training) against `eval_labels.csv`:

- **Overall accuracy: 79.0%**
- Strongest classes: Residential (100% recall), AnnualCrop (93% recall)
- Weakest: Highway (40% recall -- most often confused with River, which
  makes sense at 64x64: both are long, thin, linear features)

Full classification report and confusion matrix are in the training log;
per-tile predictions with full confidence distributions are in
`results/small_cnn_predictions.csv`.

`train_finetune.py`'s architecture and full data pipeline (at the 224x224
input size ResNet18 expects) were validated to run correctly here too --
forward pass, parameter freezing (8.4M/11.2M trainable with the default
frozen-backbone setting), checkpoint format all confirmed working. The only
piece not run end-to-end here is the actual weight download + fine-tuning
loop, since this sandbox can't reach `download.pytorch.org` -- that will
just work on Colab or your own machine with normal internet access.
