# Melanoma CNN detector

Binary classifier (benign vs malignant) using a Keras `Sequential` CNN.

## Dataset (Kaggle)

Uses the local Kaggle layout:

```
data/melanoma_cancer_dataset/
  train/              # original images
    benign/
    malignant/
  train_augmented/    # augmented copies (from team GitHub main)
    benign/
    malignant/
  test/               # held-out evaluation
    benign/
    malignant/
```

Get `train_augmented` from [GitHub main](https://github.com/ManilMehta/team-manil-aisc/tree/main/data/melanoma_cancer_dataset):

```bash
git fetch origin main
git checkout origin/main -- data/melanoma_cancer_dataset/train_augmented
```

Training uses **`train/` + `train_augmented/`** by default; validation uses **`test/`** only.

## Setup

```bash
pip install -r requirements.txt
```

## Train and test (20 epochs default)

```powershell
python train_melanoma.py
```

- Trains on **train + train_augmented**
- Evaluates on **test** (classification report, confusion matrix, ROC AUC)
- Saves **`checkpoints/melanoma_best.keras`**

Original images only:

```powershell
python train_melanoma.py --no-augmented
```

Load saved model later:

```python
import tensorflow as tf
model = tf.keras.models.load_model("checkpoints/melanoma_best.keras")
```

## Project layout

| Path | Purpose |
|------|---------|
| `models/cnn.py` | CNN definition, compile, training helper |
| `datasets/load.py` | Load Kaggle folder layout into `tf.data` |
| `train_melanoma.py` | Train + evaluate |
| `setup_dataset.sh` | Kaggle download helper |
Dataset: [melanoma-skin-cancer-dataset-of-10000-images](https://www.kaggle.com/datasets/hasnainjaved/melanoma-skin-cancer-dataset-of-10000-images)

