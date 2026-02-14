# TrueFruitFly Classification

A multi-angle image dataset and deep learning pipeline for **classifying 26 quarantine true fruit flies** (Diptera: Tephritidae). This repository provides scripts for training, inference, and Grad-CAM visualization used in the associated paper.

All models were trained on **CAU_HPC** (China Agricultural University).  
Dataset and publication links will be added upon release.

---

## Environment Setup

Create a virtual environment and install dependencies:

```bash
pip install -r requirements.txt
```

**Dependencies** (see `requirements.txt`):

| Package     | Version  |
|------------|----------|
| torch      | 2.4.1    |
| torchvision| 0.19.1   |
| matplotlib | 3.7.5    |
| numpy      | 1.24.4   |
| pandas     | 2.0.3    |
| Pillow     | 11.3.0   |
| scikit-learn | 1.3.2  |
| seaborn    | 0.13.2   |
| tqdm       | 4.67.1   |

**Note:** GPU with CUDA is recommended for training and inference.

---

## Dataset Structure

### Training (`Train.py`)

Place your data under a root directory with three subfolders:

```
data_dir/
├── train/          # Training images (one subfolder per class)
│   ├── class_1/
│   ├── class_2/
│   └── ...
├── val/            # Validation images (same structure)
└── test/           # Test images (same structure)
```

### Prediction & Grad-CAM

- **Predict.py:** Test set root contains one subfolder per class, each with images (e.g. `.png`, `.jpg`, `.jpeg`).
- **Grad-CAM.py:** Same layout — one directory per class, images inside.

---

## Usage

### 1. Training

Train a model with configurable architecture, hyperparameters, and output directory.

```bash
python Train.py --data_dir <path_to_dataset> --output_dir <path_to_results> --model_type <model> [options]
```

**Main arguments:**

| Argument        | Default | Description |
|----------------|--------|-------------|
| `--data_dir`   | (see code) | Dataset root (must contain `train/`, `val/`, `test/`) |
| `--output_dir` | `outputs_lr0001/` | Base directory for run outputs |
| `--model_type` | `resnet` | One of: `resnet`, `efficientnet`, `swin`, `mobilenet`, `inception`, `vit`, `convnext` |
| `--batch_size` | 32 | Batch size |
| `--num_epochs` | 30 | Number of epochs |
| `--lr`         | 0.0005 | Learning rate |
| `--device`     | auto   | e.g. `cuda:0` or `cpu` |
| `--save_interval` | 5  | Save checkpoint every N epochs (0 to disable) |

**Example:**

```bash
python Train.py --data_dir ./dataset --output_dir ./results --model_type resnet --batch_size 32 --num_epochs 20 --lr 0.0005
```

**Outputs (under `output_dir/<model_type>_<timestamp>/`):**

- `best_model.pth` — best validation checkpoint  
- `history.json` — train/val loss and accuracy per epoch  
- `training_history.png` / `.svg` — loss and accuracy curves  
- `confusion_matrix.png` / `.svg` — test set confusion matrix  
- `test_results.json` — test metrics and per-class stats  
- `classification_report.txt` — text classification report  
- `classes.json` — class names  
- `epoch_*.pth` — periodic checkpoints (if `save_interval` > 0)

---

### 2. Prediction (Testing)

Run inference on one or more test sets, with one or more trained models.

```bash
python Predict.py --model_path <path_to_.pth> --test_root <path_to_test_sets> --model_type <model> --results_dir <path_to_results> [options]
```

**Main arguments:**

| Argument        | Description |
|----------------|-------------|
| `--model_path` | Single model file (e.g. `best_model.pth`) |
| `--model_dir`  | Directory of model files (e.g. all `*.pth`) |
| `--model_root` | Root directory; script walks subdirs for model files |
| `--test_dir`   | Single test dataset directory (class subfolders) |
| `--test_dirs`  | Multiple test directories (space-separated) |
| `--test_root`  | Root directory containing multiple test set folders |
| `--model_type` | `resnet`, `efficientnet`, `swin`, `mobilenet`, `convnext`, or `auto` |
| `--class_names`| Space-separated class names (same order as training) |
| `--results_dir`| Directory for all result subfolders |
| `--batch_size` | Default 32 |
| `--model_pattern` | Glob for model files under `model_dir`/`model_root`, e.g. `*.pth` |

**Example (single model, single test set):**

```bash
python Predict.py --model_path ./results/resnet_xxx/best_model.pth --test_dir ./test_dataset --model_type resnet --results_dir ./predict_results
```

**Example (multiple models/datasets via root):**

```bash
python Predict.py --model_root ./trained_models --test_root ./test_sets --model_type resnet --results_dir ./predict_results
```

**Outputs (per model–dataset pair, e.g. `results_dir/<model_name>-pre-<test_name>/`):**

- `test_results.csv` — per-image path, true/pred class, confidence, correct flag  
- `test_classification_report.csv` — precision, recall, F1 (macro/micro)  
- `confusion_matrix.png`  
- `test_summary.txt` — summary and main metrics  
- `misclassified/` — copies of misclassified images by true/pred class  
- `misclassified_samples.png` — sample grid of errors  
- `misclassified_details.csv`, `error_statistics.csv`, `error_rate_by_class.png`  
- `all_results_summary.csv` and accuracy comparison plots in `results_dir` when multiple runs are done  

---

### 3. Grad-CAM Visualization

Generate Grad-CAM heatmaps for a **ConvNeXt** model on a test set (by default, 26 classes).

```bash
python Grad-CAM.py --test_dir <path_to_test_images> --output_dir <path_to_output> --model_path <path_to_.pth> [options]
```

**Main arguments:**

| Argument               | Default   | Description |
|------------------------|-----------|-------------|
| `--test_dir`           | `test_dir`| Test images root (one subfolder per class) |
| `--output_dir`         | `output_dir` | Where to save Grad-CAM images and info files |
| `--model_path`         | `model.pth` | Trained ConvNeXt checkpoint (`.pth`) |
| `--num_classes`        | 26        | Number of classes (must match model) |
| `--num_images_per_class` | 10     | Images to process per class; use `-1` for all |
| `--class_names_file`   | None      | JSON file with list of class names (same order as training) |
| `--device`             | auto      | `cuda` or `cpu` |

**Example:**

```bash
python Grad-CAM.py --test_dir ./test_images --output_dir ./gradcam_results --model_path ./best_model.pth --num_classes 26 --num_images_per_class 10
```

**Outputs:**

- Under `output_dir/<class>/`:  
  - `<image_base>_gradcam.jpg` — overlay of heatmap on original image  
  - `<image_base>_info.txt` — true class, predicted class, confidence  
- `output_dir/errors/<true_class>_to_<pred_class>/` — Grad-CAM images for misclassified samples  

**Note:** The script is written for **ConvNeXt**; the target layer is set to `model.features[-1][-1].block[0]`. Other backbones would require adapting the model loading and target layer.

---

## Output Summary

| Script     | Typical outputs |
|-----------|------------------|
| **Train** | Training/validation curves, best model, test confusion matrix, classification report, per-class metrics, history JSON |
| **Predict** | CSV results, confusion matrix, misclassified samples and statistics, optional multi-run summary and accuracy plots |
| **Grad-CAM** | Heatmap overlays per image, prediction info files, error folder for misclassifications |

---

## Citation & Data

- **Training environment:** CAU_HPC (China Agricultural University).  
- **Paper:** Link will be added when published.  
- **Datasets (Dataset 1–3, Tephritid26):** Link will be added when the figshare repository is published.

---

