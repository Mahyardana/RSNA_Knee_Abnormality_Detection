# RSNA Knee Abnormality Detection

This repository contains experiments for detecting abnormalities in knee MRI studies for the RSNA Knee Abnormality Detection competition. The training pipeline reads DICOM series, combines image features with series metadata, and predicts 12 study-level labels.

## Competition result

My highest AUC score in the competition was **0.868 on the public leaderboard**.

## Approach

The main training entry point, [`training_multi_kfold_slow_single.py`](training_multi_kfold_slow_single.py), implements a multi-label training pipeline with:

- DICOM series loading and preprocessing, including resizing and foreground intensity normalization.
- Image features from a ResNet-50 backbone and attention-based pooling across slices.
- Five-fold cross-validation and soft-voting ensemble evaluation.
- Mixed-precision training when CUDA is available.

The 12 prediction labels cover ACL, MCL, medial and lateral meniscus, medial/lateral/patellofemoral osteoarthritis, effusion, synovitis, Baker's cyst, contusion, and fracture.

## Data

The competition data and labels are not included in this repository. Obtain the data through the competition, and prepare:

- Training labels with `StudyInstanceUID` and the 12 target columns.
- Validation labels in the same format.
- Series metadata with `StudyInstanceUID`, `SeriesInstanceUID`, `Anatomical_Plane`, `Fluid_Sensitive`, and `Fat_Suppression`.
- The training DICOM series organized under a directory containing study and series folders.

The script also expects a locally prepared training-label CSV (configured as `TRAIN_CSV`), validation-label CSV (`VAL_CSV`), series metadata CSV (`SERIES_CSV`), and DICOM directory (`TRAIN_IMG_DIR`).

## Setup and running

Use Python with compatible PyTorch and Torchvision versions. Install the remaining packages used by the training pipeline:

```bash
pip install numpy pandas opencv-python pydicom monai scikit-learn scipy timm \
  iterative-stratification tensorboard tqdm zstandard traitlets
```

Update the paths in the `Config` class in `training_multi_kfold_slow_single.py` to match your data and output locations, then run:

```bash
python training_multi_kfold_slow_single.py
```

Training may require a CUDA-enabled GPU and substantial disk space for the MRI data and model checkpoints. The script's configuration, dataset labels, and checkpoint paths reflect this project's local workflow and may need adapting for another environment.
