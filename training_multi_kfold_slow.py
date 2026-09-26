# !pip install pydicom timm albumentations scikit-learn tqdm

import os, math, random, warnings
import cv2
import numpy as np
import pandas as pd
import pydicom
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import io
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from monai.networks.nets import VNet
from monai.transforms import AsDiscrete, generate_spatial_bounding_box
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from scipy.ndimage import zoom
import scipy.ndimage as ndi
import timm
from typing import Dict, Optional, Union
from torch.utils.tensorboard import SummaryWriter

from scipy.ndimage import label, find_objects
from tqdm import tqdm
import multiprocessing as mp
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from torch.utils.data import DataLoader, Dataset, RandomSampler, WeightedRandomSampler
from traitlets import Long
import zstandard as zstd
import gc

try:
    mp.set_start_method('fork', force=True)
except RuntimeError:
    pass
from model_dino_simplified import KneeAbnormalityModelRefactored

torch.set_float32_matmul_precision('high')
warnings.filterwarnings('ignore')


class Config:
    KAGGLE_DIR = Path('/run/media/mahyar/Samsung-990-PRO/Projects/Kaggle/rsna-knee-abnormality-detection')
    PYTHON_DIR = Path('/run/media/mahyar/Samsung-990-PRO/Projects/Python/RSNA Knee Abnormality Detection')
    TRAIN_CSV = PYTHON_DIR / 'llm_generated_labels.csv'
    VAL_CSV = PYTHON_DIR / 'val_gold.csv'
    SERIES_CSV = PYTHON_DIR / 'train_series.csv'
    TRAIN_IMG_DIR = KAGGLE_DIR / 'train_series'
    TRAINING_LABEL = 0
    SAVE_PATH = str(PYTHON_DIR / f"best_model_dino_auc.pth")
    SAVE_PATH2 = str(PYTHON_DIR / f"best_model_dino_loss.pth")
    LOAD_PATH = PYTHON_DIR / f"best_model_dino_label.pth"

    # ── Image processing ──────────────────────────────────────────────
    MAX_SLICES_PER_SERIES = -1
    MAX_SERIES_PER_STUDY = 32
    LABELS = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus',
              'Medial OA', 'Lateral OA', 'PF OA', 'Effusion',
              'Synovitis', "Baker's", 'Contusion', 'Fracture']
    NUM_LABELS = 12

    PLANE_TO_IDX = {'Sagittal': 0, 'Axial': 1, 'Coronal': 2}
    NUM_PLANES = 3

    # ── Model dimensions ──────────────────────────────────────────────
    SLICE_DIM = 518
    META_DIM = 256
    SLICE_ENC_HEADS = 8
    SLICE_ENC_LAYERS = 2
    DROPOUT = 0.3
    CROP_SIZE_MM = 80.0

    # ── K-Fold Cross Validation ───────────────────────────────────────
    NUM_FOLDS = 5
    VAL_FRACTION = 0.10

    # ── Training hyperparameters ──────────────────────────────────────
    BATCH_SIZE = 1
    NUM_WORKERS = 4
    PREFETCH_FACTOR = 8
    GRAD_ACCUM_STEPS = 64
    WARMUP_EPOCHS = 5
    EPOCHS = 100
    LR = 1e-3
    MIN_LR = 1e-5
    WEIGHT_DECAY = 0.01
    USE_AMP = False

    DEBUG_SUBSET = None  # Set to e.g. 100 for fast debugging
    SEED = 32
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    LABEL_CLAMP_THRESHOLD_UPPER = 0.99
    LABEL_CLAMP_THRESHOLD_LOWER = 0.14
    LABEL_SMOOTHENING_THRESHOLD = 0.1


cfg = Config()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(cfg.SEED)


class TorchScriptROIExtractor:

    def __init__(
            self,
            axial_model_path: str,
            sagittal_model_path: str,
            coronal_model_path: str,
            class_maps: Optional[Dict[str, Dict[int, str]]] = None,
            device: str = "cuda",
    ):
        self.device = device

        self.models = {
            "axial": torch.jit.load(
                axial_model_path, map_location=device
            ).eval(),
            "sagittal": torch.jit.load(
                sagittal_model_path, map_location=device
            ).eval(),
            "coronal": torch.jit.load(
                coronal_model_path, map_location=device
            ).eval(),
        }

        self.class_maps = class_maps or {
            "axial": {
                0: "whole",
                1: "anterior",
                2: "medial",
                3: "lateral",
                4: "back",
                5: "front"
            },
            "sagittal": {
                0: "whole",
                1: "anterior",
                2: "medial",
                3: "lateral",
                4: "back",
                5: "front"
            },
            "coronal": {
                0: "whole",
                1: "anterior",
                2: "medial",
                3: "lateral",
                4: "back",
                5: "front"
            },
        }

    @staticmethod
    def get_center_out_slice_indices(num_slices: int):
        mid = num_slices // 2
        yield mid
        for offset in range(1, (num_slices // 2) + 1):
            if mid + offset < num_slices:
                yield mid + offset
            if mid - offset >= 0:
                yield mid - offset

    def crop_subregion(
            self, slice_tensor: torch.Tensor, bbox: torch.Tensor
    ) -> torch.Tensor:
        if bbox is None or len(bbox) == 0:
            return slice_tensor

        x1, y1, x2, y2 = map(int, bbox[:4])
        _, _, h, w = slice_tensor.shape

        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        cropped = slice_tensor[:, :, y1:y2, x1:x2]
        return cropped if cropped.numel() > 0 else slice_tensor

    def decode_yolo_output(
            self, raw_output: torch.Tensor, conf_thresh: float
    ) -> torch.Tensor:
        if raw_output.dim() == 3:
            raw_output = raw_output.squeeze(0)

        preds = raw_output.permute(1, 0)
        boxes_cxcywh = preds[:, :4]
        class_probs = preds[:, 4:]

        if class_probs.max() > 1.0 or class_probs.min() < 0.0:
            class_probs = torch.sigmoid(class_probs)

        scores, class_ids = torch.max(class_probs, dim=1)

        mask = scores >= conf_thresh
        if not mask.any():
            return torch.empty((0, 6), device=self.device)

        boxes_cxcywh = boxes_cxcywh[mask]
        scores = scores[mask]
        class_ids = class_ids[mask]

        x1 = boxes_cxcywh[:, 0] - (boxes_cxcywh[:, 2] / 2)
        y1 = boxes_cxcywh[:, 1] - (boxes_cxcywh[:, 3] / 2)
        x2 = boxes_cxcywh[:, 0] + (boxes_cxcywh[:, 2] / 2)
        y2 = boxes_cxcywh[:, 1] + (boxes_cxcywh[:, 3] / 2)
        boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=1)

        keep_indices = torchvision.ops.batched_nms(
            boxes_xyxy, scores, class_ids, iou_threshold=0.0
        )

        boxes_xyxy = boxes_xyxy[keep_indices]
        scores = scores[keep_indices]
        class_ids = class_ids[keep_indices]

        unique_classes = torch.unique(class_ids)
        final_detections = []

        for c in unique_classes:
            class_mask = class_ids == c
            cls_scores = scores[class_mask]
            cls_boxes = boxes_xyxy[class_mask]

            best_idx = torch.argmax(cls_scores)
            final_detections.append(
                torch.tensor(
                    [
                        cls_boxes[best_idx, 0].item(),
                        cls_boxes[best_idx, 1].item(),
                        cls_boxes[best_idx, 2].item(),
                        cls_boxes[best_idx, 3].item(),
                        cls_scores[best_idx].item(),
                        c.item(),
                    ],
                    device=self.device,
                )
            )

        return torch.stack(final_detections) if final_detections else torch.empty((0, 6), device=self.device)


    @torch.no_grad()
    def extract_best_roi_crops(
            self,
            volume_tensor: torch.Tensor,
            plane: str,
            conf_thresh: float = 0.45,
    ) -> Dict[Union[int, str], torch.Tensor]:
        if plane not in self.models:
            raise ValueError(
                f"Invalid plane '{plane}'. Must be one of {list(self.models.keys())}"
            )

        yolo_model = self.models[plane]
        class_map = self.class_maps.get(plane, {})
        num_slices = volume_tensor.shape[0]
        orig_hpad = 0
        orig_wpad = 0
        orig_w = volume_tensor.shape[-1]
        orig_h = volume_tensor.shape[-2]
        if volume_tensor.shape[-2] != volume_tensor.shape[-1]:
            orig_hpad = volume_tensor.shape[-1] - volume_tensor.shape[-2] if volume_tensor.shape[-1] > volume_tensor.shape[-2] else 0
            orig_wpad = volume_tensor.shape[-2] - volume_tensor.shape[-1] if volume_tensor.shape[-2] > volume_tensor.shape[-1] else 0
            volume_tensor = F.pad(
                volume_tensor,
                (0, orig_wpad, 0, orig_hpad),
                mode="constant",
                value=0)
        volume_tensor = F.interpolate(
            volume_tensor,
            (512, 512),
            mode="bilinear",
            align_corners=False,
        )
        volume_tensor = F.pad(
            volume_tensor,
            (0, 128, 0, 128),
            mode="constant",
            value=0)

        best_slice_idx = num_slices // 2
        best_crops = {}
        highest_conf = -1.0

        for idx in self.get_center_out_slice_indices(num_slices):
            slice_tensor = volume_tensor[idx: idx + 1].to(self.device)
            if cfg.USE_AMP:
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    predictions = yolo_model(slice_tensor)[0]
            else:
                predictions = yolo_model(slice_tensor)[0]

            decodedyolo = self.decode_yolo_output(predictions, conf_thresh)
            if len(decodedyolo) == 0:
                continue
            max_box_conf = decodedyolo[:, 4].max().item()

            if max_box_conf > highest_conf:
                highest_conf = max_box_conf
                best_slice_idx = idx

            if max_box_conf >= conf_thresh:
                best_slice_idx = idx
                break

        best_slice_tensor = volume_tensor[best_slice_idx: best_slice_idx + 1].to(self.device)
        predictions = yolo_model(best_slice_tensor)[0]

        effective_thresh = (
            conf_thresh if highest_conf >= conf_thresh else highest_conf
        )
        decodedyolo = self.decode_yolo_output(predictions, conf_thresh)
        scale_w = (orig_w + orig_wpad) / 640.0
        scale_h = (orig_h + orig_hpad) / 640.0
        scale_tensor = torch.tensor([scale_w, scale_h, scale_w, scale_h], device=decodedyolo.device)
        for pred in decodedyolo:
            if pred[4].item() >= effective_thresh:
                class_id = int(pred[5].item())
                category_key = class_map.get(class_id, class_id)
                best_crops[category_key] = (pred[:4] * scale_tensor)
        del best_slice_tensor, predictions
        return best_crops


# ── Training / Validation Dataset ─────────────────────────────────────────────
class KneeStudyTrainDataset(Dataset):
    def __init__(self, df_studies: pd.DataFrame, df_series: pd.DataFrame, img_dir: Path, cfg, is_train: bool = True):
        self.cfg = cfg
        self.vit = None
        self.roiextractor = None
        self.img_dir = img_dir
        self.is_train = is_train
        self.cache = {}
        maxlen = 0
        self.labels=[]

        self.studies = []
        for _, study_row in df_studies.iterrows():
            study_id = str(study_row['StudyInstanceUID'])

            study_series = df_series[df_series['StudyInstanceUID'] == study_id]
            if len(study_series) > maxlen:
                maxlen = len(study_series)
            if len(study_series) == 0:
                continue

            series_list = []
            for _, row in study_series.iterrows():
                series_list.append({
                    'series_id': str(row['SeriesInstanceUID']),
                    'plane_id': int(cfg.PLANE_TO_IDX.get(str(row['Anatomical_Plane']), 0)),
                    'fluid_sensitive': int(row.get('Fluid_Sensitive', 0)),
                    'fat_suppression': int(row.get('Fat_Suppression', 0))
                })

            labels = study_row[cfg.LABELS].fillna(0).values.astype(np.float32)
            self.labels.append(torch.tensor(labels))
            self.studies.append({
                'study_id': study_id,
                'series': series_list,
                'labels': labels
            })

        print(f"  {'Train' if is_train else 'Val'}: {len(self.studies)} studies")

    def __len__(self):
        return len(self.studies)

    @staticmethod
    def resize_shorter_side_scipy(img, target_short=518):
        h, w = img.shape[:2]
        scaleh = target_short / h
        scalew = target_short / w
        if len(img.shape) == 3:
            zoom_factors = (scaleh, scalew, 1)
        else:
            zoom_factors = (scaleh, scalew)
        return zoom(img, zoom_factors, order=3)

    @staticmethod
    def pad_to_patch_multiple(series_tensor: torch.Tensor, patch_size: int = 14) -> torch.Tensor:
        has_channel = series_tensor.ndim == 4
        if not has_channel:
            series_tensor = series_tensor.unsqueeze(0)

        _, S, H, W = series_tensor.shape
        target_H = math.ceil(H / patch_size) * patch_size
        target_W = math.ceil(W / patch_size) * patch_size

        pad_H = target_H - H
        pad_W = target_W - W
        pad_left = pad_W // 2
        pad_right = pad_W - pad_left
        pad_top = pad_H // 2
        pad_bottom = pad_H - pad_top

        padded = F.pad(series_tensor, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=0)
        return padded if has_channel else padded.squeeze(0)

    @staticmethod
    def safe_load_dicom_array(dcm_path: str):
        try:
            ds = pydicom.dcmread(dcm_path)
            arr = ds.pixel_array.astype(np.float32)
            if ds.get("PhotometricInterpretation", "MONOCHROME2") == "MONOCHROME1":
                arr = arr.max() - arr
            return arr
        except Exception:
            return None

    def __getitem__(self, idx):
        study = self.studies[idx]
        study_id = study['study_id']
        labels = study['labels']

        def identify_sequence_with_binaries(seq_type, fluid_sensitive, fat_suppressed, seq_desc, seq_name):
            if fluid_sensitive and fat_suppressed:
                if 'IR' in seq_type:
                    return 5
            if 'GR' in seq_type:
                return 4
            if 'SE' in seq_type:
                if fluid_sensitive and not fat_suppressed:
                    return 2
                elif not fluid_sensitive and not fat_suppressed:
                    if 'pd' in seq_desc.lower():
                        return 3
                    return 1
            return 0

        all_slices, all_plane_ids, all_fluids, all_fats, all_stype, all_spacing = [], [], [], [], [], []
        for sinfo in study['series']:
            path = self.img_dir / study_id / sinfo['series_id']
            if not path.exists():
                continue
            alldcms = [pydicom.dcmread(str(dcm), stop_before_pixels=True) for dcm in sorted(path.glob('*.dcm'))]
            valid_slices = [
                ds for ds in alldcms
                if "ImagePositionPatient" in ds and "ImageOrientationPatient" in ds
            ]

            if not valid_slices:
                raise ValueError("No valid spatial metadata found in the provided DICOM slices.")

            orient = valid_slices[0].ImageOrientationPatient
            row_cosines = np.array(orient[:3])
            col_cosines = np.array(orient[3:])
            slice_normal = np.cross(row_cosines, col_cosines)

            sorted_slices = sorted(
                valid_slices,
                key=lambda ds: np.dot(np.array(ds.ImagePositionPatient), slice_normal)
            )
            alldcms = sorted_slices
            if self.cfg.MAX_SLICES_PER_SERIES > 0:
                if len(alldcms) > self.cfg.MAX_SLICES_PER_SERIES:
                    indices = np.round(np.linspace(0, len(alldcms) - 1, num=self.cfg.MAX_SLICES_PER_SERIES)).astype(int)
                    dcm_files = [alldcms[i] for i in indices]
                else:
                    dcm_files = alldcms
            else:
                dcm_files = alldcms
            slices = []
            for dcm_file in dcm_files:
                try:
                    img = self.safe_load_dicom_array(dcm_file.filename)
                    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
                    img = np.clip(img, 0.0, 1.0)
                    img = np.stack([img, img, img], axis=0)
                    slices.append(torch.from_numpy(img))
                except Exception:
                    pass

            dcm = dcm_files[0]
            all_spacing.append(dcm.PixelSpacing)
            serietype = identify_sequence_with_binaries(getattr(dcm, "ScanningSequence", ""), sinfo['fluid_sensitive'], sinfo['fat_suppression'], getattr(dcm, "SeriesDescription", ""), getattr(dcm, "SequenceName", ""))
            all_plane_ids.append(sinfo['plane_id'])
            all_fluids.append(sinfo['fluid_sensitive'])
            all_fats.append(sinfo['fat_suppression'])
            all_slices.append(torch.stack(slices))
            all_stype.append(serietype)

        return {
            'study_id': study_id,
            'slices': all_slices,
            'plane_ids': torch.tensor(all_plane_ids, dtype=torch.long),
            'fluid_sensitive': torch.tensor(all_fluids, dtype=torch.long),
            'fat_suppression': torch.tensor(all_fats, dtype=torch.long),
            'serie_type': torch.tensor(all_stype, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.float32),
            'pixel_spacing': all_spacing,
        }


def collate_fn(batch: List[Dict]) -> Dict:
    studies = [x["study_id"] for x in batch]
    slices = [x["slices"] for x in batch]
    plane_ids = [x["plane_ids"] for x in batch]
    fluids = [x["fluid_sensitive"] for x in batch]
    fats = [x["fat_suppression"] for x in batch]
    stype = [x["serie_type"] for x in batch]
    labels = [x["labels"] for x in batch]
    pixel_spacing = [x["pixel_spacing"] for x in batch]
    return {
        "study_id": studies,
        "labels": torch.stack(labels, dim=0),
        "plane_ids": plane_ids,
        "fluid_sensitive": fluids,
        "fat_suppression": fats,
        "serie_type": stype,
        "slices": slices,
        "pixel_spacing": pixel_spacing,
    }


class WeightedBCELoss(nn.Module):
    def __init__(self, pos_weight=None):
        super().__init__()
        self.pos_weight = pos_weight

    def forward(self, pred, target):
        if self.pos_weight is not None:
            return F.binary_cross_entropy_with_logits(pred, target, pos_weight=self.pos_weight)
        return F.binary_cross_entropy_with_logits(pred, target)


def compute_class_weights(df, cfg):
    weights = []
    for label in cfg.LABELS:
        pos = (df[label] == 1).sum()
        neg = (df[label] == 0).sum()
        weights.append(neg / (pos + 1e-6))
    return torch.tensor(weights, dtype=torch.float32)


featurecache = {}

def extract_image_features(batch, visionmodel, roiextractor, caching=True):
    batched_slices = batch["slices"]
    batched_plane_ids = batch["plane_ids"]
    batched_fluids = batch["fluid_sensitive"]
    batched_fats = batch["fat_suppression"]
    batched_stype = batch["serie_type"]
    batched_labels = batch["labels"]
    batched_studies = batch["study_id"]
    batched_spacing = batch["pixel_spacing"]

    studies, slices, slicemasks, plane_ids, fluids, fats, stype, plane_crop_mixed = [], [], [], [], [], [], [], []
    for study, all_slices, all_plane_ids, all_fluids, all_fats, all_stype, all_spacing in zip(batched_studies, batched_slices, batched_plane_ids, batched_fluids, batched_fats, batched_stype, batched_spacing):
        study_id = study
        if study_id not in featurecache:
            # xxxxxx = [
            #     roiextractor.extract_best_roi_crops(x, "sagittal" if y == 0 else "axial" if y == 1 else "coronal")
            #     for x, y in zip(all_slices, all_plane_ids)
            # ]

            def crop_and_resize_box(
                    input_tensor: torch.Tensor,
                    box: list = None,
                    pixelspacing: list = [0.5, 0.5],
                    sizemm: float = 140.0,
                    boundary_expansion: float = 0.0
            ):
                if input_tensor.dim() == 3:
                    input_tensor = input_tensor.unsqueeze(0)

                _, _, h_img, w_img = input_tensor.shape

                # 1. Parse bounding box or default to full image
                if box is not None and len(box) >= 4:
                    minx, minh, maxx, maxh = box[:4]
                    xlen, hlen = maxx - minx, maxh - minh
                    minx = max(0, int(minx - xlen * boundary_expansion))
                    minh = max(0, int(minh - hlen * boundary_expansion))
                    maxx = min(w_img, int(maxx + xlen * boundary_expansion))
                    maxh = min(h_img, int(maxh + hlen * boundary_expansion))
                    cropped = input_tensor[:, :, minh:maxh, minx:maxx]
                    if cropped.shape[-2] == 0 or cropped.shape[-1] == 0:
                        cropped = input_tensor
                else:
                    cropped = input_tensor

                c_h, c_w = cropped.shape[-2], cropped.shape[-1]
                device = input_tensor.device
                mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

                # Helper function: Centered replicate padding (prevents high-contrast black edges)
                def pad_to_square(img):
                    h, w = img.shape[-2], img.shape[-1]
                    if h == w:
                        return img
                    diff = abs(h - w)
                    if h < w:
                        pad_top = diff // 2
                        pad_bottom = diff - pad_top
                        return F.pad(img, (0, 0, pad_top, pad_bottom), mode='replicate')
                    else:
                        pad_left = diff // 2
                        pad_right = diff - pad_left
                        return F.pad(img, (pad_left, pad_right, 0, 0), mode='replicate')

                listofcrops = []

                # 2. Whole Image / Full Bounding Box Crop
                squared_whole = pad_to_square(cropped)
                resized_whole = F.interpolate(squared_whole, (512, 512), mode='bilinear', align_corners=False)
                listofcrops.append((resized_whole - mean) / std)

                # # 3. Calculate 140mm Patch Dimensions in Pixels
                # crop_h = int(sizemm / pixelspacing[0])
                # crop_w = int(sizemm / pixelspacing[1])
                #
                # # If image dimension is smaller than patch size, return whole image crop
                # if c_h < crop_h or c_w < crop_w:
                #     return listofcrops
                #
                # stride_h = crop_h // 2
                # stride_w = crop_w // 2
                #
                # # 4. Generate Coordinates with Boundary Clamping (Guarantees exact 140mm patches)
                # h_starts = list(range(0, c_h - crop_h + 1, stride_h))
                # if h_starts[-1] + crop_h < c_h:
                #     h_starts.append(c_h - crop_h)
                #
                # w_starts = list(range(0, c_w - crop_w + 1, stride_w))
                # if w_starts[-1] + crop_w < c_w:
                #     w_starts.append(c_w - crop_w)
                #
                # # 5. Extract Sliding 140mm Patches
                # for h in h_starts:
                #     for w in w_starts:
                #         patch = cropped[:, :, h: h + crop_h, w: w + crop_w]
                #         resized_patch = F.interpolate(patch, (224, 224), mode='bilinear', align_corners=False)
                #         listofcrops.append((resized_patch - mean) / std)

                return listofcrops

            slices_feats, new_plane_crop_mixed, new_plane_ids, new_fluids, new_fats, new_stype = [], [], [], [], [], []
            rev_class_map = {"whole": 0, "anterior": 1, "medial": 2, "lateral": 3, "back": 4, "front": 5}
            # expansionfactors=[0.1]#-0.1,0.0,
            # for expansionfactor in expansionfactors:
            for idx, (y, z) in enumerate(zip(all_slices, all_spacing)):
                # for k, v in x.items():
                #     tovit = crop_and_resize_box(y, v, z, boundryexpantion=expansionfactor)
                #     if tovit is not None:
                #         slices_feats.append(tovit.detach().cpu())
                #         new_plane_crop_mixed.append([all_plane_ids[idx], rev_class_map[k]])
                #         new_plane_ids.append(all_plane_ids[idx])
                #         new_fluids.append(all_fluids[idx])
                #         new_fats.append(all_fats[idx])
                #         new_stype.append(all_stype[idx])
                #         del tovit
                tovit = crop_and_resize_box(y, torch.tensor([0, 0, y.shape[-1], y.shape[-2]], dtype=torch.float), z,sizemm=cfg.CROP_SIZE_MM)
                for vitout in tovit:
                    slices_feats.append(vitout.detach().cpu())
                    new_plane_crop_mixed.append([all_plane_ids[idx], rev_class_map['whole']])
                    new_plane_ids.append(all_plane_ids[idx])
                    new_fluids.append(all_fluids[idx])
                    new_fats.append(all_fats[idx])
                    new_stype.append(all_stype[idx])
                del tovit

            all_plane_ids = new_plane_ids
            all_fluids = new_fluids
            all_fats = new_fats
            all_stype = new_stype
            seriesizes = [slices_feats[x].shape[0] for x in range(len(slices_feats))]
            CHUNK_SIZE = 64
            tovit = torch.cat(slices_feats, dim=0).to(cfg.DEVICE)
            dino_tokens = []
            with torch.no_grad():
                for i in range(0, tovit.size(0), CHUNK_SIZE):
                    chunk = tovit[i: i + CHUNK_SIZE]
                    if cfg.USE_AMP:
                        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                            chunk_out = visionmodel.forward_features(chunk)
                    else:
                        chunk_out = visionmodel.forward_features(chunk)
                    dino_tokens.append(chunk_out.detach().cpu())
                    # patches = chunk_out[:, 1:, :]
                    #
                    # # Rearrange to spatial grid: (Batch, Channels, Height, Width)
                    # # (B, 196, 384) -> (B, 384, 196) -> (B, 384, 14, 14)
                    # spatial_grid = patches.transpose(1, 2).view(-1, 384, 14, 14)
                    # top_left = spatial_grid[:, :, 0:7, 0:7]  # Rows 0-6, Cols 0-6
                    # top_right = spatial_grid[:, :, 0:7, 7:14]  # Rows 0-6, Cols 7-13
                    # bottom_left = spatial_grid[:, :, 7:14, 0:7]  # Rows 7-13, Cols 0-6
                    # bottom_right = spatial_grid[:, :, 7:14, 7:14]  # Rows 7-13, Cols 7-13
                    # middle = spatial_grid[:, :, 3:10, 3:10]  # Rows 3-9, Cols 3-9
                    # avg_top_left = top_left.mean(dim=[-2, -1])
                    # avg_top_right = top_right.mean(dim=[-2, -1])
                    # avg_bottom_left = bottom_left.mean(dim=[-2, -1])
                    # avg_bottom_right = bottom_right.mean(dim=[-2, -1])
                    # avg_middle = middle.mean(dim=[-2, -1])
                    # cls_token = chunk_out[:, 0, :]
                    # combined_features = torch.cat([
                    #     cls_token,
                    #     avg_middle,
                    #     avg_top_left,
                    #     avg_top_right,
                    #     avg_bottom_left,
                    #     avg_bottom_right
                    # ], dim=-1)
                    # dino_tokens.append(combined_features.detach().cpu())
            dino_out = torch.cat(dino_tokens, dim=0).cpu()
            slices_feats_tmp = list(torch.split(dino_out, seriesizes, dim=0))

            slices_feats=slices_feats_tmp
            # for dinofeats in slices_feats_tmp:
            #     clstoken=dinofeats[:,0,:]
            #     tokens=[]
            #     dinotrans=dinofeats[:,1:,:].view(dinofeats.shape[0], 32,32,384)
            #     for x in range(0,32,8):
            #         for y in range(0,32,8):
            #             ct=dinotrans[:,x:x+8,y:y+8,:]
            #             ctavg=ct.mean(dim=[-3,-2])
            #             tokens.append(ctavg.unsqueeze(1))
            #     tokens.append(clstoken.unsqueeze(1))
            #     slices_feats.append(torch.cat(tokens, dim=1))
            del tovit
            if caching:
                featurecache[study_id] = slices_feats, new_plane_crop_mixed, new_plane_ids, new_fluids, new_fats, new_stype
        else:
            slices_feats, new_plane_crop_mixed, new_plane_ids, new_fluids, new_fats, new_stype = featurecache[study_id]
        if len(all_slices) == 0:
            all_slices.append(torch.zeros(3, 224, 224))
            all_plane_ids.append(0)
            all_fluids.append(0)
            all_fats.append(0)
            all_stype.append(0)

        max_slice = max(slice.shape[0] for slice in all_slices)
        masks = []
        for idx in range(len(slices_feats)):
            if slices_feats[idx].shape[0] < max_slice:
                s = slices_feats[idx].shape[0]
                sizediff = max_slice - s
                slices_feats[idx] = torch.cat([slices_feats[idx], torch.zeros((sizediff, slices_feats[idx].shape[-2], slices_feats[idx].shape[-1]))], dim=0)
                masks.append(torch.cat([torch.ones(s, dtype=torch.bool), torch.zeros(sizediff, dtype=torch.bool)]))
            else:
                masks.append(torch.ones(slices_feats[idx].shape[0], dtype=torch.bool))
        incpufeats = torch.stack(slices_feats).cpu()
        studies.append(study_id)
        slices.append(incpufeats)
        slicemasks.append(torch.stack(masks))
        plane_ids.append(torch.tensor(new_plane_ids, dtype=torch.long))
        fats.append(torch.tensor(new_fats, dtype=torch.long))
        fluids.append(torch.tensor(new_fluids, dtype=torch.long))
        stype.append(torch.tensor(new_stype, dtype=torch.long))
        plane_crop_mixed.append(torch.tensor(new_plane_crop_mixed, dtype=torch.long))
    maxslicesize = max([slice.shape[-3] for slice in slices])
    maxseriesize = max([slice.shape[-4] for slice in slices])
    paddedslicemasks, paddedplane_ids, paddedfluids, paddedfats, paddedstype, paddedplane_crop_mixed, seriemasks = [], [], [], [], [], [], []
    for i in range(len(slices)):
        newslice, newslicemask = slices[i], slicemasks[i]
        newplane_id, newfluid, newfat = plane_ids[i], fluids[i], fats[i]
        newplane_crop_mixed, newstype = plane_crop_mixed[i], stype[i]
        newseriemask = torch.ones_like(newplane_id, dtype=torch.bool)
        if newslice.shape[1] < maxslicesize:
            slicediffer = maxslicesize - newslice.shape[1]
            newslicemask = torch.cat(
                (newslicemask, torch.zeros((newslicemask.shape[0], slicediffer), dtype=torch.bool)), dim=1)
        if newslice.shape[0] < maxseriesize:
            s = newslice.shape[0]
            seriediffer = maxseriesize - s
            newslice = torch.cat([newslice, torch.zeros((seriediffer, newslice.shape[-3], newslice.shape[-2], newslice.shape[-1]))],
                                 dim=0)
            newslicemask = torch.cat(
                (newslicemask, torch.zeros((seriediffer, newslicemask.shape[1]), dtype=torch.bool)), dim=0)
            paddingones = torch.ones([seriediffer])
            newplane_id = torch.cat((newplane_id, paddingones * 10.0), dim=0)
            newfluid = torch.cat((newfluid, paddingones * 10.0), dim=0)
            newfat = torch.cat((newfat, paddingones * 10.0), dim=0)
            newstype = torch.cat((newstype, paddingones * 10.0), dim=0)
            newseriemask = torch.cat((newseriemask, torch.zeros(seriediffer, dtype=torch.bool)), dim=0)
            newplane_crop_mixed = torch.cat((newplane_crop_mixed, torch.ones((seriediffer, 2)) * 10.0), dim=0)
        paddedslicemasks.append(newslicemask)
        paddedplane_ids.append(newplane_id)
        paddedfluids.append(newfluid)
        paddedfats.append(newfat)
        seriemasks.append(newseriemask)
        paddedstype.append(newstype)
        paddedplane_crop_mixed.append(newplane_crop_mixed)
    kfold_padded_slices=[]
    paddedslices = []
    for i in range(len(slices)):
        newslice = slices[i]
        if newslice.shape[1] < maxslicesize:
            slicediffer = maxslicesize - newslice.shape[1]
            newslice = torch.cat([newslice, torch.zeros((newslice.shape[0], slicediffer, newslice.shape[-2], newslice.shape[-1]))], dim=1)
        if newslice.shape[0] < maxseriesize:
            s = newslice.shape[0]
            seriediffer = maxseriesize - s
            newslice = torch.cat([newslice, torch.zeros((seriediffer, newslice.shape[1], newslice.shape[-2], newslice.shape[-1]))], dim=0)
        paddedslices.append(newslice)

    return {
        "study_id": studies,
        "labels": batched_labels,
        "slicemasks": torch.stack(paddedslicemasks, dim=0),
        "plane_ids": torch.stack(paddedplane_ids, dim=0).long(),
        "fluid_sensitive": torch.stack(paddedfluids, dim=0).long(),
        "fat_suppression": torch.stack(paddedfats, dim=0).long(),
        "serie_type": torch.stack(paddedstype, dim=0).long(),
        "seriemasks": torch.stack(seriemasks, dim=0),
        "slices": torch.stack(paddedslices, dim=0),
        "plane_crop_mixed": torch.stack(paddedplane_crop_mixed, dim=0).long(),
    }


# ── Train & Eval Epochs ───────────────────────────────────────────────────────
def train_epoch(model, loader, visionmodel, roiextractor, optimizer, criterion, scaler, cfg, iscached=True,dinofold=0):
    model.train()
    losses = []
    optimizer.zero_grad()

    pbar = tqdm(loader, desc='Train', leave=False)
    for batch_idx, batch in enumerate(pbar):
        if not iscached:
            with torch.no_grad():
                batched_features = extract_image_features(batch, visionmodel, roiextractor, caching=False)
            batch = batched_features

        labels = batch["labels"].to(cfg.DEVICE)
        # labels[labels >= cfg.LABEL_CLAMP_THRESHOLD_UPPER] = 1.0
        # labels[labels <= cfg.LABEL_CLAMP_THRESHOLD_LOWER] = 0.0
        # labelsmasks = (labels >= cfg.LABEL_CLAMP_THRESHOLD_UPPER) | (labels <= cfg.LABEL_CLAMP_THRESHOLD_LOWER)
        # labels[~labelsmasks] = 0.5
        slices = batch["slices"].to(cfg.DEVICE)
        slicemasks = batch["slicemasks"].to(cfg.DEVICE)
        fats = batch["fat_suppression"].to(cfg.DEVICE)
        fluids = batch["fluid_sensitive"].to(cfg.DEVICE)
        plane_ids = batch["plane_ids"].to(cfg.DEVICE)
        plane_crop_mixed = batch["plane_crop_mixed"].to(cfg.DEVICE)
        seriemasks = batch["seriemasks"].to(cfg.DEVICE) if "seriemasks" in batch else slicemasks[:, :, 0]
        serietype = batch["serie_type"].to(cfg.DEVICE)

        if torch.isnan(slices).any():
            print("NaN Inputs!")
        else:
            if cfg.USE_AMP:
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    logits = model.forward(slices, slicemasks, seriemasks, plane_ids, fluids, fats, plane_crop_mixed, serietype)
            else:
                logits = model.forward(slices, slicemasks, seriemasks, plane_ids, fluids, fats, plane_crop_mixed, serietype)
            loss = criterion(logits, labels)
            loss = loss / cfg.GRAD_ACCUM_STEPS
            loss.backward()
            if (batch_idx + 1) % cfg.GRAD_ACCUM_STEPS == 0 or (batch_idx + 1) == len(loader):
                optimizer.step()
                optimizer.zero_grad()

            if torch.isnan(logits).any():
                print("❌ Model produced NaN outputs!")
            losses.append(loss.item() * cfg.GRAD_ACCUM_STEPS)
            pbar.set_postfix({'loss': f'{np.mean(losses):.4f}'})

    return np.mean(losses)


@torch.no_grad()
def eval_epoch(model, loader, visionmodel, roiextractor, criterion, cfg, iscached=True,dinofold=0):
    model.eval()
    losses = []
    all_preds, all_labels = [], []

    for batch in tqdm(loader, desc="Val", leave=False):
        if not iscached:
            with torch.no_grad():
                batched_features = extract_image_features(batch, visionmodel, roiextractor, caching=False)
            batch = batched_features

        labels = batch["labels"].to(cfg.DEVICE)
        slices = batch["slices"].to(cfg.DEVICE)
        slicemasks = batch["slicemasks"].to(cfg.DEVICE)
        fats = batch["fat_suppression"].to(cfg.DEVICE)
        fluids = batch["fluid_sensitive"].to(cfg.DEVICE)
        plane_ids = batch["plane_ids"].to(cfg.DEVICE)
        plane_crop_mixed = batch["plane_crop_mixed"].to(cfg.DEVICE)
        seriemasks = batch["seriemasks"].to(cfg.DEVICE) if "seriemasks" in batch else slicemasks[:, :, 0]
        serietype = batch["serie_type"].to(cfg.DEVICE)

        if cfg.USE_AMP:
            with torch.amp.autocast("cuda", dtype=torch.float16):
                logits = model.forward(slices, slicemasks, seriemasks, plane_ids, fluids, fats, plane_crop_mixed, serietype)
        else:
            logits = model.forward(slices, slicemasks, seriemasks, plane_ids, fluids, fats, plane_crop_mixed, serietype)
        loss = criterion(logits.float(), labels)
        losses.append(loss.item())

        if torch.isnan(logits).any():
            print("❌ Model produced NaN outputs!")

        preds = torch.sigmoid(logits).cpu().numpy()
        all_preds.append(preds)
        all_labels.append(labels.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    binary_labels = (all_labels >= 0.5).astype(int)
    val_loss = np.mean(losses)
    try:
        val_auc = roc_auc_score(binary_labels, all_preds)
    except ValueError:
        val_auc = 0.5

    return val_loss, val_auc, all_preds, all_labels


class AsymmetricLossOptimized(nn.Module):
    """
    Numerically stable Asymmetric Loss for Continuous / Soft Targets (y in [0.0, 1.0]).
    Processes soft labels (e.g. 0.5, 0.15) naturally without zeroing them out.
    """

    def __init__(self, gamma_neg: float = 4.0, gamma_pos: float = 1.0, clip: float = 0.05, eps: float = 1e-5):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Cast to float32 to prevent FP16 underflow (-inf) under AMP
        x = x.float()
        y = y.float()

        # 1. Compute probabilities
        p = torch.sigmoid(x)
        xs_neg = 1.0 - p

        # 2. Asymmetric Margin Clipping for Negatives
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1.0)

        # 3. Compute soft log-loss components
        # Positive term uses F.logsigmoid to safely handle negative logits
        loss_pos = -y * F.logsigmoid(x)
        loss_neg = -(1.0 - y) * torch.log(xs_neg.clamp(min=self.eps))

        # 4. Apply Asymmetric Focusing Weights
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            with torch.no_grad():
                # Positive focusing weight: (1 - p)^gamma_pos
                w_pos = torch.pow((1.0 - p).clamp(min=0.0), self.gamma_pos)

                # Negative focusing weight: (1 - xs_neg)^gamma_neg
                w_neg = torch.pow((1.0 - xs_neg).clamp(min=0.0), self.gamma_neg)

            loss_pos = loss_pos * w_pos
            loss_neg = loss_neg * w_neg

        # Combined soft target loss
        loss = loss_pos + loss_neg

        return loss.mean()


def build_optimizer_with_decay_exclusion(model, lr=2e-4, weight_decay=0.01):
    decay_params, no_decay_params = [], []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim < 2 or name.endswith(".bias") or "norm" in name.lower():
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optimizer_grouped_parameters = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(optimizer_grouped_parameters, lr=lr)


class CachedEmbeddingDataset(torch.utils.data.Dataset):
    def __init__(self, pt_file, cfg, label_filter=True, labels_replacement=None):
        self.slicesdir = "slicesfeats/"
        self.valid_masks = None
        data = torch.load(pt_file)

        studies = data["study_id"]
        slicemasks = data["slicemasks"] if "slicemasks" in data else data["masks"]
        plane_ids = data["plane_ids"]
        fluids = data["fluid_sensitive"]
        fats = data["fat_suppression"]
        stypes = data["serie_type"]
        if labels_replacement is None:
            labels = data["labels"]
            labels = torch.cat(labels, dim=0)
        else:
            labels = []
            for study in studies:
                filtered_df = labels_replacement.loc[labels_replacement['StudyInstanceUID'] == study[0]]
                labels.append(torch.tensor(filtered_df[cfg.LABELS].fillna(0).values.astype(np.float32)))
            labels = torch.cat(labels, dim=0)

        plane_crop_mixed = data["plane_crop_mixed"]
        if label_filter:
            mask = (labels[:, cfg.TRAINING_LABEL] >= 0.99) | (labels[:, cfg.TRAINING_LABEL] <= 0.15)
        else:
            mask = torch.ones(labels.shape[0], dtype=torch.bool)

        self.studies, self.slicemasks, self.plane_ids, self.fluids, self.fats, self.stype, self.labels, self.plane_crop_mixed = [], [], [], [], [], [], [], []
        self.cache=[]

        for i in range(len(studies)):
            self.studies.append(studies[i][0])
            self.slicemasks.append(slicemasks[i].squeeze(0))
            self.plane_ids.append(plane_ids[i].squeeze(0))
            self.fluids.append(fluids[i].squeeze(0))
            self.fats.append(fats[i].squeeze(0))
            self.labels.append(labels[i].squeeze(0))
            self.plane_crop_mixed.append(plane_crop_mixed[i].squeeze(0))
            self.stype.append(stypes[i].squeeze(0))
            with open(self.slicesdir + f"{studies[i][0]}.pt", "rb") as f:
                buffer = io.BytesIO(f.read())
                slice = torch.load(buffer)
                self.cache.append(slice.squeeze(0))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # if idx in self.cache:
        #     return self.cache[idx]
        # if self.studies[idx] in self.cache:
        #     slice = self.cache[self.studies[idx]]
        # else:
        # with open(self.slicesdir + f"{self.studies[idx]}.pt", "rb") as f:
        #     buffer = io.BytesIO(f.read())
        #     slice = torch.load(buffer)
        # filtermask=self.plane_crop_mixed[idx][:,1]==0
        # slice=slice.squeeze(0)[filtermask]
        return self.studies[idx], self.cache[idx], self.slicemasks[idx], self.plane_ids[idx], self.fluids[idx], \
            self.fats[idx], self.labels[idx], self.plane_crop_mixed[idx], self.stype[idx], self.valid_masks[idx] if self.valid_masks is not None else None
        # return self.cache[idx]


def cache_collate_fn(batch):
    # Single-pass unpacking via zip (faster than list comprehensions)
    (studies, kfold_slices, slicemasks, plane_ids,
     fluids, fats, labels, plane_crop_mixed, stype, valid_masks) = zip(*batch)

    # Fast path for batch size 1
    if len(batch) == 1:
        newplane_ids = torch.stack(plane_ids, dim=0).long()
        slices=torch.stack(kfold_slices, dim=0)
        return {
            "study_id": list(studies),
            "labels": torch.stack(labels, dim=0),
            "slicemasks": torch.stack(slicemasks, dim=0),
            "plane_ids": newplane_ids,
            "fluid_sensitive": torch.stack(fluids, dim=0).long(),
            "fat_suppression": torch.stack(fats, dim=0).long(),
            "seriemasks": torch.ones_like(newplane_ids, dtype=torch.bool),
            "slices": slices,
            "plane_crop_mixed": torch.stack(plane_crop_mixed, dim=0).long(),
            "serie_type": torch.stack(stype, dim=0).long(),
            "valid_masks": torch.stack(valid_masks, dim=0) if valid_masks[0] is not None else None,
        }

    # High-performance path for Batch Size > 1
    B = len(batch)

    # Calculate maximum sequence lengths across the batch
    maxseriesize = max(s.shape[0] for s in kfold_slices)
    maxslicesize = max(s.shape[1] for s in kfold_slices)
    channels=kfold_slices[0].shape[-1]
    patches=kfold_slices[0].shape[-2]
    device=kfold_slices[0].device
    first_slice=kfold_slices[0]

    # 1. Pre-allocate target tensors filled with default pad values
    padded_slices = torch.zeros((B, maxseriesize, maxslicesize, patches, channels), dtype=first_slice.dtype,
                                device=device)
    padded_slicemasks = torch.zeros((B, maxseriesize, maxslicesize), dtype=torch.bool, device=device)
    seriemasks = torch.zeros((B, maxseriesize), dtype=torch.bool, device=device)

    # 2. Pre-allocate integer metadata directly as long with pad value 10
    padded_plane_ids = torch.full((B, maxseriesize), 10, dtype=torch.long, device=device)
    padded_fluids = torch.full((B, maxseriesize), 10, dtype=torch.long, device=device)
    padded_fats = torch.full((B, maxseriesize), 10, dtype=torch.long, device=device)
    padded_stype = torch.full((B, maxseriesize), 10, dtype=torch.long, device=device)
    padded_plane_crop = torch.full((B, maxseriesize, 2), 10, dtype=torch.long, device=device)

    # 3. Populate pre-allocated tensors via direct in-place indexing
    for i in range(B):
        s_len = plane_ids[i].shape[0]
        l_len = slicemasks[i].shape[1]

        # Copy metadata & masks directly into target positions
        padded_plane_ids[i, :s_len] = plane_ids[i].long()
        padded_fluids[i, :s_len] = fluids[i].long()
        padded_fats[i, :s_len] = fats[i].long()
        padded_stype[i, :s_len] = stype[i].long()
        padded_plane_crop[i, :s_len] = plane_crop_mixed[i].long()
        seriemasks[i, :s_len] = True
        padded_slicemasks[i, :s_len, :l_len] = slicemasks[i]
        padded_slices[i, :s_len, :l_len, :] = kfold_slices[i]

    return {
        "study_id": list(studies),
        "labels": torch.stack(labels, dim=0),
        "slicemasks": padded_slicemasks,
        "plane_ids": padded_plane_ids,
        "fluid_sensitive": padded_fluids,
        "fat_suppression": padded_fats,
        "seriemasks": seriemasks,
        "slices": padded_slices,
        "plane_crop_mixed": padded_plane_crop,
        "serie_type": padded_stype,
        "valid_masks": torch.stack(valid_masks, dim=0) if valid_masks[0] is not None else None,
    }
# def cache_collate_fn(batch):
#     studies = [x[0] for x in batch]
#     slices = [x[1] for x in batch]
#     slicemasks = [x[2] for x in batch]
#     plane_ids = [x[3] for x in batch]
#     fluids = [x[4] for x in batch]
#     fats = [x[5] for x in batch]
#     labels = [x[6] for x in batch]
#     plane_crop_mixed = [x[7] for x in batch]
#     stype = [x[8] for x in batch]
#     valid_masks = [x[9] for x in batch]
#     if cfg.BATCH_SIZE==1:
#         newplane_ids=torch.stack(plane_ids, dim=0).long()
#         return {
#             "study_id": studies,
#             "labels": torch.stack(labels, dim=0),
#             "slicemasks": torch.stack(slicemasks, dim=0),
#             "plane_ids": newplane_ids,
#             "fluid_sensitive": torch.stack(fluids, dim=0).long(),
#             "fat_suppression": torch.stack(fats, dim=0).long(),
#             "seriemasks": torch.ones_like(newplane_ids, dtype=torch.bool),
#             "slices": torch.stack(slices, dim=0),
#             "plane_crop_mixed": torch.stack(plane_crop_mixed, dim=0).long(),
#             "serie_type": torch.stack(stype, dim=0).long(),
#             "valid_masks": torch.stack(valid_masks, dim=0) if valid_masks[0] is not None else None,
#         }
#
#     # for i in range(len(slices)):
#     #     unpaddedmask = (slicemasks[i][:, 0] == True)
#     #     slices[i] = torch.stack([slices[i][x][unpaddedmask] for x in range(len(slices[i]))])
#     #     slicemasks[i] = slicemasks[i][unpaddedmask]
#     #     fluids[i] = fluids[i][unpaddedmask]
#     #     fats[i] = fats[i][unpaddedmask]
#     #     plane_ids[i] = plane_ids[i][unpaddedmask]
#     #     stype[i] = stype[i][unpaddedmask]
#     #     plane_crop_mixed[i] = plane_crop_mixed[i][unpaddedmask]
#     #     maxstudyslices = max([torch.sum(slicemask, dim=0).item() for slicemask in slicemasks[i]])
#     #     slices[i] = slices[i][:,:, :maxstudyslices,:]
#     #     slicemasks[i] = slicemasks[i][:, :maxstudyslices]
#
#     maxslicesize = max([slice.shape[-2] for slice in slices])
#     maxseriesize = max([slice.shape[-3] for slice in slices])
#     kfold_slices=slices
#     paddedslicemasks, paddedplane_ids, paddedfluids, paddedfats, paddedstype, paddedplane_crop_mixed, seriemasks = [], [], [], [], [], [], []
#     slices=[x[0] for x in slices]
#     for i in range(len(slices)):
#         newslice, newslicemask = slices[i], slicemasks[i]
#         newplane_id, newfluid, newfat = plane_ids[i], fluids[i], fats[i]
#         newplane_crop_mixed, newstype = plane_crop_mixed[i], stype[i]
#         newseriemask = torch.ones_like(newplane_id, dtype=torch.bool)
#         if newslice.shape[1] < maxslicesize:
#             slicediffer = maxslicesize - newslice.shape[1]
#             newslicemask = torch.cat(
#                 (newslicemask, torch.zeros((newslicemask.shape[0], slicediffer), dtype=torch.bool)), dim=1)
#         if newslice.shape[0] < maxseriesize:
#             s = newslice.shape[0]
#             seriediffer = maxseriesize - s
#             newslice = torch.cat([newslice, torch.zeros((seriediffer, newslice.shape[1], newslice.shape[-1]))],
#                                  dim=0)
#             newslicemask = torch.cat(
#                 (newslicemask, torch.zeros((seriediffer, newslicemask.shape[1]), dtype=torch.bool)), dim=0)
#             paddingones = torch.ones([seriediffer])
#             newplane_id = torch.cat((newplane_id, paddingones * 10.0), dim=0)
#             newfluid = torch.cat((newfluid, paddingones * 10.0), dim=0)
#             newfat = torch.cat((newfat, paddingones * 10.0), dim=0)
#             newstype = torch.cat((newstype, paddingones * 10.0), dim=0)
#             newseriemask = torch.cat((newseriemask, torch.zeros(seriediffer, dtype=torch.bool)), dim=0)
#             newplane_crop_mixed = torch.cat((newplane_crop_mixed, torch.ones((seriediffer, 2)) * 10.0), dim=0)
#         paddedslicemasks.append(newslicemask)
#         paddedplane_ids.append(newplane_id)
#         paddedfluids.append(newfluid)
#         paddedfats.append(newfat)
#         seriemasks.append(newseriemask)
#         paddedstype.append(newstype)
#         paddedplane_crop_mixed.append(newplane_crop_mixed)
#     kfold_padded_slices=[]
#     for slices in kfold_slices:
#         paddedslices = []
#         for i in range(len(slices)):
#             newslice = slices[i]
#             if newslice.shape[1] < maxslicesize:
#                 slicediffer = maxslicesize - newslice.shape[1]
#                 newslice = torch.cat([newslice, torch.zeros((newslice.shape[0], slicediffer, newslice.shape[-1]))], dim=1)
#             if newslice.shape[0] < maxseriesize:
#                 s = newslice.shape[0]
#                 seriediffer = maxseriesize - s
#                 newslice = torch.cat([newslice, torch.zeros((seriediffer, newslice.shape[1], newslice.shape[-1]))], dim=0)
#             paddedslices.append(newslice)
#         kfold_padded_slices.append(paddedslices)
#     paddedslices=[torch.stack(x) for x in kfold_padded_slices]
#
#     return {
#         "study_id": studies,
#         "labels": torch.stack(labels, dim=0),
#         "slicemasks": torch.stack(paddedslicemasks, dim=0),
#         "plane_ids": torch.stack(paddedplane_ids, dim=0).long(),
#         "fluid_sensitive": torch.stack(paddedfluids, dim=0).long(),
#         "fat_suppression": torch.stack(paddedfats, dim=0).long(),
#         "seriemasks": torch.stack(seriemasks, dim=0),
#         "slices": torch.stack(paddedslices, dim=0),
#         "plane_crop_mixed": torch.stack(paddedplane_crop_mixed, dim=0).long(),
#         "serie_type": torch.stack(paddedstype, dim=0).long(),
#         "valid_masks": torch.stack(valid_masks, dim=0) if valid_masks[0] is not None else None,
#     }


@torch.no_grad()
def precompute_and_save(model, visionmodel, roiextractor, loader, save_path, cfg):
    model.eval()
    all_labels, all_study, all_masks, all_plane_ids, all_fluid, all_fat, all_planecrop, all_stype = [], [], [], [], [], [], [], []
    for batch in tqdm(loader, desc="Caching Features"):
        with torch.no_grad():
            batched_features = extract_image_features(batch, visionmodel, roiextractor, caching=False)

        for i in range(len(batched_features["study_id"])):
            all_labels.append(batched_features["labels"][i].unsqueeze(0).detach().cpu())
            all_study.append([batched_features["study_id"][i]])
            all_masks.append(batched_features["slicemasks"][i].unsqueeze(0).detach().cpu())
            all_plane_ids.append(batched_features["plane_ids"][i].unsqueeze(0).detach().cpu())
            all_fluid.append(batched_features["fluid_sensitive"][i].unsqueeze(0).detach().cpu())
            all_fat.append(batched_features["fat_suppression"][i].unsqueeze(0).detach().cpu())
            all_planecrop.append(batched_features["plane_crop_mixed"][i].unsqueeze(0).detach().cpu())
            all_stype.append(batched_features["serie_type"][i].unsqueeze(0).detach().cpu())
            with open(f"slicesfeats/{batched_features['study_id'][i]}.pt", "wb") as f:
                catted = batched_features["slices"][i].unsqueeze(0)
                torch.save(catted.detach().cpu(), f)
        gc.collect()

    torch.save(
        {
            "study_id": all_study,
            "masks": all_masks,
            "plane_ids": all_plane_ids,
            "fat_suppression": all_fat,
            "fluid_sensitive": all_fluid,
            "labels": all_labels,
            "plane_crop_mixed": all_planecrop,
            "serie_type": all_stype,
        },
        str(save_path),
    )
    print(f"✅ Extracted features saved to {save_path}")


def main():
    print(torch.__version__)
    print('✓ All imports successful')
    print('Loading metadata...')
    print(f"Device : {cfg.DEVICE}")
    df_train = pd.read_csv(cfg.TRAIN_CSV)
    df_val = pd.read_csv(cfg.VAL_CSV)
    df_tnew = df_train[~df_train["StudyInstanceUID"].isin(df_val["StudyInstanceUID"])]
    df_train = df_tnew
    df_series = pd.read_csv(cfg.SERIES_CSV)

    if cfg.DEBUG_SUBSET:
        df_train = df_train.head(cfg.DEBUG_SUBSET)

    # ── 1. Feature Precomputation (Vision & YOLO left intact) ────────────────
    # if not (cfg.PYTHON_DIR / "train_feats.pt").exists():
    #     dummy_model = KneeAbnormalityModelRefactored(cfg,inputdim=384).to(cfg.DEVICE)
    #     # visionmodel = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=0, dynamic_img_size=True)
    #     visionmodel = timm.create_model("vit_small_patch16_224", pretrained=True, num_classes=0,
    #                                     dynamic_img_size=True, checkpoint_path="radiodinos16.safetensors")
    #     # visionmodel.load_state_dict(torch.load(f"radiodinov2_ssl_finetuned_backbone.pth")["model_state_dict"])
    #     visionmodel.eval()
    #     visionmodel.requires_grad_(False)
    #     visionmodel.to(cfg.DEVICE)
    #
    #     roiextractor = TorchScriptROIExtractor(
    #         "yolo_axial.torchscript",
    #         "yolo_sagittal.torchscript",
    #         "yolo_coronal.torchscript",
    #         {
    #             "axial": {0: "whole", 1: "anterior", 2: "medial", 3: "lateral", 4: "back", 5: "front"},
    #             "sagittal": {0: "whole", 1: "anterior", 2: "medial", 3: "lateral", 4: "back", 5: "front"},
    #             "coronal": {0: "whole", 1: "anterior", 2: "medial", 3: "lateral", 4: "back", 5: "front"}
    #         }
    #     )
    #     ds_train = KneeStudyTrainDataset(df_train, df_series, cfg.TRAIN_IMG_DIR, cfg, is_train=True)
    #     ld_train = DataLoader(ds_train, batch_size=1, num_workers=8, shuffle=False, pin_memory=False, collate_fn=collate_fn, prefetch_factor=4)
    #     precompute_and_save(dummy_model, visionmodel, roiextractor, ld_train, (cfg.PYTHON_DIR / "train_feats.pt"), cfg)
    #     del dummy_model, visionmodel, roiextractor
    #     gc.collect()
    #     torch.cuda.empty_cache()

    train_cached_ds = KneeStudyTrainDataset(df_train, df_series, cfg.TRAIN_IMG_DIR, cfg, is_train=True)#CachedEmbeddingDataset("train_feats.pt", cfg, labels_replacement=df_train, label_filter=False)
    # if not (cfg.PYTHON_DIR / "val_feats.pt").exists():
    #     print("Caching features for Gold Validation set...")
    #     dummy_model = KneeAbnormalityModelRefactored(cfg,inputdim=384).to(cfg.DEVICE)
    #     # visionmodel = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=0,dynamic_img_size=True)
    #     visionmodel = timm.create_model("vit_small_patch16_224", pretrained=True, num_classes=0,
    #                                     dynamic_img_size=True, checkpoint_path="radiodinos16.safetensors")
    #     # visionmodel.load_state_dict(torch.load(f"radiodinov2_ssl_finetuned_backbone.pth")["model_state_dict"])
    #     visionmodel.eval()
    #     visionmodel.requires_grad_(False)
    #     visionmodel.to(cfg.DEVICE)
    #
    #     roiextractor = TorchScriptROIExtractor(
    #         "yolo_axial.torchscript", "yolo_sagittal.torchscript", "yolo_coronal.torchscript"
    #     )
    #     ds_gold = KneeStudyTrainDataset(df_val, df_series, cfg.TRAIN_IMG_DIR, cfg, is_train=False)
    #     ld_gold_raw = DataLoader(ds_gold, batch_size=1, num_workers=8 ,shuffle=False, collate_fn=collate_fn, prefetch_factor=4)
    #
    #     precompute_and_save(dummy_model, visionmodel, roiextractor, ld_gold_raw, (cfg.PYTHON_DIR / "val_feats.pt"), cfg)
    #     del dummy_model, visionmodel, roiextractor
    #     gc.collect()
    #     torch.cuda.empty_cache()
    # Create Gold Val DataLoader from cached features
    # visionmodel = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=0,dynamic_img_size=True)
    visionmodel = timm.create_model("vit_small_patch16_224", pretrained=True, num_classes=0,
                                    dynamic_img_size=True, checkpoint_path="radiodinos16.safetensors")
    # visionmodel.load_state_dict(torch.load(f"radiodinov2_ssl_finetuned_backbone.pth")["model_state_dict"])
    visionmodel.eval()
    visionmodel.requires_grad_(False)
    visionmodel.to(cfg.DEVICE)
    val_cached_ds = KneeStudyTrainDataset(df_val, df_series, cfg.TRAIN_IMG_DIR, cfg, is_train=False)#CachedEmbeddingDataset("val_feats.pt", cfg, labels_replacement=df_val, label_filter=False)
    ld_val_gold = DataLoader(
        val_cached_ds,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=cfg.NUM_WORKERS,
        # prefetch_factor=cfg.PREFETCH_FACTOR
    )
    # ── 2. Setup Multilabel Stratified K-Fold for Main Model ────────────────
    labels_matrix = torch.stack(train_cached_ds.labels).cpu().numpy()
    binary_targets = (labels_matrix >= 0.5).astype(int)
    X_dummy = np.zeros(len(train_cached_ds))

    mskf = MultilabelStratifiedKFold(n_splits=cfg.NUM_FOLDS, shuffle=True, random_state=cfg.SEED)

    oof_preds = np.zeros_like(labels_matrix)
    oof_targets = np.zeros_like(labels_matrix)

    print(f"\n🚀 Starting {cfg.NUM_FOLDS}-Fold Cross Validation for Main Model...")

    # ── 3. K-Fold Training Loop ──────────────────────────────────────────────
    for dinofold in range(1):
        for fold, (train_idx, val_idx) in enumerate(mskf.split(X_dummy, binary_targets)):
            print(f"\n========================== FOLD {fold + 1}/{cfg.NUM_FOLDS} ==========================")

            train_sub = Subset(train_cached_ds, train_idx)
            val_sub = Subset(train_cached_ds, val_idx)

            ld_train = DataLoader(
                train_sub,
                batch_size=cfg.BATCH_SIZE,
                shuffle=True,
                collate_fn=collate_fn,
                num_workers=cfg.NUM_WORKERS,
                persistent_workers=True if cfg.NUM_WORKERS > 0 else False,
                pin_memory=False,
                prefetch_factor=cfg.PREFETCH_FACTOR
            )
            ld_val = DataLoader(
                val_sub,
                batch_size=cfg.BATCH_SIZE,
                shuffle=False,
                collate_fn=collate_fn,
                num_workers=cfg.NUM_WORKERS,
                persistent_workers=True if cfg.NUM_WORKERS > 0 else False,
                pin_memory=False,
                prefetch_factor=cfg.PREFETCH_FACTOR
            )

            print(f"Fold {dinofold}-{fold + 1} - Train samples: {len(train_sub)} | Val samples: {len(val_sub)}")

            # Fresh Main Model Instance per Fold
            model = KneeAbnormalityModelRefactored(cfg,inputdim=384).to(cfg.DEVICE)
            model.train()
            if cfg.LOAD_PATH.exists():
                model.load_state_dict(torch.load(cfg.LOAD_PATH, weights_only=True)["model_state_dict"], strict=False)

            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in model.parameters())
            print(f"Fold {dinofold}-{fold + 1} Model Params: Trainable {trainable_params:,} / Total {total_params:,}")

            optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
            scheduler_cosine = CosineAnnealingLR(optimizer, T_max=cfg.EPOCHS, eta_min=cfg.MIN_LR)
            scheduler_warmup = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=cfg.WARMUP_EPOCHS)
            scheduler = SequentialLR(optimizer, schedulers=[scheduler_warmup, scheduler_cosine], milestones=[cfg.WARMUP_EPOCHS])
            scaler = torch.amp.GradScaler(cfg.DEVICE)
            criterion = AsymmetricLossOptimized(gamma_neg=4.0, gamma_pos=0.0, clip=0.05)
            valcriterion= torch.nn.MSELoss(reduction="sum")

            save_path_fold = str(cfg.PYTHON_DIR / f"best_model_dino_fold_{dinofold}_{fold}.pth")
            writer = SummaryWriter(comment=f"_Fold_{dinofold}_{fold}")

            best_loss = float("inf")
            best_auc = 0.0
            best_val_preds = None
            best_val_targets = None
            without_save_epochs = 0
            patience = 10

            for epoch in range(cfg.EPOCHS):
                print(f"\n--- Fold {dinofold}-{fold + 1} | Epoch {epoch + 1}/{cfg.EPOCHS} ---")
                without_save_epochs += 1

                train_loss = train_epoch(model, ld_train, visionmodel, None, optimizer, criterion, scaler, cfg, iscached=False,dinofold=dinofold)
                val_loss, val_auc, val_preds, val_targets = eval_epoch(model, ld_val, visionmodel, None, criterion, cfg, iscached=False,dinofold=dinofold)
                gval_loss, gval_auc, gval_preds, gval_targets = eval_epoch(model, ld_val_gold, visionmodel, None, criterion, cfg, iscached=False,dinofold=dinofold)

                current_lr = optimizer.param_groups[0]["lr"]
                scheduler.step()

                print(f"Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f} | Val AUC: {val_auc:.4f}")

                writer.add_scalar("Loss/Train", train_loss, epoch)
                writer.add_scalar("Loss/Val", val_loss, epoch)
                writer.add_scalar("AUC/Val", val_auc, epoch)
                writer.add_scalar("GoldAUC/Val", gval_auc, epoch)
                writer.add_scalar("Params/Learning_Rate", current_lr, epoch)

                if val_loss < best_loss:
                    best_loss = val_loss


                if gval_auc > best_auc:
                    best_auc = gval_auc
                    without_save_epochs = 0
                    best_val_preds = val_preds
                    best_val_targets = val_targets
                    torch.save({
                        'fold': fold,
                        'dinofold': dinofold,
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'val_auc': float(gval_auc),
                        'val_loss': float(val_loss),
                    }, save_path_fold)
                    print(f"  -> Saved new best checkpoint for Fold {dinofold}-{fold + 1} (GOLD AUC: {gval_auc:.4f}) -> {save_path_fold}")
                if without_save_epochs >= patience:
                    print(f"Early stopping triggered for Fold {dinofold}-{fold + 1} at epoch {epoch + 1}")
                    break

            writer.close()
            oof_preds[val_idx] = best_val_preds
            oof_targets[val_idx] = best_val_targets
            print(f"Fold {fold + 1} finished. Best Val AUC: {best_auc:.4f}")

    # ── 4. Overall Out-Of-Fold Evaluation ───────────────────────────────────
    print("\n========================== OVERALL OOF EVALUATION ==========================")
    oof_binary_targets = (oof_targets >= 0.5).astype(int)
    try:
        oof_auc = roc_auc_score(oof_binary_targets, oof_preds)
        print(f"Overall Out-of-Fold AUC: {oof_auc:.4f}")
    except Exception as e:
        print(f"Could not calculate overall OOF AUC: {e}")

    for idx, label_name in enumerate(cfg.LABELS):
        try:
            label_auc = roc_auc_score(oof_binary_targets[:, idx], oof_preds[:, idx])
            print(f"  - {label_name:20s} AUC: {label_auc:.4f}")
        except Exception:
            pass
    df_train = pd.read_csv(cfg.TRAIN_CSV)
    df_val = pd.read_csv(cfg.VAL_CSV)
    df_tnew = df_train[~df_train["StudyInstanceUID"].isin(df_val["StudyInstanceUID"])]
    df_train = df_tnew
    df_series = pd.read_csv(cfg.SERIES_CSV)
    print(f"Synovitis: {df_val['Synovitis'].value_counts()}")
    if cfg.DEBUG_SUBSET:
        df_train = df_train.head(cfg.DEBUG_SUBSET)
    # ── Feature Caching for Holdout Gold Validation Set ───────────────────────
    if not (cfg.PYTHON_DIR / "val_feats.pt").exists():
        print("Caching features for Gold Validation set...")
        dummy_model = KneeAbnormalityModelRefactored(cfg,inputdim=384).to(cfg.DEVICE)
        # visionmodel = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=0,dynamic_img_size=True)
        visionmodel = timm.create_model("vit_small_patch16_224", pretrained=True, num_classes=0,
                                        dynamic_img_size=True, checkpoint_path="radiodinos16.safetensors")
        # visionmodel.load_state_dict(torch.load(f"radiodinov2_ssl_finetuned_backbone.pth")["model_state_dict"])
        visionmodel.eval()
        visionmodel.requires_grad_(False)
        visionmodel.to(cfg.DEVICE)

        roiextractor = TorchScriptROIExtractor(
            "yolo_axial.torchscript", "yolo_sagittal.torchscript", "yolo_coronal.torchscript"
        )
        ds_gold = KneeStudyTrainDataset(df_val, df_series, cfg.TRAIN_IMG_DIR, cfg, is_train=False)
        ld_gold_raw = DataLoader(ds_gold, batch_size=1, num_workers=8 ,shuffle=False, collate_fn=collate_fn, prefetch_factor=4)

        precompute_and_save(dummy_model, visionmodel, roiextractor, ld_gold_raw, (cfg.PYTHON_DIR / "val_feats.pt"), cfg)
        del dummy_model, visionmodel, roiextractor
        gc.collect()
        torch.cuda.empty_cache()

    # Create Gold Val DataLoader from cached features
    val_cached_ds = CachedEmbeddingDataset("val_feats.pt", cfg, labels_replacement=df_val, label_filter=False)
    ld_val_gold = DataLoader(
        val_cached_ds,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        collate_fn=cache_collate_fn,
        num_workers=cfg.NUM_WORKERS
    )

    # Run the modular evaluation
    criterion = AsymmetricLossOptimized(gamma_neg=4.0, gamma_pos=0.0, clip=0.05)
    overall_auc, label_aucs, gold_predictions = evaluate_ensemble(
        model_class=KneeAbnormalityModelRefactored,
        val_loader=ld_val_gold,
        criterion=criterion,
        cfg=cfg
    )


def evaluate_ensemble(
        model_class,
        val_loader: DataLoader,
        criterion: torch.nn.Module,
        cfg: Config,
        fold_paths: Optional[List[str]] = None
) -> Tuple[float, Dict[str, float], np.ndarray]:
    """
    Loads saved fold checkpoints, runs inference on `val_loader`,
    averages probability predictions (Soft Voting), and calculates metrics.

    Returns:
        tuple: (overall_ensemble_auc, per_label_aucs_dict, ensemble_predictions)
    """
    if fold_paths is None:
        fold_paths=[]
        for dinofold in range(cfg.NUM_FOLDS):
            fold_paths += [str(cfg.PYTHON_DIR / f"best_model_dino_fold_{dinofold}_{fold}.pth") for fold in range(cfg.NUM_FOLDS)]

    all_fold_preds = []
    gold_targets_matrix = None

    print(f"\n========================== ENSEMBLE EVALUATION ({len(fold_paths)} Folds) ==========================")
    for dinofold in range(1):
        for fold_idx in range(cfg.NUM_FOLDS):
            path=str(cfg.PYTHON_DIR / f"best_model_dino_fold_{dinofold}_{fold_idx}.pth")
            if not Path(path).exists():
                print(f"⚠️ Warning: Checkpoint for fold {fold_idx + 1} not found at {path}. Skipping.")
                continue

            print(f"Loading checkpoint [{fold_idx+dinofold + 1}/{len(fold_paths)}]: {path}")
            model = model_class(cfg,inputdim=384).to(cfg.DEVICE)
            checkpoint = torch.load(path, map_location=cfg.DEVICE)
            model.load_state_dict(checkpoint['model_state_dict'])

            # Predict using current fold model
            val_loss, val_auc, fold_preds, fold_targets = eval_epoch(
                model, val_loader, visionmodel=None, roiextractor=None, criterion=criterion, cfg=cfg, iscached=True, dinofold=dinofold
            )

            all_fold_preds.append(fold_preds)
            if gold_targets_matrix is None:
                gold_targets_matrix = (fold_targets >= 0.5).astype(int)

            print(f"  └─ Fold {fold_idx + 1} Single-Model AUC: {val_auc:.4f} | Loss: {val_loss:.4f}")

            # Explicit memory release per iteration
            del model
            gc.collect()
            torch.cuda.empty_cache()

    if not all_fold_preds:
        raise RuntimeError("No fold checkpoints were successfully loaded for evaluation.")

    # ── Soft Voting: Average probabilities across all K models ─────────────────
    ensemble_preds = np.mean(all_fold_preds, axis=0)
    # Calculate overall AUC
    try:
        overall_ensemble_auc = roc_auc_score(gold_targets_matrix, ensemble_preds)
    except ValueError:
        overall_ensemble_auc = 0.5

    print(f"\n🎯 Final Ensemble Gold Validation AUC: {overall_ensemble_auc:.4f}\n")
    print("Per-Label Performance:")
    print("-" * 50)

    per_label_aucs = {}
    for idx, label_name in enumerate(cfg.LABELS):
        try:
            lbl_auc = roc_auc_score(gold_targets_matrix[:, idx], ensemble_preds[:, idx])
            per_label_aucs[label_name] = float(lbl_auc)
            print(f"  - {label_name:22s} AUC: {lbl_auc:.4f}")
        except ValueError:
            per_label_aucs[label_name] = 0.5
            print(f"  - {label_name:22s} AUC: N/A (Single class present)")

    print("-" * 50)
    return overall_ensemble_auc, per_label_aucs, ensemble_preds

if __name__ == '__main__':
    # df_train = pd.read_csv(cfg.TRAIN_CSV)
    # df_val = pd.read_csv(cfg.VAL_CSV)
    # df_tnew = df_train[~df_train["StudyInstanceUID"].isin(df_val["StudyInstanceUID"])]
    # df_train = df_tnew
    # df_series = pd.read_csv(cfg.SERIES_CSV)
    # print(f"Synovitis: {df_val['Synovitis'].value_counts()}")
    # if cfg.DEBUG_SUBSET:
    #     df_train = df_train.head(cfg.DEBUG_SUBSET)
    # # ── Feature Caching for Holdout Gold Validation Set ───────────────────────
    # if not (cfg.PYTHON_DIR / "val_feats.pt").exists():
    #     print("Caching features for Gold Validation set...")
    #     dummy_model = KneeAbnormalityModelRefactored(cfg, inputdim=384).to(cfg.DEVICE)
    #     # visionmodel = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=0,dynamic_img_size=True)
    #     for idx in range(5):
    #         visionmodel = timm.create_model("vit_small_patch14_reg4_dinov2.lvd142m", pretrained=True, num_classes=0,
    #                                         dynamic_img_size=True, checkpoint_path="vit_small_patch14_reg4_dinov2.safetensors")
    #         #visionmodel.load_state_dict(torch.load(f"radiodinov2_ssl_backbone_fold_{idx}.pth")["model_state_dict"])
    #         visionmodel.eval()
    #         visionmodel.requires_grad_(False)
    #         visionmodel.to(cfg.DEVICE)
    #
    #     roiextractor = TorchScriptROIExtractor(
    #         "yolo_axial.torchscript", "yolo_sagittal.torchscript", "yolo_coronal.torchscript"
    #     )
    #     ds_gold = KneeStudyTrainDataset(df_val, df_series, cfg.TRAIN_IMG_DIR, cfg, is_train=False)
    #     ld_gold_raw = DataLoader(ds_gold, batch_size=1, num_workers=16, shuffle=False, collate_fn=collate_fn)
    #
    #     precompute_and_save(dummy_model, visionmodel, roiextractor, ld_gold_raw, (cfg.PYTHON_DIR / "val_feats.pt"),
    #                         cfg)
    #     del dummy_model, visionmodel, roiextractor
    #     gc.collect()
    #     torch.cuda.empty_cache()
    #
    # # Create Gold Val DataLoader from cached features
    # val_cached_ds = CachedEmbeddingDataset("val_feats.pt", cfg, labels_replacement=df_val, label_filter=False)
    # ld_val_gold = DataLoader(
    #     val_cached_ds,
    #     batch_size=cfg.BATCH_SIZE,
    #     shuffle=False,
    #     collate_fn=cache_collate_fn,
    #     num_workers=cfg.NUM_WORKERS
    # )
    #
    # # Run the modular evaluation
    # criterion = AsymmetricLossOptimized(gamma_neg=4.0, gamma_pos=0.0, clip=0.05)
    # overall_auc, label_aucs, gold_predictions = evaluate_ensemble(
    #     model_class=KneeAbnormalityModelRefactored,
    #     val_loader=ld_val_gold,
    #     criterion=criterion,
    #     cfg=cfg
    # )
    main()
