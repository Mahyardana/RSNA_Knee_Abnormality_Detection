# !pip install pydicom timm albumentations scikit-learn tqdm

import os, math, random, warnings
import cv2
import numpy as np
import pandas as pd
import pydicom
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from monai.networks.nets import VNet
from monai.transforms import AsDiscrete, generate_spatial_bounding_box
from torch.utils.data import Dataset, DataLoader, random_split
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from scipy.ndimage import zoom
import scipy.ndimage as ndi
import timm
from typing import Dict, Optional, Union

from scipy.ndimage import label, find_objects
from tqdm import tqdm
import multiprocessing as mp

from traitlets import Long

# try:
#     mp.set_start_method('fork', force=True)
# except RuntimeError:
#     pass
print(torch.__version__)
from model_dino_simplified import KneeAbnormalityModelWithDinoMulti, pad_image, KneeAbnormalityModelRefactored
torch.set_float32_matmul_precision('high')
warnings.filterwarnings('ignore')
print('✓ All imports successful')

class Config:
    KAGGLE_DIR    = Path('/run/media/mahyar/Samsung-990-PRO/Projects/Kaggle/rsna-knee-abnormality-detection')
    PYTHON_DIR    = Path('/run/media/mahyar/Samsung-990-PRO/Projects/Python/RSNA Knee Abnormality Detection')
    TRAIN_CSV     = PYTHON_DIR / 'llm_generated_labels.csv'
    VAL_CSV     = PYTHON_DIR / 'val_gold.csv'
    SERIES_CSV    = PYTHON_DIR / 'train_series.csv'
    TRAIN_IMG_DIR = KAGGLE_DIR / 'train_series'
    TRAINING_LABEL = None
    SAVE_PATH     = str(PYTHON_DIR / f"best_model_dino_auc.pth")
    SAVE_PATH2     = str(PYTHON_DIR / f"best_model_dino_loss.pth")
    LOAD_PATH     = PYTHON_DIR / f"best_model_dino_label.pth"

    # ── Image processing ──────────────────────────────────────────────
    MAX_SLICES_PER_SERIES = -1
    MAX_SERIES_PER_STUDY = 16
    LABELS = ['ACL','MCL','Medial Meniscus','Lateral Meniscus',
              'Medial OA','Lateral OA','PF OA','Effusion',
              'Synovitis',"Baker's",'Contusion','Fracture']
    NUM_LABELS = 12

    PLANE_TO_IDX = {'Sagittal': 0, 'Axial': 1, 'Coronal': 2}
    NUM_PLANES   = 3

    # ── Model dimensions ──────────────────────────────────────────────
    SLICE_DIM    = 518
    META_DIM     = 256
    SLICE_ENC_HEADS  = 8
    SLICE_ENC_LAYERS = 2
    DROPOUT      = 0.3

    # ── Train / val split ─────────────────────────────────────────────
    VAL_FRACTION = 0.10   # 20% val, 80% train

    # ── Training hyperparameters ──────────────────────────────────────
    BATCH_SIZE   = 1
    NUM_WORKERS  = 0
    GRAD_ACCUM_STEPS = 8
    EPOCHS       = 100
    LR           = 2e-4
    MIN_LR       = 1e-5
    WEIGHT_DECAY = 0.03
    USE_AMP      = True

    DEBUG_SUBSET = None # Set to e.g. 100 for fast debugging
    SEED         = 42
    DEVICE       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    LABEL_CLAMP_THRESHOLD = 0.8

cfg = Config()
print(f"Device : {cfg.DEVICE}")

def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

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

        # Load plane-specific TorchScript YOLO detectors
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

        # Class ID to anatomical category label mappings
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
        """Generates slice indices starting from the middle and alternating outward."""
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
        """Crops 4D tensor [1, C, H, W] using bounding box [x1, y1, x2, y2]."""
        if bbox is None or len(bbox) == 0:
            return slice_tensor

        x1, y1, x2, y2 = map(int, bbox[:4])
        _, _, h, w = slice_tensor.shape

        # Clamp boundaries to slice spatial dimensions
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        cropped = slice_tensor[:, :, y1:y2, x1:x2]
        return cropped if cropped.numel() > 0 else slice_tensor

    def decode_yolo_output(
            self, raw_output: torch.Tensor, conf_thresh: float
    ) -> torch.Tensor:
        """Decodes YOLOv8/v11 output [10, 8400] and preserves overlapping sub-regions across different classes."""
        if raw_output.dim() == 3:
            raw_output = raw_output.squeeze(0)

        preds = raw_output.permute(1, 0)  # Shape: [8400, 10]
        boxes_cxcywh = preds[:, :4]
        class_probs = preds[:, 4:]

        # Apply sigmoid if raw logits are returned
        if class_probs.max() > 1.0 or class_probs.min() < 0.0:
            class_probs = torch.sigmoid(class_probs)

        scores, class_ids = torch.max(class_probs, dim=1)

        # Filter out low-confidence background anchors
        mask = scores >= conf_thresh
        if not mask.any():
            return torch.empty((0, 6), device=self.device)

        boxes_cxcywh = boxes_cxcywh[mask]
        scores = scores[mask]
        class_ids = class_ids[mask]

        # Convert center [cx, cy, w, h] to corner [x1, y1, x2, y2]
        x1 = boxes_cxcywh[:, 0] - (boxes_cxcywh[:, 2] / 2)
        y1 = boxes_cxcywh[:, 1] - (boxes_cxcywh[:, 3] / 2)
        x2 = boxes_cxcywh[:, 0] + (boxes_cxcywh[:, 2] / 2)
        y2 = boxes_cxcywh[:, 1] + (boxes_cxcywh[:, 3] / 2)
        boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=1)

        # 1. BATCHED NMS: Keeps overlaps between different classes (e.g., 'whole' & 'anterior')
        keep_indices = torchvision.ops.batched_nms(
            boxes_xyxy, scores, class_ids, iou_threshold=0.0
        )

        boxes_xyxy = boxes_xyxy[keep_indices]
        scores = scores[keep_indices]
        class_ids = class_ids[keep_indices]

        # 2. TOP-1 PER CLASS: Select highest confidence box per anatomical label
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
        volume_tensor: torch.Tensor,  # Shape: [Slices, C, H, W]
        plane: str,
        conf_thresh: float = 0.45,
    ) -> Dict[Union[int, str], torch.Tensor]:
        """Scans 3D volume slices center-outward to find highest-confidence ROI bounding boxes."""
        if plane not in self.models:
            raise ValueError(
                f"Invalid plane '{plane}'. Must be one of"
                f" {list(self.models.keys())}"
            )

        yolo_model = self.models[plane]
        class_map = self.class_maps.get(plane, {})
        num_slices = volume_tensor.shape[0]
        volume_tensor=F.interpolate(
            volume_tensor,
            (512,512),
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

        # 1. Search center-outward for a slice satisfying confidence threshold
        for idx in self.get_center_out_slice_indices(num_slices):
            slice_tensor = volume_tensor[idx : idx + 1].to(
                self.device
            )  # [1, C, H, W]
            if cfg.USE_AMP:
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    predictions = yolo_model(slice_tensor)[0]  # Shape: [num_boxes, 6]
            else:
                predictions = yolo_model(slice_tensor)[0]

            decodedyolo=self.decode_yolo_output(predictions, conf_thresh)
            if len(decodedyolo) == 0:
                continue
            # Compute highest box confidence in current slice
            max_box_conf = decodedyolo[:, 4].max().item()

            # Track fallback slice with highest overall detection confidence
            if max_box_conf > highest_conf:
                highest_conf = max_box_conf
                best_slice_idx = idx

            # Terminate search early once threshold criteria is met
            if max_box_conf >= conf_thresh:
                best_slice_idx = idx
                break

        # 2. Extract sub-region crops from the optimal candidate slice
        best_slice_tensor = volume_tensor[
            best_slice_idx : best_slice_idx + 1
        ].to(self.device)
        predictions = yolo_model(best_slice_tensor)[0]

        effective_thresh = (
            conf_thresh if highest_conf >= conf_thresh else highest_conf
        )
        decodedyolo = self.decode_yolo_output(predictions, conf_thresh)
        for pred in decodedyolo:
            if pred[4].item() >= effective_thresh:
                class_id = int(pred[5].item())
                category_key = class_map.get(class_id, class_id)
                best_crops[category_key] = pred[:4]/512

        return best_crops
# ── Training / Validation Dataset ─────────────────────────────────────────────
class KneeStudyTrainDataset(Dataset):
    '''
    Groups all series for a study together for training.
    Returns slices, metadata, and 12-class target labels.
    '''
    def __init__(self, df_studies: pd.DataFrame, df_series: pd.DataFrame, img_dir: Path, cfg, is_train: bool = True):
        self.cfg = cfg
        self.vit = timm.create_model("hf_hub:Snarcy/RadioDino-b16",
        pretrained=True,
        dynamic_img_size=True,
        num_classes=0)
        self.vit.eval()
        self.vit.requires_grad_(False)
        self.vit.to(cfg.DEVICE)
        self.roiextractor=TorchScriptROIExtractor(
            "yolo_axial.torchscript",
            "yolo_sagittal.torchscript",
            "yolo_coronal.torchscript",
            {
                "axial":{
                    0:"whole",
                    1:"anterior",
                    2:"medial",
                    3:"lateral",
                    4:"back",
                    5:"front"
                },
                "sagittal":{
                    0:"whole",
                    1:"anterior",
                    2:"medial",
                    3:"lateral",
                    4:"back",
                    5:"front"
                },
                "coronal":{
                    0:"whole",
                    1:"anterior",
                    2:"medial",
                    3:"lateral",
                    4:"back",
                    5:"front"
                }
            }
        )
        weights_path = "vnet_skmtea_standalone.pt"
        model_monai = torch.jit.load(weights_path)
        model_monai.eval()
        self.model_monai=model_monai.to(cfg.DEVICE)
        self.img_dir = img_dir
        self.is_train = is_train
        self.cache={}
        maxlen=0
        
        # Group series by study_id
        self.studies = []
        for _, study_row in df_studies.iterrows():
            study_id = str(study_row['StudyInstanceUID'])
            
            # Get series for this study
            study_series = df_series[df_series['StudyInstanceUID'] == study_id]
            if(len(study_series)>maxlen):
                maxlen=len(study_series)
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
                
            # Extract labels (nan to 0 if any missing)
            labels = study_row[cfg.LABELS].fillna(0).values.astype(np.float32)
            
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

        # Calculate scale factor
        scaleh = target_short / h
        scalew = target_short / w

        # Check if image has channels (e.g., 3D array for RGB or 2D for Grayscale DICOM)
        if len(img.shape) == 3:
            zoom_factors = (scaleh, scalew, 1)  # Don't scale channels
        else:
            zoom_factors = (scaleh, scalew)

        # order=3 specifies a Cubic B-Spline interpolation
        resized_img = zoom(img, zoom_factors, order=3)
        return resized_img

    @staticmethod
    def pad_to_patch_multiple(
            series_tensor: torch.Tensor, patch_size: int = 14
    ) -> torch.Tensor:
        """Pads an MRI series to the nearest higher multiple of 14 without resizing.

        Args:
            series_tensor: Tensor of shape (Slices, H, W) or (C, Slices, H, W)
            patch_size: ViT patch size (default: 14 for DINOv2)

        Returns:
            Zero-padded tensor with H and W as exact multiples of patch_size.
        """
        has_channel = series_tensor.ndim == 4
        if not has_channel:
            series_tensor = series_tensor.unsqueeze(0)  # (1, Slices, H, W)

        _, S, H, W = series_tensor.shape

        # Calculate minimum target shape (nearest higher multiple of 14)
        target_H = math.ceil(H / patch_size) * patch_size
        target_W = math.ceil(W / patch_size) * patch_size

        # Calculate symmetric padding
        pad_H = target_H - H
        pad_W = target_W - W

        pad_left = pad_W // 2
        pad_right = pad_W - pad_left
        pad_top = pad_H // 2
        pad_bottom = pad_H - pad_top

        # F.pad format: (left, right, top, bottom)
        padded = F.pad(
            series_tensor,
            (pad_left, pad_right, pad_top, pad_bottom),
            mode="constant",
            value=0,
        )

        return padded if has_channel else padded.squeeze(0)

    @torch.no_grad()
    @staticmethod
    def extract_multilayer_features(model, slice_chunk):
        """Extracts CLS embeddings from the top 4 layers of RadioDINO.

        Args:
            slice_chunk: Shape (Chunk_Size, 3, 384, 384)
        Returns:
            Tensor of shape (Chunk_Size, 768 * 4) -> (Chunk_Size, 3072)
        """
        # Returns intermediate representations from blocks -4, -3, -2, -1
        intermediates = model.forward_intermediates(
            slice_chunk, indices=[-4, -3, -2, -1], norm=True
        )
        layer_vectors = []
        layer_feature_maps = intermediates[1]
        for fmap in layer_feature_maps:
            # fmap shape: (B, 768, 48, 48)
            # 1. Global Average Pool across 48x48 spatial grid (macro texture)
            avg_feat = fmap.mean(dim=(2, 3))  # (B, 768)

            # 2. Global Max Pool across 48x48 spatial grid (peak local injury)
            max_feat = fmap.amax(dim=(2, 3))  # (B, 768)

            layer_vectors.append(avg_feat + max_feat)

        # Concatenate top 4 blocks -> Final Slice Vector Shape: (B, 3072)
        slice_embedding = torch.cat(layer_vectors, dim=-1)
        return slice_embedding

    def preprocess_rsna_slice_for_skmtea(slice_2d: np.ndarray) -> torch.Tensor:
        # Clip scanner artifacts
        p1, p99 = np.percentile(slice_2d, (1, 99))
        clipped = np.clip(slice_2d, p1, p99)

        # Slice-wise Z-score normalization
        mean, std = clipped.mean(), clipped.std() + 1e-8
        normed = (clipped - mean) / std

        tensor = torch.from_numpy(normed).float()
        return tensor.cuda()  # Shape: (1, 1, H, W)
    @torch.no_grad()
    def infer_volumetric_category_bboxes_robust(
            mri_volume: torch.Tensor,
            vnet_model: torch.nn.Module,
            batch_size: int = 16,
            prob_threshold: float = 0.50,
            min_cluster_size: int = 150,  # Ignore noise clusters < 150 pixels
    ) -> dict[str, tuple[float, float, float, float]]:
        mri_volume=KneeStudyTrainDataset.preprocess_rsna_slice_for_skmtea(mri_volume.cpu().numpy())
        # 1. Normalize Intensity [0.0, 1.0]
        v_min, v_max = mri_volume.min(), mri_volume.max()
        if v_max > v_min:
            mri_volume = (mri_volume - v_min) / (v_max - v_min)

        # 2. Reformat Shapes
        if mri_volume.ndim == 4 and mri_volume.shape[1] != 1:
            mri_volume = mri_volume[:, 0:1, :, :]
        elif mri_volume.ndim == 3:
            mri_volume = mri_volume.unsqueeze(1)

        if mri_volume.shape[-2:] != (512, 512):
            mri_volume = F.interpolate(
                mri_volume, size=(512, 512), mode="bilinear", align_corners=False
            )

        S, C, H, W = mri_volume.shape
        vnet_model.eval()

        # 3. Restrict Inference to Central 80% Slices to Avoid Edge/Coil Artifacts
        slice_start = int(S * 0.10)
        slice_end = max(slice_start + 1, int(S * 0.90))
        core_volume = mri_volume[slice_start:slice_end]

        slice_preds = []
        for i in range(0, core_volume.shape[0], batch_size):
            chunk = core_volume[i: i + batch_size].cuda()
            logits = vnet_model(chunk)
            probs = F.softmax(logits, dim=1)

            max_probs, preds = torch.max(probs, dim=1)
            preds[max_probs < prob_threshold] = 0
            slice_preds.append(preds.cpu())

        full_mask = torch.cat(slice_preds, dim=0).numpy()

        class_map = {1: "Patellar", 2: "Femoral", 3: "Tibial_Meniscus"}
        category_bboxes = {}

        for class_idx, class_name in class_map.items():
            binary_mask = full_mask == class_idx

            # 4. Filter Small Noise Clusters
            labeled_mask, num_features = ndi.label(binary_mask)
            if num_features > 0:
                component_sizes = ndi.sum(
                    binary_mask, labeled_mask, range(1, num_features + 1)
                )
                # Retain only large valid tissue clusters
                valid_components = np.where(component_sizes >= min_cluster_size)[
                                       0
                                   ] + 1
                clean_mask = np.isin(labeled_mask, valid_components)
            else:
                clean_mask = binary_mask

            if not clean_mask.any():
                category_bboxes[class_name] = (0.0, 0.0, 0.0, 0.0)
            else:
                _, y_indices, x_indices = np.where(clean_mask)

                ymin = float(np.percentile(y_indices, 2.0) / H)
                ymax = float(np.percentile(y_indices, 98.0) / H)
                xmin = float(np.percentile(x_indices, 2.0) / W)
                xmax = float(np.percentile(x_indices, 98.0) / W)

                category_bboxes[class_name] = (
                    round(ymin, 4),
                    round(ymax, 4),
                    round(xmin, 4),
                    round(xmax, 4),
                )

        return category_bboxes
    @staticmethod
    def prepare_mri_for_radiodino(
            series_tensor: torch.Tensor,
            target_size: int = 224,
            threshold: float = 10.0,
    ) -> torch.Tensor:
        """Safely processes an MRI series for RadioDINO without destroying features:

        1. ROI Cropping: Isolates joint and removes dark background.
        2. Square Padding: Symmetric padding to a square aspect ratio to prevent distortion.
        3. Bilinear Resizing: Scales spatial dimensions to (target_size, target_size).
        4. Channel Expansion: Converts 1-channel grayscale slices to 3-channel RGB.
        5. ImageNet Normalization: Scales intensities to match RadioDINO's expected distribution.

        Args:
            series_tensor: Shape (Slices, H, W) or (1, Slices, H, W)
            target_size: Target height and width (default: 224)
            threshold: Intensity cutoff to identify background scanner noise

        Returns:
            Tensor of shape (Slices, 3, target_size, target_size) ready for RadioDINO
        """
        if series_tensor.ndim == 3:
            series_tensor = series_tensor.unsqueeze(0)  # (1, Slices, H, W)

        _, S, H, W = series_tensor.shape

        # 1. Unified 3D ROI Bounding Box
        mask = series_tensor > threshold
        if not mask.any():
            ymin, ymax, xmin, xmax = 0, H, 0, W
        else:
            nonzero = torch.nonzero(mask)
            ymin = nonzero[:, -2].min().item()
            ymax = nonzero[:, -2].max().item() + 1
            xmin = nonzero[:, -1].min().item()
            xmax = nonzero[:, -1].max().item() + 1

        cropped = series_tensor[..., ymin:ymax, xmin:xmax]  # (1, S, crop_H, crop_W)

        # # Center crop (middle 50%)
        # c_h, c_w = cropped.shape[2], cropped.shape[3]
        # ymin, ymax = c_h // 4, (c_h // 4) * 3
        # xmin, xmax = c_w // 4, (c_w // 4) * 3
        # cropped = cropped[..., ymin:ymax, xmin:xmax]

        _, _, crop_H, crop_W = cropped.shape

        # 2. Symmetric Padding to Square (preserves aspect ratio)
        max_dim = max(crop_H, crop_W)
        pad_h = max_dim - crop_H
        pad_w = max_dim - crop_W

        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        squared = F.pad(
            cropped,
            (pad_left, pad_right, pad_top, pad_bottom),
            mode="constant",
            value=0,
        )  # (1, S, max_dim, max_dim)
        #
        # # 3. Resize to 224x224
        # resized = F.interpolate(
        #     squared,
        #     size=(target_size, target_size),
        #     mode="bilinear",
        #     align_corners=False,
        # )  # (1, S, target_size, target_size)

        # 4. Format Shape to (Slices, 3, target_size, target_size)
        slices_tensor = squared.squeeze(0).unsqueeze(1)  # (Slices, 1, 224, 224)
        slices_tensor = slices_tensor.repeat(
            1, 3, 1, 1
        )  # Grayscale to 3-channel RGB

        # 5. Normalize (0-1 MinMax + ImageNet Stats)
        if slices_tensor.max() > 1.0:
            slices_tensor = slices_tensor / 255.0

        mean = torch.tensor(
            [0.485, 0.456, 0.406], device=slices_tensor.device
        ).view(1, 3, 1, 1)
        std = torch.tensor(
            [0.229, 0.224, 0.225], device=slices_tensor.device
        ).view(1, 3, 1, 1)

        return (slices_tensor - mean) / std
    # @staticmethod
    # def prepare_mri_for_radiodino(
    #         series_tensor: torch.Tensor, patch_size: int = 16, threshold: float = 10.0
    # ) -> torch.Tensor:
    #     """Safely processes an MRI series for RadioDINO without destroying features:
    #
    #     1. ROI Cropping: Isolates joint and removes dark background.
    #     2. Symmetric Padding: Pads height/width to nearest multiple of 16 (0%
    #     resampling blur).
    #     3. Channel Expansion: Converts 1-channel grayscale slices to 3-channel RGB.
    #     4. ImageNet Normalization: Scales intensities to match RadioDINO's expected
    #     distribution.
    #
    #     Args:
    #         series_tensor: Shape (Slices, H, W) or (1, Slices, H, W)
    #         patch_size: RadioDINO patch multiple (16)
    #         threshold: Intensity cutoff to identify background scanner noise
    #
    #     Returns:
    #         Tensor of shape (Slices, 3, H_padded, W_padded) ready for RadioDINO
    #     """
    #     if series_tensor.ndim == 3:
    #         series_tensor = series_tensor.unsqueeze(0)  # (1, Slices, H, W)
    #
    #     _, S, H, W = series_tensor.shape
    #
    #     # 1. Unified 3D ROI Bounding Box
    #     mask = series_tensor > threshold
    #     if not mask.any():
    #         ymin, ymax, xmin, xmax = 0, H, 0, W
    #     else:
    #         nonzero = torch.nonzero(mask)
    #         ymin = nonzero[:, -2].min().item()
    #         ymax = nonzero[:, -2].max().item() + 1
    #         xmin = nonzero[:, -1].min().item()
    #         xmax = nonzero[:, -1].max().item() + 1
    #
    #     cropped = series_tensor[..., ymin:ymax, xmin:xmax]  # (1, S, crop_H, crop_W)
    #     ymin = cropped.shape[2]//4
    #     ymax = ymin*3
    #     xmin = cropped.shape[3]//4
    #     xmax = xmin*3
    #
    #     cropped = cropped[..., ymin:ymax, xmin:xmax]  # (1, S, crop_H, crop_W)
    #     _, _, crop_H, crop_W = cropped.shape
    #
    #     # 2. Symmetric Padding to Multiple of 16
    #     target_H = math.ceil(crop_H / patch_size) * patch_size
    #     target_W = math.ceil(crop_W / patch_size) * patch_size
    #
    #     pad_top = (target_H - crop_H) // 2
    #     pad_bottom = (target_H - crop_H) - pad_top
    #     pad_left = (target_W - crop_W) // 2
    #     pad_right = (target_W - crop_W) - pad_left
    #
    #     padded = F.pad(
    #         cropped,
    #         (pad_left, pad_right, pad_top, pad_bottom),
    #         mode="constant",
    #         value=0,
    #     )
    #
    #     # 3. Format Shape to (Slices, 3, H_padded, W_padded)
    #     slices_tensor = padded.squeeze(0).unsqueeze(1)  # (Slices, 1, H, W)
    #     slices_tensor = slices_tensor.repeat(
    #         1, 3, 1, 1
    #     )  # Grayscale to 3-channel RGB
    #
    #     # 4. Normalize (0-1 MinMax + ImageNet Stats)
    #     if slices_tensor.max() > 1.0:
    #         slices_tensor = slices_tensor / 255.0
    #
    #     mean = torch.tensor([0.485, 0.456, 0.406], device=slices_tensor.device).view(
    #         1, 3, 1, 1
    #     )
    #     std = torch.tensor([0.229, 0.224, 0.225], device=slices_tensor.device).view(
    #         1, 3, 1, 1
    #     )
    #
    #     return (slices_tensor - mean) / std
    @staticmethod
    def preprocess_mri_series(
            series_tensor: torch.Tensor,
            threshold: float = 10.0,
    ) -> torch.Tensor:
        """Preprocesses a 3D MRI volume by:

        1. Finding the ROI bounding box across all slices (thresholding black background).
        2. Cropping all slices to the shared bounding box.
        3. Padding the cropped region symmetrically to a square aspect ratio.
        4. Resizing to target spatial resolution using bilinear interpolation.

        Args:
            series_tensor: PyTorch tensor of shape (Slices, H, W) or (C, Slices, H,
              W)
            target_size: Output spatial grid (H_out, W_out)
            threshold: Intensity cutoff to identify background scanner noise

        Returns:
            Preprocessed PyTorch tensor of shape (Slices, H_out, W_out) or (C,
            Slices, H_out, W_out)
        """
        has_channel = series_tensor.ndim == 4
        if not has_channel:
            # Temporary add channel dim: (Slices, H, W) -> (1, Slices, H, W)
            series_tensor = series_tensor.unsqueeze(0)

        # 1. Unified 3D Bounding Box Calculation
        mask = series_tensor > threshold
        if not mask.any():
            # Fallback if image is entirely dark
            coords_h = torch.arange(series_tensor.shape[-2])
            coords_w = torch.arange(series_tensor.shape[-1])
            ymin, ymax = 0, series_tensor.shape[-2]
            xmin, xmax = 0, series_tensor.shape[-1]
        else:
            nonzero_coords = torch.nonzero(mask)
            ymin, ymax = (
                nonzero_coords[:, -2].min().item(),
                nonzero_coords[:, -2].max().item() + 1,
            )
            xmin, xmax = (
                nonzero_coords[:, -1].min().item(),
                nonzero_coords[:, -1].max().item() + 1,
            )

        # Crop ROI across all slices simultaneously
        cropped = series_tensor[..., ymin:ymax, xmin:xmax]

        padded=KneeStudyTrainDataset.pad_to_patch_multiple(cropped,16)

        # Reshape back to (C, Slices, H_out, W_out)
        output = padded

        return output if has_channel else output.squeeze(0)
    @staticmethod
    def safe_load_dicom_array(dcm_path: str):
        """
        Handles uncompressed (Explicit/Implicit VR) and compressed
        (JPEG Lossless, JPEG 2000) DICOM transfer syntaxes automatically.
        """
        try:
            ds = pydicom.dcmread(dcm_path)
            # Forces pylibjpeg/openjpeg decompression backend
            arr = ds.pixel_array.astype(np.float32)

            # Check for MONOCHROME1 pixel inversion
            if ds.get("PhotometricInterpretation", "MONOCHROME2") == "MONOCHROME1":
                arr = arr.max() - arr

            return arr
        except Exception as e:
            # Fallback or corrupted DICOM handling
            return None
    def __getitem__(self, idx):
        study = self.studies[idx]
        study_id = study['study_id']
        labels = study['labels']
        
        all_slices = []
        all_plane_ids = []
        all_fluids = []
        all_fats = []
        for sinfo in study['series']:
            path = self.img_dir / study_id / sinfo['series_id']
            if not path.exists():
                continue
            dcm_files = []
            alldcms=[pydicom.dcmread(str(dcm),stop_before_pixels=True) for dcm in sorted(path.glob('*.dcm'))]
            # Filter out files missing critical spatial metadata
            valid_slices = [
                ds for ds in alldcms
                if "ImagePositionPatient" in ds and "ImageOrientationPatient" in ds
            ]

            if not valid_slices:
                raise ValueError("No valid spatial metadata found in the provided DICOM slices.")

            # Extract orientation vectors from the first slice
            orient = valid_slices[0].ImageOrientationPatient
            row_cosines = np.array(orient[:3])
            col_cosines = np.array(orient[3:])

            # Compute the normal vector (orthogonal to the slice plane)
            slice_normal = np.cross(row_cosines, col_cosines)

            # Sort based on the projection of ImagePosition onto the slice normal
            sorted_slices = sorted(
                valid_slices,
                key=lambda ds: np.dot(np.array(ds.ImagePositionPatient), slice_normal)
            )
            alldcms=sorted_slices
            if self.cfg.MAX_SLICES_PER_SERIES>0:
                skipnum = math.ceil(len(alldcms) / self.cfg.MAX_SLICES_PER_SERIES)
                # for i in range(len(alldcms)):
                #     if (i % skipnum) == 0:
                #         dcm_files.append(alldcms[i])
                #         # if i==0:
                #         #     dcm_files.append(alldcms[i])
                #         #     dcm_files.append(alldcms[i+1])
                #         #     dcm_files.append(alldcms[i+2])
                #         # elif i==len(alldcms)-1:
                #         #     dcm_files.append(alldcms[i-2])
                #         #     dcm_files.append(alldcms[i-1])
                #         #     dcm_files.append(alldcms[i])
                #         # else:
                #         #     dcm_files.append(alldcms[i-1])
                #         #     dcm_files.append(alldcms[i])
                #         #     dcm_files.append(alldcms[i+1])
                middleidx=int(len(alldcms) / 2)
                onefourth=int(middleidx/2)
                threefourth=int(middleidx+onefourth)
                dcm_files.append(alldcms[onefourth])
                dcm_files.append(alldcms[middleidx])
                dcm_files.append(alldcms[threefourth])
            else:
                dcm_files = alldcms
            slices=[]
            # tempimgs=[]
            for dcm_file in dcm_files:
                # dcm = pydicom.dcmread(str(dcm_file))
                dcm=dcm_file
                try:
                    img = self.safe_load_dicom_array(dcm.filename)
                    img = (img - img.min()) / (img.max() - img.min())

                    # Very simple augmentations for training
                    # if self.is_train and random.random() > 0.5:
                    #     img = np.fliplr(img).copy()
                    # cv2.resize
                    if img.shape[0]!=img.shape[1]:
                        pass
                    # padded_img = cv2.copyMakeBorder(
                    #     img,
                    #     0,
                    #     bottom,
                    #     0,
                    #     right,
                    #     borderType=cv2.BORDER_CONSTANT,
                    #     value=[0, 0, 0]
                    # )
                    # img = self.resize_shorter_side_scipy(img)
                    #img = cv2.resize(img, (518, 518), interpolation=cv2.INTER_CUBIC) # Base size, will be padded natively
                    img = np.clip(img, 0.0, 1.0)
                    img = np.stack([img, img, img], axis=0)  # (3, H, W)
                    slices.append(torch.from_numpy(img))
                except:
                    pass
                # if len(tempimgs)==3:
                #     slices.append(torch.stack(tempimgs, dim=0))
                #     tempimgs=[]
            # if len(tempimgs)>0:
            #     toadd=3-len(tempimgs)
            #     newtempimgs=torch.cat([torch.stack(tempimgs,dim=0),torch.zeros(toadd, 518, 518)],dim=0)
            #     slices.append(newtempimgs)

            all_plane_ids.append(sinfo['plane_id'])
            all_fluids.append(sinfo['fluid_sensitive'])
            all_fats.append(sinfo['fat_suppression'])
            all_slices.append(torch.stack(slices))
        if study_id not in self.cache:

            xxxxxx = [self.roiextractor.extract_best_roi_crops(x,"sagittal" if y==0 else "axial" if y==1 else "coronal") for x,y in zip(all_slices,all_plane_ids)]
            all_slices_preprocessed=[KneeStudyTrainDataset.prepare_mri_for_radiodino(x[:,0,:,:]).to(self.cfg.DEVICE) for x in all_slices]
            testimage= all_slices_preprocessed[0][all_slices_preprocessed[0].shape[0]//2].detach().cpu().numpy()[0]
            h=testimage.shape[-2]
            w=testimage.shape[-1]
            testimages1=[]
            for i,xxx in enumerate(xxxxxx):
                testimages2=[]
                for k,v in xxx.items():
                    minx = int(v[0] * w)
                    minh = int(v[1] * h)
                    maxx = int(v[2] * w)
                    maxh = int(v[3] * h)
                    testimages2.append (all_slices_preprocessed[i][all_slices_preprocessed[i].shape[0] // 2].detach().cpu().numpy()[0, minh:maxh, minx:maxx])
                testimages1.append(testimages2)
            def crop_and_resize_box(input,box):
                w=input.shape[-1]
                h=input.shape[-2]
                minx = int(box[0] if box[0]>0 else 0 * w)
                minh = int(box[1] if box[1]>0 else 0 * h)
                maxx = int(box[2] * w)
                maxh = int(box[3] * h)
                cropped=input[:,:, minh:maxh, minx:maxx]
                if cropped.shape[-1]==0 or cropped.shape[-2]==0:
                    return None
                return F.interpolate(cropped,(224,224),mode='bilinear',align_corners=False)
            slices_feats = []
            plane_crop_mixed=[]
            new_plane_ids=[]
            new_fluids=[]
            new_fats=[]
            CHUNK_SIZE = 32  # Process max 32 images at a time through Dino
            rev_class_map={
                "whole":0,
                "anterior":1,
                "medial":2,
                "lateral":3,
                "back":4,
                "front":5
            }
            for idx,(x,y) in enumerate(zip(xxxxxx,all_slices_preprocessed)):
                xdict={}
                for k,v in x.items():
                    tovit=crop_and_resize_box(y,v)
                    if tovit is not None:
                        dino_tokens = []
                        with torch.no_grad():
                            for i in range(0, tovit.size(0), CHUNK_SIZE):
                                chunk = tovit[i: i + CHUNK_SIZE]
                                if cfg.USE_AMP:
                                    with torch.amp.autocast("cuda", dtype=torch.float16):
                                        chunk_out = self.vit(chunk)
                                else:
                                    chunk_out = self.vit(chunk)
                                dino_tokens.append(chunk_out)
                        dino_out = torch.cat(dino_tokens, dim=0)
                        slices_feats.append(dino_out.cpu())
                        plane_crop_mixed.append([all_plane_ids[idx], rev_class_map[k]])
                        new_plane_ids.append(all_plane_ids[idx])
                        new_fluids.append(all_fluids[idx])
                        new_fats.append(all_fats[idx])
            self.cache[study_id]=slices_feats,plane_crop_mixed,new_plane_ids,new_fluids,new_fats
        else:
            slices_feats,plane_crop_mixed,new_plane_ids,new_fluids,new_fats = self.cache[study_id]
        all_plane_ids=new_plane_ids
        all_fluids=new_fluids
        all_fats=new_fats
        if len(all_slices) == 0:
            # Fallback if no images found
            all_slices.append(torch.zeros(3, 224, 224))
            all_plane_ids.append(0)
            all_fluids.append(0)
            all_fats.append(0)
        max_slice=max(slice.shape[0] for slice in all_slices)
        # max_height=max(slices.shape[2] for slices in all_slices)
        # max_width=max(slices.shape[3] for slices in all_slices)
        # if max_height%14!=0:
        #     max_height+=14-(max_height%14)
        # if max_width%14!=0:
        #     max_width+=14-(max_width%14)
        # if max_width>1000 or max_height>1000:
        #     pass
        masks=[]
        for idx in range(len(slices_feats)):
            # if all_slices[idx].shape[2]<max_height or all_slices[idx].shape[3]<max_width:
            #     pad_height=max_height - all_slices[idx].shape[2]
            #     pad_width=max_width - all_slices[idx].shape[3]
            #     all_slices[idx]=F.pad(all_slices[idx], (0, pad_width, 0, pad_height),value=0, mode="constant")
            if slices_feats[idx].shape[0] < max_slice:
                s=slices_feats[idx].shape[0]
                sizediff = max_slice - s
                slices_feats[idx] = torch.cat([slices_feats[idx], torch.zeros((sizediff, 768))], dim=0)
                masks.append(torch.cat([torch.ones(s, dtype=torch.bool), torch.zeros(sizediff, dtype=torch.bool)]))
            else:
                masks.append(torch.ones(slices_feats[idx].shape[0], dtype=torch.bool))
        incpufeats=torch.stack(slices_feats).cpu()

        return {
            'study_id': study_id,
            'slices': incpufeats, # (S, 3, H, W)
            'slicemasks': torch.stack(masks),
            'plane_ids': torch.tensor(all_plane_ids, dtype=torch.long),
            'fluid_sensitive': torch.tensor(all_fluids, dtype=torch.long),
            'fat_suppression': torch.tensor(all_fats, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.float32),
            'plane_crop_mixed': torch.tensor(plane_crop_mixed, dtype=torch.long),
        }

# ── Collate function ──────────────────────────────────────────────────────────
def collate_fn(batch: List[Dict]) -> Dict:
    max_s = cfg.MAX_SERIES_PER_STUDY
    max_slice = cfg.MAX_SLICES_PER_SERIES#max(b['slices'].shape[1] for b in batch)#cfg.MAX_SLICES_PER_SERIES
    max_height = max(b['slices'].shape[-2] for b in batch)
    max_width = max(b['slices'].shape[-1] for b in batch)
    padded_slices = []
    padded_planes = []
    padded_fluids = []
    padded_fats = []
    slice_masks = []
    study_ids = []
    labels = []
    placeholder_planes=10.0
    placeholder_fluids=10.0
    placeholder_fats=10.0
    for b in batch:
        s = b['slices'].shape[0]
        slice_len=b['slices'].shape[1]
        study_ids.append(b['study_id'])
        labels.append(b['labels'])
        # padded_slice = b['slices']
        # if padded_slice.shape[-2] < max_height or padded_slice.shape[-1] < max_width:
        #     pad_height = max_height - padded_slice.shape[-2]
        #     pad_width = max_width - padded_slice.shape[-1]
        #     padded_slice = F.pad(padded_slice, (0, pad_width, 0, pad_height), value=0, mode="constant")
        # if slice_len<max_slice:
        #     sizediff = max_slice - slice_len
        #     padded_slice = torch.cat([padded_slice, torch.zeros((s,sizediff, 3, max_height, max_width))],dim=1)
        # if s < max_s:
        #     pad_img = torch.zeros(max_s - s, *padded_slice.shape[1:])
        #     padded_slices.append(torch.cat([padded_slice, pad_img], dim=0))
        #     pad_idx = torch.ones(max_s - s)
        #     padded_planes.append(torch.cat([b['plane_ids'], pad_idx*placeholder_planes], dim=0))
        #     padded_fluids.append(torch.cat([b['fluid_sensitive'], pad_idx*placeholder_fluids], dim=0))
        #     padded_fats.append(torch.cat([b['fat_suppression'], pad_idx*placeholder_fats], dim=0))
        #
        #     mask = torch.cat([torch.zeros(s, dtype=torch.bool), torch.ones(max_s - s, dtype=torch.bool)])
        #     slice_masks.append(mask)
        # else:
        #     padded_slices.append(padded_slice)
        #     padded_planes.append(b['plane_ids'])
        #     padded_fluids.append(b['fluid_sensitive'])
        #     padded_fats.append(b['fat_suppression'])
        #     slice_masks.append(torch.zeros(s, dtype=torch.bool))
        # max_slice = max(slices.shape[0] for slices in all_slices)
        # max_height = max(slices.shape[2] for slices in all_slices)
        # max_width = max(slices.shape[3] for slices in all_slices)
        # if max_height % 14 != 0:
        #     max_height += 14 - (max_height % 14)
        # if max_width % 14 != 0:
        #     max_width += 14 - (max_width % 14)
        #
        # for idx in range(len(all_slices)):
        #     if all_slices[idx].shape[2] < max_height or all_slices[idx].shape[3] < max_width:
        #         pad_height = max_height - all_slices[idx].shape[2]
        #         pad_width = max_width - all_slices[idx].shape[3]
        #         all_slices[idx] = F.pad(all_slices[idx], (0, pad_width, 0, pad_height), value=0, mode="constant")
        #     if all_slices[idx].shape[0] < max_slice:
        #         sizediff = max_slice - all_slices[idx].shape[0]
        #         all_slices[idx] = torch.cat([all_slices[idx], torch.zeros((sizediff, 3, max_height, max_width))],
        #                                     dim=0)
    # images = [[torch.stack(padded_slice) for padded_slice in padded_slice_outer] for padded_slice_outer in padded_slices] # (B, max_s, 3, 224, 224)
    # Pad natively for RadDino to 14x multiples
    images = padded_slices # (B, max_s, 3, H, W)
    
    return {
        'study_ids': study_ids,
        'slice_features': torch.stack(images),
        'slice_mask': torch.stack(slice_masks),
        'plane_ids': torch.stack(padded_planes),
        'fluid_sensitive': torch.stack(padded_fluids),
        'fat_suppression': torch.stack(padded_fats),
        'labels': torch.stack(labels),
    }

# ── Loss ──────────────────────────────────────────────────────────────────────
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

# ── Train & Eval Epochs ───────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, criterion,scaler, cfg):
    model.train()
    losses = []
    optimizer.zero_grad()

    active_label_idx = cfg.TRAINING_LABEL
    pbar = tqdm(loader, desc='Train')
    for batch_idx, batch in enumerate(pbar):
        # images = batch['images'].to(cfg.DEVICE)
        # slice_mask = batch['slice_mask'].to(cfg.DEVICE)
        # plane_ids = batch['plane_ids'].to(cfg.DEVICE)
        # fluid_sensitive = batch['fluid_sensitive'].to(cfg.DEVICE)
        # fat_suppression = batch['fat_suppression'].to(cfg.DEVICE)
        labels = batch["labels"].to(cfg.DEVICE)

        labels[labels>=cfg.LABEL_CLAMP_THRESHOLD]=1.0
        labels[labels<cfg.LABEL_CLAMP_THRESHOLD]=0.0
        slices = batch["slices"].to(cfg.DEVICE)
        slicemasks = batch["slicemasks"].to(cfg.DEVICE)
        fats = batch["fat_suppression"].to(cfg.DEVICE)
        fluids = batch["fluid_sensitive"].to(cfg.DEVICE)
        plane_ids = batch["plane_ids"].to(cfg.DEVICE)
        plane_crop_mixed = batch["plane_crop_mixed"].to(cfg.DEVICE)
        seriemasks = batch["seriemasks"].to(cfg.DEVICE) if "seriemasks" in batch else slicemasks[:,:,0]

        if torch.isnan(slices).any():
            print(
                "NaN Inputs!"
            )
        else:
            # optimizer.zero_grad()
            if cfg.USE_AMP:
                with torch.amp.autocast("cuda",dtype=torch.float16):
                    logits = model.forward(slices, slicemasks, seriemasks, plane_ids, fluids, fats, plane_crop_mixed)
                    loss = criterion(logits, labels)
                    loss = loss / cfg.GRAD_ACCUM_STEPS

                scaler.scale(loss).backward()
                if (batch_idx + 1) % cfg.GRAD_ACCUM_STEPS == 0 or (batch_idx + 1) == len(loader):
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            else:
                logits = model.forward(slices, slicemasks, seriemasks, plane_ids, fluids, fats, plane_crop_mixed)

                loss = criterion(logits, labels)
                loss = loss / cfg.GRAD_ACCUM_STEPS
                loss.backward()
                if (batch_idx + 1) % cfg.GRAD_ACCUM_STEPS == 0 or (batch_idx + 1) == len(loader):
                    optimizer.step()
                    optimizer.zero_grad()

            if torch.isnan(logits).any():
                print(
                    "❌ Model produced NaN outputs! Check learning rate or gradient clipping."
                )
            losses.append(loss.item() * cfg.GRAD_ACCUM_STEPS)
            pbar.set_postfix({'loss': f'{np.mean(losses):.4f}'})

    return np.mean(losses)

@torch.no_grad()
def eval_epoch(model, loader, criterion, cfg):
    model.eval()
    losses = []
    all_preds, all_labels = [], []
    active_label_idx = cfg.TRAINING_LABEL

    for batch in tqdm(loader, desc="Val"):
        # images = batch["images"].to(cfg.DEVICE)
        # slice_mask = batch["slice_mask"].to(cfg.DEVICE)
        # plane_ids = batch["plane_ids"].to(cfg.DEVICE)
        # fluid_sensitive = batch["fluid_sensitive"].to(cfg.DEVICE)
        # fat_suppression = batch["fat_suppression"].to(cfg.DEVICE)

        # Target label shape: (B,)
        labels = batch["labels"].to(cfg.DEVICE)
        slices = batch["slices"].to(cfg.DEVICE)
        slicemasks = batch["slicemasks"].to(cfg.DEVICE)
        fats = batch["fat_suppression"].to(cfg.DEVICE)
        fluids = batch["fluid_sensitive"].to(cfg.DEVICE)
        plane_ids = batch["plane_ids"].to(cfg.DEVICE)
        plane_crop_mixed=batch["plane_crop_mixed"].to(cfg.DEVICE)
        seriemasks = batch["seriemasks"].to(cfg.DEVICE) if "seriemasks" in batch else slicemasks[:,:,0]
        if cfg.USE_AMP:
            with torch.amp.autocast("cuda",dtype=torch.float16):
                logits = model.forward(slices,slicemasks,seriemasks,plane_ids,fluids,fats,plane_crop_mixed)
        else:
            logits = model.forward(slices, slicemasks, seriemasks, plane_ids, fluids, fats, plane_crop_mixed)

        # Cast logits to float32 for numerically stable loss calculation
        loss = criterion(logits.float(), labels)
        losses.append(loss.item())

        if torch.isnan(logits).any():
            print(
                "❌ Model produced NaN outputs! Check learning rate or gradient clipping."
            )

        # Compute probabilities for active label only & cast to float32 before numpy conversion
        preds = torch.sigmoid(logits).cpu().numpy()

        # preds[preds<=0.50]=0.0
        # preds[preds>0.50]=1.0

        all_preds.append(preds)
        all_labels.append(labels.cpu().numpy())

    # Concatenate 1D batch vectors into flat 1D arrays of shape (N_samples,)
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    # all_preds[all_preds >= 0.50]=1.0
    # all_preds[all_preds < 0.50]=0.0
    binary_labels = (all_labels >= cfg.LABEL_CLAMP_THRESHOLD).astype(int)
    val_loss = np.mean(losses)
    try:
        val_auc = roc_auc_score(binary_labels, all_preds)
    except ValueError:
        val_auc = 0.5

    return val_loss, val_auc


class AsymmetricLossOptimized(nn.Module):
    """
    Numerically stable Asymmetric Loss for Multi-Label Classification.
    """
    def __init__(self, gamma_neg: float = 4.0, gamma_pos: float = 1.0, clip: float = 0.05, eps: float = 1e-5):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # 1. Compute probabilities from raw logits
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1.0 - x_sigmoid

        # 2. Asymmetric Margin Clipping for Negatives
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1.0)

        # 3. Calculate basic log loss components safely
        loss_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        loss_neg = (1.0 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = loss_pos + loss_neg

        # 4. Apply Asymmetric Focusing Weights without backpropagating through the weights
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            with torch.no_grad():
                pt0 = xs_pos * y
                pt1 = xs_neg * (1.0 - y)
                pt = pt0 + pt1
                one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1.0 - y)
                # Clamp (1.0 - pt) to 0.0 to prevent negative bases in pow
                one_sided_w = torch.pow((1.0 - pt).clamp(min=0.0), one_sided_gamma)

            loss = loss * one_sided_w

        return -loss.mean()
def build_optimizer_with_decay_exclusion(model, lr=2e-4, weight_decay=0.01):
    decay_params, no_decay_params = [], []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # Exclude biases and 1D norm parameters from decay
        if param.ndim < 2 or name.endswith(".bias") or "norm" in name.lower():
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optimizer_grouped_parameters = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(optimizer_grouped_parameters, lr=lr)
import torch
from tqdm import tqdm

class CachedEmbeddingDataset(torch.utils.data.Dataset):

    def __init__(self, pt_file,cfg):
        data = torch.load(pt_file)
        studies = data["study_id"]  # Tensor shape: (N, 264)
        slices = data["slices"]  # Tensor shape: (N, 264)
        slicemasks = data["slicemasks"] if "slicemasks" in data else data["masks"]  # Tensor shape: (N, 264)
        plane_ids = data["plane_ids"]  # Tensor shape: (N, 264)
        fluids = data["fluid_sensitive"]  # Tensor shape: (N, 264)
        fats = data["fat_suppression"]  # Tensor shape: (N, 264)
        labels = data["labels"]
        plane_crop_mixed = data["plane_crop_mixed"]
        labels=torch.cat(labels,dim=0)# Tensor shape: (N, 12)
        # mask = (labels[:, cfg.TRAINING_LABEL] > 0.90) | (labels[:, cfg.TRAINING_LABEL] < 0.05)
        # masknp=mask.numpy()
        splittedfeatures=[]
        self.studies=[]
        self.slices=[]
        self.slicemasks=[]
        self.plane_ids=[]
        self.fluids=[]
        self.fats=[]
        self.labels=[]
        self.plane_crop_mixed=[]
        # for i in range(len(features)):
        #     elem0=torch.split(features[i][0],16,0)
        #     elem1=features[i][1][1:]
        #     elem2=torch.split(features[i][2],1,0)
        #     elem3=torch.split(features[i][3],1,0)
        #     elem4=torch.split(features[i][4],1,0)
        #     elem5=torch.split(features[i][5],1,0)
        #
        #     for j in range(4):
        #         splittedfeatures.append((elem0[j],elem1[j],elem2[j].squeeze(0),elem3[j].squeeze(0),elem4[j].squeeze(0),elem5[j].squeeze(0)))
        #         if j==0 and i==len(features)-1:
        #             break
        nanslices=0
        for i in range(len(studies)):
            if torch.isnan(slices[i]).any():
                nanslices+=1
                continue
            self.studies.append(studies[i][0])
            self.slices.append(slices[i].squeeze(0))
            self.slicemasks.append(slicemasks[i].squeeze(0))
            self.plane_ids.append(plane_ids[i].squeeze(0))
            self.fluids.append(fluids[i].squeeze(0))
            self.fats.append(fats[i].squeeze(0))
            self.labels.append(labels[i].squeeze(0))
            self.plane_crop_mixed.append(plane_crop_mixed[i].squeeze(0))
        # self.features = features[mask]  # Tensor shape: (N, 264)
        # self.labels = labels  # Tensor shape: (N, 12)
        print(f"Nanslices: {nanslices}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.studies[idx],self.slices[idx],self.slicemasks[idx],self.plane_ids[idx],self.fluids[idx],self.fats[idx], self.labels[idx],self.plane_crop_mixed[idx]
def cache_collate_fn(batch):
    studies=[x[0] for x in batch]
    slices=[x[1] for x in batch]
    slicemasks=[x[2] for x in batch]
    plane_ids=[x[3] for x in batch]
    fluids=[x[4] for x in batch]
    fats=[x[5] for x in batch]
    labels=[x[6] for x in batch]
    plane_crop_mixed=[x[7] for x in batch]
    maxslicesize=max([slice.shape[1] for slice in slices])
    maxseriesize=max([slice.shape[0] for slice in slices])
    paddedslices=[]
    paddedslicemasks=[]
    paddedplane_ids=[]
    paddedfluids=[]
    paddedfats=[]
    paddedplane_crop_mixed=[]
    seriemasks=[]

    for i in range(len(slices)):
        newslice=slices[i]
        newslicemask=slicemasks[i]
        newplane_id=plane_ids[i]
        newfluid=fluids[i]
        newfat=fats[i]
        newplane_crop_mixed=plane_crop_mixed[i]
        newseriemask=torch.ones_like(newplane_id,dtype=torch.bool)
        if newslice.shape[1]<maxslicesize:
            slicediffer=maxslicesize-newslice.shape[1]
            newslice=torch.cat([newslice,torch.zeros((newslice.shape[0],slicediffer,newslice.shape[-1]))],dim=1)
            newslicemask=torch.cat((newslicemask,torch.zeros((newslicemask.shape[0],slicediffer),dtype=torch.bool)),dim=1)
        if newslice.shape[0]<maxseriesize:
            s=newslice.shape[0]
            seriediffer=maxseriesize-s
            newslice=torch.cat([newslice,torch.zeros((seriediffer,newslice.shape[1],newslice.shape[-1]))],dim=0)
            newslicemask=torch.cat((newslicemask,torch.zeros((seriediffer,newslicemask.shape[1]),dtype=torch.bool)),dim=0)
            paddingones= torch.ones([seriediffer])
            newplane_id=torch.cat((newplane_id,paddingones*10.0),dim=0)
            newfluid=torch.cat((newfluid,paddingones*10.0),dim=0)
            newfat=torch.cat((newfat,paddingones*10.0),dim=0)
            newseriemask=torch.cat((newseriemask,torch.zeros(seriediffer,dtype=torch.bool)),dim=0)
            newplane_crop_mixed=torch.cat((newplane_crop_mixed,torch.ones((seriediffer,2))*10.0),dim=0)
        paddedslices.append(newslice)
        paddedslicemasks.append(newslicemask)
        paddedplane_ids.append(newplane_id)
        paddedfluids.append(newfluid)
        paddedfats.append(newfat)
        seriemasks.append(newseriemask)
        paddedplane_crop_mixed.append(newplane_crop_mixed)
    return {
        "studies": studies,
        "labels": torch.stack(labels,dim=0),
        "slicemasks": torch.stack(paddedslicemasks,dim=0),
        "plane_ids": torch.stack(paddedplane_ids,dim=0).long(),
        "fluid_sensitive": torch.stack(paddedfluids,dim=0).long(),
        "fat_suppression": torch.stack(paddedfats,dim=0).long(),
        "seriemasks": torch.stack(seriemasks,dim=0),
        "slices": torch.stack(paddedslices,dim=0),
        "plane_crop_mixed":torch.stack(paddedplane_crop_mixed,dim=0).long(),
    }
    pass
@torch.no_grad()
def precompute_and_save(model, loader, save_path, cfg):
    model.eval()
    all_labels,all_study,all_slice,all_masks,all_plane_ids,all_fluid,all_fat,all_planecrop = [], [],[],[],[],[],[],[]

    for batch in tqdm(loader, desc="Caching Features"):
        #
        # # Extract (B, 264) vectors
        # with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        #     feats = model.extract_features(
        #         images, slice_mask, plane_ids, fluid_sens, fat_supp
        #     )
        # newfeats=[]
        # for i in range(len(feats)):
        #     if torch.is_tensor(feats[i]):
        #         newfeats.append(feats[i].cpu())
        #     else:
        #         newfeats.append(feats[i])

        all_labels.append(batch["labels"].cpu())
        all_study.append(batch["study_id"])
        all_slice.append(batch["slices"].cpu())
        all_masks.append(batch["masks"].cpu())
        all_plane_ids.append(batch["plane_ids"].cpu())
        all_fluid.append(batch["fluid_sensitive"].cpu())
        all_fat.append(batch["fat_suppression"].cpu())
        all_planecrop.append(batch["plane_crop_mixed"].cpu())


    torch.save(
        {
            "study_id": all_study,
            "masks": all_masks,
            "plane_ids": all_plane_ids,
            "fat_suppression": all_fat,
            "fluid_sensitive": all_fluid,
            "slices": all_slice,  # Tensor shape: (N, 264)
            "labels": all_labels,
            "plane_crop_mixed": all_planecrop,
        },
        save_path,
    )
    print(f"✅ Extracted features saved to {save_path}")
def main():
    print('Loading metadata...')
    df_train = pd.read_csv(cfg.TRAIN_CSV)
    df_val = pd.read_csv(cfg.VAL_CSV)
    df_tnew= df_train[~df_train["StudyInstanceUID"].isin(df_val["StudyInstanceUID"])]
    # label_col = cfg.LABELS[cfg.TRAINING_LABEL]

    # Filters explicit positives (>= 0.90) and explicit negatives (<= 0.10)
    df_train = df_tnew
    # [
    #     (df_tnew[label_col] >= 0.90) | (df_tnew[label_col] <= 0.10)
    # ].reset_index(drop=True)
    df_series = pd.read_csv(cfg.SERIES_CSV)
    
    if cfg.DEBUG_SUBSET:
        df_train = df_train.head(cfg.DEBUG_SUBSET)
        
    pos_weights = compute_class_weights(df_train, cfg).to(cfg.DEVICE)
    #
    # Split by study
    train_studies, val_studies = train_test_split(
        df_train, test_size=cfg.VAL_FRACTION, random_state=cfg.SEED
    )
    ds_train = KneeStudyTrainDataset(df_train, df_series, cfg.TRAIN_IMG_DIR, cfg, is_train=True)
    ds_val = KneeStudyTrainDataset(df_val, df_series, cfg.TRAIN_IMG_DIR, cfg, is_train=False)
    
    ld_train = DataLoader(ds_train, batch_size=cfg.BATCH_SIZE, shuffle=False,
                          num_workers=cfg.NUM_WORKERS, pin_memory=True,
                          collate_fn=None)
    ld_val = DataLoader(ds_val, batch_size=cfg.BATCH_SIZE, shuffle=False,
                        num_workers=cfg.NUM_WORKERS, pin_memory=True,
                        collate_fn=None)


    print('\nInitializing model...')
    model = KneeAbnormalityModelRefactored(cfg).to(cfg.DEVICE)
    model.train()
    if cfg.LOAD_PATH.exists():
        loadout=model.load_state_dict(torch.load(cfg.LOAD_PATH,weights_only=True)["model_state_dict"],strict=False)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f'Trainable: {trainable_params:,} / Total: {total_params:,}  ({100*trainable_params/total_params:.1f}%)')
    print('RadDino backbone frozen ✓')
    optimizer = torch.optim.AdamW(model.parameters(),lr=cfg.LR,weight_decay=cfg.WEIGHT_DECAY)#build_optimizer_with_decay_exclusion(model, lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.EPOCHS, eta_min=cfg.MIN_LR)
    criterion = nn.BCEWithLogitsLoss()#AsymmetricLossOptimized(gamma_neg=4.0, gamma_pos=1.0, clip=0.05)#WeightedBCELoss(pos_weight=pos_weights)##nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler(cfg.DEVICE)
    if not (cfg.PYTHON_DIR / "train_feats.pt").exists():
        precompute_and_save(model,ld_train,(cfg.PYTHON_DIR / "train_feats.pt"),cfg)
    train_cached_ds = CachedEmbeddingDataset("train_feats.pt",cfg)

    # Calculate split sizes (e.g., 20% validation)
    val_size = int(len(train_cached_ds) * cfg.VAL_FRACTION)
    train_size = len(train_cached_ds) - val_size

    # Split with reproducible seed
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(
        train_cached_ds, [train_size, val_size], generator=generator
    )
    ld_train = DataLoader(train_dataset, batch_size=16, shuffle=True,collate_fn=cache_collate_fn)
    ld_val2 = DataLoader(val_dataset, batch_size=16, shuffle=False,collate_fn=cache_collate_fn)
    best_auc2 = 0.0
    best_auc=0
    for epoch in range(cfg.EPOCHS):
        print(f'\n--- Epoch {epoch+1}/{cfg.EPOCHS} ---')
        
        train_loss = train_epoch(model, ld_train, optimizer, criterion,scaler, cfg)
        val_loss, val_auc = eval_epoch(model, ld_val, criterion, cfg)
        val_loss2, val_auc2 = eval_epoch(model, ld_val2, criterion, cfg)
        scheduler.step()
        avgauc=(val_auc+val_auc2)/2
        print(f'Train loss: {train_loss:.4f}  Val loss: {val_loss:.4f}  Val AUC: {val_auc:.4f}  Val loss2: {val_loss2:.4f}  Val AUC2: {val_auc2:.4f}')
        if avgauc > best_auc:
            best_auc = avgauc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_auc': float(val_auc),
                'val_auc2': float(val_auc2)
            }, cfg.SAVE_PATH)
            print(f'✓ Checkpoint saved (AUC: {avgauc:.4f})')
        # if val_auc2 > best_auc2:
        #     best_auc2 = val_auc2
        #     torch.save({
        #         'epoch': epoch,
        #         'model_state_dict': model.state_dict(),
        #         'val_auc': val_auc,
        #         'val_auc2': val_auc2
        #     }, cfg.SAVE_PATH2)
        #     print(f'✓ Checkpoint saved (AUC2: {val_auc2:.4f})')
    print(f"Best AUC2: {best_auc2:.4f}")
    print(f"Best AUC: {best_auc:.4f}")
if __name__ == '__main__':
    main()

    pass
