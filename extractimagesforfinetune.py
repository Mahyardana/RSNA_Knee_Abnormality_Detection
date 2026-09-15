import io
import math
import warnings
from pathlib import Path
from typing import Dict, Tuple, Optional, Union

import cv2
import numpy as np
import pandas as pd
import pydicom
import torch
import torch.nn.functional as F
import torchvision
from tqdm import tqdm

warnings.filterwarnings('ignore')


class ExtractionConfig:
    KAGGLE_DIR = Path('/run/media/mahyar/Samsung-990-PRO/Projects/Kaggle/rsna-knee-abnormality-detection')
    PYTHON_DIR = Path('/run/media/mahyar/Samsung-990-PRO/Projects/Python/RSNA Knee Abnormality Detection')

    TRAIN_CSV = PYTHON_DIR / 'llm_labels_v4_blend.csv'
    SERIES_CSV = PYTHON_DIR / 'train_series.csv'
    TRAIN_IMG_DIR = KAGGLE_DIR / 'train_series'

    # Directory where DINOv2 JPEG dataset will be exported
    OUTPUT_DIR = PYTHON_DIR / 'dinov2_dataset'

    PLANE_TO_IDX = {'Sagittal': 0, 'Axial': 1, 'Coronal': 2}
    PLANE_NAMES = {0: 'sagittal', 1: 'axial', 2: 'coronal'}

    # Target resolution for DINOv2 (224x224 or 518x518)
    TARGET_SIZE = (224, 224)
    EXPANSION_FACTORS = [0.1, 0.0 , -0.1]

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


cfg = ExtractionConfig()


class TorchScriptROIExtractor:
    def __init__(
            self,
            axial_model_path: str,
            sagittal_model_path: str,
            coronal_model_path: str,
            device: str = "cuda",
    ):
        self.device = device
        self.models = {
            "axial": torch.jit.load(axial_model_path, map_location=device).eval(),
            "sagittal": torch.jit.load(sagittal_model_path, map_location=device).eval(),
            "coronal": torch.jit.load(coronal_model_path, map_location=device).eval(),
        }
        self.class_maps = {
            "axial": {0: "whole", 1: "anterior", 2: "medial", 3: "lateral", 4: "back", 5: "front"},
            "sagittal": {0: "whole", 1: "anterior", 2: "medial", 3: "lateral", 4: "back", 5: "front"},
            "coronal": {0: "whole", 1: "anterior", 2: "medial", 3: "lateral", 4: "back", 5: "front"},
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

    def decode_yolo_output(self, raw_output: torch.Tensor, conf_thresh: float) -> torch.Tensor:
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

        keep_indices = torchvision.ops.batched_nms(boxes_xyxy, scores, class_ids, iou_threshold=0.0)
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
            self, volume_tensor: torch.Tensor, plane: str, conf_thresh: float = 0.45
    ) -> Tuple[Dict[str, torch.Tensor], int]:
        yolo_model = self.models[plane]
        class_map = self.class_maps.get(plane, {})
        num_slices = volume_tensor.shape[0]

        orig_h, orig_w = volume_tensor.shape[-2], volume_tensor.shape[-1]
        orig_hpad = orig_w - orig_h if orig_w > orig_h else 0
        orig_wpad = orig_h - orig_w if orig_h > orig_w else 0

        padded_vol = F.pad(volume_tensor, (0, orig_wpad, 0, orig_hpad), mode="constant", value=0)
        resized_vol = F.interpolate(padded_vol, (512, 512), mode="bilinear", align_corners=False)
        padded_512 = F.pad(resized_vol, (0, 128, 0, 128), mode="constant", value=0)

        best_slice_idx = num_slices // 2
        highest_conf = -1.0

        for idx in self.get_center_out_slice_indices(num_slices):
            slice_tensor = padded_512[idx: idx + 1].to(self.device)
            with torch.amp.autocast("cuda", dtype=torch.float16):
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

        best_slice_tensor = padded_512[best_slice_idx: best_slice_idx + 1].to(self.device)
        predictions = yolo_model(best_slice_tensor)[0]

        effective_thresh = conf_thresh if highest_conf >= conf_thresh else highest_conf
        decodedyolo = self.decode_yolo_output(predictions, effective_thresh)

        scale_w = (orig_w + orig_wpad) / 640.0
        scale_h = (orig_h + orig_hpad) / 640.0
        scale_tensor = torch.tensor([scale_w, scale_h, scale_w, scale_h], device=decodedyolo.device)

        best_crops = {}
        for pred in decodedyolo:
            if pred[4].item() >= effective_thresh:
                class_id = int(pred[5].item())
                category_key = class_map.get(class_id, str(class_id))
                best_crops[category_key] = pred[:4] * scale_tensor

        return best_crops, best_slice_idx


def crop_and_resize_box(
        slice_img: torch.Tensor,
        box: torch.Tensor,
        target_size: Tuple[int, int] = (224, 224),
        boundary_expansion: float = 0.0,
) -> torch.Tensor:
    """
    Crops slice_img using box coordinates [x1, y1, x2, y2] with optional boundary expansion,
    pads to square, and resizes to target_size.
    """
    _, h_img, w_img = slice_img.shape
    minx, minh = int(max(0, box[0])), int(max(0, box[1]))
    maxx, maxh = int(box[2]), int(box[3])

    xlen = maxx - minx
    hlen = maxh - minh

    maxx = min(w_img, int(maxx + xlen * boundary_expansion))
    minx = max(0, int(minx - xlen * boundary_expansion))
    maxh = min(h_img, int(maxh + hlen * boundary_expansion))
    minh = max(0, int(minh - hlen * boundary_expansion))

    cropped = slice_img[:, minh:maxh, minx:maxx]
    if cropped.shape[-1] == 0 or cropped.shape[-2] == 0:
        cropped = slice_img

    paddingbottom = max(0, cropped.shape[-1] - cropped.shape[-2])
    paddingright = max(0, cropped.shape[-2] - cropped.shape[-1])

    if paddingbottom > 0 or paddingright > 0:
        cropped = F.pad(cropped, (0, paddingright, 0, paddingbottom), "constant", 0)

    resized = F.interpolate(cropped.unsqueeze(0), size=target_size, mode="bilinear", align_corners=False).squeeze(0)
    return resized


def save_tensor_as_jpg(tensor_img: torch.Tensor, output_path: Path):
    """
    Converts (3, H, W) float tensor [0.0, 1.0] to uint8 JPEG.
    """
    np_img = tensor_img.detach().cpu().numpy().transpose(1, 2, 0)
    np_img = np.clip(np_img * 255.0, 0, 255).astype(np.uint8)
    bgr_img = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), bgr_img)


def load_dicom_series(series_path: Path) -> Tuple[Optional[torch.Tensor], Optional[int]]:
    alldcms = [pydicom.dcmread(str(dcm), stop_before_pixels=True) for dcm in sorted(series_path.glob('*.dcm'))]
    valid_slices = [ds for ds in alldcms if "ImagePositionPatient" in ds and "ImageOrientationPatient" in ds]

    if not valid_slices:
        return None, None

    orient = valid_slices[0].ImageOrientationPatient
    row_cosines = np.array(orient[:3])
    col_cosines = np.array(orient[3:])
    slice_normal = np.cross(row_cosines, col_cosines)

    sorted_slices = sorted(
        valid_slices, key=lambda ds: np.dot(np.array(ds.ImagePositionPatient), slice_normal)
    )

    slices = []
    for dcm_meta in sorted_slices:
        try:
            ds = pydicom.dcmread(dcm_meta.filename)
            arr = ds.pixel_array.astype(np.float32)
            if ds.get("PhotometricInterpretation", "MONOCHROME2") == "MONOCHROME1":
                arr = arr.max() - arr
            arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
            arr = np.clip(arr, 0.0, 1.0)
            arr_3ch = np.stack([arr, arr, arr], axis=0)
            slices.append(torch.from_numpy(arr_3ch))
        except Exception:
            pass

    if not slices:
        return None, None

    return torch.stack(slices), len(slices)


def main():
    # Setup PyTorch ImageFolder hierarchy: output_dir/train/0/*.jpg
    export_dir = cfg.OUTPUT_DIR / "train" / "0"
    export_dir.mkdir(parents=True, exist_ok=True)

    print(f"📁 Exporting DINOv2 JPEG dataset to: {export_dir}")

    df_train = pd.read_csv(cfg.TRAIN_CSV)
    df_series = pd.read_csv(cfg.SERIES_CSV)

    roiextractor = TorchScriptROIExtractor(
        "yolo_axial.torchscript",
        "yolo_sagittal.torchscript",
        "yolo_coronal.torchscript",
        device=str(cfg.DEVICE),
    )

    extracted_count = 0

    for _, study_row in tqdm(df_train.iterrows(), total=len(df_train), desc="Extracting DICOM & YOLO Crops"):
        study_id = str(study_row['StudyInstanceUID'])
        study_series = df_series[df_series['StudyInstanceUID'] == study_id]

        for _, series_row in study_series.iterrows():
            series_id = str(series_row['SeriesInstanceUID'])
            plane_str = str(series_row['Anatomical_Plane']).lower()
            if plane_str not in ["axial", "sagittal", "coronal"]:
                continue

            series_path = cfg.TRAIN_IMG_DIR / study_id / series_id
            if not series_path.exists():
                continue

            volume_tensor, num_slices = load_dicom_series(series_path)
            if volume_tensor is None:
                continue

            # Detect ROIs and retrieve target slice index
            crops_dict, best_slice_idx = roiextractor.extract_best_roi_crops(
                volume_tensor, plane_str, conf_thresh=0.45
            )

            target_slice = volume_tensor[best_slice_idx]
            for currentidx,target_slice in enumerate(volume_tensor):
                for expansion in cfg.EXPANSION_FACTORS:
                    exp_str = f"exp{int(expansion * 100)}"

                    # 1. Save Detected Regional Bounding Box Crops
                    for roi_name, bbox in crops_dict.items():
                        crop_tensor = crop_and_resize_box(
                            target_slice, bbox, target_size=cfg.TARGET_SIZE, boundary_expansion=expansion
                        )
                        file_name = f"{study_id}_{series_id}_{plane_str}_s{currentidx}_{roi_name}_{exp_str}.jpg"
                        save_tensor_as_jpg(crop_tensor, export_dir / file_name)
                        extracted_count += 1

                # 2. Save Full Uncropped Slice
                full_bbox = torch.tensor([0, 0, target_slice.shape[-1], target_slice.shape[-2]], dtype=torch.float)
                full_crop_tensor = crop_and_resize_box(
                    target_slice, full_bbox, target_size=cfg.TARGET_SIZE, boundary_expansion=expansion
                )
                file_name = f"{study_id}_{series_id}_{plane_str}_s{currentidx}_orig_{exp_str}.jpg"
                save_tensor_as_jpg(full_crop_tensor, export_dir / file_name)
                extracted_count += 1

    print(f"\n✅ Extraction Complete! Saved {extracted_count} `.jpg` images to `{export_dir}`.")


if __name__ == "__main__":
    main()