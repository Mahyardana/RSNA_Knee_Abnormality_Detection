import os
import cv2
import numpy as np
import pandas as pd
import pydicom
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Tuple, Optional, List
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# Prevent multi-threading thrashing across worker processes
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"


class ExtractionConfig:
    KAGGLE_DIR = Path('/run/media/mahyar/Samsung-990-PRO/Projects/Kaggle/rsna-knee-abnormality-detection')
    PYTHON_DIR = Path('/run/media/mahyar/Samsung-990-PRO/Projects/Python/RSNA Knee Abnormality Detection')

    TRAIN_CSV = PYTHON_DIR / 'llm_labels_v4_blend.csv'
    SERIES_CSV = PYTHON_DIR / 'train_series.csv'
    TRAIN_IMG_DIR = KAGGLE_DIR / 'train_series'

    # 🔹 Output directory for plain JPEG files on disk
    OUTPUT_DIR = KAGGLE_DIR / 'extracted_jpg_dataset'

    TARGET_SIZE = (224, 224)
    CROP_SIZE_MM = 80.0

    # 🔹 Subsampling Config: Keeps 1 out of every 2 slices (50% dataset reduction)
    SUBSAMPLE_STEP = 2

    # 9950X has 32 threads; setting 28 to 32 workers maximizes CPU throughput
    NUM_WORKERS = min(32, os.cpu_count() or 16)


cfg = ExtractionConfig()


def pad_and_resize_center(slice_img: torch.Tensor, target_size: Tuple[int, int] = (224, 224)) -> torch.Tensor:
    _, h, w = slice_img.shape
    if h != w:
        diff = abs(h - w)
        pad_1 = diff // 2
        pad_2 = diff - pad_1
        if h < w:
            slice_img = F.pad(slice_img, (0, 0, pad_1, pad_2), mode='replicate')
        else:
            slice_img = F.pad(slice_img, (pad_1, pad_2, 0, 0), mode='replicate')

    return F.interpolate(slice_img.unsqueeze(0), size=target_size, mode="bilinear", align_corners=False).squeeze(0)


def extract_mm_crops(slice_img: torch.Tensor, pixel_spacing: Tuple[float, float], crop_size_mm: float,
                     target_size: Tuple[int, int]) -> List[torch.Tensor]:
    _, c_h, c_w = slice_img.shape
    crop_h = int(crop_size_mm / pixel_spacing[0])
    crop_w = int(crop_size_mm / pixel_spacing[1])

    if c_h < crop_h or c_w < crop_w:
        return []

    stride_h, stride_w = crop_h // 2, crop_w // 2

    h_starts = list(range(0, c_h - crop_h + 1, stride_h))
    if h_starts[-1] + crop_h < c_h:
        h_starts.append(c_h - crop_h)

    w_starts = list(range(0, c_w - crop_w + 1, stride_w))
    if w_starts[-1] + crop_w < c_w:
        w_starts.append(c_w - crop_w)

    patches = []
    for h in h_starts:
        for w in w_starts:
            patch = slice_img[:, h:h + crop_h, w:w + crop_w]
            resized_patch = F.interpolate(patch.unsqueeze(0), size=target_size, mode="bilinear",
                                          align_corners=False).squeeze(0)
            patches.append(resized_patch)

    return patches


def save_tensor_to_jpg(tensor_img: torch.Tensor, out_path: Path):
    """Saves a PyTorch image tensor directly to disk as JPEG."""
    np_img = tensor_img.detach().cpu().numpy().transpose(1, 2, 0)
    np_img = np.clip(np_img * 255.0, 0, 255).astype(np.uint8)
    bgr_img = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(out_path), bgr_img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])


def load_dicom_series(series_path: Path) -> Tuple[Optional[torch.Tensor], Tuple[float, float]]:
    alldcms = [pydicom.dcmread(str(dcm), stop_before_pixels=True) for dcm in sorted(series_path.glob('*.dcm'))]
    valid_slices = [ds for ds in alldcms if "ImagePositionPatient" in ds and "ImageOrientationPatient" in ds]

    if not valid_slices:
        return None, (0.5, 0.5)

    pixel_spacing = (0.5, 0.5)
    if "PixelSpacing" in valid_slices[0]:
        pixel_spacing = (float(valid_slices[0].PixelSpacing[0]), float(valid_slices[0].PixelSpacing[1]))

    orient = valid_slices[0].ImageOrientationPatient
    row_cosines, col_cosines = np.array(orient[:3]), np.array(orient[3:])
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
        return None, pixel_spacing

    return torch.stack(slices), pixel_spacing


def process_series_task(task_args: tuple) -> int:
    """Worker process writing JPEG files directly to disk, returning count of saved images."""
    study_id, series_id, plane_str, series_path_str, output_dir_str, target_size, crop_size_mm, subsample_step = task_args

    # Enforce strict single-threaded libraries per worker process
    torch.set_num_threads(1)
    cv2.setNumThreads(1)

    series_path = Path(series_path_str)
    output_dir = Path(output_dir_str)

    volume_tensor, pixel_spacing = load_dicom_series(series_path)
    if volume_tensor is None:
        return 0

    saved_count = 0

    # 🔹 Slice Subsampling: Stepping through volume using subsample_step
    for idx in range(0, len(volume_tensor), subsample_step):
        slice_tensor = volume_tensor[idx]

        # 1. Whole Symmetrically Padded Slice
        full_img = pad_and_resize_center(slice_tensor, target_size)
        fname_orig = f"{study_id}_{series_id}_{plane_str}_s{idx}_orig.jpg"
        save_tensor_to_jpg(full_img, output_dir / fname_orig)
        saved_count += 1

        # 2. Local 80mm Sliding Window Patches
        patches = extract_mm_crops(slice_tensor, pixel_spacing, crop_size_mm, target_size)
        for p_idx, patch in enumerate(patches):
            fname_patch = f"{study_id}_{series_id}_{plane_str}_s{idx}_p{p_idx}.jpg"
            save_tensor_to_jpg(patch, output_dir / fname_patch)
            saved_count += 1

    return saved_count


def main():
    cfg.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df_train = pd.read_csv(cfg.TRAIN_CSV)
    df_series = pd.read_csv(cfg.SERIES_CSV)

    # Pre-build task argument tuples
    tasks = []
    for _, study_row in df_train.iterrows():
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

            tasks.append((
                study_id,
                series_id,
                plane_str,
                str(series_path),
                str(cfg.OUTPUT_DIR),
                cfg.TARGET_SIZE,
                cfg.CROP_SIZE_MM,
                cfg.SUBSAMPLE_STEP
            ))

    print(f"🚀 Dispatching {len(tasks)} series tasks across {cfg.NUM_WORKERS} workers...")
    print(f"📁 Saving direct JPEGs to: `{cfg.OUTPUT_DIR}`")
    print(f"ℹ️  Subsample Step: {cfg.SUBSAMPLE_STEP} (extracting every {cfg.SUBSAMPLE_STEP}nd slice)")

    total_extracted = 0
    with ProcessPoolExecutor(max_workers=cfg.NUM_WORKERS) as executor:
        futures = [executor.submit(process_series_task, task) for task in tasks]

        for future in tqdm(as_completed(futures), total=len(futures), desc="Extracting JPEGs"):
            try:
                saved_count = future.result()
                total_extracted += saved_count
            except Exception as e:
                print(f"Error processing series task: {e}")

    print(f"\n✅ Complete! Extracted {total_extracted} `.jpg` images directly into `{cfg.OUTPUT_DIR}`.")


if __name__ == "__main__":
    main()