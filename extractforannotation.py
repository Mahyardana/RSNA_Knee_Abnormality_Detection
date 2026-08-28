import os
import glob
import numpy as np
import pandas as pd
import cv2
import pydicom
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm


def process_single_series(args):
    series_dir, output_root, info = args

    dcm_files = glob.glob(os.path.join(series_dir, "*.dcm"))
    if not dcm_files:
        return

    # Sort files by InstanceNumber to reliably extract the middle slice
    def get_instance_num(filepath):
        try:
            ds = pydicom.dcmread(filepath, stop_before_pixels=True)
            return int(getattr(ds, 'InstanceNumber', 0))
        except Exception:
            return 0

    dcm_files.sort(key=get_instance_num)
    mid_file = dcm_files[len(dcm_files) // 2]

    # Read DICOM and scale intensity to 8-bit
    ds = pydicom.dcmread(mid_file)
    img = ds.pixel_array.astype(float)
    img_min, img_max = img.min(), img.max()
    img_scaled = (
        ((img - img_min) / (img_max - img_min) * 255.0).astype(np.uint8)
        if img_max > img_min
        else np.zeros(img.shape, dtype=np.uint8)
    )

    # Categorize output folder structure
    out_dir = os.path.join(
        output_root,
        str(info['Anatomical_Plane']),
        f"fat_{info['Fat_Suppression']}_fluid_{info['Fluid_Sensitive']}"
    )
    os.makedirs(out_dir, exist_ok=True)

    # Save as JPEG
    cv2.imwrite(
        os.path.join(out_dir, f"{info['SeriesInstanceUID']}.jpg"),
        img_scaled,
        [cv2.IMWRITE_JPEG_QUALITY, 92]
    )


def batch_process(dataset_root, output_root, csv_path):
    df = pd.read_csv(csv_path)

    tasks = []
    for _, row in df.iterrows():
        # Build path: train_series/{StudyInstanceUID}/{SeriesInstanceUID}
        series_dir = os.path.join(dataset_root, str(row['StudyInstanceUID']), str(row['SeriesInstanceUID']))
        info = {
            'SeriesInstanceUID': str(row['SeriesInstanceUID']),
            'Anatomical_Plane': row['Anatomical_Plane'],
            'Fat_Suppression': row['Fat_Suppression'],
            'Fluid_Sensitive': row['Fluid_Sensitive']
        }
        tasks.append((series_dir, output_root, info))

    print(f"Processing {len(tasks)} series using available CPU cores...")

    # Process in parallel for high speed across 24k+ series
    with ProcessPoolExecutor() as executor:
        list(tqdm(executor.map(process_single_series, tasks), total=len(tasks)))


if __name__ == "__main__":
    DATASET_ROOT = "/run/media/mahyar/Samsung-990-PRO/Projects/Kaggle/rsna-knee-abnormality-detection/train_series"
    OUTPUT_ROOT = "/run/media/mahyar/Samsung-990-PRO/Projects/Kaggle/rsna-knee-abnormality-detection/output"
    CSV_PATH = "train_series.csv"

    batch_process(DATASET_ROOT, OUTPUT_ROOT, CSV_PATH)