import nibabel as nib
import numpy as np
from totalsegmentator.python_api import totalsegmentator
from totalsegmentator.map_to_binary import class_map
import os
from pathlib import Path

import multiprocessing as mp
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass
custom_temp_dir = Path("./tmp").resolve()
custom_temp_dir.mkdir(parents=True, exist_ok=True)

# Set environment variables to override system /tmp
os.environ["TMPDIR"] = str(custom_temp_dir)
os.environ["TEMP"] = str(custom_temp_dir)
os.environ["TMP"] = str(custom_temp_dir)
os.environ["TORCH_HOME"] = str(custom_temp_dir / "torch_cache")
# 1. Path to your RSNA clinical knee MRI (NIfTI or DICOM folder)
input_nifti_path = "/run/media/mahyar/Samsung-990-PRO/Projects/Kaggle/rsna-knee-abnormality-detection/train_series/1.2.826.0.1.3680043.8.498.10004873229099053869093324292195817260/1.2.826.0.1.3680043.8.498.12343110195036213483454091715412333772"


# 4. Derive 3D Bounding Box
def get_3d_bbox(
        mask: np.ndarray, padding: int = 10
) -> tuple[slice, slice, slice]:
    nonzero = np.where(mask > 0)
    if len(nonzero[0]) == 0:
        raise ValueError(
            "No knee structures detected in the MRI volume."
        )  # Fallback if unsegmented

    z_min = max(0, int(nonzero[0].min() - padding))
    z_max = min(mask.shape[0], int(nonzero[0].max() + padding))
    y_min = max(0, int(nonzero[1].min() - padding))
    y_max = min(mask.shape[1], int(nonzero[1].max() + padding))
    x_min = max(0, int(nonzero[2].min() - padding))
    x_max = min(mask.shape[2], int(nonzero[2].max() + padding))

    return (slice(z_min, z_max), slice(y_min, y_max), slice(x_min, x_max))
if __name__ == "__main__":
    # 1. Run appendicular_bones_mr WITHOUT roi_subset
    mask_img = totalsegmentator(
        input=input_nifti_path,
        task="appendicular_bones_mr",
    )

    # 2. Extract multi-label array
    mask_data = mask_img.get_fdata()

    # Class IDs for appendicular_bones_mr:
    # 1: patella | 2: tibia | 3: fibula
    bone_map = class_map["appendicular_bones_mr"]
    newbone_map = class_map["total_mr"]
    patella_id = [k for k, v in bone_map.items() if v == "patella"][0]  # 1
    tibia_id = [k for k, v in bone_map.items() if v == "tibia"][0]  # 2
    fibula_id = [k for k, v in bone_map.items() if v == "fibula"][0]

    # 3. Create binary combined knee mask (Tibia + Patella)
    knee_bones_mask = np.isin(mask_data,[patella_id, tibia_id, fibula_id])

    # 4. Extract 3D bounding box coordinates
    nonzero = np.where(knee_bones_mask > 0)
    if len(nonzero[0]) > 0:
        z_min, z_max = nonzero[0].min(), nonzero[0].max()
        y_min, y_max = nonzero[1].min(), nonzero[1].max()
        x_min, x_max = nonzero[2].min(), nonzero[2].max()
        print(f"[SUCCESS] Knee BBox Limits -> Z: [{z_min}:{z_max}], Y: [{y_min}:{y_max}], X: [{x_min}:{x_max}]")