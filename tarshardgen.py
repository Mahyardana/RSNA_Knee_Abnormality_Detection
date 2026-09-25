import tarfile
from pathlib import Path
from tqdm import tqdm

# Input loose files directory and output shard directory
IMAGE_DIR = Path('/run/media/mahyar/Samsung-990-PRO/Projects/Kaggle/rsna-knee-abnormality-detection/dinov2_dataset/train/0')
SHARD_OUT_DIR = Path('/run/media/mahyar/Samsung-990-PRO/Projects/Kaggle/rsna-knee-abnormality-detection/dinov2_webdataset')
SHARD_OUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGES_PER_SHARD = 5000

print("Scanning image files...")
all_images = sorted(list(IMAGE_DIR.glob("*.jpg")))
total_images = len(all_images)
print(f"Found {total_images} images. Packing into tar shards of {IMAGES_PER_SHARD} items each...")

shard_idx = 0
for i in tqdm(range(0, total_images, IMAGES_PER_SHARD), desc="Creating Tar Shards"):
    batch_files = all_images[i:i + IMAGES_PER_SHARD]
    tar_path = SHARD_OUT_DIR / f"shard_{shard_idx:05d}.tar"

    with tarfile.open(tar_path, "w") as tar:
        for img_path in batch_files:
            # Store image in tar using unique filename key
            tar.add(img_path, arcname=img_path.name)

    shard_idx += 1

print(f"\n✅ Pack complete: Created {shard_idx} shards in `{SHARD_OUT_DIR}`.")