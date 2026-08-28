import glob
import copy
import os
import numpy as np
import pydicom
import timm
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import random
from collections import defaultdict
from torch.utils.data import Dataset
# --- 1. Custom DICOM Dataset ---
def get_spatial_slice_position(dcm_path):
    """
    Computes the absolute physical 3D position of a DICOM slice
    projected along its slice normal vector.
    """
    try:
        header = pydicom.dcmread(dcm_path, stop_before_pixels=True)

        if "ImagePositionPatient" in header and "ImageOrientationPatient" in header:
            pos = np.array(header.ImagePositionPatient, dtype=float)
            ori = np.array(header.ImageOrientationPatient, dtype=float)

            row_cosine = ori[:3]
            col_cosine = ori[3:]
            slice_normal = np.cross(row_cosine, col_cosine)

            return float(np.dot(pos, slice_normal))

        elif "SliceLocation" in header:
            return float(header.SliceLocation)

        elif "InstanceNumber" in header:
            return float(header.InstanceNumber)

    except Exception:
        pass
    return 0.0


class RSNAMiddleSliceDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.transform = transform
        self.series_map = self._index_and_filter_series(root_dir)
        self.series_dirs = list(self.series_map.keys())
        print(f"\nDataset ready: {len(self.series_dirs)} series indexed.")

    def _index_and_filter_series(self, root_dir):
        print("Discovering DICOM files...")
        all_files = glob.glob(os.path.join(root_dir, "**/*.dcm"), recursive=True)
        print(f"Found {len(all_files)} DICOM files.")

        # 1. Group files by series directory with tqdm progress bar
        raw_series = defaultdict(list)
        for path in tqdm(all_files, desc="Grouping DICOMs by Series"):
            raw_series[os.path.dirname(path)].append(path)

        filtered_series = {}

        # 2. Sort and trim series with tqdm progress bar
        for series_dir, file_paths in tqdm(raw_series.items(), desc="Sorting Slices (3D Space)"):
            if not file_paths:
                continue

            # Sort slices geometrically along the scan plane normal
            sorted_files = sorted(file_paths, key=get_spatial_slice_position)

            # Extract middle 60% slice range
            n_slices = len(sorted_files)
            start_idx = int(n_slices * 0.20)
            end_idx = int(n_slices * 0.80)

            middle_candidates = sorted_files[start_idx:end_idx]

            if not middle_candidates:
                middle_candidates = sorted_files

            filtered_series[series_dir] = middle_candidates

        return filtered_series

    def __len__(self):
        return len(self.series_dirs)

    def __getitem__(self, idx):
        series_dir = self.series_dirs[idx]
        candidates = self.series_map[series_dir]

        # Randomly select 1 slice from middle 60% per epoch
        dcm_path = random.choice(candidates)

        dicom = pydicom.dcmread(dcm_path)
        img = dicom.pixel_array.astype(np.float32)

        if getattr(dicom, "PhotometricInterpretation", "") == "MONOCHROME1":
            img = np.amax(img) - img

        img_min, img_max = img.min(), img.max()
        if img_max > img_min:
            img = (img - img_min) / (img_max - img_min) * 255.0
        else:
            img = np.zeros_like(img)
        img = img.astype(np.uint8)

        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)

        pil_img = Image.fromarray(img)

        if self.transform:
            return self.transform(pil_img)
        return pil_img

# --- 2. Multi-Crop Augmentation Pipeline for MRI ---
class DINOMultiCropTransform:
    def __init__(self):
        self.global_transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.5, 1.0), ratio=(0.85, 1.15)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.local_transform = transforms.Compose([
            transforms.RandomResizedCrop(96, scale=(0.15, 0.5), ratio=(0.85, 1.15)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __call__(self, image):
        crops = [self.global_transform(image), self.global_transform(image)]
        for _ in range(4):
            crops.append(self.local_transform(image))
        return crops

# --- 3. DINO Loss Function with Dynamic Centering ---
class DINOLoss(nn.Module):
    def __init__(self, out_dim=2048, teacher_temp=0.04, student_temp=0.1, center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim))

    def forward(self, student_out, teacher_out):
        student_logits = student_out / self.student_temp
        teacher_probs = F.softmax((teacher_out - self.center) / self.teacher_temp, dim=-1).detach()
        loss = torch.sum(-teacher_probs * F.log_softmax(student_logits, dim=-1), dim=-1).mean()
        return loss

    @torch.no_grad()
    def update_center(self, teacher_preds_all):
        batch_center = torch.sum(teacher_preds_all, dim=0, keepdim=True)
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)

# --- 4. DINOv2 Architecture Wrapper ---
class DINOv2Wrapper(nn.Module):
    def __init__(self, backbone, embed_dim, proj_dim=2048):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Sequential(
            nn.Linear(embed_dim, 2048),
            nn.GELU(),
            nn.Linear(2048, proj_dim)
        )

    def forward(self, x):
        feats = self.backbone(x)
        return self.head(feats)

# --- 5. Main Fine-Tuning Execution ---
def train_dinov2_dicom():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Output directory setup
    output_dir = "./checkpoints"
    os.makedirs(output_dir, exist_ok=True)

    # Load dataset paths
    root_dir = "/run/media/mahyar/Samsung-990-PRO/Projects/Kaggle/rsna-knee-abnormality-detection/train_series/"
    dicom_paths = glob.glob(os.path.join(root_dir, "**/*.dcm"), recursive=True)
    print(f"Found {len(dicom_paths)} DICOM files.")

    dataset = RSNAMiddleSliceDataset(root_dir, transform=DINOMultiCropTransform())
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4, drop_last=True)

    # Load RadioDINO backbone
    pretrained_backbone = timm.create_model(
        "hf_hub:Snarcy/RadioDino-b16",
        pretrained=True,
        dynamic_img_size=True,
        num_classes=0
    )
    embed_dim = pretrained_backbone.embed_dim

    # Build Student and Teacher Networks
    student = DINOv2Wrapper(pretrained_backbone, embed_dim, proj_dim=2048).to(device)
    teacher = copy.deepcopy(student).to(device)

    for p in teacher.parameters():
        p.requires_grad = False

    optimizer = torch.optim.AdamW(student.parameters(), lr=1e-5, weight_decay=0.04)
    dino_loss_fn = DINOLoss(out_dim=2048).to(device)
    ema_momentum = 0.996

    num_epochs = 10
    print("\nStarting Self-Supervised Fine-Tuning...")

    for epoch in range(num_epochs):
        student.train()
        teacher.train()
        total_loss = 0.0

        pbar = tqdm(dataloader, desc=f"Epoch [{epoch+1}/{num_epochs}]", leave=True)

        for step, crops in enumerate(pbar):
            global_crops = [c.to(device) for c in crops[:2]]
            local_crops = [c.to(device) for c in crops[2:]]

            # 1. Forward Teacher on Global crops (returns list of 2 tensors [B, 2048])
            with torch.no_grad():
                teacher_outputs = [teacher(gc) for gc in global_crops]

            # 2. Forward Student on all crops (returns list of 6 tensors [B, 2048])
            all_crops = global_crops + local_crops
            student_outputs = [student(c) for c in all_crops]

            # 3. Pairwise DINO Loss calculation
            loss = 0
            n_loss_terms = 0
            for s_idx, s_out in enumerate(student_outputs):
                for t_idx, t_out in enumerate(teacher_outputs):
                    if s_idx == t_idx:
                        continue  # Skip identical view comparisons
                    loss += dino_loss_fn(s_out, t_out)
                    n_loss_terms += 1

            loss = loss / n_loss_terms

            # Update loss center buffer
            with torch.no_grad():
                dino_loss_fn.update_center(torch.cat(teacher_outputs, dim=0))

            # Backpropagation & Gradient Clipping
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=3.0)
            optimizer.step()

            # 4. EMA Update for Teacher weights
            with torch.no_grad():
                for param_s, param_t in zip(student.parameters(), teacher.parameters()):
                    param_t.data.mul_(ema_momentum).add_(param_s.data, alpha=1 - ema_momentum)

            total_loss += loss.item()
            running_avg_loss = total_loss / (step + 1)

            # Update progress bar metrics
            pbar.set_postfix({
                "batch_loss": f"{loss.item():.4f}",
                "avg_loss": f"{running_avg_loss:.4f}"
            })

        epoch_loss = total_loss / len(dataloader)
        print(f"Summary Epoch [{epoch+1}/{num_epochs}] - Loss: {epoch_loss:.4f}")

        # Save intermediate epoch checkpoint
        ckpt_path = os.path.join(output_dir, f"radiodino_epoch_{epoch+1}.pth")
        torch.save(student.backbone.state_dict(), ckpt_path)
        print(f"Saved backbone weights to {ckpt_path}\n")

    # Save final model state
    final_path = os.path.join(output_dir, "radiodino_final.pth")
    torch.save(student.backbone.state_dict(), final_path)
    print(f"Training Complete. Final model saved to {final_path}")

if __name__ == "__main__":
    train_dinov2_dicom()