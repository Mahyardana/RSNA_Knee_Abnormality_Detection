import os
import gc
from pathlib import Path
from typing import List, Dict
from collections import defaultdict
from tqdm import tqdm
from PIL import Image
import numpy as np
import pandas as pd
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm
from sklearn.model_selection import KFold

# ── 1. System & Environment Optimization ──────────────────────────────────────
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

torch.backends.cudnn.benchmark = True
# Enable TF32 & BF16 optimizations for RTX 3090 (Ampere)
torch.set_float32_matmul_precision('high')


# ── 2. Configuration ──────────────────────────────────────────────────────────
class SSLConfig:
    PYTHON_DIR = Path('/run/media/mahyar/Samsung-990-PRO/Projects/Python/RSNA Knee Abnormality Detection')
    TRAIN_CSV = PYTHON_DIR / 'llm_labels_v4_blend.csv'
    VAL_CSV = PYTHON_DIR / 'val_gold.csv'
    IMAGE_DIR = Path(
        '/run/media/mahyar/Samsung-990-PRO/Projects/Kaggle/rsna-knee-abnormality-detection/') / 'extracted_jpg_dataset'

    CHECKPOINT_PATH = "dinov2_vits14_pretrain.pth"

    NUM_FOLDS = 5
    BATCH_SIZE = 128  # Batch size per view (256 total per concatenated pass)
    NUM_WORKERS = 8
    PREFETCH_FACTOR = 4
    EPOCHS = 8
    SUBSAMPLE_STEP = 1

    # Learning Rates
    HEAD_LR = 1e-4        # Higher LR for random projection head
    BACKBONE_LR = 1e-6    # Ultra-low LR to protect pretrained RadioDINO weights
    WEIGHT_DECAY = 0.04
    IMAGE_SIZE = 224
    EMBED_DIM = 384
    OUT_DIM = 4096
    EMA_MOMENTUM = 0.996
    CLIP_GRAD = 3.0

    WARMUP_TEACHER_TEMP = 0.04
    TEACHER_TEMP = 0.07
    STUDENT_TEMP = 0.1

    SEED = 32
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


cfg = SSLConfig()


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(cfg.SEED)


# ── 3. Medical-Safe Transforms ────────────────────────────────────────────────
class MedicalDINOTransforms:
    def __init__(self, img_size=224):
        self.transform = transforms.Compose([
            # 1. Conservative crop: Preserves 75% to 100% of the joint space
            transforms.RandomResizedCrop(
                img_size,
                scale=(0.75, 1.0),
                ratio=(0.9, 1.1),
                interpolation=transforms.InterpolationMode.BICUBIC
            ),
            # 2. Subtle Affine: Simulates slight positioning/rotation variation
            transforms.RandomAffine(
                degrees=(-7, 7),
                translate=(0.05, 0.05),
                scale=(0.95, 1.05),
                interpolation=transforms.InterpolationMode.BICUBIC
            ),
            # 3. Micro Color Jitter: Protects MRI fluid/edema signal intensity
            transforms.ColorJitter(brightness=0.08, contrast=0.08),

            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __call__(self, img: Image.Image):
        return self.transform(img), self.transform(img)


# ── 4. Dataset ────────────────────────────────────────────────────────────────
class FastSSLDataset(Dataset):
    def __init__(self, study_ids: List[str], study_file_map: Dict[str, List[Path]], transform=None, step: int = 1):
        self.transform = transform
        self.samples = []

        for s_id in study_ids:
            if s_id in study_file_map:
                files = study_file_map[s_id][::step]
                self.samples.extend(files)

        print(f"Loaded {len(self.samples)} active image crops across {len(study_ids)} studies (Step: {step}).")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path = str(self.samples[idx])
        img_bgr = cv2.imread(img_path)

        if img_bgr is None:
            img = Image.new("RGB", (cfg.IMAGE_SIZE, cfg.IMAGE_SIZE))
        else:
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img_rgb)

        if self.transform:
            return self.transform(img)
        return img, img


# ── 5. DINO Architecture & Loss ────────────────────────────────────────────────
class DINOHead(nn.Module):
    def __init__(self, in_dim=384, out_dim=4096, hidden_dim=2048, bottleneck_dim=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, bottleneck_dim),
        )
        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))

    def forward(self, x):
        x = self.mlp(x)
        x = F.normalize(x, dim=-1, p=2)
        return self.last_layer(x)


class DINOLoss(nn.Module):
    def __init__(self, out_dim, student_temp=0.1, center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim))

    def forward(self, student_output: torch.Tensor, teacher_output: torch.Tensor, teacher_temp: float):
        # Force FP32 inside loss to guarantee numerical stability across small temperatures
        student_out = student_output.float() / self.student_temp
        teacher_out = teacher_output.float()

        teacher_out = F.softmax((teacher_out - self.center) / teacher_temp, dim=-1).detach()
        loss = torch.sum(-teacher_out * F.log_softmax(student_out, dim=-1), dim=-1).mean()

        self.update_center(teacher_output)
        return loss

    @torch.no_grad()
    def update_center(self, teacher_output: torch.Tensor):
        batch_center = torch.mean(teacher_output.float(), dim=0, keepdim=True)
        self.center = self.center * self.center_momentum + batch_center * (1.0 - self.center_momentum)


class RadioDINOv2SSLWrapper(nn.Module):
    def __init__(self, checkpoint_path: str, in_dim=384, out_dim=4096):
        super().__init__()
        self.backbone = timm.create_model(
            "vit_small_patch14_224",
            pretrained=True,
            num_classes=0,
            dynamic_img_size=True,
            checkpoint_path=checkpoint_path
        )
        self.head = DINOHead(in_dim=in_dim, out_dim=out_dim)

    def forward(self, x):
        feats = self.backbone(x)
        return self.head(feats)


# ── 6. Native BFloat16 Loops for RTX 3090 ──────────────────────────────────────
def train_ssl_epoch_bf16(student, teacher, loader, optimizer, dino_loss, teacher_temp):
    student.train()
    teacher.eval()
    running_loss = 0.0
    pbar = tqdm(loader, desc="SSL Train BF16", leave=False)

    for view1, view2 in pbar:
        view1 = view1.to(cfg.DEVICE, non_blocking=True)
        view2 = view2.to(cfg.DEVICE, non_blocking=True)

        views = torch.cat([view1, view2], dim=0)
        optimizer.zero_grad(set_to_none=True)

        # Native BFloat16 Autocast (No GradScaler required)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            s_out = student(views)
            with torch.no_grad():
                t_out = teacher(views)

            s_out1, s_out2 = s_out.chunk(2, dim=0)
            t_out1, t_out2 = t_out.chunk(2, dim=0)

            loss1 = dino_loss(s_out1, t_out2, teacher_temp)
            loss2 = dino_loss(s_out2, t_out1, teacher_temp)
            loss = (loss1 + loss2) / 2.0

        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), cfg.CLIP_GRAD)
        optimizer.step()

        # Update Teacher via EMA
        with torch.no_grad():
            m = cfg.EMA_MOMENTUM
            for param_s, param_t in zip(student.parameters(), teacher.parameters()):
                param_t.data.mul_(m).add_((1.0 - m) * param_s.detach().data)

        running_loss += loss.item()
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    return running_loss / len(loader)


@torch.no_grad()
def eval_ssl_epoch_bf16(student, teacher, loader, dino_loss, teacher_temp):
    student.eval()
    teacher.eval()
    running_loss = 0.0

    for view1, view2 in tqdm(loader, desc="SSL Val BF16", leave=False):
        view1 = view1.to(cfg.DEVICE, non_blocking=True)
        view2 = view2.to(cfg.DEVICE, non_blocking=True)

        views = torch.cat([view1, view2], dim=0)

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            s_out = student(views)
            t_out = teacher(views)

            s_out1, s_out2 = s_out.chunk(2, dim=0)
            t_out1, t_out2 = t_out.chunk(2, dim=0)

            loss1 = dino_loss(s_out1, t_out2, teacher_temp)
            loss2 = dino_loss(s_out2, t_out1, teacher_temp)
            loss = (loss1 + loss2) / 2.0

        running_loss += loss.item()

    return running_loss / len(loader)


# ── 7. Pipeline Execution ─────────────────────────────────────────────────────
def main():
    df_train = pd.read_csv(cfg.TRAIN_CSV)
    df_val_gold = pd.read_csv(cfg.VAL_CSV)
    df_train = df_train[~df_train["StudyInstanceUID"].isin(df_val_gold["StudyInstanceUID"])].reset_index(drop=True)

    print("Indexing disk directory structure into RAM...")
    study_file_map = defaultdict(list)
    for p in cfg.IMAGE_DIR.glob("*.jpg"):
        study_id = p.name.split('_')[0]
        study_file_map[study_id].append(p)

    unique_studies = df_train['StudyInstanceUID'].unique()
    kf = KFold(n_splits=cfg.NUM_FOLDS, shuffle=True, random_state=cfg.SEED)

    print(f"\n🚀 Launching BFloat16 Self-Supervised Fine-Tuning Pipeline (RTX 3090 Optimized)...")

    for fold, (train_idx, val_idx) in enumerate(kf.split(unique_studies)):
        print(f"\n========================== FOLD {fold + 1}/{cfg.NUM_FOLDS} ==========================")

        train_studies = unique_studies[train_idx]
        val_studies = unique_studies[val_idx]

        ds_train = FastSSLDataset(train_studies, study_file_map, transform=MedicalDINOTransforms(cfg.IMAGE_SIZE),
                                  step=cfg.SUBSAMPLE_STEP)
        ds_val = FastSSLDataset(val_studies, study_file_map, transform=MedicalDINOTransforms(cfg.IMAGE_SIZE),
                                step=cfg.SUBSAMPLE_STEP)

        ld_train = DataLoader(
            ds_train,
            batch_size=cfg.BATCH_SIZE,
            shuffle=True,
            num_workers=cfg.NUM_WORKERS,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=cfg.PREFETCH_FACTOR
        )
        ld_val = DataLoader(
            ds_val,
            batch_size=cfg.BATCH_SIZE,
            shuffle=False,
            num_workers=cfg.NUM_WORKERS,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=cfg.PREFETCH_FACTOR
        )

        student = RadioDINOv2SSLWrapper(cfg.CHECKPOINT_PATH, in_dim=cfg.EMBED_DIM, out_dim=cfg.OUT_DIM).to(cfg.DEVICE)
        teacher = RadioDINOv2SSLWrapper(cfg.CHECKPOINT_PATH, in_dim=cfg.EMBED_DIM, out_dim=cfg.OUT_DIM).to(cfg.DEVICE)

        teacher.load_state_dict(student.state_dict())
        for p in teacher.parameters():
            p.requires_grad = False

        dino_loss = DINOLoss(out_dim=cfg.OUT_DIM, student_temp=cfg.STUDENT_TEMP).to(cfg.DEVICE)

        # 1. Warmup Setup: Freeze Backbone for Epoch 0 to train ONLY the random head
        for param in student.backbone.parameters():
            param.requires_grad = False

        optimizer = torch.optim.AdamW(student.head.parameters(), lr=cfg.HEAD_LR, weight_decay=cfg.WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.EPOCHS, eta_min=1e-6)

        best_val_loss = float('inf')
        save_path = cfg.PYTHON_DIR / "radiodinov2_ssl_finetuned_backbone.pth"

        for epoch in range(cfg.EPOCHS):
            # 2. Unfreeze Backbone starting at Epoch 1 with Differential Learning Rates
            if epoch == 1:
                print(" Unfreezing backbone with differential learning rates...")
                for param in student.backbone.parameters():
                    param.requires_grad = True

                optimizer = torch.optim.AdamW([
                    {'params': student.backbone.parameters(), 'lr': cfg.BACKBONE_LR},  # 1e-6 for backbone
                    {'params': student.head.parameters(), 'lr': cfg.HEAD_LR}            # 1e-4 for head
                ], weight_decay=cfg.WEIGHT_DECAY)

                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.EPOCHS - 1, eta_min=1e-7)

            teacher_temp = (
                cfg.WARMUP_TEACHER_TEMP + (cfg.TEACHER_TEMP - cfg.WARMUP_TEACHER_TEMP) * (epoch / cfg.EPOCHS)
            )

            train_loss = train_ssl_epoch_bf16(student, teacher, ld_train, optimizer, dino_loss, teacher_temp)
            val_loss = eval_ssl_epoch_bf16(student, teacher, ld_val, dino_loss, teacher_temp)
            scheduler.step()

            print(
                f"Epoch {epoch + 1:02d}/{cfg.EPOCHS:02d} | Train SSL Loss: {train_loss:.4f} | Val SSL Loss: {val_loss:.4f}"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'model_state_dict': teacher.backbone.state_dict(),
                    'val_loss': val_loss
                }, save_path)
                print(f"  -> Saved updated backbone checkpoint to: {save_path.name}")

        del student, teacher, optimizer, scheduler, ld_train, ld_val
        gc.collect()
        torch.cuda.empty_cache()

        # Stop after Fold 0 to produce single fine-tuned backbone
        break


if __name__ == '__main__':
    main()