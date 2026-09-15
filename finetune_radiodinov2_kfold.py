import os
import gc
import copy
import math
import warnings
from pathlib import Path
from typing import List, Tuple
from tqdm import tqdm
from PIL import Image
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm
from sklearn.model_selection import KFold

warnings.filterwarnings("ignore")


# ── 1. Configuration ──────────────────────────────────────────────────────────
class SSLConfig:
    PYTHON_DIR = Path('/run/media/mahyar/Samsung-990-PRO/Projects/Python/RSNA Knee Abnormality Detection')
    TRAIN_CSV = PYTHON_DIR / 'llm_labels_v4_blend.csv'
    VAL_CSV = PYTHON_DIR / 'val_gold.csv'
    IMAGE_DIR = Path('/run/media/mahyar/Samsung-990-PRO/Projects/Kaggle/rsna-knee-abnormality-detection/') / 'dinov2_dataset' / 'train' / '0'

    CHECKPOINT_PATH = "radiodinos16.safetensors"

    NUM_FOLDS = 5
    BATCH_SIZE = 256
    NUM_WORKERS = 32
    EPOCHS = 10

    # SSL Hyperparameters
    LR = 5e-5
    WEIGHT_DECAY = 0.04
    IMAGE_SIZE = 224
    EMBED_DIM = 384  # ViT-S/16 embedding dimension
    OUT_DIM = 4096  # DINO projection head output dimension
    EMA_MOMENTUM = 0.996  # Teacher update momentum

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


# ── 2. Data Augmentations for SSL (Two Global Views) ──────────────────────────
class DINOTransforms:
    """
    Generates two distinct augmented views of the same image slice for SSL.
    """

    def __init__(self, img_size=224):
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.4, 1.0),
                                         interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.3, contrast=0.3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __call__(self, img):
        view1 = self.transform(img)
        view2 = self.transform(img)
        return view1, view2


# ── 3. Dataset ─────────────────────────────────────────────────────────────────
class SSLDataset(Dataset):
    def __init__(self, study_ids: List[str], image_dir: Path, transform=None):
        self.transform = transform
        self.samples = []
        study_set = set(study_ids)

        for img_path in image_dir.glob("*_orig_*.jpg"):
            study_id = img_path.name.split('_')[0]
            if study_id in study_set:
                self.samples.append(img_path)

        print(f"Loaded {len(self.samples)} image crops across {len(study_ids)} studies.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path = self.samples[idx]
        img = Image.open(img_path).convert('RGB')

        if self.transform:
            view1, view2 = self.transform(img)
            return view1, view2
        return img, img


# ── 4. DINO Loss & Projection Head ─────────────────────────────────────────────
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
        x = self.last_layer(x)
        return x


class DINOLoss(nn.Module):
    def __init__(self, out_dim, student_temp=0.1, center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim))

    def forward(self, student_output, teacher_output, teacher_temp):
        student_out = student_output / self.student_temp

        # Center and softmax teacher output
        teacher_out = F.softmax((teacher_output - self.center) / teacher_temp, dim=-1).detach()

        # Cross entropy loss between teacher and student
        loss = torch.sum(-teacher_out * F.log_softmax(student_out, dim=-1), dim=-1).mean()

        # Update center
        self.update_center(teacher_output)
        return loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        batch_center = torch.mean(teacher_output, dim=0, keepdim=True)
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)


# ── 5. Full Student-Teacher Model Wrapper ──────────────────────────────────────
class RadioDINOv2SSLWrapper(nn.Module):
    def __init__(self, checkpoint_path: str, in_dim=384, out_dim=4096):
        super().__init__()
        self.backbone = timm.create_model(
            "vit_small_patch16_224",
            pretrained=True,
            num_classes=0,
            dynamic_img_size=True,
            checkpoint_path=checkpoint_path
        )
        self.head = DINOHead(in_dim=in_dim, out_dim=out_dim)

    def forward(self, x):
        feats = self.backbone(x)
        logits = self.head(feats)
        return logits


# ── 6. Training & Validation Loop ─────────────────────────────────────────────
def train_ssl_epoch(student, teacher, loader, optimizer, dino_loss, teacher_temp, scaler):
    student.train()
    teacher.eval()
    running_loss = 0.0
    pbar = tqdm(loader, desc="SSL Train", leave=False)
    for view1, view2 in pbar:
        view1, view2 = view1.to(cfg.DEVICE), view2.to(cfg.DEVICE)
        optimizer.zero_grad()

        with torch.amp.autocast('cuda', dtype=torch.float16):
            # Student forward pass on both views
            s_out1 = student(view1)
            s_out2 = student(view2)

            # Teacher forward pass (no grads)
            with torch.no_grad():
                t_out1 = teacher(view1)
                t_out2 = teacher(view2)

            # Cross-view self-distillation loss
            loss1 = dino_loss(s_out1, t_out2, teacher_temp)
            loss2 = dino_loss(s_out2, t_out1, teacher_temp)
            loss = (loss1 + loss2) / 2.0

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Update Teacher weights via EMA
        with torch.no_grad():
            m = cfg.EMA_MOMENTUM
            for param_s, param_t in zip(student.parameters(), teacher.parameters()):
                param_t.data.mul_(m).add_((1 - m) * param_s.detach().data)

        running_loss += loss.item()

        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    return running_loss / len(loader)


@torch.no_grad()
def eval_ssl_epoch(student, teacher, loader, dino_loss, teacher_temp):
    student.eval()
    teacher.eval()
    running_loss = 0.0

    for view1, view2 in tqdm(loader, desc="SSL Val", leave=False):
        view1, view2 = view1.to(cfg.DEVICE), view2.to(cfg.DEVICE)

        with torch.amp.autocast('cuda', dtype=torch.float16):
            s_out1, s_out2 = student(view1), student(view2)
            t_out1, t_out2 = teacher(view1), teacher(view2)

            loss1 = dino_loss(s_out1, t_out2, teacher_temp)
            loss2 = dino_loss(s_out2, t_out1, teacher_temp)
            loss = (loss1 + loss2) / 2.0

        running_loss += loss.item()

    return running_loss / len(loader)


# ── 7. Main Pipeline ──────────────────────────────────────────────────────────
def main():
    df_train = pd.read_csv(cfg.TRAIN_CSV)
    df_val_gold = pd.read_csv(cfg.VAL_CSV)
    df_train = df_train[~df_train["StudyInstanceUID"].isin(df_val_gold["StudyInstanceUID"])].reset_index(drop=True)

    unique_studies = df_train['StudyInstanceUID'].unique()
    kf = KFold(n_splits=cfg.NUM_FOLDS, shuffle=True, random_state=cfg.SEED)

    print(f"\n🚀 Launching {cfg.NUM_FOLDS}-Fold Self-Supervised (DINO) Fine-Tuning Pipeline...")

    for fold, (train_idx, val_idx) in enumerate(kf.split(unique_studies)):
        print(f"\n========================== FOLD {fold + 1}/{cfg.NUM_FOLDS} ==========================")

        train_studies = unique_studies[train_idx]
        val_studies = unique_studies[val_idx]

        ds_train = SSLDataset(train_studies, cfg.IMAGE_DIR, transform=DINOTransforms(cfg.IMAGE_SIZE))
        ds_val = SSLDataset(val_studies, cfg.IMAGE_DIR, transform=DINOTransforms(cfg.IMAGE_SIZE))

        ld_train = DataLoader(ds_train, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=cfg.NUM_WORKERS,
                              pin_memory=True)
        ld_val = DataLoader(ds_val, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=cfg.NUM_WORKERS,
                            pin_memory=True)

        # Initialize Student and Teacher Networks
        student = RadioDINOv2SSLWrapper(cfg.CHECKPOINT_PATH, in_dim=cfg.EMBED_DIM, out_dim=cfg.OUT_DIM).to(cfg.DEVICE)
        teacher = RadioDINOv2SSLWrapper(cfg.CHECKPOINT_PATH, in_dim=cfg.EMBED_DIM, out_dim=cfg.OUT_DIM).to(cfg.DEVICE)

        # Initialize Teacher parameters equal to Student
        teacher.load_state_dict(student.state_dict())
        for p in teacher.parameters():
            p.requires_grad = False

        optimizer = torch.optim.AdamW(student.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.EPOCHS, eta_min=1e-6)
        dino_loss = DINOLoss(out_dim=cfg.OUT_DIM, student_temp=cfg.STUDENT_TEMP).to(cfg.DEVICE)
        scaler = torch.amp.GradScaler('cuda')

        best_val_loss = float('inf')
        save_path = cfg.PYTHON_DIR / f"radiodinov2_ssl_backbone_fold_{fold}.pth"

        for epoch in range(cfg.EPOCHS):
            # Warmup teacher temperature schedule
            teacher_temp = (
                    cfg.WARMUP_TEACHER_TEMP + (cfg.TEACHER_TEMP - cfg.WARMUP_TEACHER_TEMP) * (epoch / cfg.EPOCHS)
            )

            train_loss = train_ssl_epoch(student, teacher, ld_train, optimizer, dino_loss, teacher_temp, scaler)
            val_loss = eval_ssl_epoch(student, teacher, ld_val, dino_loss, teacher_temp)
            scheduler.step()

            print(
                f"Fold {fold + 1} | Epoch {epoch + 1:02d}/{cfg.EPOCHS:02d} | Train SSL Loss: {train_loss:.4f} | Val SSL Loss: {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                # Save ONLY the fine-tuned ViT backbone (discard DINO projection heads)
                torch.save({
                    'model_state_dict': teacher.backbone.state_dict(),
                    'val_loss': val_loss
                }, save_path)
                print(f"  -> Saved best backbone checkpoint: {save_path.name}")

        del student, teacher, optimizer, scheduler, ld_train, ld_val
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()