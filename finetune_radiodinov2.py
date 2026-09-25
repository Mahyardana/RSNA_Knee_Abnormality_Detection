import os
import gc
import glob
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import numpy as np
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm
from torch.utils.data import DataLoader
from torchvision import transforms
import timm
import webdataset as wds

# ── 1. Threading Guardrails ───────────────────────────────────────────────────
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

torch.backends.cudnn.benchmark = True


# ── 1. Lower Configuration Memory Overhead ─────────────────────────────────────
class SSLConfig:
    KAGGLE_DIR = Path('/run/media/mahyar/Samsung-990-PRO/Projects/Kaggle/rsna-knee-abnormality-detection')
    PYTHON_DIR = Path('/run/media/mahyar/Samsung-990-PRO/Projects/Python/RSNA Knee Abnormality Detection')

    SHARD_DIR = KAGGLE_DIR / 'dinov2_webdataset'
    CHECKPOINT_PATH = "radiodinos16.safetensors"

    BATCH_SIZE = 256
    NUM_WORKERS = 4         # 👈 Reduced from 12 to 4 to drastically cut worker RAM duplicate allocation
    PREFETCH_FACTOR = 2     # 👈 Reduced from 4 to 2 (keeps shared memory lean)
    EPOCHS = 8

    # Optimization Hyperparameters
    LR = 5e-5
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


# ── 2. Memory-Capped WebDataset Loader ─────────────────────────────────────────
def build_webdataset_loader(shard_paths, batch_size, is_train=True, steps_per_epoch=2000):
    if is_train:
        dataset = (
            wds.WebDataset(shard_paths, resampled=True, shardshuffle=True)
            .shuffle(300)   # 👈 Reduced from 2000 to 300 per worker (4 * 300 = 1200 images in RAM)
            .map(decode_and_augment, handler=wds.ignore_and_continue)
            .with_epoch(steps_per_epoch)
        )
    else:
        dataset = (
            wds.WebDataset(shard_paths, shardshuffle=False)
            .compose(wds.split_by_worker)
            .map(decode_and_augment, handler=wds.ignore_and_continue)
        )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True,
        prefetch_factor=cfg.PREFETCH_FACTOR,
        persistent_workers=True # Keeps workers alive across epochs without re-allocating memory
    )
    return loader

cfg = SSLConfig()


# ── 3. DINO Augmentations & Stream Decoding ───────────────────────────────────
class DINOTransforms:
    def __init__(self, img_size=224):
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.4, 1.0),
                                         interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.3, contrast=0.3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __call__(self, img: Image.Image):
        return self.transform(img), self.transform(img)


dino_aug = DINOTransforms(cfg.IMAGE_SIZE)


def decode_and_augment(sample):
    """Safely decodes image bytes with fallback on corrupt samples."""
    try:
        # Extract binary payload key ignoring metadata keys starting with '__'
        payload_key = [k for k in sample.keys() if not k.startswith("__")][0]
        img_bytes = sample[payload_key]

        np_arr = np.frombuffer(img_bytes, np.uint8)
        img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img_bgr is None:
            img = Image.new("RGB", (cfg.IMAGE_SIZE, cfg.IMAGE_SIZE))
        else:
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img_rgb)

        return dino_aug(img)
    except Exception:
        # Fallback dummy sample to prevent thread crash
        dummy = Image.new("RGB", (cfg.IMAGE_SIZE, cfg.IMAGE_SIZE))
        return dino_aug(dummy)



# ── 4. Network Heads & Loss Modules ───────────────────────────────────────────
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
        # Updated parametrization to remove PyTorch weight_norm warning
        self.last_layer = weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))

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
            "vit_small_patch16_224",
            pretrained=True,
            num_classes=0,
            dynamic_img_size=True,
            checkpoint_path=checkpoint_path
        )
        self.head = DINOHead(in_dim=in_dim, out_dim=out_dim)

    def forward(self, x):
        feats = self.backbone(x)
        return self.head(feats)


# ── 5. FP16 WebDataset Epoch Step ─────────────────────────────────────────────
def train_ssl_epoch(student, teacher, loader, optimizer, dino_loss, teacher_temp, scaler):
    student.train()
    teacher.eval()
    running_loss = 0.0
    steps = 0
    pbar = tqdm(loader, desc="WebDataset Train FP16", leave=False)

    for view1, view2 in pbar:
        view1 = view1.to(cfg.DEVICE, non_blocking=True)
        view2 = view2.to(cfg.DEVICE, non_blocking=True)

        views = torch.cat([view1, view2], dim=0)
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast('cuda', dtype=torch.float16):
            s_out = student(views)
            with torch.no_grad():
                t_out = teacher(views)

            s_out1, s_out2 = s_out.chunk(2, dim=0)
            t_out1, t_out2 = t_out.chunk(2, dim=0)

            loss1 = dino_loss(s_out1, t_out2, teacher_temp)
            loss2 = dino_loss(s_out2, t_out1, teacher_temp)
            loss = (loss1 + loss2) / 2.0

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(student.parameters(), cfg.CLIP_GRAD)
        scaler.step(optimizer)
        scaler.update()

        with torch.no_grad():
            m = cfg.EMA_MOMENTUM
            for param_s, param_t in zip(student.parameters(), teacher.parameters()):
                param_t.data.mul_(m).add_((1.0 - m) * param_s.detach().data)

        running_loss += loss.item()
        steps += 1
        pbar.set_postfix({"loss": f"{loss.item():.4f}", "steps": steps})

    return running_loss / max(steps, 1)


# ── 6. Main Execution Loop ─────────────────────────────────────────────────────
def main():
    all_shards = sorted(glob.glob(str(cfg.SHARD_DIR / "shard_*.tar")))
    if not all_shards:
        raise FileNotFoundError(f"No tar shards found in `{cfg.SHARD_DIR}`. Run `extract_images_to_webdataset.py` first.")

    split_idx = int(len(all_shards) * 0.9)
    train_shards = all_shards[:split_idx]
    val_shards = all_shards[split_idx:]

    print(f"🚀 Found {len(all_shards)} shards on disk.")
    print(f"📦 Streaming WebDataset ({len(train_shards)} train shards, {len(val_shards)} val shards)...")

    student = RadioDINOv2SSLWrapper(cfg.CHECKPOINT_PATH, in_dim=cfg.EMBED_DIM, out_dim=cfg.OUT_DIM).to(cfg.DEVICE)
    teacher = RadioDINOv2SSLWrapper(cfg.CHECKPOINT_PATH, in_dim=cfg.EMBED_DIM, out_dim=cfg.OUT_DIM).to(cfg.DEVICE)

    teacher.load_state_dict(student.state_dict())
    for p in teacher.parameters():
        p.requires_grad = False

    optimizer = torch.optim.AdamW(student.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.EPOCHS, eta_min=1e-6)
    dino_loss = DINOLoss(out_dim=cfg.OUT_DIM, student_temp=cfg.STUDENT_TEMP).to(cfg.DEVICE)
    scaler = torch.amp.GradScaler('cuda')

    save_path = cfg.PYTHON_DIR / "radiodinov2_ssl_finetuned_webdataset.pth"

    for epoch in range(cfg.EPOCHS):
        # 🔹 Re-build DataLoader each epoch to reset WebDataset stream
        ld_train = build_webdataset_loader(train_shards, cfg.BATCH_SIZE, is_train=True)

        teacher_temp = (
            cfg.WARMUP_TEACHER_TEMP + (cfg.TEACHER_TEMP - cfg.WARMUP_TEACHER_TEMP) * (epoch / cfg.EPOCHS)
        )

        train_loss = train_ssl_epoch(student, teacher, ld_train, optimizer, dino_loss, teacher_temp, scaler)
        scheduler.step()

        print(f"Epoch {epoch + 1:02d}/{cfg.EPOCHS:02d} | Train SSL Loss: {train_loss:.4f}")

        torch.save({
            'model_state_dict': teacher.backbone.state_dict(),
            'val_loss': train_loss
        }, save_path)
        print(f"  -> Saved checkpoint to `{save_path.name}`")


if __name__ == '__main__':
    main()