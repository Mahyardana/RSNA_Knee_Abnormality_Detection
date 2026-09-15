import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import timm
from tqdm import tqdm
import multiprocessing as mp
try:
    mp.set_start_method('fork', force=True)
except RuntimeError:
    pass
# ── Configuration ─────────────────────────────────────────────────────────────
DATASET_DIR = "./dinov2_dataset"  # Folder containing extracted JPEGs
CHECKPOINT_PATH = "radiodinos16.safetensors"
SAVE_PATH = "radiodinov2_s16_finetuned.pth"

BATCH_SIZE = 256
EPOCHS = 10
LR = 1e-5
WEIGHT_DECAY = 0.01
IMAGE_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Data Augmentations ────────────────────────────────────────────────────────
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

dataset = datasets.ImageFolder(root=DATASET_DIR, transform=train_transforms)
dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=32,
    pin_memory=True,
    drop_last=True
)

# ── Load RadioDINOv2 Backbone via timm ─────────────────────────────────────────
print(f"Loading RadioDINOv2 backbone from {CHECKPOINT_PATH}...")
model = timm.create_model(
    "vit_small_patch16_224",
    pretrained=True,
    num_classes=0,  # Feature extraction / embedding output
    dynamic_img_size=True,
    checkpoint_path=CHECKPOINT_PATH
).to(DEVICE)

# ── Optimization Setup ────────────────────────────────────────────────────────
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-7)
scaler = torch.amp.GradScaler('cuda')

# ── Fine-Tuning Loop ──────────────────────────────────────────────────────────
model.train()
print("Starting RadioDINOv2 Fine-Tuning...")

for epoch in range(EPOCHS):
    total_loss = 0.0
    pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{EPOCHS}")

    for images, _ in pbar:
        images = images.to(DEVICE)
        optimizer.zero_grad()

        with torch.amp.autocast('cuda', dtype=torch.float16):
            # Forward pass: extract features
            features = model(images)

            # Reconstruction / Feature Loss or custom objective
            # Here we compute variance regularization across feature dimensions to prevent collapse
            std_loss = torch.mean(torch.relu(1.0 - torch.std(features, dim=0)))
            loss = std_loss

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    scheduler.step()
    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch + 1} Complete | Average Loss: {avg_loss:.4f}")

# ── Save Updated Weights ──────────────────────────────────────────────────────
torch.save({"model_state_dict": model.state_dict()}, SAVE_PATH)
print(f"✅ Fine-tuned RadioDINOv2 checkpoint saved to {SAVE_PATH}")