# MedDINOv3 Integration Guide

This guide explains the integration of **MedDINOv3** into the RSNA Knee Abnormality Detection model.

## Overview

The original model used **ResNet50** as the backbone for patch feature extraction. This has been replaced with **MedDINOv3** (ViT-B), a medical vision foundation model trained on 3M CT images. MedDINOv3 weights are **frozen** during training, acting as a feature extractor only.

### Key Changes

| Aspect | Before (ResNet50) | After (MedDINOv3) |
|--------|------------------|-------------------|
| Backbone | ResNet50 (CNN) | ViT-B (Transformer) |
| Feature Dimension | 2048-d | 768-d |
| Patch Processing | Direct feature extraction | Patch embedding + transformer blocks |
| Medical Data | ImageNet pretraining | CT-3M (3.8M CT slices) |
| Training | Backbone trainable | **Backbone frozen** |
| Model Size | ~94 MB | ~276 MB (frozen) |

## Architecture Details

### Modified Forward Pass

```
Input: patches (B, S, P, 3, 224, 224)
  ↓
[1] MedDINOv3 Backbone (FROZEN)
    - 768-d feature extraction per patch
    - All weights frozen, no gradient updates
  ↓
[2] Patch Projection
    - 768 → 512 (SLICE_DIM)
    - Trainable
  ↓
[3] Patch Transformer Encoder
    - Self-attention over patches within each slice
    - Trainable
  ↓
[4] Slice Encoder
    - Self-attention over slices within each series
    - Trainable
  ↓
[5] Metadata Fusion
    - Fuse with plane, fluid-sensitive, fat-suppression
    - Trainable
  ↓
[6] Classification Head (MLP)
    - 512 → 256 → 12 logits
    - Trainable
```

### Parameter Breakdown

- **Total Parameters**: 99.9M
- **Frozen (MedDINOv3)**: 86.6M (86.7%)
- **Trainable**: 13.4M (13.3%)

## Model Files

### Main Implementation
- **`model_with_meddinov3.py`** - Complete model implementation
  - `MedDINOv3Backbone` - Wrapper around ViT-B with frozen weights
  - `KneeAbnormalityModelWithMedDINOv3` - Full pipeline

### Example Usage

```python
import torch
from model_with_meddinov3 import KneeAbnormalityModelWithMedDINOv3

class Config:
    SLICE_DIM    = 512
    META_DIM     = 256
    NUM_LABELS   = 12
    NUM_PLANES   = 3
    PATCH_ENC_HEADS  = 8
    PATCH_ENC_LAYERS = 2
    SLICE_ENC_HEADS  = 8
    SLICE_ENC_LAYERS = 2
    DROPOUT      = 0.3

cfg = Config()

# Initialize model
model = KneeAbnormalityModelWithMedDINOv3(cfg)

# Forward pass
patches = torch.randn(2, 4, 9, 3, 224, 224)  # (batch, slices, patches, C, H, W)
slice_mask = torch.zeros(2, 4, dtype=torch.bool)
patch_valid = torch.ones(2, 4, 9, dtype=torch.bool)
plane_id = torch.tensor([0, 1], dtype=torch.long)
fluid_sensitive = torch.tensor([0, 1], dtype=torch.long)
fat_suppression = torch.tensor([1, 0], dtype=torch.long)

model.eval()
with torch.no_grad():
    logits = model(patches, slice_mask, patch_valid, 
                   plane_id, fluid_sensitive, fat_suppression)

print(logits.shape)  # torch.Size([2, 12])
```

## Training Notes

### Backbone Frozen
The `KneeAbnormalityModelWithMedDINOv3` automatically freezes MedDINOv3 weights:
- Even in `.train()` mode, the backbone remains in evaluation mode
- No gradients are computed for backbone parameters
- This saves memory and computation time during training

### Advantages
1. **Leverages Medical Foundation Model**: MedDINOv3 is trained on 3.8M CT images
2. **Faster Training**: Frozen backbone = fewer parameters to update
3. **Better Initialization**: Medical imaging features from day 1
4. **Stable Features**: Backbone doesn't drift during training

### Considerations
1. Feature dimension reduced (2048 → 768) - patch encoder output stays at 512
2. May need to adjust learning rates for remaining trainable layers
3. MedDINOv3 is CT-specific - may need fine-tuning for your specific MRI data

## Loading Pretrained Models

### Loading from Checkpoint
If you have a saved checkpoint from the new model:

```python
model = KneeAbnormalityModelWithMedDINOv3(cfg)
checkpoint = torch.load('model_checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'], strict=True)
model.eval()
```

### MedDINOv3 Checkpoint
The MedDINOv3 checkpoint is automatically loaded from:
```
~/.cache/huggingface/hub/models--ricklisz123--MedDINOv3-ViTB-16-CT-3M/snapshots/*/model.pth
```

If not found, the model will use random initialization (with a warning).

## Comparison with Original Model

### ResNet50 Baseline
```python
from torchvision.models import resnet50
backbone = resnet50(pretrained=False)
backbone.fc = nn.Identity()
# Output: 2048-d features
```

### MedDINOv3 New
```python
from model_with_meddinov3 import MedDINOv3Backbone
backbone = MedDINOv3Backbone(
    checkpoint_path='path/to/model.pth',
    freeze_weights=True
)
# Output: 768-d features (frozen)
```

## Migration Guide

If you have an existing trained model with ResNet50, you'll need to:

1. **Initialize fresh**: Create new model with `KneeAbnormalityModelWithMedDINOv3`
2. **Transfer only what's compatible**: 
   - The patch_encoder, slice_encoder, and classifier have the same architecture
   - You may need to retrain due to different backbone output dimensions

3. **Fine-tuning strategy** (recommended):
   - Start training from scratch with a lower learning rate
   - Or: Load old classifier weights, train with frozen backbone for 5-10 epochs

## Performance Impact

Expected changes:
- **Training time**: Slightly faster (fewer parameters to update)
- **Memory usage**: Similar or slightly lower
- **Inference speed**: Similar (frozen backbone doesn't compute gradients)
- **Accuracy**: Potentially better (medical pretraining on CT-3M)

## Citation

If using MedDINOv3, please cite:
```bibtex
@article{li2025meddinov3,
  title={MedDINOv3: How to Adapt Vision Foundation Models for Medical Image Segmentation?},
  author={Li, Yuheng and Wu, Yizhou and Lai, Yuxiang and Hu, Mingzhe and Yang, Xiaofeng},
  journal={arXiv preprint arXiv:2509.02379},
  year={2025},
  url={https://arxiv.org/abs/2509.02379}
}
```

## Troubleshooting

### MedDINOv3 checkpoint not found
```
⚠ Warning: MedDINOv3 checkpoint not found at ...
  Using random initialization instead
```
**Solution**: Download using `huggingface-hub`:
```bash
huggingface-cli download ricklisz123/MedDINOv3-ViTB-16-CT-3M
```

### Shape mismatches during loading
If you're transferring weights from a ResNet50 model:
- `patch_proj` layer needs reinitialization (768 → 512 vs 2048 → 512)
- Other layers are compatible in terms of architecture

### Out of memory
MedDINOv3 is frozen, so this shouldn't be an issue. If it is:
1. Reduce batch size
2. Use gradient checkpointing (add `GRAD_CHECKPOINT=True` to config)
3. Reduce `SLICE_DIM` or `META_DIM`

## Contact

For issues or questions about the MedDINOv3 integration, refer to:
- Original MedDINOv3 repo: https://github.com/ricklisz/MedDINOv3
- Paper: https://arxiv.org/abs/2509.02379
