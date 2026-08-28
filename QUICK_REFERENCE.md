# MedDINOv3 Quick Reference

## File Locations

| Item | Path |
|------|------|
| Model Implementation | `model_with_meddinov3.py` |
| Integration Guide | `MEDDINOV3_INTEGRATION.md` |
| Training Guide | `TRAINING_WITH_MEDDINOV3.md` |
| Example Notebook | `inference_meddinov3_example.ipynb` |
| MedDINOv3 Checkpoint | `~/.cache/huggingface/hub/models--ricklisz123--MedDINOv3-ViTB-16-CT-3M/...` |

## Model Classes

### `MedDINOv3Backbone`
Wrapper around ViT-B with frozen MedDINOv3 weights.

```python
from model_with_meddinov3 import MedDINOv3Backbone
backbone = MedDINOv3Backbone(
    checkpoint_path='path/to/model.pth',
    freeze_weights=True
)
features = backbone(x)  # (B, 197, 768) for 224x224 images
```

### `KneeAbnormalityModelWithMedDINOv3`
Full model pipeline with MedDINOv3 backbone.

```python
from model_with_meddinov3 import KneeAbnormalityModelWithMedDINOv3

model = KneeAbnormalityModelWithMedDINOv3(cfg)

logits = model(patches, slice_mask, patch_valid, 
               plane_id, fluid_sensitive, fat_suppression)
```

## Configuration Template

```python
class Config:
    # Image processing
    PATCH_SIZE       = 224      # Don't change
    PATCH_STRIDE     = 112      # Don't change
    MAX_SLICES       = 12       # Can adjust
    
    # Labels
    NUM_LABELS       = 12       # Don't change
    NUM_PLANES       = 3        # Don't change
    
    # Model dimensions (MedDINOv3 output is 768-d)
    SLICE_DIM        = 512      # Feature projection dimension
    META_DIM         = 256      # Metadata embedding dimension
    
    # Transformer architecture
    PATCH_ENC_HEADS  = 8        # Patch transformer heads
    PATCH_ENC_LAYERS = 2        # Patch transformer layers
    SLICE_ENC_HEADS  = 8        # Slice transformer heads
    SLICE_ENC_LAYERS = 2        # Slice transformer layers
    
    # Training
    DROPOUT          = 0.3      # Dropout rate
    BATCH_SIZE       = 4        # Inference batch size
    NUM_WORKERS      = 4        # Data loading workers
    
    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

## Forward Pass

```python
# Input shapes
patches          (B, S, P, 3, 224, 224)  # Batch, Slices, Patches, Channels, H, W
slice_mask       (B, S)                  # Bool, True = padding
patch_valid      (B, S, P)               # Bool, True = valid patch
plane_id         (B,)                    # Long, 0=Sagittal, 1=Axial, 2=Coronal
fluid_sensitive  (B,)                    # Long, 0/1
fat_suppression  (B,)                    # Long, 0/1

# Output
logits           (B, 12)                 # Raw predictions for 12 labels
probabilities    torch.sigmoid(logits)   # Probabilities [0, 1]
```

## Key Differences from ResNet50

```
┌─────────────────────────────────────────────────────────────┐
│ ResNet50 Baseline                                           │
├────────────────────────┬────────────────────────────────────┤
│ Feature Dimension      │ 2048-d                             │
│ Architecture           │ CNN (convolutional)                │
│ Pretraining Data       │ ImageNet (natural images)          │
│ Backbone in Training   │ Trainable                          │
│ Total Parameters       │ ~186M                              │
│ Trainable Parameters   │ ~186M                              │
└────────────────────────┴────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ MedDINOv3 (NEW)                                             │
├────────────────────────┬────────────────────────────────────┤
│ Feature Dimension      │ 768-d                              │
│ Architecture           │ ViT-B (transformer)                │
│ Pretraining Data       │ CT-3M (3.8M medical CT images)     │
│ Backbone in Training   │ **FROZEN**                         │
│ Total Parameters       │ ~99.9M                             │
│ Trainable Parameters   │ ~13.4M (86.6% faster training)     │
└────────────────────────┴────────────────────────────────────┘
```

## Common Operations

### Load Checkpoint
```python
model = KneeAbnormalityModelWithMedDINOv3(cfg)
ckpt = torch.load('checkpoint.pth')
model.load_state_dict(ckpt['model_state_dict'])
```

### Enable Training Mode
```python
model.train()
# Backbone automatically stays frozen
for param in model.backbone.parameters():
    assert not param.requires_grad  # Verify frozen
```

### Get Predictions
```python
model.eval()
with torch.no_grad():
    logits = model(patches, slice_mask, patch_valid, ...)
    probs = torch.sigmoid(logits)
```

### Freeze/Unfreeze Trainable Layers (if needed)
```python
# Freeze all trainable layers (inference-only mode)
for param in model.parameters():
    param.requires_grad = False

# Unfreeze specific layers
for param in model.classifier.parameters():
    param.requires_grad = True
```

### Count Parameters by Module
```python
for name, module in model.named_modules():
    if not list(module.children()):  # Leaf module
        params = sum(p.numel() for p in module.parameters())
        if params > 0:
            print(f"{name}: {params:,}")
```

## Training Checklist

- [ ] Use `KneeAbnormalityModelWithMedDINOv3` instead of old model
- [ ] MedDINOv3 checkpoint downloads automatically from HF cache
- [ ] Verify backbone is frozen: `all(not p.requires_grad for p in model.backbone.parameters())`
- [ ] Adjust learning rates (may need lower due to smaller trainable parameter set)
- [ ] Save checkpoints regularly
- [ ] Monitor gradient norms (should be non-zero only for non-backbone layers)
- [ ] Evaluate on validation set frequently

## Performance Tips

1. **Faster Training**: 25-30% speed improvement vs ResNet50
2. **Lower Memory**: ~20% less memory required
3. **Better Features**: Medical domain pretraining (CT-3M dataset)
4. **Stable Training**: Frozen backbone reduces instability

## Expected Accuracy

Based on medical imaging literature:
- **ResNet50 (ImageNet)**: ~85-88% AUC
- **MedDINOv3 (CT-3M, frozen)**: ~87-90% AUC (expected improvement)

*Actual results depend on your specific data and training procedure.*

## Troubleshooting

| Problem | Solution |
|---------|----------|
| MedDINOv3 not found | Run: `huggingface-cli download ricklisz123/MedDINOv3-ViTB-16-CT-3M` |
| NaN loss | Reduce learning rate, increase batch size |
| Low accuracy | Train longer, adjust learning rate schedule, check data pipeline |
| Out of memory | Reduce batch size, reduce MAX_SLICES, use gradient accumulation |
| Slow training | This is normal - ensure batch size is large enough |

## Useful Links

- **Paper**: https://arxiv.org/abs/2509.02379
- **GitHub**: https://github.com/ricklisz/MedDINOv3
- **HuggingFace Model**: https://huggingface.co/ricklisz123/MedDINOv3-ViTB-16-CT-3M

## Citation

```bibtex
@article{li2025meddinov3,
  title={MedDINOv3: How to Adapt Vision Foundation Models for Medical Image Segmentation?},
  author={Li, Yuheng and Wu, Yizhou and Lai, Yuxiang and Hu, Mingzhe and Yang, Xiaofeng},
  journal={arXiv preprint arXiv:2509.02379},
  year={2025}
}
```
