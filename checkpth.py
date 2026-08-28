import torch

# 1. Update with the exact path to your model file

# 2. Load the full checkpoint file (set weights_only=False if it contains custom objects/dictionaries)
for i in range(12):
    file_path = f"dinov2_vitb14_pretrain.pth"
    checkpoint = torch.load(file_path, map_location="cpu", weights_only=False)
    print(checkpoint['val_auc'])
pass