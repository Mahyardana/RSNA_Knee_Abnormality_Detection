import torch
import timm

# 1. Load the model as a feature extractor
# This strips the final classifier layer and returns intermediate feature maps
model = timm.create_model('coatnet_0_rw_224.sw_in1k', pretrained=True, features_only=True)
model.eval()

# 2. Check the channels of each stage
print(f"Feature channels for each stage: {model.feature_info.channels()}")
# CoAtNet-0 typically outputs: [64, 96, 192, 384, 768] corresponding to its stages

# 3. Extract multi-stage feature maps
dummy_input = torch.randn(1, 3, 224, 224)
with torch.no_grad():
    features = model(dummy_input)

for i, stage_feature in enumerate(features):
    print(f"Stage {i} Feature Map Shape: {stage_feature.shape}")
# Setting num_classes=0 retains the global pooling layer but removes the classifier
model_vector = timm.create_model('coatnet_0_rw_224.sw_in1k', pretrained=True, num_classes=0)
model_vector.eval()

with torch.no_grad():
    embedding = model_vector(dummy_input)

print(f"Global Vector Shape: {embedding.shape}")  # Outputs: [1, 768]
