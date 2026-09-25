import cv2
import matplotlib.pyplot as plt
import numpy as np
import timm
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from PIL import Image
from torchvision import transforms


def process_dino_features(feature_map, bg_threshold=0.2):
    """feature_map: Tensor of shape (14, 14, 384) or (196, 384)

    returns: Tensor of shape (224, 224, 3) ready for plt.imshow()
    """
    # 1. Reshape to flattened spatial tokens: (196, 384)
    if feature_map.dim() == 3:
        H, W, C = feature_map.shape  # 14, 14, 384
        features = feature_map.reshape(-1, C)
    else:
        features = feature_map

    features_np = features.detach().cpu().numpy()

    # 2. Compute PCA down to 3 components
    pca = PCA(n_components=3)
    pca_features = pca.fit_transform(features_np)  # Shape: (196, 3)

    # 3. Robust Min-Max Normalization (1st-99th percentile prevents outlier clipping)
    p1, p99 = np.percentile(pca_features, (1, 99), axis=0)
    pca_normalized = (pca_features - p1) / (p99 - p1 + 1e-8)
    pca_normalized = np.clip(pca_normalized, 0.0, 1.0)

    # 4. Background Masking using PC1
    # Component 0 typically captures foreground vs background contrast
    mask = pca_normalized[:, 0] > bg_threshold
    pca_normalized[~mask] = 0.0

    # 5. Reshape back to spatial grid: (14, 14, 3)
    rgb_grid = torch.tensor(
        pca_normalized, dtype=torch.float32
    ).reshape(4, 4, 3)

    # 6. Format for PyTorch Interpolation: (1, 3, 14, 14)
    rgb_tensor = rgb_grid.permute(2, 0, 1).unsqueeze(0)

    # 7. Upsample from 14x14 grid to original 224x224 image resolution
    rgb_upsampled = F.interpolate(
        rgb_tensor, size=(512, 512), mode="bicubic", align_corners=False
    )

    # 8. Format for image display: (224, 224, 3)
    rgb_out = (
        torch.clamp(rgb_upsampled.squeeze(0), 0.0, 1.0).permute(1, 2, 0).numpy()
    )

    return rgb_out


def crop_and_resize_box(
        input_tensor: torch.Tensor,
        box: list = None,
        pixelspacing: list = [0.5, 0.5],
        sizemm: float = 80.0,
        boundary_expansion: float = 0.0
):
    if input_tensor.dim() == 3:
        input_tensor = input_tensor.unsqueeze(0)

    _, _, h_img, w_img = input_tensor.shape

    # 1. Parse bounding box or default to full image
    if box is not None and len(box) >= 4:
        minx, minh, maxx, maxh = box[:4]
        xlen, hlen = maxx - minx, maxh - minh
        minx = max(0, int(minx - xlen * boundary_expansion))
        minh = max(0, int(minh - hlen * boundary_expansion))
        maxx = min(w_img, int(maxx + xlen * boundary_expansion))
        maxh = min(h_img, int(maxh + hlen * boundary_expansion))
        cropped = input_tensor[:, :, minh:maxh, minx:maxx]
        if cropped.shape[-2] == 0 or cropped.shape[-1] == 0:
            cropped = input_tensor
    else:
        cropped = input_tensor

    c_h, c_w = cropped.shape[-2], cropped.shape[-1]
    device = input_tensor.device
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    # Helper function: Centered replicate padding (prevents high-contrast black edges)
    def pad_to_square(img):
        h, w = img.shape[-2], img.shape[-1]
        if h == w:
            return img
        diff = abs(h - w)
        if h < w:
            pad_top = diff // 2
            pad_bottom = diff - pad_top
            return F.pad(img, (0, 0, pad_top, pad_bottom), mode='replicate')
        else:
            pad_left = diff // 2
            pad_right = diff - pad_left
            return F.pad(img, (pad_left, pad_right, 0, 0), mode='replicate')

    listofcrops = []

    # 2. Whole Image / Full Bounding Box Crop
    squared_whole = pad_to_square(cropped)
    resized_whole = F.interpolate(squared_whole, (512, 512), mode='bilinear', align_corners=False)
    listofcrops.append((resized_whole - mean) / std)

    # # 3. Calculate 140mm Patch Dimensions in Pixels
    # crop_h = int(sizemm / pixelspacing[0])
    # crop_w = int(sizemm / pixelspacing[1])
    #
    # # If image dimension is smaller than patch size, return whole image crop
    # if c_h < crop_h or c_w < crop_w:
    #     return listofcrops
    #
    # stride_h = crop_h // 2
    # stride_w = crop_w // 2
    #
    # # 4. Generate Coordinates with Boundary Clamping (Guarantees exact 140mm patches)
    # h_starts = list(range(0, c_h - crop_h + 1, stride_h))
    # if h_starts[-1] + crop_h < c_h:
    #     h_starts.append(c_h - crop_h)
    #
    # w_starts = list(range(0, c_w - crop_w + 1, stride_w))
    # if w_starts[-1] + crop_w < c_w:
    #     w_starts.append(c_w - crop_w)
    #
    # # 5. Extract Sliding 140mm Patches
    # for h in h_starts:
    #     for w in w_starts:
    #         patch = cropped[:, :, h: h + crop_h, w: w + crop_w]
    #         resized_patch = F.interpolate(patch, (512, 512), mode='bilinear', align_corners=False)
    #         listofcrops.append((resized_patch - mean) / std)

    return listofcrops
imgtransform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
# --- Usage Example ---
# Dummy input matching your model output shape: (14, 14, 384)
img_bgr=cv2.imread("coronaltest.jpg")
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
img = Image.fromarray(img_rgb)
imgtensor = imgtransform(img)
crops=crop_and_resize_box(imgtensor,sizemm=100)
visionmodel = timm.create_model("vit_small_patch16_224", pretrained=True, num_classes=0,
                                dynamic_img_size=True, checkpoint_path="radiodinos16.safetensors")
# visionmodel.load_state_dict(torch.load(f"radiodinov2_ssl_finetuned_backbone.pth")["model_state_dict"])
visionmodel.eval()
visionmodel.requires_grad_(False)
visionmodel.to("cuda")
for img in crops:
    model_output = visionmodel.forward_features(img.to("cuda"))
    model_output=model_output.detach().cpu()
    clstoken = model_output[:, 0, :]
    tokens = []
    dinotrans = model_output[:, 1:, :].transpose(1,2).view(384,32,32)
    for x in range(0, 32, 8):
        for y in range(0, 32, 8):
            ct = dinotrans[:,x:x + 8, y:y + 8]
            ctavg = ct.mean(dim=[-2, -1])
            tokens.append(ctavg.unsqueeze(0))
    # tokens.append(clstoken.unsqueeze(1))

    # Get RGB map
    rgb_image = process_dino_features(torch.cat(tokens,dim=0), bg_threshold=0.25)
    normalized=img.cpu()[0].permute(1, 2, 0)
    normalized=(normalized-normalized.min())/(normalized.max()-normalized.min()+1e-5)
    # Plot
    plt.subplot(1, 2, 1)
    plt.imshow(normalized.numpy())
    plt.title("original")
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(rgb_image)
    plt.title("segmented")
    plt.axis("off")
    plt.show()