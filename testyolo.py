import cv2
import numpy as np
import torch
import torchvision
from PIL import ImageOps
from typer.colors import BLACK

# Define your intended class map to verify index alignment
CLASS_MAP = {
    0: "whole",
    1: "anterior",
    2: "medial",
    3: "lateral",
    4: "back",
    5: "front",
}

import torch.nn.functional as F
from ultralytics import YOLO


def decode_yolo_output(
        raw_output: torch.Tensor, conf_thresh: float
) -> torch.Tensor:
    """Decodes YOLOv8/v11 output [10, 8400] and preserves overlapping sub-regions across different classes."""
    if raw_output.dim() == 3:
        raw_output = raw_output.squeeze(0)

    preds = raw_output.permute(1, 0)  # Shape: [8400, 10]
    boxes_cxcywh = preds[:, :4]
    class_probs = preds[:, 4:]

    # Apply sigmoid if raw logits are returned
    if class_probs.max() > 1.0 or class_probs.min() < 0.0:
        class_probs = torch.sigmoid(class_probs)

    scores, class_ids = torch.max(class_probs, dim=1)

    # Filter out low-confidence background anchors
    mask = scores >= conf_thresh
    if not mask.any():
        return torch.empty((0, 6))

    boxes_cxcywh = boxes_cxcywh[mask]
    scores = scores[mask]
    class_ids = class_ids[mask]

    # Convert center [cx, cy, w, h] to corner [x1, y1, x2, y2]
    x1 = boxes_cxcywh[:, 0] - (boxes_cxcywh[:, 2] / 2)
    y1 = boxes_cxcywh[:, 1] - (boxes_cxcywh[:, 3] / 2)
    x2 = boxes_cxcywh[:, 0] + (boxes_cxcywh[:, 2] / 2)
    y2 = boxes_cxcywh[:, 1] + (boxes_cxcywh[:, 3] / 2)
    boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=1)

    # 1. BATCHED NMS: Keeps overlaps between different classes (e.g., 'whole' & 'anterior')
    keep_indices = torchvision.ops.batched_nms(
        boxes_xyxy, scores, class_ids, iou_threshold=0.0
    )

    boxes_xyxy = boxes_xyxy[keep_indices]
    scores = scores[keep_indices]
    class_ids = class_ids[keep_indices]

    # 2. TOP-1 PER CLASS: Select highest confidence box per anatomical label
    unique_classes = torch.unique(class_ids)
    final_detections = []

    for c in unique_classes:
        class_mask = class_ids == c
        cls_scores = scores[class_mask]
        cls_boxes = boxes_xyxy[class_mask]

        best_idx = torch.argmax(cls_scores)
        final_detections.append(
            torch.tensor(
                [
                    cls_boxes[best_idx, 0].item(),
                    cls_boxes[best_idx, 1].item(),
                    cls_boxes[best_idx, 2].item(),
                    cls_boxes[best_idx, 3].item(),
                    cls_scores[best_idx].item(),
                    c.item(),
                ],
            )
        )

    return torch.stack(final_detections) if final_detections else torch.empty((0, 6))

def padding(img, expected_size):
    desired_size = expected_size
    delta_width = desired_size[0] - img.size[0]
    delta_height = desired_size[1] - img.size[1]
    pad_width = delta_width // 2
    pad_height = delta_height // 2
    padding = (pad_width, pad_height, delta_width - pad_width, delta_height - pad_height)
    return ImageOps.expand(img, padding)


def resize_with_padding(img, expected_size):
    img.thumbnail((expected_size[0], expected_size[1]))
    # print(img.size)
    delta_width = expected_size[0] - img.size[0]
    delta_height = expected_size[1] - img.size[1]
    pad_width = delta_width // 2
    pad_height = delta_height // 2
    padding = (pad_width, pad_height, delta_width - pad_width, delta_height - pad_height)
    return ImageOps.expand(img, padding)
def test_jpeg_yolo(model, image_path: str, conf_thresh: float = 0.3):
    # 1. Load JPEG image
    orig_img = cv2.imread(image_path)
    h, w, _ = orig_img.shape
    desired_size = 640

    resizedimage=(orig_img-orig_img.min())/(orig_img.max()-orig_img.min())
    input_tensor = (
        torch.from_numpy(resizedimage)
        .permute(2, 0, 1)
        .float()
        .unsqueeze(0)
    )
    orig_hpad=0
    orig_wpad=0
    if input_tensor.shape[-2]!=input_tensor.shape[-1]:
        orig_hpad=input_tensor.shape[-1]- input_tensor.shape[-2] if input_tensor.shape[-1]>input_tensor.shape[-2] else 0
        orig_wpad=input_tensor.shape[-2]- input_tensor.shape[-1] if input_tensor.shape[-2]>input_tensor.shape[-1] else 0
        input_tensor = F.pad(
            input_tensor,
            (0, orig_wpad, 0, orig_hpad),
            mode="constant",
            value=0)
    input_tensor = F.interpolate(
        input_tensor,
        (512, 512),
        mode="bilinear",
        align_corners=False,
    )
    input_tensor = F.pad(
        input_tensor,
        (0, 128, 0, 128),
        mode="constant",
        value=0)

    # 3. Model Inference
    with torch.no_grad():
        raw_output = model(input_tensor)

    # 4. Custom Decoder (Passes through your decode_yolo_output logic)
    detections = decode_yolo_output(raw_output, conf_thresh=conf_thresh)

    # 5. Draw Boxes on Image
    for det in detections:
        x1, y1, x2, y2, conf, cls_id = det.tolist()
        cls_id = int(cls_id)

        # Scale coordinates back to original image shape
        px1, py1 = int(x1 * (w+orig_wpad) / 512), int(y1 * (h+orig_hpad) / 512)
        px2, py2 = int(x2 * (w+orig_wpad) / 512), int(y2 * (h+orig_hpad) / 512)

        label = f"{CLASS_MAP.get(cls_id, 'Unknown')}: {conf:.2f}"

        # Draw box and class text
        cv2.rectangle(orig_img, (px1, py1), (px2, py2), (0, 255, 0), 2)
        cv2.putText(
            orig_img,
            label,
            (px1, max(py1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

    # 6. Save visualization
    cv2.imwrite("yolo_verification_result.jpg", orig_img)
    print("Saved annotated image to yolo_verification_result.jpg")
if __name__ == "__main__":
    # model = YOLO("yolo_axial.pt")
    # model.export(format="torchscript",imgsz=(640, 640))
    model=torch.jit.load("yolo_axial.torchscript")
    test_jpeg_yolo(model,"axial2test.jpg")