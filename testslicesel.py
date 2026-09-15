from pathlib import Path
from typing import Optional, Dict, Union, List

import numpy as np
import pandas as pd
import pydicom
import timm
import torch
import torch.nn.functional as F
import torchvision
from torch.utils.data import Dataset, DataLoader


class Config:
    KAGGLE_DIR = Path('/run/media/mahyar/Samsung-990-PRO/Projects/Kaggle/rsna-knee-abnormality-detection')
    PYTHON_DIR = Path('/run/media/mahyar/Samsung-990-PRO/Projects/Python/RSNA Knee Abnormality Detection')
    TRAIN_CSV = PYTHON_DIR / 'llm_generated_labels.csv'
    VAL_CSV = PYTHON_DIR / 'val_gold.csv'
    SERIES_CSV = PYTHON_DIR / 'train_series.csv'
    TRAIN_IMG_DIR = KAGGLE_DIR / 'train_series'
    TRAINING_LABEL = 0
    SAVE_PATH = str(PYTHON_DIR / f"best_model_dino_auc.pth")
    SAVE_PATH2 = str(PYTHON_DIR / f"best_model_dino_loss.pth")
    LOAD_PATH = PYTHON_DIR / f"best_model_dino_label.pth"

    # ── Image processing ──────────────────────────────────────────────
    MAX_SLICES_PER_SERIES = -1
    MAX_SERIES_PER_STUDY = 32
    LABELS = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus',
              'Medial OA', 'Lateral OA', 'PF OA', 'Effusion',
              'Synovitis', "Baker's", 'Contusion', 'Fracture']
    NUM_LABELS = 12

    PLANE_TO_IDX = {'Sagittal': 0, 'Axial': 1, 'Coronal': 2}
    NUM_PLANES = 3

    # ── Model dimensions ──────────────────────────────────────────────
    SLICE_DIM = 518
    META_DIM = 256
    SLICE_ENC_HEADS = 8
    SLICE_ENC_LAYERS = 2
    DROPOUT = 0.3

    # ── Train / val split ─────────────────────────────────────────────
    VAL_FRACTION = 0.10  # 20% val, 80% train

    # ── Training hyperparameters ──────────────────────────────────────
    BATCH_SIZE = 1
    NUM_WORKERS = 4
    GRAD_ACCUM_STEPS = 1
    WARMUP_EPOCHS = 5
    EPOCHS = 100
    LR = 1e-3
    MIN_LR = 1e-5
    WEIGHT_DECAY = 0.01
    USE_AMP = False

    DEBUG_SUBSET = None  # Set to e.g. 100 for fast debugging
    SEED = 32
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    LABEL_CLAMP_THRESHOLD_UPPER = 0.9
    LABEL_CLAMP_THRESHOLD_LOWER = 0.14
    LABEL_SMOOTHENING_THRESHOLD = 0.1


cfg = Config()
class TorchScriptROIExtractor:

    def __init__(
            self,
            axial_model_path: str,
            sagittal_model_path: str,
            coronal_model_path: str,
            class_maps: Optional[Dict[str, Dict[int, str]]] = None,
            device: str = "cuda",
    ):
        self.device = device

        # Load plane-specific TorchScript YOLO detectors
        self.models = {
            "axial": torch.jit.load(
                axial_model_path, map_location=device
            ).eval(),
            "sagittal": torch.jit.load(
                sagittal_model_path, map_location=device
            ).eval(),
            "coronal": torch.jit.load(
                coronal_model_path, map_location=device
            ).eval(),
        }

        # Class ID to anatomical category label mappings
        self.class_maps = class_maps or {
            "axial": {
                0: "whole",
                1: "anterior",
                2: "medial",
                3: "lateral",
                4: "back",
                5: "front"
            },
            "sagittal": {
                0: "whole",
                1: "anterior",
                2: "medial",
                3: "lateral",
                4: "back",
                5: "front"
            },
            "coronal": {
                0: "whole",
                1: "anterior",
                2: "medial",
                3: "lateral",
                4: "back",
                5: "front"
            },
        }

    @staticmethod
    def get_center_out_slice_indices(num_slices: int):
        """Generates slice indices starting from the middle and alternating outward."""
        mid = num_slices // 2
        yield mid
        for offset in range(1, (num_slices // 2) + 1):
            if mid + offset < num_slices:
                yield mid + offset
            if mid - offset >= 0:
                yield mid - offset

    def crop_subregion(
            self, slice_tensor: torch.Tensor, bbox: torch.Tensor
    ) -> torch.Tensor:
        """Crops 4D tensor [1, C, H, W] using bounding box [x1, y1, x2, y2]."""
        if bbox is None or len(bbox) == 0:
            return slice_tensor

        x1, y1, x2, y2 = map(int, bbox[:4])
        _, _, h, w = slice_tensor.shape

        # Clamp boundaries to slice spatial dimensions
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        cropped = slice_tensor[:, :, y1:y2, x1:x2]
        return cropped if cropped.numel() > 0 else slice_tensor

    def decode_yolo_output(
            self, raw_output: torch.Tensor, conf_thresh: float
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
            return torch.empty((0, 6), device=self.device)

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
                    device=self.device,
                )
            )

        return torch.stack(final_detections) if final_detections else torch.empty((0, 6), device=self.device)

    @torch.no_grad()
    def extract_best_roi_crops(
            self,
            volume_tensor: torch.Tensor,  # Shape: [Slices, C, H, W]
            plane: str,
            conf_thresh: float = 0.60,
    ) -> Dict[Union[int, str], torch.Tensor]:
        """Scans 3D volume slices center-outward to find highest-confidence ROI bounding boxes."""
        if plane not in self.models:
            raise ValueError(
                f"Invalid plane '{plane}'. Must be one of"
                f" {list(self.models.keys())}"
            )

        yolo_model = self.models[plane]
        class_map = self.class_maps.get(plane, {})
        num_slices = volume_tensor.shape[0]
        orig_hpad = 0
        orig_wpad = 0
        orig_w=volume_tensor.shape[-1]
        orig_h=volume_tensor.shape[-2]
        if volume_tensor.shape[-2] != volume_tensor.shape[-1]:
            orig_hpad = volume_tensor.shape[-1] - volume_tensor.shape[-2] if volume_tensor.shape[-1] > volume_tensor.shape[
                -2] else 0
            orig_wpad = volume_tensor.shape[-2] - volume_tensor.shape[-1] if volume_tensor.shape[-2] > volume_tensor.shape[
                -1] else 0
            volume_tensor = F.pad(
                volume_tensor,
                (0, orig_wpad, 0, orig_hpad),
                mode="constant",
                value=0)
        volume_tensor = F.interpolate(
            volume_tensor,
            (512, 512),
            mode="bilinear",
            align_corners=False,
        )
        volume_tensor = F.pad(
            volume_tensor,
            (0, 128, 0, 128),
            mode="constant",
            value=0)

        best_slice_idx = num_slices // 2
        best_crops = {}
        highest_conf = -1.0
        yoloout=yolo_model(volume_tensor.to(self.device))
        yoloboxes=[self.decode_yolo_output(x,conf_thresh) for x in yoloout]
        # 1. Search center-outward for a slice satisfying confidence threshold
        for idx in self.get_center_out_slice_indices(num_slices):
            slice_tensor = volume_tensor[idx: idx + 1].to(
                self.device
            )  # [1, C, H, W]
            if cfg.USE_AMP:
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    predictions = yolo_model(slice_tensor)[0]  # Shape: [num_boxes, 6]
            else:
                predictions = yolo_model(slice_tensor)[0]

            decodedyolo = self.decode_yolo_output(predictions, conf_thresh)
            if len(decodedyolo) == 0:
                continue
            # Compute highest box confidence in current slice
            max_box_conf = decodedyolo[:, 4].max().item()

            # Track fallback slice with highest overall detection confidence
            if max_box_conf > highest_conf:
                highest_conf = max_box_conf
                best_slice_idx = idx

            # Terminate search early once threshold criteria is met
            if max_box_conf >= conf_thresh:
                best_slice_idx = idx
                break

        # 2. Extract sub-region crops from the optimal candidate slice
        best_slice_tensor = volume_tensor[
            best_slice_idx: best_slice_idx + 1
        ].to(self.device)
        predictions = yolo_model(best_slice_tensor)[0]

        effective_thresh = (
            conf_thresh if highest_conf >= conf_thresh else highest_conf
        )
        decodedyolo = self.decode_yolo_output(predictions, conf_thresh)
        scale_w = (orig_w + orig_wpad) / 640.0
        scale_h = (orig_h + orig_hpad) / 640.0
        scale_tensor = torch.tensor([scale_w, scale_h, scale_w, scale_h], device=decodedyolo.device)
        for pred in decodedyolo:
            if pred[4].item() >= effective_thresh:
                class_id = int(pred[5].item())
                category_key = class_map.get(class_id, class_id)

                best_crops[category_key] = (pred[:4]*scale_tensor)
        del best_slice_tensor, predictions
        return best_crops, yoloboxes


class KneeStudyTrainDataset(Dataset):
    '''
    Groups all series for a study together for training.
    Returns slices, metadata, and 12-class target labels.
    '''

    def __init__(self, df_studies: pd.DataFrame, df_series: pd.DataFrame, img_dir: Path, cfg, is_train: bool = True):
        self.cfg = cfg
        self.vit = None
        self.roiextractor = None
        self.img_dir = img_dir
        self.is_train = is_train
        self.cache = {}
        maxlen = 0

        # Group series by study_id
        self.studies = []
        for _, study_row in df_studies.iterrows():
            study_id = str(study_row['StudyInstanceUID'])

            # Get series for this study
            study_series = df_series[df_series['StudyInstanceUID'] == study_id]
            if (len(study_series) > maxlen):
                maxlen = len(study_series)
            if len(study_series) == 0:
                continue

            series_list = []
            for _, row in study_series.iterrows():
                series_list.append({
                    'series_id': str(row['SeriesInstanceUID']),
                    'plane_id': int(cfg.PLANE_TO_IDX.get(str(row['Anatomical_Plane']), 0)),
                    'fluid_sensitive': int(row.get('Fluid_Sensitive', 0)),
                    'fat_suppression': int(row.get('Fat_Suppression', 0))
                })

            # Extract labels (nan to 0 if any missing)
            labels = study_row[cfg.LABELS].fillna(0).values.astype(np.float32)

            self.studies.append({
                'study_id': study_id,
                'series': series_list,
                'labels': labels
            })

        print(f"  {'Train' if is_train else 'Val'}: {len(self.studies)} studies")

    def __len__(self):
        return len(self.studies)

    @staticmethod
    def safe_load_dicom_array(dcm_path: str):
        """
        Handles uncompressed (Explicit/Implicit VR) and compressed
        (JPEG Lossless, JPEG 2000) DICOM transfer syntaxes automatically.
        """
        try:
            ds = pydicom.dcmread(dcm_path)
            # Forces pylibjpeg/openjpeg decompression backend
            arr = ds.pixel_array.astype(np.float32)

            # Check for MONOCHROME1 pixel inversion
            if ds.get("PhotometricInterpretation", "MONOCHROME2") == "MONOCHROME1":
                arr = arr.max() - arr

            return arr
        except Exception as e:
            # Fallback or corrupted DICOM handling
            return None


    def __getitem__(self, idx):
        study = self.studies[idx]
        study_id = study['study_id']
        labels = study['labels']

        def identify_sequence_with_binaries(seq_type, fluid_sensitive, fat_suppressed,seq_desc,seq_name):
            """
            tr: float (Repetition Time)
            te: float (Echo Time)
            ti: float (Inversion Time)
            fa: float (Flip Angle)
            seq_type: str (Scanning Sequence tag e.g., 'SE', 'GR', 'IR')
            fluid_sensitive: bool (Your binary flag)
            fat_suppressed: bool (Your binary flag)
            """

            # 1. Isolate STIR instantly using flags
            if fluid_sensitive and fat_suppressed:
                # Double check with TI or IR string to ensure it's not a T2-FLAIR
                if 'IR' in seq_type:
                    return 5

            # 2. Isolate GRE / T2* (Low flip angle, no fat suppression)
            if 'GR' in seq_type:
                return 4

            # 3. Handle Spin Echo (TSE/SE) variants
            if 'SE' in seq_type:
                if fluid_sensitive and not fat_suppressed:
                    return 2

                elif not fluid_sensitive and not fat_suppressed:
                    if 'pd' in seq_desc.lower():
                        return 3

                    return 1


            return 0
        all_slices = []
        all_plane_ids = []
        all_fluids = []
        all_fats = []
        all_stype=[]
        for sinfo in study['series']:
            path = self.img_dir / study_id / sinfo['series_id']
            if not path.exists():
                continue
            dcm_files = []
            alldcms = [pydicom.dcmread(str(dcm), stop_before_pixels=True) for dcm in sorted(path.glob('*.dcm'))]
            # Filter out files missing critical spatial metadata
            valid_slices = [
                ds for ds in alldcms
                if "ImagePositionPatient" in ds and "ImageOrientationPatient" in ds
            ]

            if not valid_slices:
                raise ValueError("No valid spatial metadata found in the provided DICOM slices.")

            # Extract orientation vectors from the first slice
            orient = valid_slices[0].ImageOrientationPatient
            row_cosines = np.array(orient[:3])
            col_cosines = np.array(orient[3:])

            # Compute the normal vector (orthogonal to the slice plane)
            slice_normal = np.cross(row_cosines, col_cosines)

            # Sort based on the projection of ImagePosition onto the slice normal
            sorted_slices = sorted(
                valid_slices,
                key=lambda ds: np.dot(np.array(ds.ImagePositionPatient), slice_normal)
            )
            alldcms = sorted_slices
            if self.cfg.MAX_SLICES_PER_SERIES > 0:
                if len(alldcms) > self.cfg.MAX_SLICES_PER_SERIES:
                    indices = np.round(np.linspace(0, len(alldcms) - 1, num=self.cfg.MAX_SLICES_PER_SERIES)).astype(int)
                    dcm_files = [alldcms[i] for i in indices]
                else:
                    dcm_files = alldcms
                # skipnum = math.ceil(len(alldcms) / self.cfg.MAX_SLICES_PER_SERIES)
                # startidx=0
                # endidx=len(alldcms)
                # slicethickness=float(alldcms[0].SliceThickness)
                # if slicethickness < 2.0:
                #     skipnum = math.ceil(3.0/slicethickness)
                # if len(alldcms)>self.cfg.MAX_SLICES_PER_SERIES*skipnum:
                #     idxoffset=(len(alldcms)-self.cfg.MAX_SLICES_PER_SERIES*skipnum)//2
                #     startidx=idxoffset
                #     endidx=len(alldcms)-idxoffset
                # for i in range(startidx,endidx):
                #     if (i % skipnum) == 0:
                #         dcm_files.append(alldcms[i])
                        # if i==0:
                        #     dcm_files.append(alldcms[i])
                        #     dcm_files.append(alldcms[i+1])
                        #     dcm_files.append(alldcms[i+2])
                        # elif i==len(alldcms)-1:
                        #     dcm_files.append(alldcms[i-2])
                        #     dcm_files.append(alldcms[i-1])
                        #     dcm_files.append(alldcms[i])
                        # else:
                        #     dcm_files.append(alldcms[i-1])
                        #     dcm_files.append(alldcms[i])
                        #     dcm_files.append(alldcms[i+1])
                # middleidx=int(len(alldcms) / 2)
                # onefourth=int(middleidx/2)
                # threefourth=int(middleidx+onefourth)
                # dcm_files.append(alldcms[onefourth])
                # dcm_files.append(alldcms[middleidx])
                # dcm_files.append(alldcms[threefourth])
            else:
                dcm_files = alldcms
            slices = []
            imagetypes=[]
            # tempimgs=[]
            for dcm_file in dcm_files:
                # dcm = pydicom.dcmread(str(dcm_file))
                dcm = dcm_file
                try:
                    img = self.safe_load_dicom_array(dcm.filename)
                    img = (img - img.min()) / (img.max() - img.min()+1e-8)

                    # Very simple augmentations for training
                    # if self.is_train and random.random() > 0.5:
                    #     img = np.fliplr(img).copy()
                    # cv2.resize
                    if img.shape[0] != img.shape[1]:
                        pass
                    # padded_img = cv2.copyMakeBorder(
                    #     img,
                    #     0,
                    #     bottom,
                    #     0,
                    #     right,
                    #     borderType=cv2.BORDER_CONSTANT,
                    #     value=[0, 0, 0]
                    # )
                    # img = self.resize_shorter_side_scipy(img)
                    # img = cv2.resize(img, (518, 518), interpolation=cv2.INTER_CUBIC) # Base size, will be padded natively
                    img = np.clip(img, 0.0, 1.0)
                    img = np.stack([img, img, img], axis=0)  # (3, H, W)
                    slices.append(torch.from_numpy(img))

                except:
                    pass
                # if len(tempimgs)==3:
                #     slices.append(torch.stack(tempimgs, dim=0))
                #     tempimgs=[]
            # if len(tempimgs)>0:
            #     toadd=3-len(tempimgs)
            #     newtempimgs=torch.cat([torch.stack(tempimgs,dim=0),torch.zeros(toadd, 518, 518)],dim=0)
            #     slices.append(newtempimgs)
            dcm = dcm_files[0]
            serietype = identify_sequence_with_binaries(getattr(dcm,"ScanningSequence",""),sinfo['fluid_sensitive'],sinfo['fat_suppression'],getattr(dcm,"SeriesDescription",""),getattr(dcm,"SequenceName",""))
            all_plane_ids.append(sinfo['plane_id'])
            all_fluids.append(sinfo['fluid_sensitive'])
            all_fats.append(sinfo['fat_suppression'])
            all_slices.append(torch.stack(slices))
            all_stype.append(serietype)


        return {
            'study_id': study_id,
            'slices': all_slices,  # (S, 3, H, W)
            'plane_ids': torch.tensor(all_plane_ids, dtype=torch.long),
            'fluid_sensitive': torch.tensor(all_fluids, dtype=torch.long),
            'fat_suppression': torch.tensor(all_fats, dtype=torch.long),
            'serie_type': torch.tensor(all_stype, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.float32),
        }
featurecache={}
def extract_image_features(batch,visionmodel,roiextractor,caching=True):
    batched_slices=batch["slices"]
    batched_plane_ids=batch["plane_ids"]
    batched_fluids=batch["fluid_sensitive"]
    batched_fats=batch["fat_suppression"]
    batched_stype=batch["serie_type"]
    batched_labels=batch["labels"]
    batched_studies=batch["study_id"]


    studies = []
    slices = []
    slicemasks = []
    plane_ids = []
    fluids = []
    fats = []
    stype = []
    plane_crop_mixed = []
    for study,all_slices,all_plane_ids,all_fluids,all_fats,all_stype in zip(batched_studies,batched_slices,batched_plane_ids,batched_fluids,batched_fats,batched_stype):
        study_id = study
        if study_id not in featurecache:

            xxxxxx = [
                roiextractor.extract_best_roi_crops(x, "sagittal" if y == 0 else "axial" if y == 1 else "coronal")
                for x, y in zip(all_slices, all_plane_ids)]
            # all_slices_preprocessed = [
            #     KneeStudyTrainDataset.prepare_mri_for_radiodino(x[:, 0, :, :]).to(self.cfg.DEVICE) for x in all_slices]
            testimage = all_slices[0][all_slices[0].shape[0] // 2].detach().cpu().numpy()[0]
            h = testimage.shape[-2]
            w = testimage.shape[-1]
            if abs(h-w)>300:
                pass
            testimages1 = []
            for i, xxx in enumerate(xxxxxx):
                testimages2 = []
                for k, v in xxx.items():
                    minx = int(v[0])
                    minh = int(v[1])
                    maxx = int(v[2])
                    maxh = int(v[3])
                    testimages2.append(
                        all_slices[i][all_slices[i].shape[0] // 2].detach().cpu().numpy()[
                            0, minh:maxh, minx:maxx])
                testimages1.append(testimages2)

            def crop_and_resize_box(input, box):
                w = input.shape[-1]
                h = input.shape[-2]
                minx = int(box[0] if box[0] > 0 else 0)
                minh = int(box[1] if box[1] > 0 else 0)
                maxx = int(box[2])
                maxh = int(box[3])
                cropped = input[:, :, minh:maxh, minx:maxx]
                if cropped.shape[-1] == 0 or cropped.shape[-2] == 0:
                    return None
                if cropped.shape[-1] < 224 and cropped.shape[-2] < 224 and False:
                    paddingbottom = 16 - cropped.shape[-2] % 16
                    paddingright = 16 - cropped.shape[-1] % 16
                    toreturn = F.pad(cropped, (0, paddingright, 0, paddingbottom), "constant", 0)
                else:
                    paddingbottom = 0
                    paddingright = 0
                    if cropped.shape[-2] < cropped.shape[-1]:
                        paddingbottom = cropped.shape[-1] - cropped.shape[-2]
                    else:
                        paddingright = cropped.shape[-2] - cropped.shape[-1]
                    if paddingbottom != 0 or paddingright != 0:
                        cropped = F.pad(cropped, (0, paddingright, 0, paddingbottom), "constant", 0)
                    toreturn = F.interpolate(cropped, (224, 224), mode='bilinear', align_corners=False)
                mean = torch.tensor(
                    [0.485, 0.456, 0.406], device=toreturn.device
                ).view(1, 3, 1, 1)
                std = torch.tensor(
                    [0.229, 0.224, 0.225], device=toreturn.device
                ).view(1, 3, 1, 1)

                return (toreturn - mean) / std

            slices_feats = []
            new_plane_crop_mixed = []
            new_plane_ids = []
            new_fluids = []
            new_fats = []
            new_stype = []
            rev_class_map = {
                "whole": 0,
                "anterior": 1,
                "medial": 2,
                "lateral": 3,
                "back": 4,
                "front": 5
            }
            for idx, (x, y) in enumerate(zip(xxxxxx, all_slices)):
                xdict = {}
                for k, v in x.items():
                    tovit = crop_and_resize_box(y, v)
                    if tovit is not None:
                        slices_feats.append(tovit.detach().cpu())
                        new_plane_crop_mixed.append([all_plane_ids[idx], rev_class_map[k]])
                        new_plane_ids.append(all_plane_ids[idx])
                        new_fluids.append(all_fluids[idx])
                        new_fats.append(all_fats[idx])
                        new_stype.append(all_stype[idx])
                        del tovit
                if "whole" not in x:
                    tovit = crop_and_resize_box(y, torch.tensor([0, 1, 0, 1], dtype=torch.float))
                    if tovit is not None:
                        slices_feats.append(tovit.detach().cpu())
                        new_plane_crop_mixed.append([all_plane_ids[idx], rev_class_map['whole']])
                        new_plane_ids.append(all_plane_ids[idx])
                        new_fluids.append(all_fluids[idx])
                        new_fats.append(all_fats[idx])
                        new_stype.append(all_stype[idx])
                        del tovit
            all_plane_ids = new_plane_ids
            all_fluids = new_fluids
            all_fats = new_fats
            all_stype = new_stype
            seriesizes=[]
            for x in range(len(slices_feats)):
                seriesizes.append(slices_feats[x].shape[0])
            CHUNK_SIZE = 128  # Process max 32 images at a time through Dino
            tovit = torch.cat(slices_feats,dim=0).to(cfg.DEVICE)
            dino_tokens = []
            with torch.no_grad():
                for i in range(0, tovit.size(0), CHUNK_SIZE):
                    chunk = tovit[i: i + CHUNK_SIZE]
                    if cfg.USE_AMP:
                        with torch.amp.autocast("cuda", dtype=torch.float16):
                            chunk_out = visionmodel(chunk)
                    else:
                        chunk_out = visionmodel(chunk)
                    dino_tokens.append(chunk_out)
            dino_out = torch.cat(dino_tokens, dim=0).cpu()
            slices_feats=list(torch.split(dino_out, seriesizes, dim=0))
            del tovit
            if caching:
                featurecache[study_id] = slices_feats, new_plane_crop_mixed, new_plane_ids, new_fluids, new_fats, new_stype
        else:
            slices_feats, new_plane_crop_mixed, new_plane_ids, new_fluids, new_fats, new_stype = featurecache[study_id]

        if len(all_slices) == 0:
            # Fallback if no images found
            all_slices.append(torch.zeros(3, 224, 224))
            all_plane_ids.append(0)
            all_fluids.append(0)
            all_fats.append(0)
            all_stype.append(0)
        max_slice = max(slice.shape[0] for slice in all_slices)
        # max_height=max(slices.shape[2] for slices in all_slices)
        # max_width=max(slices.shape[3] for slices in all_slices)
        # if max_height%14!=0:
        #     max_height+=14-(max_height%14)
        # if max_width%14!=0:
        #     max_width+=14-(max_width%14)
        # if max_width>1000 or max_height>1000:
        #     pass
        masks = []
        for idx in range(len(slices_feats)):
            # if all_slices[idx].shape[2]<max_height or all_slices[idx].shape[3]<max_width:
            #     pad_height=max_height - all_slices[idx].shape[2]
            #     pad_width=max_width - all_slices[idx].shape[3]
            #     all_slices[idx]=F.pad(all_slices[idx], (0, pad_width, 0, pad_height),value=0, mode="constant")
            if slices_feats[idx].shape[0] < max_slice:
                s = slices_feats[idx].shape[0]
                sizediff = max_slice - s
                slices_feats[idx] = torch.cat([slices_feats[idx], torch.zeros((sizediff, slices_feats[idx].shape[-1]))], dim=0)
                masks.append(torch.cat([torch.ones(s, dtype=torch.bool), torch.zeros(sizediff, dtype=torch.bool)]))
            else:
                masks.append(torch.ones(slices_feats[idx].shape[0], dtype=torch.bool))
        incpufeats = torch.stack(slices_feats).cpu()
        studies.append(study_id)
        slices.append(incpufeats)
        slicemasks.append(torch.stack(masks))
        plane_ids.append(torch.tensor(new_plane_ids, dtype=torch.long),)
        fats.append(torch.tensor(new_fats, dtype=torch.long))
        fluids.append(torch.tensor(new_fluids, dtype=torch.long))
        stype.append(torch.tensor(new_stype, dtype=torch.long))
        plane_crop_mixed.append(torch.tensor(new_plane_crop_mixed, dtype=torch.long))

    maxslicesize = max([slice.shape[1] for slice in slices])
    maxseriesize = max([slice.shape[0] for slice in slices])
    paddedslices = []
    paddedslicemasks = []
    paddedplane_ids = []
    paddedfluids = []
    paddedfats = []
    paddedstype = []
    paddedplane_crop_mixed = []
    seriemasks = []

    for i in range(len(slices)):
        newslice = slices[i]
        newslicemask = slicemasks[i]
        newplane_id = plane_ids[i]
        newfluid = fluids[i]
        newfat = fats[i]
        newplane_crop_mixed = plane_crop_mixed[i]
        newseriemask = torch.ones_like(newplane_id, dtype=torch.bool)
        newstype = stype[i]
        if newslice.shape[1] < maxslicesize:
            slicediffer = maxslicesize - newslice.shape[1]
            newslice = torch.cat([newslice, torch.zeros((newslice.shape[0], slicediffer, newslice.shape[-1]))], dim=1)
            newslicemask = torch.cat(
                (newslicemask, torch.zeros((newslicemask.shape[0], slicediffer), dtype=torch.bool)), dim=1)
        if newslice.shape[0] < maxseriesize:
            s = newslice.shape[0]
            seriediffer = maxseriesize - s
            newslice = torch.cat([newslice, torch.zeros((seriediffer, newslice.shape[1],  newslice.shape[-1]))], dim=0)
            newslicemask = torch.cat(
                (newslicemask, torch.zeros((seriediffer, newslicemask.shape[1]), dtype=torch.bool)), dim=0)
            paddingones = torch.ones([seriediffer])
            newplane_id = torch.cat((newplane_id, paddingones * 10.0), dim=0)
            newfluid = torch.cat((newfluid, paddingones * 10.0), dim=0)
            newfat = torch.cat((newfat, paddingones * 10.0), dim=0)
            newstype = torch.cat((newstype, paddingones * 10.0), dim=0)
            newseriemask = torch.cat((newseriemask, torch.zeros(seriediffer, dtype=torch.bool)), dim=0)
            newplane_crop_mixed = torch.cat((newplane_crop_mixed, torch.ones((seriediffer, 2)) * 10.0), dim=0)
        paddedslices.append(newslice)
        paddedslicemasks.append(newslicemask)
        paddedplane_ids.append(newplane_id)
        paddedfluids.append(newfluid)
        paddedfats.append(newfat)
        seriemasks.append(newseriemask)
        paddedstype.append(newstype)
        paddedplane_crop_mixed.append(newplane_crop_mixed)
    return {
        "study_id": studies,
        "labels": batched_labels,
        "slicemasks": torch.stack(paddedslicemasks, dim=0),
        "plane_ids": torch.stack(paddedplane_ids, dim=0).long(),
        "fluid_sensitive": torch.stack(paddedfluids, dim=0).long(),
        "fat_suppression": torch.stack(paddedfats, dim=0).long(),
        "serie_type": torch.stack(paddedstype, dim=0).long(),
        "seriemasks": torch.stack(seriemasks, dim=0),
        "slices": torch.stack(paddedslices, dim=0),
        "plane_crop_mixed": torch.stack(paddedplane_crop_mixed, dim=0).long(),
    }
df_train = pd.read_csv(cfg.TRAIN_CSV)
df_val = pd.read_csv(cfg.VAL_CSV)
df_tnew = df_train[~df_train["StudyInstanceUID"].isin(df_val["StudyInstanceUID"])]
# label_col = cfg.LABELS[cfg.TRAINING_LABEL]

# Filters explicit positives (>= 0.90) and explicit negatives (<= 0.10)
df_train = df_tnew
# [
#     (df_tnew[label_col] >= 0.90) | (df_tnew[label_col] <= 0.10)
# ].reset_index(drop=True)
df_series = pd.read_csv(cfg.SERIES_CSV)

if cfg.DEBUG_SUBSET:
    df_train = df_train.head(cfg.DEBUG_SUBSET)

roiextractor = TorchScriptROIExtractor(
    "yolo_axial.torchscript",
    "yolo_sagittal.torchscript",
    "yolo_coronal.torchscript",
    {
        "axial": {
            0: "whole",
            1: "anterior",
            2: "medial",
            3: "lateral",
            4: "back",
            5: "front"
        },
        "sagittal": {
            0: "whole",
            1: "anterior",
            2: "medial",
            3: "lateral",
            4: "back",
            5: "front"
        },
        "coronal": {
            0: "whole",
            1: "anterior",
            2: "medial",
            3: "lateral",
            4: "back",
            5: "front"
        }
    }
)
def collate_fn(batch: List[Dict]) -> Dict:
    studies = [x["study_id"] for x in batch]
    slices = [x["slices"] for x in batch]
    plane_ids = [x["plane_ids"] for x in batch]
    fluids = [x["fluid_sensitive"] for x in batch]
    fats = [x["fat_suppression"] for x in batch]
    stype = [x["serie_type"] for x in batch]
    labels = [x["labels"] for x in batch]
    return {
        "study_id": studies,
        "labels": torch.stack(labels, dim=0),
        "plane_ids": plane_ids,
        "fluid_sensitive": fluids,
        "fat_suppression": fats,
        "serie_type": stype,
        "slices": slices,
    }
ds_val = KneeStudyTrainDataset(df_val, df_series, cfg.TRAIN_IMG_DIR, cfg, is_train=False)
ld_val = DataLoader(ds_val, batch_size=cfg.BATCH_SIZE, shuffle=False, pin_memory=False,
                    # num_workers=cfg.NUM_WORKERS,
                    collate_fn=collate_fn)
num_positives = 0
num_negatives = 0
for item in ld_val:
    extract_image_features(item,visionmodel=None,roiextractor=roiextractor)
    pass
