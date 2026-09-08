import os
import torch
import zstandard as zstd
pt_file="train_feats.pt"
data = {}
data["study_id"] = []  # Tensor shape: (N, 264)
data["slices"] = []  # Tensor shape: (N, 264)
data["slicemasks"] = []  # Tensor shape: (N, 264)
data["plane_ids"] = []  # Tensor shape: (N, 264)
data["fluid_sensitive"] = []  # Tensor shape: (N, 264)
data["fat_suppression"] = []  # Tensor shape: (N, 264)
data["serie_type"] = []
data["labels"] = []
data["plane_crop_mixed"] = []
for i in range(1, 100):
    if not os.path.exists(pt_file + f".part{i}"):
        break
    loadedpart = torch.load(pt_file + f".part{i}")
    data["study_id"] = data["study_id"] + loadedpart["study_id"]
    # data["slices"]=data["slices"]+loadedpart["slices"]
    for studyid,slice in zip(loadedpart["study_id"],loadedpart["slices"]):
        with zstd.open(f"slicesfeats/{studyid[0]}.pt", "wb") as f:
            torch.save(slice,f)
    data["slicemasks"] = data["slicemasks"] + (
        loadedpart["slicemasks"] if "slicemasks" in loadedpart else loadedpart["masks"])
    data["plane_ids"] = data["plane_ids"] + loadedpart["plane_ids"]
    data["fluid_sensitive"] = data["fluid_sensitive"] + loadedpart["fluid_sensitive"]
    data["fat_suppression"] = data["fat_suppression"] + loadedpart["fat_suppression"]
    data["serie_type"] = data["serie_type"] + loadedpart["serie_type"]
    data["labels"] = data["labels"] + loadedpart["labels"]
    data["plane_crop_mixed"] = data["plane_crop_mixed"] + loadedpart["plane_crop_mixed"]
torch.save(data,pt_file)