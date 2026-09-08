
from peft import LoraConfig, get_peft_model
import timm
import torch
visionmodel = timm.create_model("vit_base_patch16_224",
                                pretrained=False,
                                num_classes=0,
                                dynamic_img_size=True)
# visionmodel.eval()
# visionmodel.requires_grad_(False)
# Recompute activations during backward pass to save VRAM
visionmodel.set_grad_checkpointing(enable=True)
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["qkv", "proj"],
    lora_dropout=0.1,
    bias="none"
)
visionmodel = get_peft_model(visionmodel, lora_config)
visionmodel.load_state_dict(torch.load("finetuned/6-v.pth")["model_state_dict"])
visionmodel=visionmodel.merge_and_unload()
torch.save({"model_state_dict": visionmodel.state_dict()}, "best_model_dino_finetuned.pth")
pass