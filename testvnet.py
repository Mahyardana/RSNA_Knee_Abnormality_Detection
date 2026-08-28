import numpy as np
import skm_tea as st
import torch
import torch.nn as nn
import monai.networks.nets as monai_nets
_original_torch_load = torch.load


def patched_torch_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _original_torch_load(*args, **kwargs)


torch.load = patched_torch_load
# # ---------------------------------------------------------
# # 1. MONAI VNet (With Bias Fix Enabled)
# # ---------------------------------------------------------
# model_monai = monai_nets.VNet(
#     spatial_dims=2,
#     in_channels=1,
#     out_channels=4,
#     act=("elu", {"inplace": True}),
# )


# def enable_conv_bias(module):
#     if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
#         if module.bias is None:
#             module.bias = nn.Parameter(
#                 torch.zeros(
#                     module.out_channels, device=module.weight.device
#                 )
#             )
#

# model_monai.apply(enable_conv_bias)

# ckpt = torch.load("skm-tea-unet.ckpt", map_location="cpu",weights_only=False)
# state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
# clean_dict = {
#     k.replace("model.", "").replace("net.", ""): v for k, v in state_dict.items()
# }

# model_monai.load_state_dict(clean_dict, strict=True)
# model_monai.eval().cuda()

# ---------------------------------------------------------
# 2. Native SKM-TEA Core Model
# ---------------------------------------------------------
model_skmtea_wrapper = st.get_model_from_zoo(
    cfg_or_file="config-unet.yaml", weights_path="skm-tea-unet.ckpt"
)

# Unwrap Lightning module -> VNetMONAI -> VNet
if hasattr(model_skmtea_wrapper, "model") and hasattr(
    model_skmtea_wrapper.model, "net"
):
    model_skmtea = model_skmtea_wrapper.model.net
elif hasattr(model_skmtea_wrapper, "model"):
    model_skmtea = model_skmtea_wrapper.model
else:
    model_skmtea = model_skmtea_wrapper.net

model_skmtea = model_skmtea.eval().cuda()

# ---------------------------------------------------------
# 3. Test Alignment & Export TorchScript
# ---------------------------------------------------------
dummy_input = torch.randn(8, 1, 512, 512).cuda()

with torch.no_grad():
    # logits_monai = model_monai(dummy_input)
    logits_skmtea = model_skmtea(dummy_input)

# max_logit_diff = torch.max(torch.abs(logits_monai - logits_skmtea)).item()
# pixel_agreement = (
#     (
#         torch.argmax(logits_monai, dim=1) == torch.argmax(logits_skmtea, dim=1)
#     )
#     .float()
#     .mean()
#     .item()
#     * 100.0
# )

# print(f"Max Absolute Logit Diff: {max_logit_diff:.6e}")
# print(f"Pixel Mask Agreement:    {pixel_agreement:.2f}%")

# if max_logit_diff < 1e-4:
traced_model = torch.jit.trace(model_skmtea, dummy_input[0:1])
traced_model.save("vnet_skmtea_standalone.pt")
print(
    "\n[SUCCESS] Exported 'vnet_skmtea_standalone.pt' for your Python"
    " 3.12 environment."
)