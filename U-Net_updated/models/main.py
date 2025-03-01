import torch
import torchprofile
from models.unet.model import UNetModel

config = {
    "image_size": 32,
    "in_channels": 8,
    "model_channels": 320,
    "out_channels": 4,
    "num_res_blocks": 2,
    "attention_resolutions": [4, 2, 1],
    "dropout": 0,
    "channel_mult": [1, 2, 4, 4],
    "conv_resample": True,
    "dims": 2,
    "num_classes": None,
    "use_checkpoint": True,
    "use_fp16": False,
    "default_eps": False,
    "force_type_convert": True,
    "num_heads": 8,
    "num_head_channels": -1,
    "num_heads_upsample": -1,
    "use_scale_shift_norm": False,
    "resblock_updown": False,
    "use_new_attention_order": False,
    "use_spatial_transformer": True,
    "transformer_depth": 1,
    "context_dim": 768,
    "n_embed": None,
    "legacy": False,
    "disable_dual_context": False,
}

unet_model = UNetModel(**config)
input_tensor = torch.randn(1, 8, 64, 64)
torchprofile.profile_macs(unet_model, (input_tensor, torch.tensor([1]), torch.randn(2, 768)))