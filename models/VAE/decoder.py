import torch
import torch.nn as nn
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

from models.VAE.encoder import make_attn, ResNetBlock

class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x

def Normalize(in_channels, default_eps, num_groups=32):
    if default_eps:
        return torch.nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, affine=True)
    else:
        return torch.nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)

def nonlinearity(x):
    # swish
    return x*torch.sigmoid(x)

class Decoder(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1,2,4,8), num_res_blocks,
                 attn_resolutions, default_eps=False, force_type_convert=False, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, z_channels, give_pre_end=False, tanh_out=False, use_linear_attn=False,
                 attn_type="vanilla", **ignorekwargs):
        super().__init__()
        if use_linear_attn: attn_type = "linear"
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end
        self.tanh_out = tanh_out

        # compute in_ch_mult, block_in and curr_res at lowest res
        in_ch_mult = (1,)+tuple(ch_mult)
        block_in = ch*ch_mult[self.num_resolutions-1]
        curr_res = resolution // 2**(self.num_resolutions-1)
        self.z_shape = (1,z_channels,curr_res,curr_res)
        print("Working with z of shape {} = {} dimensions.".format(
            self.z_shape, np.prod(self.z_shape)))

        # z to block_in
        self.conv_in = torch.nn.Conv2d(z_channels,
                                       block_in,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResNetBlock(in_channels=block_in,
                                       default_eps=default_eps,
                                       force_type_convert=force_type_convert,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = make_attn(block_in, default_eps, force_type_convert, attn_type=attn_type)
        self.mid.block_2 = ResNetBlock(in_channels=block_in,
                                       default_eps=default_eps,
                                       force_type_convert=force_type_convert,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks+1):
                block.append(ResNetBlock(in_channels=block_in,
                                         default_eps=default_eps,
                                         force_type_convert=force_type_convert,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, default_eps, force_type_convert, attn_type=attn_type))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up) # prepend to get consistent order

        # end
        self.force_type_convert = force_type_convert
        self.norm_out = Normalize(block_in, default_eps)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        out_ch,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, z, hs=None):
        logger.info(f"Input z shape: {z.shape}")
        self.last_z_shape = z.shape

        temb = None
        h = self.conv_in(z)
        logger.info(f"After conv_in: {h.shape}")

        logger.info("--- Mid Block ---")
        h = self.mid.block_1(h, temb)
        logger.info(f"After Mid Block 1: {h.shape}")

        h = self.mid.attn_1(h)
        logger.info(f"After Mid Attention: {h.shape}")

        h = self.mid.block_2(h, temb)
        logger.info(f"After Mid Block 2: {h.shape}")

        for i_level in reversed(range(self.num_resolutions)):
            logger.info(f"--- Level {i_level} ---")
            for i_block in range(self.num_res_blocks+1):
                if hs:
                    h = self.up[i_level].block[i_block](h, temb, hs.pop())  
                else:
                    h = self.up[i_level].block[i_block](h, temb)
                logger.info(f"  ResBlock {i_block}: {h.shape}")

                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
                    logger.info(f"  Attn Block {i_block}: {h.shape}")

            if i_level != 0:
                h = self.up[i_level].upsample(h)
                logger.info(f"  After Upsample: {h.shape}")

        if self.give_pre_end:
            return h

        h = self.norm_out(h)
        logger.info(f"Before Activation: {h.shape}")

        h = nonlinearity(h)
        logger.info(f"Before conv_out: {h.shape}")

        h = self.conv_out(h)
        logger.info(f"Final Output: {h.shape}")

        if self.tanh_out:
            h = torch.tanh(h)
            logger.info(f"After Tanh: {h.shape}")

        return h
