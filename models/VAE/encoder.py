from typing import Optional, Tuple, List
import torch
import torch.nn as nn
import logging

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

class AttnBlock(nn.Module):
    """
    Simplied version of the Attention block used in the promptfix codebase.
    """

    def __init__(self, in_channels: int, default_eps: bool, force_type_convert: bool) -> None:
        super(AttnBlock, self).__init__()  # type: ignore

        self.norm = nn.GroupNorm(num_groups=32, num_channels=in_channels, affine=True) if default_eps else torch.nn.GroupNorm(
            num_groups=32, num_channels=in_channels, affine=True, eps=1e-6)
        self.force_type_convert = force_type_convert

        self.query = nn.Conv2d(in_channels, in_channels, 1, 1, 0)
        self.key = nn.Conv2d(in_channels, in_channels, 1, 1, 0)
        self.value = nn.Conv2d(in_channels, in_channels, 1, 1, 0)

        self.proj_out = nn.Conv2d(in_channels, in_channels, 1, 1, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        if self.force_type_convert:
            h = self.norm.float()(h.float())
            h = h.half()
        else:
            h = self.norm(h)

        query = self.query(h)
        key = self.key(h)
        value = self.value(h)

        batch, channel, height, width = query.shape
        query = query.reshape(batch, channel, height*width).permute(0, 2, 1)
        key = key.reshape(batch, channel, height*width)

        weights = torch.bmm(query, key)
        weights = weights * (int(channel)**(-0.5))
        weights = torch.nn.functional.softmax(weights, dim=2)

        value = value.reshape(batch, channel, height*width)
        weights = weights.permute(0, 2, 1)

        h = torch.bmm(value, weights)
        h = h.reshape(batch, channel, height, width)
        h = self.proj_out(h)

        return x+h


def make_attn(in_channels: int, default_eps: bool, force_type_convert: bool, attn_type: str = "vanilla") -> nn.Module:
    if attn_type == "vanilla":
        return AttnBlock(in_channels, default_eps, force_type_convert)
    return nn.Module()  # TODO: Add other blocks if required


class DownSample(nn.Module):
    def __init__(self, in_channels: int, with_conv_layer: bool) -> None:
        super().__init__()  # type: ignore
        self.with_conv = with_conv_layer
        if (self.with_conv):
            self.conv = nn.Conv2d(in_channels, in_channels, 3, 2, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.with_conv:
            pad = (0, 1, 0, 1)
            x = nn.functional.pad(x, pad, mode='constant', value=0)
            x = self.conv(x)
        else:
            x = nn.functional.avg_pool2d(x, 2, 2)
        return x


class ResNetBlock(nn.Module):
    """
    Modified version of the ResNetBlock used in the promptfix codebase.
    """

    def __init__(self, *, in_channels: int,
                 default_eps: bool, force_type_convert: bool,
                 out_channels: Optional[int] = None, conv_shortcut: bool = False,
                 dropout: float, temb_channels: int = 512) -> None:

        super(ResNetBlock, self).__init__()  # type: ignore

        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.use_conv_shortcut = conv_shortcut
        self.force_type_convert = force_type_convert

        self.norm1 = nn.GroupNorm(num_groups=32, num_channels=self.in_channels, affine=True) if default_eps else torch.nn.GroupNorm(
            num_groups=32, num_channels=self.in_channels, affine=True, eps=1e-6)
        self.conv1 = nn.Conv2d(in_channels, self.out_channels, 3, 1, 1)
        self.temb_channels = temb_channels
        self.default_eps = default_eps

        if temb_channels > 0:
            self.temb_proj = nn.Linear(temb_channels, in_channels)

        self.norm2 = nn.GroupNorm(num_groups=32, num_channels=self.out_channels, affine=True) if default_eps else torch.nn.GroupNorm(
            num_groups=32, num_channels=self.out_channels, affine=True, eps=1e-6)
        self.droupout_percent = dropout
        self.droupout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = nn.Conv2d(
                    self.in_channels, self.out_channels, 3, 1, 1)
            else:
                self.nin_shortcut = nn.Conv2d(
                    self.in_channels, self.out_channels, 1, 1, 0)

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:  # Skip_x was removed
        h = x
        if self.force_type_convert:
            h = self.norm1.float()(h.float())
            h = h.half()
        else:
            h = self.norm1(h)

        h = h*torch.sigmoid(h)
        h = self.conv1(h)

        if temb is not None:
            h = h + self.temb_proj(temb*torch.sigmoid(temb))[:, :, None, None]

        if self.force_type_convert:
            h = self.norm2.float()(h.float())
            h = h.half()
        else:
            h = self.norm2(h)

        h = h*torch.sigmoid(h)
        h = self.droupout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x+h


class Encoder(nn.Module):
    """
    Encoder to be used in the VQ-VAE model. 
    """

    def __init__(self, *, ch: int, out_ch: int,
                 ch_mult,
                #  ch_mult: Tuple[int, ...] | List[int] = (1, 2, 4, 8),
                 num_res_blocks: int, attn_resolutions: list[int],
                 default_eps: bool = False, force_type_convert: bool = False,
                 dropout: float = 0.0, resample_with_conv: bool = True,
                 in_channels: int, resolution: int,
                 z_channels: int, double_z: bool = True,
                 use_linear_attn: bool = False,
                 attn_type: str = "vanilla", **kwargs: object) -> None:

        super().__init__()  # type: ignore

        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        self.conv_in = nn.Conv2d(in_channels, self.ch, 3, 1, 1)

        self.input_channel_mult = (1,)+tuple(ch_mult)

        self.down = nn.ModuleList()

        curr_res = resolution

        block_in: int = 0
        block_out: int = 0

        for level in range(self.num_resolutions):

            block = nn.ModuleList()
            attn = nn.ModuleList()

            block_in = self.ch*self.input_channel_mult[level]
            block_out = self.ch*self.input_channel_mult[level+1]

            for _ in range(self.num_res_blocks):
                block.append(ResNetBlock(in_channels=block_in, default_eps=default_eps, force_type_convert=force_type_convert,
                             out_channels=block_out, dropout=dropout, temb_channels=self.temb_ch))
                block_in = block_out

                if resolution in attn_resolutions:
                    attn.append(make_attn(block_in, default_eps,
                                force_type_convert, attn_type=attn_type))

            down = nn.Module()
            down.block = block
            down.attn = attn
            if level != self.num_resolutions - 1:
                down.downsample = DownSample(block_in, resample_with_conv)
                curr_res = curr_res//2
            self.down.append(down)

        self.mid = nn.Module()
        self.mid.block_1 = ResNetBlock(in_channels=block_in, out_channels=block_out, default_eps=default_eps,
                                       force_type_convert=force_type_convert, dropout=dropout, temb_channels=self.temb_ch)
        self.mid.attn_1 = make_attn(
            block_in, default_eps, force_type_convert, attn_type=attn_type)
        self.mid.block_2 = ResNetBlock(in_channels=block_out, out_channels=block_out, default_eps=default_eps,
                                       force_type_convert=force_type_convert, dropout=dropout, temb_channels=self.temb_ch)

        self.force_type_convert = force_type_convert
        self.norm_out = nn.GroupNorm(num_groups=32, num_channels=block_out, affine=True) if default_eps else torch.nn.GroupNorm(
            num_groups=32, num_channels=block_out, affine=True, eps=1e-6)

        self.conv_out = nn.Conv2d(
            block_in, 2*z_channels if double_z else z_channels, 3, 1, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        logger.info(f"Input x shape: {x.shape}")

        temb = None
        hs: List[torch.Tensor] = [self.conv_in(x)]
        logger.info(f"After conv_in: {hs[-1].shape}")

        h: torch.Tensor = hs[-1]
        for level in range(self.num_resolutions):
            logger.info(f"--- Level {level} ---")
            for itr_block in range(self.num_res_blocks):
                h = self.down[level].block[itr_block](h, temb)  # type: ignore
                logger.info(f"  ResBlock {itr_block}: {h.shape}")

                if len(self.down[level].attn) > 0:  # type: ignore
                    h = self.down[level].attn[itr_block](h)  # type: ignore
                    logger.info(f"  Attn Block {itr_block}: {h.shape}")

                hs.append(h)

            if level != self.num_resolutions - 1:
                hs.append(self.down[level].downsample(hs[-1]))  # type: ignore
                logger.info(f"  After Downsample: {hs[-1].shape}")

        logger.info("--- Mid Block ---")
        h = self.mid.block_1(h, temb)
        logger.info(f"After Mid Block 1: {h.shape}")

        h = self.mid.attn_1(h)
        logger.info(f"After Mid Attention: {h.shape}")

        h = self.mid.block_2(h, temb)
        logger.info(f"After Mid Block 2: {h.shape}")

        h = self.norm_out(h)
        logger.info(f"Before Activation: {h.shape}")

        h = h * torch.sigmoid(h)
        logger.info(f"Before conv_out: {h.shape}")

        h = self.conv_out(h)
        logger.info(f"Final Output: {h.shape}")

        return h, hs