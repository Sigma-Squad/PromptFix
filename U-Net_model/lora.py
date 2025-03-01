import torch
import torch.nn as nn

class LoRAConv(nn.Module):
    def __init__(self, in_channels, out_channels, rank=4):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.lora = nn.Linear(in_channels, out_channels)
        self.rank = rank

    def forward(self, x):
        lora_update = self.lora(x.mean(dim=(2, 3))) / self.rank
        lora_update = lora_update.view(x.shape[0], -1, 1, 1)  # Ensure correct shape
        return self.conv(x) + lora_update

