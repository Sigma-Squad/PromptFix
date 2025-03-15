import torch
import torch.nn as nn

class PromptFix(nn.Module):
    def __init__(self, encdec, unet, device='cpu'):
        super(PromptFix, self).__init__()
        self.encdec = encdec
        self.unet = unet
        self.device = device
        
    def forward(self, input_imgs, instruction_embeddings, prompt_embeddings):
        # encoder
        posterior = self.encdec.encode(input_imgs)
        z = posterior.sample()
        
        # unet
        timesteps = torch.tensor([1]).to(device=self.device)
        context = torch.cat([instruction_embeddings, prompt_embeddings], dim=1)
        z = self.unet(z, timesteps, context)
        
        # decoder
        reconstructed_imgs = self.encdec.decode(z)
        
        return reconstructed_imgs