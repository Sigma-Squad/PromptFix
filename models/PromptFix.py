import torch
import torch.nn as nn

class PromptFix(nn.Module):
    def __init__(self, encoder, unet, decoder, device='cpu'):
        super(PromptFix, self).__init__()
        self.encoder = encoder
        self.unet = unet
        self.decoder = decoder
        self.device = device
        
    def forward(self, input_imgs, instruction_embeddings, prompt_embeddings):
        # encoder
        z = self.encoder(input_imgs)
        
        # unet
        timesteps = torch.tensor([1]).to(device=self.device)
        context = torch.cat([instruction_embeddings, prompt_embeddings], dim=0)
        z = self.unet(z, timesteps, context)
        
        # decoder
        
        pass