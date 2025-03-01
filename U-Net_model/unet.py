import torch
import torch.nn as nn
import torch.nn.functional as F
from attention import CrossAttentionBlock
from lora import LoRAConv
from hgs import high_freq_loss

class Unet(nn.Module):
    def __init__(self,input_channels=4, base_channels=64, no_layers=4, text_embed_dim=768):
        super().__init__()
        
        # for Encoder layers
        self.encoder_layers = nn.ModuleList([
                              nn.Conv2d(input_channels if i==0 else base_channels*(2**i), base_channels*(2**(i+1)),kernel_size =3, padding=1, stride=2)
                              for i in range(no_layers)
        ])
        
        # for bottleneck layer
        self.bottleneck_layer = nn.Conv2d(base_channels * (2**no_layers), base_channels * (2**no_layers), 
                                    kernel_size=3, padding=1)
        
        # for decoder layers
        self.decoder_layers =  nn.ModuleList([
                               nn.ConvTranspose2d(base_channels * (2**(i+1)), base_channels * (2**i), kernel_size=4, 
                                            stride=2, padding=1)
                               for i in reversed(range(no_layers))
        ])
        
        # for cross attention layers
        self.cross_attentions = nn.ModuleList([
            CrossAttentionBlock(text_embed_dim, base_channels * (2**(i+1)))
            for i in reversed(range(no_layers))
        ])
        
        # for lora layers
        self.lora_layers = nn.ModuleList([
            LoRAConv(base_channels * (2**(i+1)), base_channels * (2**(i+1)))
            for i in reversed(range(no_layers))
        ])
        
        self.relu = nn.ReLU()
    
    def forward(self,x,instruction_embedding,prompt_embedding):
        encoded_features = []
        for layer in self.encoder_layers:
            x = self.relu(layer(x))
            encoded_features.append(x)
        
        x = self.relu(self.bottleneck_layer(x))
        
        for decoder_layer, attention_layer, lora_layer, encoded_feature in zip(self.decoder_layers, self.cross_attentions, self.lora_layers, reversed(encoded_features)):
            # print(x.shape)
            # x = decoder_layer(x)
            # print(x.shape)
            # x_instEmbed = attention_layer(x, instruction_embedding)
            # x_proEmbed = attention_layer(x,prompt_embedding)
            
            # x = (x_instEmbed + x_proEmbed) / 2
            
            # x = lora_layer(x) + x
            # x = self.relu(x + encoded_feature)
            print(x.shape)

            x_instEmbed = attention_layer(x, instruction_embedding)
            x_proEmbed = attention_layer(x, prompt_embedding)
            x = (x_instEmbed + x_proEmbed) / 2

            x = lora_layer(x) + x

            x = decoder_layer(x)
            print(x.shape)

            x = self.relu(x + encoded_feature)
                
        return x
    
     
            