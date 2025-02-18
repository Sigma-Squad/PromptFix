import torch
import torch.nn as nn

class CrossAttentionBlock(nn.Module):
    def __init__(self, text_dim, image_dim):
        super().__init__()
        
        self.image_dim = image_dim
        self.q_proj = nn.Linear(image_dim, text_dim)
        self.k_proj = nn.Linear(text_dim, text_dim)
        self.v_proj = nn.Linear(text_dim, image_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, image_features, text_embedding):
        B, C, H, W = image_features.shape  
        print(image_features.shape,self.image_dim)
        image_features = image_features.view(B, C, -1).permute(0, 2, 1)  
        # print(image_features.shape,self.image_dim)
        q = self.q_proj(image_features)  
        k = self.k_proj(text_embedding).unsqueeze(1)  
        v = self.v_proj(text_embedding).unsqueeze(1)  

        attn_weights = self.softmax(torch.matmul(q, k.transpose(-2, -1)))  
        print(attn_weights.shape)
        print(v.shape)
        print(torch.matmul(attn_weights, v).shape)
        attn_output = torch.matmul(attn_weights, v).permute(0, 2, 1).view(B, C, H, W)  

        return image_features.view(B, C, H, W) + attn_output
