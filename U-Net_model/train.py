import torch
import torch.optim as optim
from unet import Unet
from hgs import high_freq_loss
from torchvision import transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Unet().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)


latent_images = torch.randn(8, 4, 128, 128).to(device) 
instruction_embedding = torch.randn(8, 1, 768).to(device)  
auxiliary_embedding = torch.randn(8, 1, 768).to(device)
gt_images = torch.randn(8, 4, 128, 128).to(device)  

epochs = 10
for epoch in range(epochs):
    optimizer.zero_grad()
    
    output = model(latent_images, instruction_embedding, auxiliary_embedding) 
    
 
    diffusion_loss = torch.nn.functional.mse_loss(output, gt_images)
    hf_loss = high_freq_loss(gt_images, output)
    
    loss = diffusion_loss + 0.001 * hf_loss 
    loss.backward()
    optimizer.step()
    
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")
