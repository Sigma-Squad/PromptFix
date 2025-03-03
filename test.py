from PIL import Image
from torchvision import transforms as T
from datasets import load_dataset

from models.VAE.autoencoder import AutoencoderKL

transforms = T.Compose([
    T.Resize((128, 128)),
    T.ToTensor()
])

dataset = load_dataset("yeates/PromptfixData", split="train", streaming=True)

num_points = 1
data_points = dataset.take(num_points)

autoencoder = AutoencoderKL(
    ddconfig={
        'double_z': True,
        'z_channels': 4,
        'resolution': 256,
        'in_channels': 3,
        'out_ch': 3,
        'ch': 128,
        'ch_mult': [1, 2, 4, 8],
        'num_res_blocks': 2,
        'attn_resolutions': [],
        'dropout': 0.0,
    },
    lossconfig={
        'target': 'torch.nn.Identity',
    },
    embed_dim=4,
)

for i, data in enumerate(data_points):
    input_tensor = transforms(data["input_img"])
    print(f"Image {i}: {input_tensor.shape}")
    z = autoencoder(input_tensor.unsqueeze(0))
    print(f"z: {z.shape}")
    print("done", flush=True)
    # print(output.shape, flush=True)