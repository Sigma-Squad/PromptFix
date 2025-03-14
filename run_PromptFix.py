import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from models.CLIP.clip import CLIPTextEmbedder
from utils.dataset_class import StreamingPromptDataset
from models.VAE.autoencoder import AutoencoderKL
from models.UNet.unet.model import UNetModel
import yaml

# hyperparameters
BUFFER_SIZE = 100
BATCH_SIZE = 8
NUM_EPOCHS = 50
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    # Load dataset
    dataset = load_dataset("yeates/PromptfixData", split="train", streaming=True)
    embedder = CLIPTextEmbedder()

    # Create dataset and dataloader
    prompt_dataset = StreamingPromptDataset(dataset=dataset, shuffle=True, embedder=embedder, buffer_size=BUFFER_SIZE)
    dataloader = DataLoader(prompt_dataset, batch_size=BATCH_SIZE)
    
    # instantiate model
    model_config = yaml.safe_load(open("./models/promptfix_config.yaml", "r"))
    
    # encoder
    autoencoder_config = model_config["autoencoder"]
    encoder = AutoencoderKL(**autoencoder_config)
    
    # unet
    unet_config = model_config["unet"]
    unet = UNetModel(**unet_config).half()

    # start training
    for epoch in range(NUM_EPOCHS):
        for batch in dataloader:
            input_imgs, output_imgs, instruction_embeddings, prompt_embeddings = batch.values()
            print(input_imgs.shape, output_imgs.shape, instruction_embeddings.shape, prompt_embeddings.shape)
            break
        break