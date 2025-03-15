import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from models.CLIP.clip import CLIPTextEmbedder
from utils.dataset_class import StreamingPromptDataset
from models.VAE.autoencoder import AutoencoderKL
from models.UNet.unet.model import UNetModel
from models.PromptFix import PromptFix
import yaml, os, argparse

# hyperparameters
BUFFER_SIZE = 100
BATCH_SIZE = 8
NUM_EPOCHS = 50
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train PromptFix')
    parser.add_argument('--checkpoint', type=int, required=False, help='checkpoint model weights to load')
    args = parser.parse_args()
    
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
    encdec = AutoencoderKL(**autoencoder_config)
    
    # unet
    unet_config = model_config["unet"]
    unet = UNetModel(**unet_config).half()
    
    # PromptFix model
    checkpoint = args.checkpoint
    promptfix = PromptFix(encdec=encdec, unet=unet, device=DEVICE).to(DEVICE)
    # load a trained model
    if os.path.exists(f"promptfix_weights_{checkpoint}.pt"):
        promptfix.load_state_dict(torch.load(f"promptfix_weights_{checkpoint}.pt"))
    
    # loss
    loss_fn = torch.nn.MSELoss()
    
    # optimizer
    optimizer = torch.optim.AdamW(promptfix.parameters())

    # start training
    for epoch in range(NUM_EPOCHS):
        print(F"Epoch {epoch}")
        for idx, batch in enumerate(dataloader):
            input_imgs, output_imgs, instruction_embeddings, prompt_embeddings = batch.values()
            reconstructed_imgs = promptfix(input_imgs, instruction_embeddings, prompt_embeddings)
            loss = loss_fn(reconstructed_imgs, output_imgs)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            print(f"Loss: {loss.item()}")
        
        # checkpoints
        if os.path.exists(f"promptfix_weights_{epoch-1}.pt"):
            os.remove(f"promptfix_weights_{epoch-1}.pt")
        torch.save(promptfix.state_dict(), f"promptfix_weights_{epoch}.pt")
        
    print("Training done!")