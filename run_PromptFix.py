import torch
from torch.utils.data import DataLoader
from datasets import load_from_disk, load_dataset
from utils.dataset_class import StreamingPromptFixDataset, PromptFixDataset
from models.CLIP.clip import CLIPTextEmbedder
from models.VAE.autoencoder import AutoencoderKL
from models.UNet.unet.model import UNetModel
from models.PromptFix import PromptFix
import yaml, os, argparse, time

# hyperparameters
BUFFER_SIZE = 100
BATCH_SIZE = 8
NUM_EPOCHS = 50
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train PromptFix')
    parser.add_argument('--checkpoint', type=int, required=False, help='checkpoint model weights to load')
    parser.add_argument ('-f', type=str, help='The local dataset to load')
    args = parser.parse_args()
    
    # Load dataset
    dataset = load_from_disk(args.f)
    # dataset = load_dataset("yeates/PromptfixData", split="train", streaming=True)
    embedder = CLIPTextEmbedder()

    # Create dataset and dataloader
    # prompt_dataset = StreamingPromptFixDataset(dataset=dataset,shuffle=True, embedder=embedder, buffer_size=BUFFER_SIZE)
    prompt_dataset = PromptFixDataset(dataset=dataset, shuffle=True, embedder=embedder)
    dataloader = DataLoader(prompt_dataset, batch_size=BATCH_SIZE)
    print("Created dataset and dataloader")
    
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
    print("Starting training...")
    start = time.time()
    for epoch in range(NUM_EPOCHS):
        print(F"Epoch {epoch}")
        for idx, batch in enumerate(dataloader):
            # Move all batch tensors to the device
            input_imgs = batch["input_img"].to(DEVICE)
            output_imgs = batch["output_img"].to(DEVICE)
            instruction_embeddings = batch["instruction_embedding"].to(DEVICE)
            prompt_embeddings = batch["prompt_embedding"].to(DEVICE)
            
            reconstructed_imgs = promptfix(input_imgs, instruction_embeddings, prompt_embeddings)
            loss = loss_fn(reconstructed_imgs, output_imgs)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            print(f"Loss: {loss.item()}")
        
        # checkpoints
        torch.save(promptfix.state_dict(), f"promptfix_weights_{epoch}.pt")
        if os.path.exists(f"promptfix_weights_{epoch-1}.pt"):
            os.remove(f"promptfix_weights_{epoch-1}.pt")
        
        print(f"Finished epoch {epoch} in {time.time() - start} seconds")
        
    print("Training done!")