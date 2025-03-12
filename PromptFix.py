from torch.utils.data import DataLoader
from datasets import load_dataset
from models.CLIP.clip import CLIPTextEmbedder
from utils.dataset_class import PromptDataset

if __name__ == "__main__":
    # Load dataset
    dataset = load_dataset("yeates/PromptfixData", split="train", streaming=True)
    dataset = dataset.remove_columns([col for col in dataset.column_names if col not in ["auxiliary_prompt", "instruction"]])
    embedder = CLIPTextEmbedder()

    # Create dataset and dataloader
    prompt_dataset = PromptDataset(dataset=dataset, embedder=embedder)
    batch_size = 8
    dataloader = DataLoader(prompt_dataset, batch_size=batch_size, shuffle=True)

    # Example batch
    batch = next(iter(dataloader))