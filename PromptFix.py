
import sys
import torch
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
sys.path.append("models/CLIP")

from clip import CLIPTextEmbedder

# Load dataset
dataset = load_dataset("yeates/PromptfixData", split="train", streaming = True)
dataset = dataset.remove_columns([col for col in dataset.column_names if col not in ["auxiliary_prompt", "instruction"]])
embedder = CLIPTextEmbedder()

class PromptDataset(Dataset):
    def __init__(self, dataset):
        self.prompts = dataset["prompt"]
        self.instructions = dataset["Instruction"]

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        prompt_text = self.prompts[idx]
        instruction_text = self.instructions[idx]

        # Generate embeddings separately
        prompt_embedding = embedder.get_text_embedding(prompt_text)
        instruction_embedding = embedder.get_text_embedding(instruction_text)

        return {
            "prompt_embedding": torch.tensor(prompt_embedding, dtype=torch.float32).squeeze(0),
            "instruction_embedding": torch.tensor(instruction_embedding, dtype=torch.float32).squeeze(0)
        }

# Create dataset and dataloader
prompt_dataset = PromptDataset(dataset)
batch_size = 8
dataloader = DataLoader(prompt_dataset, batch_size=batch_size, shuffle=True)

# # Example batch
# batch = next(iter(dataloader))
# print(f"Prompt Embedding Shape: {batch['prompt_embedding'].shape}") 
# print(f"Instruction Embedding Shape: {batch['instruction_embedding'].shape}") 