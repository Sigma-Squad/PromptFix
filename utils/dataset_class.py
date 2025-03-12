import torch
from torch.utils.data import Dataset

class PromptDataset(Dataset):
    def __init__(self, dataset, embedder):
        self.prompts = dataset["prompt"]
        self.instructions = dataset["Instruction"]
        self.embedder = embedder

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        prompt_text = self.prompts[idx]
        instruction_text = self.instructions[idx]

        # Generate embeddings separately
        prompt_embedding = self.embedder.get_text_embedding(prompt_text)
        instruction_embedding = self.embedder.get_text_embedding(instruction_text)

        return {
            "prompt_embedding": torch.tensor(prompt_embedding, dtype=torch.float32).squeeze(0),
            "instruction_embedding": torch.tensor(instruction_embedding, dtype=torch.float32).squeeze(0)
        }