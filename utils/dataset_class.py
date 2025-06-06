import random
import torch
from torch.utils.data import IterableDataset
import torchvision.transforms as T
from torch.utils.data import Dataset

class StreamingPromptFixDataset(IterableDataset):
    def __init__(self, dataset, embedder, shuffle=False, buffer_size=100):
        self.dataset = dataset
        self.embedder = embedder
        self.shuffle = shuffle
        self.buffer_size = buffer_size
        self.image_transform = T.Compose([
            T.Lambda(lambda img: img.convert("RGB") if img.mode in ["L", "RGBA", "CMYK", "BGRA"] else img),
            T.Resize((128, 128)),
            T.ToTensor()
        ])

    def __iter__(self):
        # Buffer for shuffling if needed
        samples = []
        dataset_iter = iter(self.dataset)
        
        # Process samples one by one - no batching in the dataset
        for example in dataset_iter:
            prompt_text = example["auxiliary_prompt"]
            instruction_text = example["instruction"]
            input_image = example["input_img"]
            output_image = example["processed_img"]

            # Apply transformations to images
            input_image = self.image_transform(input_image)
            output_image = self.image_transform(output_image)

            # Generate text embeddings
            instruction_embedding, prompt_embedding = self.embedder.get_dual_embeddings(instruction_text, prompt_text)

            sample = {
                "input_img": input_image,
                "output_img": output_image,
                "prompt_embedding": prompt_embedding,
                "instruction_embedding": instruction_embedding
            }
            
            if not self.shuffle:
                yield sample
            else:
                samples.append(sample)
                if len(samples) >= self.buffer_size:
                    random.shuffle(samples)
                    for s in samples:
                        yield s
                    samples = []
        
        # Yield any remaining samples
        if samples and self.shuffle:
            random.shuffle(samples)
            for s in samples:
                yield s

class PromptFixDataset(Dataset):
    def __init__(self, dataset, embedder, shuffle=False):
        self.dataset = list(dataset)
        if shuffle:
            random.shuffle(self.dataset)
        self.embedder = embedder
        self.image_transform = T.Compose([
            T.Lambda(lambda img: img.convert("RGB") if img.mode in ["L", "RGBA", "CMYK", "BGRA"] else img),
            T.Resize((128, 128)),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.dataset) 

    def __getitem__(self, idx):
        example = self.dataset[idx]  
        
        prompt_text = example["auxiliary_prompt"]
        instruction_text = example["instruction"]
        input_image = example["input_img"]
        output_image = example["processed_img"]

        # Apply image transformations
        input_image = self.image_transform(input_image)
        output_image = self.image_transform(output_image)

        # Generate text embeddings
        instruction_embedding, prompt_embedding = self.embedder.get_dual_embeddings(instruction_text, prompt_text)

        return {
            "input_img": input_image,
            "output_img": output_image,
            "prompt_embedding": prompt_embedding,
            "instruction_embedding": instruction_embedding
        }