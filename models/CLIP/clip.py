import torch
from transformers import CLIPTokenizer, CLIPTextModel

class CLIPTextEmbedder:
    def __init__(self, model_name="openai/clip-vit-large-patch14"):
        """
        Initializing the CLIP embeddder with the ViT-L/14 CLIP model.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self.model = CLIPTextModel.from_pretrained(model_name).to(self.device)
    
    def get_text_embedding(self, text):
        """
        Generates a text embedding of the given input string.
        :param text: Input string
        :return: Numpy array of the text embedding
        """
        inputs = self.tokenizer(text, padding=True, truncation=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            text_embedding = outputs.pooler_output
        
        return text_embedding.cpu().numpy()
    
    def get_dual_embeddings(self, instruction, text):
        """
        Generates embeddings for both an instruction and a text prompt.
        :param instruction: Instruction prompt string
        :param text: Text prompt string
        :return: Tuple of numpy arrays (embedding1, embedding2)
        """
        embedding1 = self.get_text_embedding(instruction)
        embedding2 = self.get_text_embedding(text)
        return embedding1, embedding2

if __name__ == "__main__":
    embedder = CLIPTextEmbedder()
    instruction = "Describe the image in detail."
    text = "A cat sitting on a windowsill looking at the sunset."
    embedding1, embedding2 = embedder.get_dual_embeddings(instruction, text)
    print(f"Instruction embedding shape: {embedding1.shape}")
    print(f"Text embedding shape: {embedding2.shape}")
