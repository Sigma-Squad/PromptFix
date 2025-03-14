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
    
    def get_dual_embeddings(self, instruction, prompt):
        """
        Generates embeddings for both an instruction and a text prompt.
        :param instruction: Instruction prompt string
        :param prompt: Text prompt string
        :return: Tuple of numpy arrays (instruction_embedding, prompt_embedding)
        """
        instruction_embedding = self.get_text_embedding(instruction)
        prompt_embedding = self.get_text_embedding(prompt)
        return instruction_embedding, prompt_embedding
