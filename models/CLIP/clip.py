#pass
import torch
from transformers import CLIPTokenizer,CLIPTextModel

class CLIPTextEmbedder:
   def __init__(self,model_name = "openai/clip-vit-large-patch14"):
    """
     Initializing the CLIP embeddder with the VLi14 CLIP
    """
    self.device = "cuda" if torch.cuda.is_available() else "cpu"
    self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
    self.model = CLIPTextModel.from_pretrained(model_name)
    
   def get_text_embedding(self,text):
    """
    Generates a text embedding of the given input string
    :param text: Input string
    :return: Numpy array of the text embedding
    
    """
    inputs = self.tokenizer(text,padding=True,truncation=True,return_tensors="pt").to(self.device)
    with torch.no_grad():
      outputs = self.model(**inputs)
      text_embedding = outputs.pooler_output
      
    return text_embedding.cpu().numpy()
   
# if __name__ == "__main__":
#     embedder = CLIPTextEmbedder()
#     text = "A quick brown fox jumps over a lazy dog."
#     embedding = embedder.get_text_embedding(text)
#     print(f"Text embedding shape: {embedding.shape}")
