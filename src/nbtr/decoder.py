from nbtr.model.hf_model import HfModel
from nbtr.tokenizer.tokenizer import Tokenizer
import torch
from torch.nn import functional as F

class Decoder:

    def __init__(self, repo_id) -> None:
        self.tokenizer = Tokenizer.from_pretrained(repo_id)
        self.model = HfModel.from_pretrained(repo_id).model

    def decode(self, prompt:str, max_new_tokens=256):
        input_ids = self.tokenizer.encode(prompt)['input_ids']
        
        ids = torch.tensor(input_ids).view(1, -1)
        ids = ids[:,:-1] # exclude eos token
        
        for i in range(max_new_tokens):
            logits,_ = self.model(ids)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=1)
            tok = torch.multinomial(probs, 1, replacement=True)
            
            if tok.item() == self.tokenizer.eos_id:
                break

            ids = torch.cat((ids , tok), dim=-1)

        text = self.tokenizer.decode(ids.tolist()[0])
        return text
            