import torch
from torch.nn import functional as F

class Decoder:
    
    def __init__(self, tokenizer, model):
        self.tokenizer = tokenizer
        self.model     = model
    
    def decode(self, prompt, max_tokens=50):
        prompt = torch.tensor(self.tokenizer.encode(prompt, eos=False)['input_ids']).view(1, -1)
        
        for _ in range(max_tokens):
            with torch.inference_mode():
                logits = self.model.forward(prompt)['logits']
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=1)
            tok = torch.multinomial(probs, 1, replacement=True)
            
            if tok.item() == self.tokenizer.eos_id:
                break

            prompt = torch.cat((prompt , tok), dim=-1)

        text = self.tokenizer.decode(prompt.tolist()[0])
        
        return text

    def decode_greedy(self, prompt, max_tokens=50):
        prompt = torch.tensor(self.tokenizer.encode(prompt, eos=False)['input_ids']).view(1, -1)
        
        for _ in range(max_tokens):
            with torch.inference_mode():
                logits = self.model.forward(prompt)['logits']
            logits = logits[:, -1, :]
            
            tok = torch.argmax(logits, dim=1, keepdim=True)
            
            if tok.item() == self.tokenizer.eos_id:
                break

            prompt = torch.cat((prompt , tok), dim=-1)

        text = self.tokenizer.decode(prompt.tolist()[0])
        
        return text