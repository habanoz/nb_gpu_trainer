import torch
from torch.nn import functional as F

class Decoder:
    
    def __init__(self, tokenizer, model):
        self.tokenizer = tokenizer
        self.model     = model
    
    @torch.inference_mode()
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

    @torch.inference_mode()
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
    
    
    @torch.inference_mode()
    def decode_temp(self, prompt, max_tokens=50, temperature=1.0, top_k=None):
        prompt = torch.tensor(self.tokenizer.encode(prompt, eos=False)['input_ids']).view(1, -1)
        
        for _ in range(max_tokens):
            with torch.inference_mode():
                logits = self.model.forward(prompt)['logits']
            
            logits = logits[:, -1, :] / temperature
            
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            probs = F.softmax(logits, dim=-1)
            tok = torch.multinomial(probs, num_samples=1)
            
            if tok.item() == self.tokenizer.eos_id:
                break

            prompt = torch.cat((prompt , tok), dim=-1)

        text = self.tokenizer.decode(prompt.tolist()[0])
        
        return text