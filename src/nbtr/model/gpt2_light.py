from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import math

@dataclass
class GPTConfig:
    seq_length: int =  1024
    vocab_size: int = 50304
    n_embed: int = 768
    n_head: int = 12
    n_layer: int = 12
    dropout: float = 0.1

    @staticmethod
    def from_yaml(config_file:str):
        import yaml

        with open(config_file) as f:
            doc = yaml.safe_load(f)
        
        return GPTConfig(**doc)

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )
    
def norm(x):
    return F.rms_norm(x, (x.size(-1),))
    
class MLP(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()

        self.c_fc = nn.Linear(config.n_embed, config.n_embed * 4, bias=False)
        self.c_proj = nn.Linear(config.n_embed * 4, config.n_embed, bias=False)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        
        assert config.n_embed % config.n_head == 0
        self.head_dim = config.n_embed // config.n_head
        
        self.n_head = config.n_head
        self.n_kv_head = int(config.n_head / 2) ## TODO: move to configuration
        self.n_rep = self.n_head // self.n_kv_head
        
        self.n_embed = config.n_embed
        
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embed, (self.n_head+2*self.n_kv_head)*self.head_dim, bias=False)
        
        # output projection
        self.c_proj = nn.Linear(config.n_embed, config.n_embed, bias=False)
        
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        
        self.dropout = config.dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        assert self.flash, "SDPA is required!"
        
    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embed)
        
        qkv = self.c_attn(x)
        q, k, v = qkv.split([self.n_head * self.head_dim, self.n_kv_head * self.head_dim, self.n_kv_head * self.head_dim], dim=-1)
        q, k, v = map(lambda t: t.view(B, T, -1, self.head_dim), (q, k, v))  # (B, T, NH, HD)
        
        # GQA
        ######################
        k = repeat_kv(k, self.n_rep)
        v = repeat_kv(v, self.n_rep)

        k = k.transpose(1,2) # (B, nh, T, hs)
        q = q.transpose(1,2) # (B, nh, T, hs)
        v = v.transpose(1,2) # (B, nh, T, hs)
        ######################

        # efficient attention using Flash Attention CUDA kernels
        y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
            
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        
        return y
    
class Block(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()

        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)
        
    def forward(self, x):
        x = x + self.attn(norm(x))
        x = x + self.mlp(norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self._gradient_checkpointing = False

        self.transformer = nn.ModuleDict(
            dict(
                wte = nn.Embedding(config.vocab_size, config.n_embed),
                wpe = nn.Embedding(config.seq_length, config.n_embed),
                h = nn.ModuleList(
                    [Block(config) for _ in range(config.n_layer)]
                ),
            )
        )

        self.lm_head = nn.Linear(config.n_embed, config.vocab_size, bias=False)
 
        # weight-tying
        self.transformer.wte.weight = self.lm_head.weight

        # init all weights
        self.apply(self._init_weights)

        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))
    
    @property
    def gradient_checkpointing(self):
        return self._gradient_checkpointing

    @gradient_checkpointing.setter
    def gradient_checkpointing(self, value:bool):
        self._gradient_checkpointing = value
    
    def forward(self, idx, targets=None):
        device = idx.device
        B, T = idx.size()
        assert T <= self.config.seq_length, f"Input sequence length ({T}) cannot be larger than model sequence lenfgth {self.config.seq_length} "

        pos = torch.arange(0, T, dtype=torch.long, device=device)

        tok_embd = self.transformer.wte(idx)
        pos_embd = self.transformer.wpe(pos)
        x = tok_embd + pos_embd
        
        for block in self.transformer.h:
            x = block(x)
        
        x = norm(x)
        
        logits = self.lm_head(x)
        loss = None
        
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

        return logits, loss

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params