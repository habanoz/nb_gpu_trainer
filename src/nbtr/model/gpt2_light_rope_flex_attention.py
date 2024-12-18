from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from typing import Tuple
from torch.nn.attention.flex_attention import flex_attention, create_block_mask
flex_attention = torch.compile(flex_attention, dynamic=False)
create_block_mask = torch.compile(create_block_mask, dynamic=False)

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

def apply_scaling(freqs: torch.Tensor):
    # Values obtained from grid search
    scale_factor = 8
    low_freq_factor = 1
    high_freq_factor = 4
    old_context_len = 8192  # original llama3 length

    low_freq_wavelen = old_context_len / low_freq_factor
    high_freq_wavelen = old_context_len / high_freq_factor
    new_freqs = []
    for freq in freqs:
        wavelen = 2 * math.pi / freq
        if wavelen < high_freq_wavelen:
            new_freqs.append(freq)
        elif wavelen > low_freq_wavelen:
            new_freqs.append(freq / scale_factor)
        else:
            assert low_freq_wavelen != high_freq_wavelen
            smooth = (old_context_len / wavelen - low_freq_factor) / (
                high_freq_factor - low_freq_factor
            )
            new_freqs.append((1 - smooth) * freq / scale_factor + smooth * freq)
    return torch.tensor(new_freqs, dtype=freqs.dtype, device=freqs.device)


def precompute_freqs_cis(
    dim: int, end: int, theta: float = 10000.0, use_scaled: bool = False
):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    if use_scaled:
        freqs = apply_scaling(freqs)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

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
        
    def forward(self, x, block_mask, freqs_cis: torch.Tensor):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embed)
        
        qkv = self.c_attn(x)
        q, k, v = qkv.split([self.n_head * self.head_dim, self.n_kv_head * self.head_dim, self.n_kv_head * self.head_dim], dim=-1)
        q, k, v = map(lambda t: t.view(B, T, -1, self.head_dim), (q, k, v))  # (B, T, NH, HD)
        
        # rope
        q, k = apply_rotary_emb(q, k, freqs_cis=freqs_cis)
        
        # GQA
        ######################
        k = repeat_kv(k, self.n_rep)
        v = repeat_kv(v, self.n_rep)

        k = k.transpose(1,2) # (B, nh, T, hs)
        q = q.transpose(1,2) # (B, nh, T, hs)
        v = v.transpose(1,2) # (B, nh, T, hs)
        ######################

        # efficient attention using Flash Attention CUDA kernels
        # y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        y = flex_attention(q, k, v, block_mask=block_mask)
        
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        
        return y
    
class Block(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()

        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)
        
    def forward(self, x, block_mask, freqs_cis):
        x = x + self.attn(norm(x), block_mask, freqs_cis)
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
                h = nn.ModuleList(
                    [Block(config) for _ in range(config.n_layer)]
                ),
            )
        )

        self.lm_head = nn.Linear(config.n_embed, config.vocab_size, bias=False)
 
        # weight-tying
        self.transformer.wte.weight = self.lm_head.weight
        
        freqs_cis = precompute_freqs_cis(
            config.n_embed // config.n_head,
            config.seq_length * 2,
            5e+5, # config.rope_theta,
            False #config.use_scaled_rope,
        )
        
        assert freqs_cis.requires_grad==False,"Precomputed cis tensor cannot accumulate grad"
        assert freqs_cis.dtype==torch.complex64,"Precomputed cis tensor must have complex64 dtype"
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

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
    
    def forward(self, inputs: torch.Tensor, targets=None):
        B, T = inputs.size()
        assert T <= self.config.seq_length, f"Input sequence length ({T}) cannot be larger than model sequence lenfgth {self.config.seq_length} "
        
        END_OF_TEXT_ID=2
        BLOCK_SIZE = 128
        
        seq_len = len(inputs)
        assert seq_len % BLOCK_SIZE == 0
        total_num_blocks = seq_len // BLOCK_SIZE
        assert inputs.ndim == 1
        
        docs = (inputs == END_OF_TEXT_ID).cumsum(0)
        def document_causal_mask(b, h, q_idx, kv_idx):
          causal_mask = q_idx >= kv_idx
          document_mask = docs[q_idx] == docs[kv_idx]
          window_mask = q_idx - kv_idx < 1024
          return causal_mask & document_mask & window_mask
        S = len(inputs)
        
        block_mask = create_block_mask(document_causal_mask, None, None, S, S, device="cuda", _compile=True)

        
        freqs_cis = self.freqs_cis[:T]
        
        x = self.transformer.wte(inputs)
        x = norm(x)
        
        for block in self.transformer.h:
            x = block(x, block_mask, freqs_cis)
            
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
        return n_params