from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from torch.utils.checkpoint import checkpoint as gc
import torch.version

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

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class LlamaRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        super().__init__()
        self.scaling_factor = scaling_factor
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        # For BC we register cos and sin cached
        self.max_seq_len_cached = max_position_embeddings

    @torch.no_grad()
    def forward(self, x, position_ids):
        # x: [bs, num_attention_heads, seq_len, head_size]
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        # Force float32 since bfloat16 loses precision on long contexts
        # See https://github.com/huggingface/transformers/pull/29285
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)
    
class MLP(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()

        self.c_fc = nn.Linear(config.n_embed, config.n_embed * 4, bias=False)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(config.n_embed * 4, config.n_embed, bias=False)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embed % config.n_head == 0
        self.head_dim = config.n_embed // config.n_head
        
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embed, 3 * config.n_embed, bias=False)
        # output projection
        self.c_proj = nn.Linear(config.n_embed, config.n_embed, bias=False)
        
        self.rotary_emb = LlamaRotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta,
            )
        
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embed = config.n_embed
        self.dropout = config.dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.seq_length, config.seq_length))
                                        .view(1, 1, config.seq_length, config.seq_length))

    def forward(self, x, position_ids):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embed)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embed, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        
        # rope
        cos, sin = self.rotary_emb(v, position_ids)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # gqa
        # key_states = repeat_kv(key_states, self.num_key_value_groups) # (B, num_attention_heads, T, hs)
        # value_states = repeat_kv(value_states, self.num_key_value_groups) # (B, num_attention_heads, T, hs)
        
        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y
    
class Block(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()

        self.ln_1 = nn.LayerNorm(config.n_embed, bias=False)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embed, bias=False)
        self.mlp = MLP(config)

    def forward(self, x, position_ids):
        x = x + self.attn(self.ln_1(x), position_ids)
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()

        # version check, otherwise make sure q,k,v tensors are continuous
        torch_version = int(torch.version.__version__.split("+")[0].replace(".",""))
        assert torch_version>=220,"Issue #112577 is fixed after torch version 2.2.0"
        
        self.config = config
        self._gradient_checkpointing = False

        self.transformer = nn.ModuleDict(
            dict(
                wte = nn.Embedding(config.vocab_size, config.n_embed),
                h = nn.ModuleList(
                    [Block(config) for _ in range(config.n_layer)]
                ),
                ln_f = nn.LayerNorm(config.n_embed, bias=False)
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

        position_ids = torch.arange(0, T, dtype=torch.long, device=device).unsqueeze(0)
        
        x = self.transformer.wte(idx)
        
        for block in self.transformer.h:
            if self._gradient_checkpointing and self.training:
                x = gc(block,x, position_ids, use_reentrant=True)
            else:
                x = block(x, position_ids)
            
        x = self.transformer.ln_f(x)
        
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