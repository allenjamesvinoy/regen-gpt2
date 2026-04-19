import torch
import torch.nn as nn
from torch import dropout, embedding
from dataclasses import dataclass
import torch.nn.functional as F

@dataclass
class GPT2Config:
  num_heads: int = 6
  num_layers: int = 6
  vocab_size: int = 50257
  embedding_dim: int = 768
  block_size: int = 1024
  dropout : float = .1
  weight_decay : float = .1
  pad_token_id : int = 50256

class GPT2_Baseline(nn.Module):
  #create a ModuleDict with wte, wpe, hidden layers, weight and bias
  def __init__(self, config: GPT2Config, device):
    super().__init__()
    self.config = config
    self.device = device
    self.transformer = nn.ModuleDict(
        dict(
            wte = nn.Embedding(config.vocab_size, config.embedding_dim), # how do we determine the dimensions of this?
            wpe = nn.Embedding(config.block_size, config.embedding_dim),
            h = nn.ModuleList([LayerBlock(config, device) for _ in range(config.num_layers)]),
            ln_f = nn.LayerNorm(config.embedding_dim),
            drop = nn.Dropout(config.dropout)
        )
    )
    self.lm_head = nn.Linear(config.embedding_dim, config.vocab_size, bias=False) #why is bias set as False?
    self.transformer.wte.weight = self.lm_head.weight  # weight tying



  def forward(self, x, y=None):
    _, N = x.size()
    V = self.config.vocab_size


    pos = torch.arange(0, N, dtype=torch.long).to(self.device)
    pos_emb = self.transformer.wpe(pos)

    #Combine both sequences so they can be passed through together
    x = self.transformer.wte(x) + pos_emb
    x = self.transformer.drop(x)


    for layer_block in self.transformer.h:
      x = layer_block(x)

    x = self.transformer.ln_f(x)
    logits_fwd = self.lm_head(x)


    if y is not None:
        # Forward loss: standard next-token prediction

        loss_fwd = F.cross_entropy(
            logits_fwd.view(-1, V), 
            y.view(-1),
            ignore_index=self.config.pad_token_id)

    return loss_fwd


class LayerBlock(nn.Module):
  def __init__(self, config: GPT2Config, device):
    super().__init__()
    self.ln_1 = nn.LayerNorm(config.embedding_dim)
    self.attn = AttentionMultiHeadFused(config, device)
    self.ln_2 = nn.LayerNorm(config.embedding_dim)
    self.mlp = MLP(config)
    

  def forward(self, x):
    x = x + self.attn(self.ln_1(x))
    x = x + self.mlp(self.ln_2(x)) #why do we layer-norm before passing mlp?
    return x


class MLP(nn.Module):
  def __init__(self, config: GPT2Config):
    super().__init__()
    self.c_fc = nn.Linear(config.embedding_dim, 4*config.embedding_dim)
    self.gelu = nn.GELU(approximate='tanh')
    self.c_proj = nn.Linear(4*config.embedding_dim, config.embedding_dim)
    self.drop   = nn.Dropout(config.dropout)

  def forward(self, x):
    x = self.c_fc(x)
    x = self.gelu(x)
    x = self.c_proj(x)
    x = self.drop(x)
    return x

class AttentionMultiHeadFused(nn.Module):
  def __init__(self, config: GPT2Config, device):
    super().__init__()
    self.config = config
    self.device = device
    self.w_qkv = nn.Linear(config.embedding_dim, 3*config.embedding_dim)
    self.output = nn.Linear(config.embedding_dim, config.embedding_dim)
    self.attn_drop = config.dropout  # passed to scaled_dot_product_attention
    self.resid_drop = nn.Dropout(config.dropout)  # add this

  def forward(self, x):
    B,N,ED = x.size()
    qkv = self.w_qkv(x)
    q, k, v = qkv.split(self.config.embedding_dim, dim=2)
    q = q.view(B, N, self.config.num_heads, ED // self.config.num_heads).transpose(1,2)
    k = k.view(B, N, self.config.num_heads, ED // self.config.num_heads).transpose(1,2)
    v = v.view(B, N, self.config.num_heads, ED // self.config.num_heads).transpose(1,2)

    dropout_p=self.attn_drop if self.training else 0.0

    attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=dropout_p)


    attn_out = attn_out.transpose(1,2).contiguous().view(B, N, ED)
    y = self.output(attn_out)
    y = self.resid_drop(y)
    return y