import torch
import torch.nn as nn
from torch import dropout, embedding
from dataclasses import dataclass
import torch.nn.functional as F
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from Configs import BERTConfig


#A custom dataclass to store the parameters of the model
#It's defaults are the hyperparameters associated with our trained model
# @dataclass
# class BERTConfig:
#   num_heads: int = 12
#   num_layers: int = 12
#   vocab_size: int = 50257
#   embedding_dim: int = 768
#   block_size: int = 512
#   dropout : float = .1
#   weight_decay : float = .1
#   pad_token_id : int = 50256
#   range_low : float = .5
#   range_high : float = .75


class BERT_Lag(nn.Module):
  #create a ModuleDict with wte, wpe, hidden layers, weight and bias
  def __init__(self, config: BERTConfig, device):
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




  # def forward(self, x, y=None, use_fwd=True):
  #   B, N = x.size()
  #   V = self.config.vocab_size
  #   pos = torch.arange(0, N, dtype=torch.long, device=x.device)
  #   pos_emb = self.transformer.wpe(pos)

  #   if(use_fwd):
  #     x_fwd = self.transformer.wte(x) + pos_emb
  #     x_fwd = self.transformer.drop(x_fwd)
  #     for layer_block in self.transformer.h:
  #       x_fwd = layer_block(x_fwd, mask=True)
  #     x_fwd = self.transformer.ln_f(x_fwd)
  #     logits_fwd = self.lm_head(x_fwd)
  #     if y is not None:
  #       # Forward loss: standard next-token prediction

  #       loss = F.cross_entropy(
  #           logits_fwd.view(-1, V), 
  #           y.view(-1),
  #           ignore_index=self.config.pad_token_id)
  #   else:
  #     min_pos = int(N * self.config.range_low) + 1
  #     sample_pos = torch.randint(min_pos, N + 1, (1,), device=x.device)[0]
  #     start = (sample_pos * self.config.range_low).long()
  #     end = (sample_pos * self.config.range_high).long()
  #     end = torch.where(end <= start, start + 1, end)

  #     positions = torch.arange(N, device=x.device)
  #     mask = (positions >= start) & (positions < end)  # fully tensor ops, no graph break
  #     causal_mask = positions >= sample_pos

  #     targets = y[:, mask] if y is not None else None

  #     x_bwd = x.clone()
  #     x_bwd = x_bwd.masked_fill(mask.unsqueeze(0), self.config.pad_token_id)
  #     x_bwd = x_bwd.masked_fill(causal_mask.unsqueeze(0), self.config.pad_token_id)

  #     x_bwd = self.transformer.wte(x_bwd) + pos_emb
  #     x_bwd = self.transformer.drop(x_bwd)

  #     for layer_block in self.transformer.h:
  #       x_bwd = layer_block(x_bwd, mask=False)
  #     x_bwd = self.transformer.ln_f(x_bwd)
  #     logits_bwd = self.lm_head(x_bwd[:,mask])
      
  #     if y is not None:
  #       loss = F.cross_entropy(
  #           # logits_bwd[:, start:end].reshape(-1, V),
  #           logits_bwd.reshape(-1,V),
  #           targets.reshape(-1),
  #           ignore_index=self.config.pad_token_id,
  #       )

  #   return loss





  def forward(self, x, y=None):
    #Define some useful constants for the forward pass
    B, N = x.size()
    V = self.config.vocab_size

    # Loss is defined on both tasks independently
    x_fwd = x
    # x_bwd = x.clone()
    y_fwd = y

    #Get the learned positional embeddings
    pos = torch.arange(0, N, dtype=torch.long, device=x.device)
    pos_emb = self.transformer.wpe(pos)

    #Randomly sample a position to be the simulated endpoint for group regeneration
    sample_pos = torch.randint(int(1/self.config.range_low), N + 1, (1,), device=x.device)[0]

    start = (sample_pos * self.config.range_low).long()
    end = (sample_pos * self.config.range_high).long()
    end = torch.where(end <= start, start + 1, end)

    positions = torch.arange(N, device=x.device)
    mask = (positions >= start) & (positions < end)  # fully tensor ops, no graph break
    causal_mask = positions >= sample_pos
    y_bwd = y[:, mask] if y is not None else None
    x_bwd = x.masked_fill(mask.unsqueeze(0), self.config.pad_token_id)
    x_bwd = x_bwd.masked_fill(causal_mask.unsqueeze(0), self.config.pad_token_id)

    #Combine both sequences so they can be passed through together
    x_fwd = self.transformer.wte(x_fwd) + pos_emb
    x_bwd = self.transformer.wte(x_bwd) + pos_emb
    x_fwd = self.transformer.drop(x_fwd)
    x_bwd = self.transformer.drop(x_bwd)



    for layer_block in self.transformer.h:
      x_fwd = layer_block(x_fwd, mask=True)
      x_bwd = layer_block(x_bwd, mask=False)

    x_fwd = self.transformer.ln_f(x_fwd)
    x_bwd = self.transformer.ln_f(x_bwd)


    logits_fwd = self.lm_head(x_fwd)
    logits_bwd = self.lm_head(x_bwd[:,mask])


    if y is not None:
        # Forward loss: standard next-token prediction

        loss_fwd = F.cross_entropy(
            logits_fwd.view(-1, V), 
            y_fwd.view(-1),
            ignore_index=self.config.pad_token_id)

        loss_bwd = F.cross_entropy(
            logits_bwd.reshape(-1,V),
            y_bwd.reshape(-1),
            ignore_index=self.config.pad_token_id,
        )

    return loss_fwd, loss_bwd
  


class LayerBlock(nn.Module):
  def __init__(self, config: BERTConfig, device):
    super().__init__()
    self.ln_1_fwd = nn.LayerNorm(config.embedding_dim)
    self.ln_1_bwd = nn.LayerNorm(config.embedding_dim)

    self.attn = AttentionMultiHeadFused(config, device)
    self.ln_2_fwd = nn.LayerNorm(config.embedding_dim)
    self.ln_2_bwd = nn.LayerNorm(config.embedding_dim)

    self.mlp = MLP(config)
    

  def forward(self, x, mask):
    if(mask):
      x = x + self.attn(self.ln_1_fwd(x), mask)
      x = x + self.mlp(self.ln_2_fwd(x)) 
    else:
      x = x + self.attn(self.ln_1_bwd(x), mask)
      x = x + self.mlp(self.ln_2_bwd(x)) 
    return x


class MLP(nn.Module):
  def __init__(self, config: BERTConfig):
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
  def __init__(self, config: BERTConfig, device):
    super().__init__()
    self.config = config
    self.device = device
    self.w_qkv = nn.Linear(config.embedding_dim, 3*config.embedding_dim)
    self.output = nn.Linear(config.embedding_dim, config.embedding_dim)
    self.attn_drop = config.dropout  # passed to scaled_dot_product_attention
    self.resid_drop = nn.Dropout(config.dropout)  # add this

  def forward(self, x, mask=None):
    B,N,ED = x.size()
    qkv = self.w_qkv(x)
    q, k, v = qkv.split(self.config.embedding_dim, dim=2)
    q = q.view(B, N, self.config.num_heads, ED // self.config.num_heads).transpose(1,2)
    k = k.view(B, N, self.config.num_heads, ED // self.config.num_heads).transpose(1,2)
    v = v.view(B, N, self.config.num_heads, ED // self.config.num_heads).transpose(1,2)

    dropout_p=self.attn_drop if self.training else 0.0

    if(mask):
      attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=dropout_p)
    else:
      attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=dropout_p)

    attn_out = attn_out.transpose(1,2).contiguous().view(B, N, ED)
    y = self.output(attn_out)
    y = self.resid_drop(y)
    return y
