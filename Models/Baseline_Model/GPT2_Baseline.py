import torch
import torch.nn as nn
from torch import dropout, embedding
from dataclasses import dataclass
import torch.nn.functional as F
from transformers import GPT2Tokenizer
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from Configs import BaselineConfig

class GPT2_Baseline(nn.Module):
  def __init__(self, config: BaselineConfig, device):
    super().__init__()
    self.config = config
    self.device = device
    self.transformer = nn.ModuleDict(
        dict(
            wte = nn.Embedding(config.vocab_size, config.embedding_dim),
            wpe = nn.Embedding(config.block_size, config.embedding_dim),
            h = nn.ModuleList([LayerBlock(config) for _ in range(config.num_layers)]),
            ln_f = nn.LayerNorm(config.embedding_dim),
            drop = nn.Dropout(config.dropout)
        )
    )
    self.lm_head = nn.Linear(config.embedding_dim, config.vocab_size, bias=False)
    self.transformer.wte.weight = self.lm_head.weight  # weight tying
    self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')




  def forward(self, x, y):
    _, N = x.size()
    V = self.config.vocab_size


    pos = torch.arange(0, N, dtype=torch.long, device=x.device)
    pos_emb = self.transformer.wpe(pos)

    x = self.transformer.wte(x) + pos_emb
    x = self.transformer.drop(x)


    for layer_block in self.transformer.h:
      x, _ = layer_block(x)

    x = self.transformer.ln_f(x)
    logits = self.lm_head(x)


    loss = F.cross_entropy(
        logits.view(-1, V), 
        y.view(-1),
        ignore_index=self.config.pad_token_id)

    return logits, loss

  @torch.no_grad()
  def infer(self, prompt, max_new_tokens: int, temperature: float = 1.0, top_k: int = 50):
    #Note that the infer function itself handles the KV cache management
    self.eval()
    tokens = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)

    # Prefill
    N = tokens.size(1)
    pos = torch.arange(0, N, dtype=torch.long, device=self.device)
    x = self.transformer.wte(tokens) + self.transformer.wpe(pos)
    x = self.transformer.drop(x)

    kv_caches = []
    for layer_block in self.transformer.h:
      x, cache = layer_block(x, kv_cache=None)
      kv_caches.append(cache)
    x = self.transformer.ln_f(x)
    next_token = self._sample(self.lm_head(x[:, -1, :]), temperature, top_k)
    tokens = torch.cat([tokens, next_token], dim=1)

    # Decode
    for _ in range(max_new_tokens - 1):
      pos  = torch.tensor([tokens.size(1) - 1], dtype=torch.long, device=self.device)
      x    = self.transformer.wte(next_token) + self.transformer.wpe(pos)
      x    = self.transformer.drop(x)
      new_caches = []
      for layer_block, kv_cache in zip(self.transformer.h, kv_caches):
        x, cache = layer_block(x, kv_cache=kv_cache)
        new_caches.append(cache)

      kv_caches = new_caches
      x = self.transformer.ln_f(x)

      next_token = self._sample(self.lm_head(x[:, -1, :]), temperature, top_k)
      tokens = torch.cat([tokens, next_token], dim=1)

    return self.tokenizer.decode(tokens[0].tolist())

  def _sample(self, logits, temperature, top_k=None):
    logits = logits / temperature
    if top_k is not None:
      v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
      logits[logits < v[:, [-1]]] = float('-inf')
    return torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1)
  
  @torch.no_grad()
  def prefill(self, prompt_tokens):
      self.eval()
      tokens = torch.tensor(prompt_tokens, dtype=torch.long, 
                            device=self.device).unsqueeze(0)
      N = tokens.size(1)
      pos = torch.arange(0, N, dtype=torch.long, device=self.device)
      x = self.transformer.wte(tokens) + self.transformer.wpe(pos)
      x = self.transformer.drop(x)

      kv_caches = []
      for layer_block in self.transformer.h:
          x, cache = layer_block(x, kv_cache=None)
          kv_caches.append(cache)

      x = self.transformer.ln_f(x)
      logits = self.lm_head(x)
      return logits, kv_caches

  @torch.no_grad()
  def decode_one_token(self, token, pos, kv_caches):
      self.eval()
      x = torch.tensor([[token]], dtype=torch.long, device=self.device)
      pos_emb = self.transformer.wpe(
          torch.tensor([pos], dtype=torch.long, device=self.device)
      )
      x = self.transformer.wte(x) + pos_emb
      x = self.transformer.drop(x)

      new_caches = []
      for layer_block, kv_cache in zip(self.transformer.h, kv_caches):
          x, cache = layer_block(x, kv_cache=kv_cache)
          new_caches.append(cache)

      x = self.transformer.ln_f(x)
      logits = self.lm_head(x)
      return logits, new_caches


class LayerBlock(nn.Module):
  def __init__(self, config: BaselineConfig):
      super().__init__()
      self.ln_1 = nn.LayerNorm(config.embedding_dim)
      self.attn = AttentionMultiHeadFused(config)
      self.ln_2 = nn.LayerNorm(config.embedding_dim)
      self.mlp = MLP(config)
    

  def forward(self, x, kv_cache = None):
    attn_out, new_cache = self.attn(self.ln_1(x), kv_cache)
    x = x + attn_out
    x = x + self.mlp(self.ln_2(x))
    return x, new_cache


class MLP(nn.Module):
  def __init__(self, config: BaselineConfig):
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
  def __init__(self, config: BaselineConfig):
    super().__init__()
    self.config = config
    self.w_qkv = nn.Linear(config.embedding_dim, 3 * config.embedding_dim)
    self.output = nn.Linear(config.embedding_dim, config.embedding_dim)
    self.attn_drop = config.dropout
    self.resid_drop = nn.Dropout(config.dropout)


  def forward(self, x, kv_cache=None):
    B, N, d = x.size()
    h = self.config.num_heads
    d_eff = d // h

    q, k, v = self.w_qkv(x).split(d, dim=2)
    q = q.view(B, N, h, d_eff).transpose(1, 2)
    k = k.view(B, N, h, d_eff).transpose(1, 2)
    v = v.view(B, N, h, d_eff).transpose(1, 2)

    if not self.training and kv_cache is not None:
      k_cache, v_cache = kv_cache
      k = torch.cat([k_cache, k], dim=2)
      v = torch.cat([v_cache, v], dim=2)

    new_cache = (k, v) if not self.training else None

    dropout_p = self.attn_drop if self.training else 0.0
    is_causal = kv_cache is None  # full causal mask during prefill and training
    attn_out  = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal, dropout_p=dropout_p)

    attn_out = attn_out.transpose(1, 2).contiguous().view(B, N, d)
    out = self.output(attn_out)
    out = self.resid_drop(out)
    return out, new_cache
