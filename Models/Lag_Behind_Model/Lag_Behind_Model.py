import torch
import torch.nn as nn
from torch import dropout, embedding
from dataclasses import dataclass
import torch.nn.functional as F
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from Configs import LagBehindConfig

class GPT2_Lag(nn.Module):
    def __init__(self, config: LagBehindConfig, device):
        super().__init__()
        self.config = config
        self.device = device
        self.transformer = nn.ModuleDict(dict(
            wte     = nn.Embedding(config.vocab_size, config.embedding_dim),
            wpe     = nn.Embedding(config.block_size, config.embedding_dim),
            h       = nn.ModuleList([LayerBlock(config, device)
                                     for _ in range(config.num_layers)]),
            ln_f    = nn.LayerNorm(config.embedding_dim),
            drop    = nn.Dropout(config.dropout)
        ))
        self.lm_head     = nn.Linear(config.embedding_dim, config.vocab_size, bias=False)
        self.lm_head_lag = nn.Linear(config.embedding_dim, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight  # weight tying
        self.lm_head_lag.weight = self.lm_head.weight

    def forward(self, x, y=None, lam=0.5):
        B, SL = x.size()
        assert B % 2 == 0, "Batch size must be even for batch splitting"
        V         = self.config.vocab_size
        skip_dist = self.config.lag_behind
        half      = B // 2

        pos     = torch.arange(0, SL, dtype=torch.long, device=self.device)
        pos_emb = self.transformer.wpe(pos)

        x_fwd = self.transformer.wte(x[:half]) + pos_emb
        x_lag = self.transformer.wte(x[half:]) + pos_emb

        x_fwd = self.transformer.drop(x_fwd)
        x_lag = self.transformer.drop(x_lag)

        for layer_block in self.transformer.h:
            x_fwd, _ = layer_block(x_fwd, future=False)
            x_lag, _ = layer_block(x_lag, future=True)

        x_fwd = self.transformer.ln_f(x_fwd)
        x_lag = self.transformer.ln_f(x_lag)

        logits_fwd = self.lm_head(x_fwd)
        logits_lag = self.lm_head_lag(x_lag)

        loss = None
        if y is not None:
            loss_fwd = F.cross_entropy(
                logits_fwd.view(-1, V),
                y[:half].view(-1)
            )

            bwd_targets = x[half:, :-skip_dist]
            bwd_logits  = logits_lag[:, skip_dist:]
            loss_lag = F.cross_entropy(
                bwd_logits.reshape(-1, V),
                bwd_targets.reshape(-1)
            )

            loss = (1 - lam) * loss_fwd + lam * loss_lag

        return logits_fwd, logits_lag, loss

    @torch.no_grad()
    def prefill(self, prompt_tokens):
        self.eval()
        SL = len(prompt_tokens)
        x = torch.tensor(prompt_tokens, dtype=torch.long,
                          device=self.device).unsqueeze(0)

        pos = torch.arange(0, SL, dtype=torch.long, device=self.device)
        pos_emb = self.transformer.wpe(pos)
        h = self.transformer.wte(x) + pos_emb
        h = self.transformer.drop(h)

        kv_caches = []
        for layer_block in self.transformer.h:
            h, cache = layer_block(h, future=False, kv_cache=None)
            kv_caches.append(cache)

        h = self.transformer.ln_f(h)
        logits = self.lm_head(h)
        return logits, kv_caches

    @torch.no_grad()
    def decode_one_token(self, token, pos, kv_caches):
        self.eval()
        x = torch.tensor([[token]], dtype=torch.long, device=self.device)
        pos_emb = self.transformer.wpe(
            torch.tensor([pos], dtype=torch.long, device=self.device)
        )
        h = self.transformer.wte(x) + pos_emb
        h = self.transformer.drop(h)

        new_caches = []
        for layer_block, kv_cache in zip(self.transformer.h, kv_caches):
            h, cache = layer_block(h, future=False, kv_cache=kv_cache)
            new_caches.append(cache)

        h = self.transformer.ln_f(h)
        logits = self.lm_head(h)
        return logits, new_caches

    @torch.no_grad()
    def lag_correct(self, generated_tokens):
        self.eval()
        seq = torch.tensor(generated_tokens, dtype=torch.long,
                            device=self.device).unsqueeze(0)

        pos = torch.arange(0, seq.size(1), dtype=torch.long, device=self.device)
        pos_emb = self.transformer.wpe(pos)
        x_lag = self.transformer.wte(seq) + pos_emb

        for layer_block in self.transformer.h:
            x_lag, _ = layer_block(x_lag, future=True)

        x_lag   = self.transformer.ln_f(x_lag)
        logits_lag = self.lm_head_lag(x_lag)
        return logits_lag
    
class LayerBlock(nn.Module):
  def __init__(self, config: LagBehindConfig, device):
    super().__init__()
    self.ln_1 = nn.LayerNorm(config.embedding_dim)
    self.attn = AttentionMultiHeadFused(config, device)
    self.ln_2 = nn.LayerNorm(config.embedding_dim)
    self.mlp = MLP(config)
    

  def forward(self, x, future, kv_cache=None):
    attn_out, new_cache = self.attn(self.ln_1(x), future, kv_cache)
    x = x + attn_out
    x = x + self.mlp(self.ln_2(x))
    return x, new_cache


class MLP(nn.Module):
  def __init__(self, config: LagBehindConfig):
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
  def __init__(self, config: LagBehindConfig, device):
    super().__init__()
    self.config = config
    self.device = device
    self.w_qkv = nn.Linear(config.embedding_dim, 3*config.embedding_dim)
    self.output = nn.Linear(config.embedding_dim, config.embedding_dim)
    self.attn_drop = config.dropout
    self.resid_drop = nn.Dropout(config.dropout)

  def forward(self, x, future, kv_cache=None):
    B,SL,ED = x.size()
    qkv = self.w_qkv(x)
    q, k, v = qkv.split(self.config.embedding_dim, dim=2)
    q = q.view(B, SL, self.config.num_heads, ED // self.config.num_heads).transpose(1,2)
    k = k.view(B, SL, self.config.num_heads, ED // self.config.num_heads).transpose(1,2)
    v = v.view(B, SL, self.config.num_heads, ED // self.config.num_heads).transpose(1,2)

    mask = torch.full((SL, SL), float('-inf'), device=self.device)

    # if future:
    #   diagonal = self.config.lag_behind + 1
    #   mask.fill_diagonal_(float('-inf'))
    # else:
    #   diagonal = 1
    #mask = torch.triu(mask, diagonal=diagonal)
    dropout_p=self.attn_drop if self.training else 0.0
    if future:
      skip_dist = self.config.lag_behind
      mask = torch.tril(torch.ones(SL, SL, device=x.device)).bool()
      rows = torch.arange(skip_dist, SL, device=x.device)
      mask[rows, rows - skip_dist] = False
      mask = torch.where(mask, 0.0, float('-inf'))
      attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=dropout_p)

      new_cache = None
    else:
      if kv_cache is not None:
        k_cache, v_cache = kv_cache
        k = torch.cat([k_cache, k], dim=2)
        v = torch.cat([v_cache, v], dim=2)
        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=False, dropout_p=dropout_p)
      else:
        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=dropout_p)

      new_cache = (k, v) if not self.training else None


    attn_out = attn_out.transpose(1,2).contiguous().view(B, SL, ED)
    y = self.output(attn_out)
    y = self.resid_drop(y)
    return y, new_cache