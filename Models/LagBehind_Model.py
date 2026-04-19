import torch
import torch.nn as nn
from torch import dropout, embedding
from dataclasses import dataclass
import torch.nn.functional as F

@dataclass
class LagBehindGPTConfig:
    num_heads:     int = 6
    num_layers:    int = 6
    vocab_size:    int = 50257
    embedding_dim: int = 768
    block_size:    int = 1024
    lag_behind:    int = 1
    dropout:       float = .1
    pad_token_id:  int = 50256


@dataclass
class GPTConfig:
    num_heads:     int   = 6
    num_layers:    int   = 6
    vocab_size:    int   = 50257
    embedding_dim: int   = 768
    block_size:    int   = 1024
    lag_behind:    int   = 1
    dropout:       float = 0.1
    pad_token_id:  int   = 50256


class GPT2_Lag(nn.Module):
    def __init__(self, config: GPTConfig, device):
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

    def forward(self, x, y=None, lam=0.5):
        B, SL = x.size()
        assert B % 2 == 0, "Batch size must be even for batch splitting"
        V         = self.config.vocab_size
        skip_dist = self.config.lag_behind
        half      = B // 2

        pos     = torch.arange(0, SL, dtype=torch.long, device=self.device)
        pos_emb = self.transformer.wpe(pos)

        # split batch before entering transformer
        x_fwd = self.transformer.wte(x[:half]) + pos_emb  # causal half
        x_lag = self.transformer.wte(x[half:]) + pos_emb  # punch-out half

        x_fwd = self.transformer.drop(x_fwd)
        x_lag = self.transformer.drop(x_lag)

        # separate passes through shared layers with different masks
        for layer_block in self.transformer.h:
            x_fwd = layer_block(x_fwd, future=False)  # causal mask
            x_lag = layer_block(x_lag, future=True)   # punch-out mask

        x_fwd = self.transformer.ln_f(x_fwd)
        x_lag = self.transformer.ln_f(x_lag)

        logits_fwd = self.lm_head(x_fwd)
        logits_lag = self.lm_head_lag(x_lag)

        loss = None
        if y is not None:
            # forward loss: predict next token on causal half
            loss_fwd = F.cross_entropy(
                logits_fwd.view(-1, V),
                y[:half].view(-1)
            )

            # lag loss: position i predicts token at i - skip_dist
            # targets come from x (input), not y
            bwd_targets = x[half:, :-skip_dist]           # tokens 0..SL-2
            bwd_logits  = logits_lag[:, skip_dist:]        # logits at positions 1..SL-1
            loss_lag = F.cross_entropy(
                bwd_logits.reshape(-1, V),
                bwd_targets.reshape(-1)
            )

            loss = (1 - lam) * loss_fwd + lam * loss_lag

        return logits_fwd, logits_lag, loss

class LayerBlock(nn.Module):
  def __init__(self, config: LagBehindGPTConfig, device):
    super().__init__()
    self.ln_1 = nn.LayerNorm(config.embedding_dim)
    self.attn = AttentionMultiHeadFused(config, device)
    self.ln_2 = nn.LayerNorm(config.embedding_dim)
    self.mlp = MLP(config)
    

  def forward(self, x, future):
    x = x + self.attn(self.ln_1(x), future)
    x = x + self.mlp(self.ln_2(x)) #why do we layer-norm before passing mlp?
    return x


class MLP(nn.Module):
  def __init__(self, config: LagBehindGPTConfig):
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
  def __init__(self, config: LagBehindGPTConfig, device):
    super().__init__()
    self.config = config
    self.device = device
    self.w_qkv = nn.Linear(config.embedding_dim, 3*config.embedding_dim)
    self.output = nn.Linear(config.embedding_dim, config.embedding_dim)
    self.attn_drop = config.dropout  # passed to scaled_dot_product_attention
    self.resid_drop = nn.Dropout(config.dropout)  # add this

  def forward(self, x, future):
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
      # Punch out t_{i - skip_dist} for each row i >= skip_dist
      rows = torch.arange(skip_dist, SL, device=x.device)
      mask[rows, rows - skip_dist] = False
      mask = torch.where(mask, 0.0, float('-inf'))
      attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=dropout_p)
    else:
      attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=dropout_p)


    attn_out = attn_out.transpose(1,2).contiguous().view(B, SL, ED)
    y = self.output(attn_out)
    y = self.resid_drop(y)
    return y