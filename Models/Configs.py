from dataclasses import dataclass
import math
import torch

@dataclass
class TrainConfig:
  batch_per_iter : int = 32
  grad_acc_factor : int = 16
  warmup_steps : int = 150
  num_steps_train : int = 3623
  num_steps_val : int = 32
  max_lr : float = 3e-4
  min_lr : float = 3e-5
  weight_decay : float = .1
  betas : tuple[float, float] = (.9,.95)
  lam : float = .5

  def get_lr(self, step):
    # warmup
    if step < self.warmup_steps:
        return self.max_lr * step / self.warmup_steps
    # cosine decay to min_lr
    progress = (step - self.warmup_steps) / \
               (self.num_steps_train - self.warmup_steps)
    cosine   = 0.5 * (1 + math.cos(math.pi * progress))
    return self.min_lr + cosine * (self.max_lr - self.min_lr)

  def make_optimizer(self, model):
    return torch.optim.AdamW(
      model.parameters(),
      betas=self.betas,
      lr=self.max_lr,
      weight_decay=self.weight_decay
    )
  
  def make_scheduler(self, optimizer):
    return torch.optim.lr_scheduler.LambdaLR(
      optimizer, lambda step: self.get_lr(step) / self.max_lr
    )
  
  def make_scaler(self):
    return torch.amp.GradScaler('cuda')



@dataclass
class BERTConfig:
  num_heads: int = 12
  num_layers: int = 12
  vocab_size: int = 50257
  embedding_dim: int = 768
  block_size: int = 512
  dropout : float = .1
  pad_token_id : int = 50256
  range_low : float = .5
  range_high : float = .75

