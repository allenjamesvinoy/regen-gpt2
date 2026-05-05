from datetime import datetime
from torch.amp import autocast
import torch
import torch.nn.functional as F

def save_checkpoint(model, optimizer, step, loss, path):
  torch.save({
    'step': step,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
  }, path)

def load_checkpoint(model, optimizer, path):
  checkpoint = torch.load(path)
  model.load_state_dict(checkpoint['model_state_dict'])
  optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
  return checkpoint['step'], checkpoint['loss']


@torch.no_grad()
def estimate_loss(model, loader, device, eval_iters=10):
  model.eval()
  fwd_losses = torch.zeros(eval_iters)
  bwd_losses = torch.zeros(eval_iters)
  loader.cursor = 0

  for k in range(eval_iters):
    x, y = loader.get_data()
    x, y = x.to(device), y.to(device)

    logits_fwd, logits_lag, _ = model(x, y)

    V = model.config.vocab_size
    skip_dist = model.config.lag_behind
    half = x.size(0) // 2

    fwd_losses[k] = F.cross_entropy(
      logits_fwd.view(-1, V),
      y[:half].view(-1)
    ).item()

    bwd_targets = x[half:, :-skip_dist]
    bwd_logits = logits_lag[:, skip_dist:]
    bwd_losses[k] = F.cross_entropy(
      bwd_logits.reshape(-1, V),
      bwd_targets.reshape(-1)
    ).item()

  model.train()
  avg_fwd = fwd_losses.mean().item()
  avg_bwd = bwd_losses.mean().item()
  ppl = torch.exp(torch.tensor(avg_fwd)).item()
  return avg_fwd, ppl, avg_bwd



def train_loop(model, optimizer, scheduler, scaler, device, train_loader, val_loader,
               train_config, model_config, lam=0.5, start_step=0, save_path = None):
  if save_path is None:
    save_path = f'/content/drive/MyDrive/checkpoint_step{train_config.num_steps_train}_{datetime.now():%Y%m%d_%H%M%S}.pt'
  tokens_per_step = train_loader.B * train_loader.SL * train_config.grad_acc_factor
  num_steps_train = train_config.num_steps_train
  num_steps_val = train_config.num_steps_val
  grad_acc_factor = train_config.grad_acc_factor
  print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

  history = {
    'train_steps': [],
    'train_loss':  [],
    'val_steps':   [],
    'val_loss':    [],
    'val_ppl':     [],
    'val_bwd':     []
  }

  fwd_loss, ppl, bwd_loss = estimate_loss(model, val_loader, device, num_steps_val)
  print(f"Step {start_step:4d} | Val FWD: {fwd_loss:.4f} | PPL: {ppl:.2f} | Val BWD: {bwd_loss:.4f}")
  history['val_steps'].append(start_step)
  history['val_loss'].append(fwd_loss)
  history['val_ppl'].append(ppl)
  history['val_bwd'].append(bwd_loss)

  tokens_seen = start_step * tokens_per_step
  final_loss = None
  for step in range(start_step, num_steps_train):

    if step % 100 == 0 and step > start_step:
      fwd_loss, ppl, bwd_loss = estimate_loss(model, val_loader, device, num_steps_val)
      print(f"Step {step:4d} | Val FWD: {fwd_loss:.4f} | PPL: {ppl:.2f} | Val BWD: {bwd_loss:.4f}")
      history['val_steps'].append(step)
      history['val_loss'].append(fwd_loss)
      history['val_ppl'].append(ppl)
      history['val_bwd'].append(bwd_loss)

    lr = train_config.get_lr(step)
    for param_group in optimizer.param_groups:
      param_group['lr'] = lr

    optimizer.zero_grad()
    loss_accum = 0.0
    for _ in range(train_config.grad_acc_factor):
      x, y = train_loader.get_data()
      x, y = x.to(device), y.to(device)

      with autocast(device_type=device.type):
          _, _, loss = model(x, y, lam=lam)
      scaler.scale(loss / train_config.grad_acc_factor).backward()
      loss_accum += loss.item()

    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    scaler.step(optimizer)
    scaler.update()
    scheduler.step()
    avg_loss = loss_accum / train_config.grad_acc_factor
    tokens_seen += tokens_per_step
    if step % 10 == 0:
      history['train_steps'].append(step)
      history['train_loss'].append(avg_loss)
      print(f"step {step:4d} | tokens {tokens_seen:,} | lr {lr:.2e} | train loss {avg_loss:.4f}")

    if step % 1000 == 0 and step > start_step:
      save_checkpoint(model, optimizer, step, loss, save_path)

    final_loss = loss.item()

  save_checkpoint(model, optimizer, train_config.num_steps_train, final_loss, save_path)

  return history