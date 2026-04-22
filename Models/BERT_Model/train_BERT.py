import torch
import time
from torch.nn.utils import clip_grad_norm_


@torch.no_grad()
def estimate_loss(model, loader, device, eval_iters=10, lam=.5):
    model.eval()

    losses_fwd = []
    losses_bwd = []


    for i in range(eval_iters):
        x, y, _ = loader.get_data()
        x, y = x.to(device), y.to(device)
        loss_fwd, loss_bwd = model(x, y)
        losses_fwd.append(loss_fwd.item())
        losses_bwd.append(loss_bwd.item())
        del x, y, loss_fwd, loss_bwd

    avg_fwd = sum(losses_fwd) / len(losses_fwd)
    ppl_fwd = torch.exp(torch.tensor(avg_fwd)).item()
    avg_bwd = sum(losses_bwd) / len(losses_bwd)
    ppl_bwd = torch.exp(torch.tensor(avg_bwd)).item()

    model.train()
    return avg_fwd, avg_bwd, ppl_fwd, ppl_bwd



def train_loop(model, optimizer, scheduler, scaler, device, train_loader, val_loader,
               train_config, model_config):
    num_steps_train = train_config.num_steps_train
    num_steps_val = train_config.num_steps_val
    lam = train_config.lam
    grad_acc_factor = train_config.grad_acc_factor
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    history = {
        'train_steps': [],
        'train_loss':  [],
        'val_steps':   [],
        'val_loss_fwd':[],
        'val_loss_bwd':[],
        'val_ppl_fwd': [],
        'val_ppl_bwd': []
    }
    loss_fwd, loss_bwd, ppl_fwd, ppl_bwd = estimate_loss(model, val_loader, device, num_steps_val)
    history['val_steps'].append(0)
    history['val_loss_fwd'].append(loss_fwd)
    history['val_ppl_fwd'].append(ppl_fwd)
    history['val_loss_bwd'].append(loss_bwd)
    history['val_ppl_bwd'].append(ppl_bwd)

    print(f"Step    0 | Val FWD: {loss_fwd:.4f} | PPL_FWD: {ppl_fwd:.2f} | Val BWD: {loss_bwd:.4f}| PPL_BWD: {ppl_bwd:.2f}")
    start = time.time()
    tokens_seen = 0

    for step in range(num_steps_train):
        if step % 50 == 0 and step > 0:
            if device == 'mps':
                torch.mps.empty_cache()
            elif device == 'cuda':
              torch.cuda.empty_cache()
              torch.cuda.ipc_collect()
            loss_fwd, loss_bwd, ppl_fwd, ppl_bwd = estimate_loss(model, val_loader, device, num_steps_val)

            history['val_steps'].append(step)
            history['val_loss_fwd'].append(loss_fwd)
            history['val_ppl_fwd'].append(ppl_fwd)
            history['val_loss_bwd'].append(loss_bwd)
            history['val_ppl_bwd'].append(ppl_bwd)

            print(f"Step  {step}| Val FWD: {loss_fwd:.4f} | PPL_FWD: {ppl_fwd:.2f} | Val BWD: {loss_bwd:.4f}| PPL_BWD: {ppl_bwd:.2f}")
            start = time.time()


        avg_loss = 0
        optimizer.zero_grad(set_to_none=True)
        for i in range(grad_acc_factor):
            x, y, _ = train_loader.get_data()
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)


            with torch.autocast(device_type=device.type, dtype=torch.float16):
                loss_fwd, loss_bwd = model(x, y)
                loss_fwd /= grad_acc_factor
                loss_bwd /= grad_acc_factor

                step_loss = lam * loss_fwd + (1- lam) * loss_bwd
            scaler.scale(step_loss).backward()
            avg_loss += step_loss.detach().item()  # ← scalar for logging
            del x, y, loss_fwd, loss_bwd, step_loss

        scaler.unscale_(optimizer)          # unscale before clipping
        clip_grad_norm_(model.parameters(), max_norm=2.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        tokens_seen += train_config.batch_per_iter * \
                train_config.grad_acc_factor * \
                model_config.block_size
        if step % 10 == 0:
            stop = time.time()
            history['train_steps'].append(step)
            history['train_loss'].append(avg_loss)
            print(f"step {step:4d}/{num_steps_train} | tokens {tokens_seen:,} | Train loss {avg_loss:.4f} | lr {scheduler.get_last_lr()[0]:.2e} | Time Since Last Train Print {(stop-start):.4f} seconds")
            start = time.time()
    return history
