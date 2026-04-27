import torch
import time
from torch.nn.utils import clip_grad_norm_


@torch.no_grad()
def estimate_loss(model, loader, device, eval_iters=10, lam=.5):
    model.eval()

    losses = []

    for _ in range(eval_iters):
        x, y, _ = loader.get_data()
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device.type, dtype=torch.float16):
            _, loss = model(x, y)
        losses.append(loss.item())

    model.train()

    avg = sum(losses) / len(losses)
    ppl = torch.exp(torch.tensor(avg)).item()

    return avg, ppl



def train_loop(model, optimizer, scheduler, scaler, device, train_loader, val_loader,
               train_config, model_config):
    
    num_steps_train = train_config.num_steps_train
    num_steps_val = train_config.num_steps_val
    grad_acc_factor = train_config.grad_acc_factor
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    loss, ppl= estimate_loss(model, val_loader, device, num_steps_val)
    print(f"Step    0 | Val Loss: {loss:.4f} | PPL: {ppl:.2f}")
    start = time.time()
    tokens_seen = 0

    #Define some useful constants
    for step in range(num_steps_train):
        if step % 100 == 0 and step > 0:
            loss, ppl = estimate_loss(model, val_loader, device, num_steps_val)
            print(f"Step {step:4d} | Val Loss: {loss:.4f} | PPL: {ppl:.2f}")
            if device == 'mps':
                torch.mps.empty_cache()


        avg_loss = 0
        optimizer.zero_grad(set_to_none=True)
        for _ in range(grad_acc_factor):
            x, y, _ = train_loader.get_data()
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)


            with torch.autocast(device_type=device.type, dtype=torch.float16):
                _, loss = model(x, y)
                loss /= grad_acc_factor
                avg_loss += loss.item()

            scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        tokens_seen += train_config.batch_per_iter * \
                train_config.grad_acc_factor * \
                model_config.block_size
        if step % 10 == 0:
            stop = time.time()
            print(f"step {step:4d}/{num_steps_train} | tokens {tokens_seen:,} | Train loss {avg_loss:.4f} | Time Since Last Train Print {(stop-start):.4f} seconds")
            start = time.time()

        if(step % 1000 == 0 and step > 0):
            path = "/content/gdrive/MyDrive/Junior/Second Semester/CS 6787 - Advanced ML Systems/regen-gpt2/ckpt_baseline_step_{step}.pt".format(step=step)
            torch.save({
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": avg_loss,
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict()
        }, path)

