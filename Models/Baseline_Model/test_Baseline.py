import time
import torch

def evaluate_statistical_matrics(model, val_loader, device, max_batches):
  model.eval()

  total_correct = 0
  total_tokens = 0
  total_loss = 0.0
  num_batches = 0

  with torch.no_grad():
    for x, y in val_loader:
      #val_loader can grab data infinitely so max_batches should be set
      if num_batches >= max_batches:
          break

      x, y = x.to(device), y.to(device)
      logits, loss = model(x, y)

      #Track accuracy by checking the token with the highest probability
      preds = logits.argmax(dim=-1)

      #Ignore masks (not that there should be(?))
      mask = y != model.config.pad_token_id

      total_correct += (preds[mask] == y[mask]).sum().item()
      total_tokens +=  mask.sum().item()
      total_loss += loss.item()
      num_batches += 1

  accuracy = total_correct / total_tokens if total_tokens > 0 else 0.0
  avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

  return {
    "accuracy": accuracy,
    "avg_loss": avg_loss,
    "PPL": torch.exp(torch.tensor(avg_loss)).item(),
    "tokens_seen": total_tokens,
  }


def evaluate_system_metrics(model, device, prompt,
                                    gen_lengths=[32, 64, 128, 256, 512], num_runs=5):
  results = {}

  for max_new_tokens in gen_lengths:
    latencies = []

    #Warmup the GPU
    for _ in range(2):
      model.infer(prompt, max_new_tokens=max_new_tokens)

    if device == "cuda":
      torch.cuda.synchronize()
    for _ in range(num_runs):
      if device == "cuda":
        torch.cuda.synchronize()
      start = time.perf_counter()
      model.infer(prompt, max_new_tokens=max_new_tokens)
      if device == "cuda":
        torch.cuda.synchronize()
      end = time.perf_counter()
      latencies.append(end - start)

    avg_latency = sum(latencies) / len(latencies)
    results[max_new_tokens] = {
      "throughput_tokens_per_sec": max_new_tokens / avg_latency,
      "avg_latency_ms": avg_latency * 1000,
      "ms_per_token": (avg_latency * 1000) / max_new_tokens,
    }

  return results
