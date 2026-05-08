import time
import torch
import torch.nn.functional as F

@torch.no_grad()
def evaluate_next_token_correction(vanilla_model, lag_model, loader, device,
                                   prompt_len=50, num_prompts=100,
                                   temperature=1.0, top_k=50, both_lag=False):
    vanilla_model.eval()
    lag_model.eval()

    van_correct = 0
    lag_correct = 0
    loader.idx = 0

    def sample(logits):
        if top_k is not None:
            threshold = torch.topk(logits, min(top_k, logits.size(-1))).values[-1]
            logits[logits < threshold] = float('-inf')
        return torch.multinomial(F.softmax(logits / temperature, dim=-1), 1).item()

    for _ in range(num_prompts):
        x, _ = loader.get_data()
        seq = x[0].tolist()

        k = torch.randint(1, prompt_len, (1,)).item()
        target_van_token = seq[k]
        target_lag_token = seq[k - 1]

        context = torch.tensor(seq[:k], dtype=torch.long, device=device).unsqueeze(0)

        # --- Vanilla predicts seq[k] from seq[:k] ---
        if both_lag:
            logits, _, _ = vanilla_model(context.repeat(2, 1), context.repeat(2, 1))
        else:
            logits, _ = vanilla_model(context, context)
        van_pred = sample(logits[0, -1, :].clone())
        van_correct += int(van_pred == target_van_token)

        # --- Lag predicts seq[k-1] using seq[:k] + vanilla's (imperfect) prediction ---
        full = torch.tensor(seq[:k] + [van_pred], dtype=torch.long, device=device).unsqueeze(0)
        _, logits_lag, _ = lag_model(full.repeat(2, 1))
        lag_pred = sample(logits_lag[0, -1, :].clone())
        lag_correct += int(lag_pred == target_lag_token)

    print(f"Vanilla accuracy:  {van_correct}/{num_prompts} = {van_correct/num_prompts:.4f}")
    print(f"Lag accuracy:      {lag_correct}/{num_prompts} = {lag_correct/num_prompts:.4f}")

    return van_correct, lag_correct