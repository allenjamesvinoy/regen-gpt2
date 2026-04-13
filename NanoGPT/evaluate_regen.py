"""
evaluate_regen.py - Evaluate baseline vs geometric regeneration

Place at: nanoGPT/evaluate_regen.py

This script measures next-token prediction accuracy under two conditions:
1. Baseline: standard autoregressive generation (model consumes its own outputs)
2. Regeneration: same, but with geometric regeneration passes at boundaries

Both use FREE GENERATION (not teacher-forcing), because the hypothesis
that regeneration improves downstream predictions only makes sense when
the model consumes its own (potentially improved) outputs.

Usage:
    python evaluate_regen.py --out_dir=out-simplewiki

The script loads the best checkpoint from out_dir and runs both evaluations
on the validation set.
"""

import os
import argparse
import numpy as np
import torch
import tiktoken
from contextlib import nullcontext

# These will be imported after we cd into nanoGPT
# from model import GPTConfig, GPT


def build_regen_mask(seq_len, regen_start, regen_end, device):
    """
    Build a custom attention mask for regeneration.

    During regeneration of tokens[regen_start:regen_end], the mask allows:
    - Standard causal attention within the regen window
      (token i attends to regen_start..i-1)
    - Full attention to ALL tokens outside the regen window
      (both before regen_start and after regen_end, i.e. the forward context)

    Tokens outside the regen window keep standard causal attention.

    Args:
        seq_len: total sequence length
        regen_start: start index of regeneration block (inclusive)
        regen_end: end index of regeneration block (exclusive)
        device: torch device

    Returns:
        mask: (1, 1, seq_len, seq_len) attention mask. 1=attend, 0=mask.
    """
    # Start with standard causal mask
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device))

    # For tokens in the regen window, allow attending to forward context
    for i in range(regen_start, regen_end):
        # Attend to everything before regen_start (backward context, already causal)
        # Already handled by causal mask

        # Attend to other regen tokens before this one (causal within block)
        # Already handled by causal mask

        # Attend to forward context: regen_end .. seq_len
        # This is the KEY modification: future tokens become visible
        mask[i, regen_end:seq_len] = 1.0

    return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, T, T)


def get_geometric_regen_schedule(total_tokens, block_size=32):
    """
    Compute the geometric regeneration schedule.

    After generating N tokens, regenerate a block of 32 tokens that
    is geometrically spaced:
        - At 64 tokens:  regenerate [0, 32)   using forward context [32, 64)
        - At 128 tokens: regenerate [32, 64)  using forward context [64, 128)
        - At 256 tokens: regenerate [64, 128) using forward context [128, 256)
        - At 512 tokens: regenerate [128, 256) using forward context [256, 512)
        - etc.

    Returns list of (trigger_point, regen_start, regen_end) tuples.
    """
    schedule = []
    trigger = block_size * 2  # first trigger at 64

    regen_start = 0
    while trigger <= total_tokens:
        regen_end = regen_start + block_size
        schedule.append((trigger, regen_start, regen_end))
        regen_start = regen_end
        trigger *= 2

    return schedule


@torch.no_grad()
def generate_baseline(model, prompt_tokens, num_generate, device, ctx):
    """
    Standard autoregressive generation. Model consumes its own outputs.

    Returns:
        generated: tensor of all generated token ids (excluding prompt)
    """
    idx = prompt_tokens.clone()  # (1, prompt_len)

    for _ in range(num_generate):
        # Crop to block_size if needed
        idx_cond = idx if idx.size(1) <= model.config.block_size else idx[:, -model.config.block_size:]
        with ctx:
            logits, _ = model(idx_cond)
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)  # greedy
        idx = torch.cat([idx, next_token], dim=1)

    # Return only the generated part (exclude prompt)
    return idx[:, prompt_tokens.size(1):]


@torch.no_grad()
def generate_with_regen(model, prompt_tokens, num_generate, device, ctx, block_size=32):
    """
    Generation with geometric regeneration.

    Same as baseline, but at geometric intervals, we go back and regenerate
    a block of tokens using a custom attention mask that provides forward context.

    Returns:
        generated: tensor of all generated token ids (excluding prompt)
        regen_boundaries: list of positions where regeneration occurred
    """
    prompt_len = prompt_tokens.size(1)
    idx = prompt_tokens.clone()  # (1, prompt_len)

    # First, generate all tokens (initial pass)
    for _ in range(num_generate):
        idx_cond = idx if idx.size(1) <= model.config.block_size else idx[:, -model.config.block_size:]
        with ctx:
            logits, _ = model(idx_cond)
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        idx = torch.cat([idx, next_token], dim=1)

    # Now apply geometric regeneration on the generated portion
    gen_len = num_generate
    schedule = get_geometric_regen_schedule(gen_len, block_size)
    regen_boundaries = []

    for trigger, regen_start_gen, regen_end_gen in schedule:
        # Convert from generation-relative to absolute positions
        regen_start = prompt_len + regen_start_gen
        regen_end = prompt_len + regen_end_gen

        # Make sure we have enough tokens
        if regen_end > idx.size(1):
            break

        regen_boundaries.append(regen_end_gen)

        # Determine the context window for regeneration
        # We need: everything up to some point after regen_end for forward context
        # Use up to block_size tokens total, centered around the regen window
        context_end = min(prompt_len + trigger, idx.size(1))
        context_start = max(0, context_end - model.config.block_size)
        context = idx[:, context_start:context_end].clone()

        # Adjust regen indices relative to the context window
        local_regen_start = regen_start - context_start
        local_regen_end = regen_end - context_start
        local_seq_len = context.size(1)

        # Build the custom attention mask
        mask = build_regen_mask(
            seq_len=local_seq_len,
            regen_start=local_regen_start,
            regen_end=local_regen_end,
            device=device,
        )

        # Regenerate tokens in the block, left to right
        for pos in range(local_regen_start, local_regen_end):
            with ctx:
                logits, _ = model(context, custom_mask=mask)
            new_token = logits[:, pos, :].argmax(dim=-1)
            context[:, pos] = new_token

        # Write regenerated tokens back into the full sequence
        idx[:, regen_start:regen_end] = context[:, local_regen_start:local_regen_end]

    # Return only the generated part
    return idx[:, prompt_len:], regen_boundaries


def compute_accuracy(generated_tokens, ground_truth_tokens):
    """
    Compute token-level accuracy between generated and ground truth.

    Args:
        generated_tokens: (num_tokens,) tensor
        ground_truth_tokens: (num_tokens,) tensor

    Returns:
        accuracy as a float
    """
    assert generated_tokens.shape == ground_truth_tokens.shape
    correct = (generated_tokens == ground_truth_tokens).sum().item()
    total = ground_truth_tokens.numel()
    return correct / total if total > 0 else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="out-simplewiki")
    parser.add_argument("--data_dir", type=str, default="data/simplewiki")
    parser.add_argument("--num_samples", type=int, default=500,
                        help="Number of sequences to evaluate")
    parser.add_argument("--num_generate", type=int, default=256,
                        help="Tokens to generate per sample. Must be <= block_size.")
    parser.add_argument("--prompt_len", type=int, default=32,
                        help="Number of ground-truth tokens to use as prompt")
    parser.add_argument("--regen_block", type=int, default=32,
                        help="Regeneration block size")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Import model after path is set
    from model import GPTConfig, GPT

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = args.device
    dtype = "bfloat16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "float16"
    ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
    ctx = nullcontext() if device == "cpu" else torch.amp.autocast(device_type=device, dtype=ptdtype)

    # Load model
    print(f"Loading model from {args.out_dir} ...")
    ckpt_path = os.path.join(args.out_dir, "ckpt.pt")
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint["model_args"])
    model = GPT(gptconf)
    state_dict = checkpoint["model"]
    # Fix key names (remove _orig_mod. prefix from torch.compile)
    unwanted_prefix = "_orig_mod."
    for k in list(state_dict.keys()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print(f"Model loaded: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters")

    # Load validation data
    val_data = np.memmap(
        os.path.join(args.data_dir, "val.bin"), dtype=np.uint16, mode="r"
    )
    print(f"Validation data: {len(val_data):,} tokens")

    total_seq_len = args.prompt_len + args.num_generate

    # Sanity check
    assert total_seq_len <= gptconf.block_size, \
        f"prompt_len + num_generate ({total_seq_len}) exceeds block_size ({gptconf.block_size})"
    assert total_seq_len < len(val_data), \
        f"Sequence length exceeds validation data size"

    # =========================================================================
    # Run evaluation
    # =========================================================================

    # Get geometric schedule for reporting
    schedule = get_geometric_regen_schedule(args.num_generate, args.regen_block)
    print(f"\nRegeneration schedule (relative to generation start):")
    for trigger, rs, re_ in schedule:
        print(f"  At token {trigger}: regenerate [{rs}, {re_})")

    # Track accuracies
    baseline_accs = []
    regen_accs = []

    # Per-region tracking: accuracy after each regen boundary
    baseline_post_regen_accs = {re_: [] for _, _, re_ in schedule}
    regen_post_regen_accs = {re_: [] for _, _, re_ in schedule}

    print(f"\nEvaluating {args.num_samples} samples ...")
    print(f"  Prompt length: {args.prompt_len}")
    print(f"  Generation length: {args.num_generate}")
    print(f"  Regen block size: {args.regen_block}")
    print("-" * 60)

    for i in range(args.num_samples):
        # Pick a random starting point
        start_idx = np.random.randint(0, len(val_data) - total_seq_len - 1)
        ground_truth = torch.from_numpy(
            val_data[start_idx:start_idx + total_seq_len].astype(np.int64)
        ).to(device)

        prompt = ground_truth[:args.prompt_len].unsqueeze(0)  # (1, prompt_len)
        gt_generated = ground_truth[args.prompt_len:]  # (num_generate,)

        # Baseline: standard generation
        baseline_gen = generate_baseline(
            model, prompt, args.num_generate, device, ctx
        )
        baseline_gen = baseline_gen.squeeze(0)  # (num_generate,)

        # Regeneration: generation with geometric regen
        regen_gen, boundaries = generate_with_regen(
            model, prompt, args.num_generate, device, ctx,
            block_size=args.regen_block
        )
        regen_gen = regen_gen.squeeze(0)  # (num_generate,)

        # Overall accuracy
        baseline_acc = compute_accuracy(baseline_gen, gt_generated)
        regen_acc = compute_accuracy(regen_gen, gt_generated)
        baseline_accs.append(baseline_acc)
        regen_accs.append(regen_acc)

        # Per-region accuracy: tokens AFTER each regeneration boundary
        for _, _, re_ in schedule:
            if re_ < args.num_generate:
                region_gt = gt_generated[re_:]
                bl_region = baseline_gen[re_:]
                rg_region = regen_gen[re_:]
                baseline_post_regen_accs[re_].append(
                    compute_accuracy(bl_region, region_gt)
                )
                regen_post_regen_accs[re_].append(
                    compute_accuracy(rg_region, region_gt)
                )

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{args.num_samples}] "
                  f"baseline={np.mean(baseline_accs[-50:]):.4f} "
                  f"regen={np.mean(regen_accs[-50:]):.4f}")

    # =========================================================================
    # Report results
    # =========================================================================

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    bl_mean = np.mean(baseline_accs)
    bl_std = np.std(baseline_accs)
    rg_mean = np.mean(regen_accs)
    rg_std = np.std(regen_accs)
    delta = rg_mean - bl_mean

    print(f"\nOverall next-token accuracy (free generation):")
    print(f"  Baseline:      {bl_mean:.4f} +/- {bl_std:.4f}")
    print(f"  Regeneration:  {rg_mean:.4f} +/- {rg_std:.4f}")
    print(f"  Delta:         {delta:+.4f}")

    # Statistical significance (paired t-test)
    from scipy import stats
    t_stat, p_val = stats.ttest_rel(regen_accs, baseline_accs)
    print(f"  Paired t-test: t={t_stat:.3f}, p={p_val:.6f}")
    if p_val < 0.05:
        print(f"  -> Statistically significant (p < 0.05)")
    else:
        print(f"  -> NOT statistically significant (p >= 0.05)")

    print(f"\nAccuracy on tokens AFTER each regeneration boundary:")
    for _, rs, re_ in schedule:
        if re_ in baseline_post_regen_accs and len(baseline_post_regen_accs[re_]) > 0:
            bl_r = np.mean(baseline_post_regen_accs[re_])
            rg_r = np.mean(regen_post_regen_accs[re_])
            d = rg_r - bl_r
            print(f"  After regen [{rs},{re_}): "
                  f"baseline={bl_r:.4f}, regen={rg_r:.4f}, delta={d:+.4f}")

    # Save results
    results = {
        "baseline_accs": baseline_accs,
        "regen_accs": regen_accs,
        "baseline_post_regen_accs": {k: v for k, v in baseline_post_regen_accs.items()},
        "regen_post_regen_accs": {k: v for k, v in regen_post_regen_accs.items()},
        "args": vars(args),
        "schedule": schedule,
    }
    results_path = os.path.join(args.out_dir, "regen_eval_results.pt")
    torch.save(results, results_path)
    print(f"\nDetailed results saved to {results_path}")


if __name__ == "__main__":
    main()
