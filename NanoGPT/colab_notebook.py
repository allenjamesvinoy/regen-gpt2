# =============================================================================
# Geometric Regeneration Experiment - Colab Pro Notebook
# =============================================================================
# 
# This notebook runs the full experiment:
# 1. Setup nanoGPT + dependencies
# 2. Prepare SimpleWiki data
# 3. Patch model.py for custom attention masks
# 4. Train GPT-2 124M baseline on SimpleWiki
# 5. Evaluate: baseline vs geometric regeneration
#
# Runtime: A100 GPU recommended (Colab Pro)
# Total time estimate: ~4-6 hours (3-4h training + 1-2h evaluation)
# =============================================================================


# ============================================================
# CELL 1: Setup
# ============================================================

# !git clone https://github.com/karpathy/nanoGPT.git
# %cd nanoGPT
# !pip install torch numpy transformers datasets tiktoken tqdm scipy


# ============================================================
# CELL 2: Upload experiment files
# ============================================================

# Upload these files to the nanoGPT directory:
#   - prepare_simplewiki.py  -> data/simplewiki/prepare.py
#   - train_simplewiki.py    -> config/train_simplewiki.py
#   - model_patch.py         -> model_patch.py
#   - evaluate_regen.py      -> evaluate_regen.py
#
# You can do this manually via Colab's file browser, or:

# !mkdir -p data/simplewiki config
# Upload files, then:
# !mv prepare_simplewiki.py data/simplewiki/prepare.py
# !mv train_simplewiki.py config/train_simplewiki.py


# ============================================================
# CELL 3: Prepare SimpleWiki data
# ============================================================

# This downloads ~200MB compressed, extracts and tokenizes articles.
# Takes about 5-10 minutes.

# !python data/simplewiki/prepare.py


# ============================================================
# CELL 4: Patch model.py for custom attention masks
# ============================================================

# This adds custom_mask support to nanoGPT's model.
# Original is backed up as model_original.py.

# !python model_patch.py


# ============================================================
# CELL 5: Verify the patch worked
# ============================================================

"""
import torch
from model import GPTConfig, GPT

# Quick smoke test: create model and forward pass with custom mask
config = GPTConfig(
    n_layer=2, n_head=2, n_embd=64,
    block_size=32, vocab_size=50257, dropout=0.0, bias=True
)
model = GPT(config).to("cuda")

x = torch.randint(0, 50257, (1, 16), device="cuda")

# Normal forward (should work as before)
logits1, _ = model(x)
print(f"Normal forward: logits shape = {logits1.shape}")

# Forward with custom mask (causal mask as custom to verify equivalence)
T = 16
causal = torch.tril(torch.ones(T, T, device="cuda")).unsqueeze(0).unsqueeze(0)
logits2, _ = model(x, custom_mask=causal)
print(f"Custom mask forward: logits shape = {logits2.shape}")

# They should be identical (both are causal)
diff = (logits1 - logits2).abs().max().item()
print(f"Max difference (should be ~0): {diff:.8f}")

print("Patch verification PASSED!" if diff < 1e-4 else "Patch verification FAILED!")
"""


# ============================================================
# CELL 6: Check GPU and adjust config if needed
# ============================================================

"""
import torch
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

# If V100 (16GB): edit config/train_simplewiki.py:
#   batch_size = 8
#   gradient_accumulation_steps = 16

# If T4 (16GB): edit config/train_simplewiki.py:
#   batch_size = 4
#   gradient_accumulation_steps = 32
#   max_iters = 15000
#   lr_decay_iters = 15000
"""


# ============================================================
# CELL 7: Train the model
# ============================================================

# This is the longest step. Monitor the loss to make sure it's decreasing.
# Expected: loss should drop from ~11 to ~3-4 over 30K iterations.

# !python train.py config/train_simplewiki.py


# ============================================================
# CELL 8: Quick sanity check - generate text
# ============================================================

"""
import torch
import tiktoken
from model import GPTConfig, GPT

# Load best checkpoint
ckpt = torch.load("out-simplewiki/ckpt.pt", map_location="cuda")
config = GPTConfig(**ckpt["model_args"])
model = GPT(config)
state_dict = ckpt["model"]
for k in list(state_dict.keys()):
    if k.startswith("_orig_mod."):
        state_dict[k[len("_orig_mod."):]] = state_dict.pop(k)
model.load_state_dict(state_dict)
model.to("cuda")
model.eval()

enc = tiktoken.get_encoding("gpt2")
prompt = "The history of mathematics"
tokens = enc.encode(prompt)
x = torch.tensor([tokens], dtype=torch.long, device="cuda")

with torch.no_grad():
    out = model.generate(x, max_new_tokens=100, temperature=0.8, top_k=40)
print(enc.decode(out[0].tolist()))
"""


# ============================================================
# CELL 9: Run the regeneration evaluation
# ============================================================

# This compares baseline vs regeneration accuracy.
# Takes ~1-2 hours depending on num_samples.

# !python evaluate_regen.py \
#     --out_dir=out-simplewiki \
#     --data_dir=data/simplewiki \
#     --num_samples=500 \
#     --num_generate=256 \
#     --prompt_len=32 \
#     --regen_block=32


# ============================================================
# CELL 10: Visualize results
# ============================================================

"""
import torch
import numpy as np
import matplotlib.pyplot as plt

results = torch.load("out-simplewiki/regen_eval_results.pt")

baseline_accs = results["baseline_accs"]
regen_accs = results["regen_accs"]

# Histogram of per-sample accuracy differences
diffs = [r - b for r, b in zip(regen_accs, baseline_accs)]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Distribution of accuracy differences
axes[0].hist(diffs, bins=50, edgecolor="black", alpha=0.7)
axes[0].axvline(0, color="red", linestyle="--", label="No improvement")
axes[0].axvline(np.mean(diffs), color="green", linestyle="--",
                label=f"Mean: {np.mean(diffs):+.4f}")
axes[0].set_xlabel("Accuracy difference (regen - baseline)")
axes[0].set_ylabel("Count")
axes[0].set_title("Per-sample accuracy improvement from regeneration")
axes[0].legend()

# Plot 2: Accuracy by region (after each regen boundary)
schedule = results["schedule"]
bl_post = results["baseline_post_regen_accs"]
rg_post = results["regen_post_regen_accs"]

regions = []
bl_means = []
rg_means = []
for _, rs, re_ in schedule:
    if re_ in bl_post and len(bl_post[re_]) > 0:
        regions.append(f"After [{rs},{re_})")
        bl_means.append(np.mean(bl_post[re_]))
        rg_means.append(np.mean(rg_post[re_]))

x_pos = np.arange(len(regions))
width = 0.35
axes[1].bar(x_pos - width/2, bl_means, width, label="Baseline", alpha=0.7)
axes[1].bar(x_pos + width/2, rg_means, width, label="Regeneration", alpha=0.7)
axes[1].set_xlabel("Region")
axes[1].set_ylabel("Accuracy")
axes[1].set_title("Accuracy by region after regeneration boundaries")
axes[1].set_xticks(x_pos)
axes[1].set_xticklabels(regions, rotation=30, ha="right")
axes[1].legend()

plt.tight_layout()
plt.savefig("regen_results.png", dpi=150)
plt.show()
print("Saved figure to regen_results.png")
"""
