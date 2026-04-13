# Geometric Regeneration Experiment with nanoGPT

## Hypothesis

Revisiting earlier tokens using forward and backward context at geometrically
spaced intervals (multiples of 32) improves downstream next-token prediction
accuracy compared to standard single-pass autoregressive generation.

## Regeneration Schedule

After generating N tokens, regenerate a 32-token block using a custom attention
mask that allows attending to future context:

| Trigger | Regenerate | Forward Context |
|---------|-----------|-----------------|
| 64 tokens | [0, 32) | [32, 64) |
| 128 tokens | [32, 64) | [64, 128) |
| 256 tokens | [64, 128) | [128, 256) |

Regeneration is left-to-right within each block. The custom attention mask
allows each regenerated token to see both its causal past AND the forward
context window.

## Files

| File | Place at (in nanoGPT/) | Purpose |
|------|----------------------|---------|
| `prepare_simplewiki.py` | `data/simplewiki/prepare.py` | Download + tokenize SimpleWiki |
| `train_simplewiki.py` | `config/train_simplewiki.py` | Training config for Colab Pro |
| `model_patch.py` | `model_patch.py` | Auto-patches model.py for custom masks |
| `evaluate_regen.py` | `evaluate_regen.py` | Baseline vs regeneration evaluation |
| `colab_notebook.py` | reference | Step-by-step Colab instructions |

## Quick Start

```bash
git clone https://github.com/karpathy/nanoGPT.git
cd nanoGPT
pip install torch numpy transformers datasets tiktoken tqdm scipy

# Place files, then:
python data/simplewiki/prepare.py
python model_patch.py
python train.py config/train_simplewiki.py
python evaluate_regen.py --out_dir=out-simplewiki
```

## Evaluation Design

Both conditions use FREE GENERATION (model consumes its own outputs).
This is critical: under teacher-forcing, regeneration cannot improve
downstream predictions because the input was already ground truth.

Metrics:
- Overall accuracy across all generated positions
- Accuracy specifically on tokens AFTER each regeneration boundary
- Paired t-test for statistical significance

## GPU Memory Guide

| GPU | batch_size | gradient_accumulation_steps | Estimated time |
|-----|-----------|---------------------------|----------------|
| A100 (40GB) | 16 | 8 | ~3-4h |
| V100 (16GB) | 8 | 16 | ~8-12h |
| T4 (16GB) | 4 | 32 | ~15-20h |
