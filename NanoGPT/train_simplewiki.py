"""
nanoGPT config for training GPT-2 124M on SimpleWiki.
Place at: nanoGPT/config/train_simplewiki.py

Usage: python train.py config/train_simplewiki.py

Tuned for Colab Pro with a single A100 (40GB).
For V100 (16GB): set batch_size=8, gradient_accumulation_steps=16
For T4 (16GB):   set batch_size=4, gradient_accumulation_steps=32
"""

# Output
out_dir = "out-simplewiki"
eval_interval = 500
log_interval = 50
eval_iters = 200
eval_only = False
always_save_checkpoint = True

# Data
dataset = "simplewiki"
gradient_accumulation_steps = 8
batch_size = 16  # micro batch size

# Model: GPT-2 124M
n_layer = 12
n_head = 12
n_embd = 768
block_size = 256  # reduced for Colab memory; nanoGPT default is 1024
dropout = 0.1
bias = True

# Optimizer
learning_rate = 6e-4
max_iters = 30000  # ~3-4 hours on A100; increase if you have time
weight_decay = 0.1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# LR schedule
decay_lr = True
warmup_iters = 2000
lr_decay_iters = 30000  # should match max_iters
min_lr = 6e-5

# System
device = "cuda"
dtype = "bfloat16"  # change to "float16" if no bfloat16 support
compile = True

# Init
init_from = "scratch"
