# Token-Level Refinement in Autoregressive Language Models

This repository contains code for an alternative inference procedure for decoder-only language models. The project introduces the Kth-Previous Prediction (KPP) objective and trains a GPT-2 scale model using both next-token prediction (NTP) and KPP objectives.

The repository also includes:
- a baseline autoregressive GPT-2 model
- experiments on token-level refinement
- an exploratory masked language modeling approach for multi-token regeneration

## Repository Structure

- `Models/Lag_Behind_Model/`  
  Training and evaluation code for regen-gpt2.

- `Models/Baseline_Model/`  
  Training and evaluation code for the autoregressive baseline model.

- `Models/BERT_Model/`  
  Training code for the masked language modeling experiments.

- `Datasets/Dataloader.py`  
  Custom dataloader implementation using the FineWeb dataset.

## Dataset

This project uses the FineWeb dataset from Hugging Face:

- [FineWeb Dataset](https://huggingface.co/datasets/HuggingFaceFW/fineweb?utm_source=chatgpt.com)
