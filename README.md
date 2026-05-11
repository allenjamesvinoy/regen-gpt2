#Token-Level Refinement in Autoregressive Language Models

This repository presents code for an alternative to typical autoregressive inference for decoder-only language models. The project introduces the Kth-Previous Prediction objective and trains a GPT-2 scale model on both next token prediction and kth-previous prediction, as well as a baseline. It also includes a failed attempt at regenerating multiple tokens at once using a masked language modeling task.

##Github Contents
Models/Lag_Behind_Model   Folder for our training/testing code for the alternative model
Models/Baseline_Model     Folder for our training/testing code for the baseline model
Models/BERT_Model         Folder for our training code for the masked language modeling model
Datasets/Dataloader.py    Custom Dataloader for our project - uses the fineweb (https://huggingface.co/datasets/HuggingFaceFW/fineweb) dataset
