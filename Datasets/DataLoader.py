import tiktoken
import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from Models.GPT_Model import GPTConfig


class TinyShakespeareDataLoader:
  def __init__(self, B, SL, path, config : GPTConfig, split='train'):
    self.B = B
    self.SL = SL
    self.config = config

    with open(path, 'r') as f:
      text = f.read()

    enc = tiktoken.get_encoding('gpt2')
    tokens = enc.encode(text)

    # Split the data: 90% train, 10% validation
    n = int(0.9 * len(tokens))
    if split == 'train':
        self.tokens = torch.tensor(tokens[:n])
    else:
        self.tokens = torch.tensor(tokens[n:])

    self.cursor = 0

  def get_data(self):
    k = self.config.lag_behind

    buf = self.tokens[self.cursor : self.cursor + self.B * self.SL + 1]
    x = buf[:-1].view(self.B, self.SL)
    y = buf[1:].view(self.B, self.SL)

    # y2[t] = token at (cursor + t + 1 - k), i.e. k steps behind y
    lag_start = self.cursor + 1 - k
    lag_end   = self.cursor + self.B * self.SL + 1 - k

    if lag_start < 0:
        pad = torch.full((-lag_start,), -1, dtype=self.tokens.dtype, device=self.tokens.device)
        y2 = torch.cat([pad, self.tokens[0:lag_end]]).view(self.B, self.SL)
    else:
        y2 = self.tokens[lag_start:lag_end].view(self.B, self.SL)

    self.cursor += self.B * self.SL
    if self.cursor + (self.B * self.SL + 1) > len(self.tokens):
        self.cursor = 0

    return x, y, y2

class TinyStoriesDataLoader:
    def __init__(self, tokenizer_name="gpt2", max_length=512, batch_size=32, num_train=100000, num_eval=10000):
        self.max_length = max_length
        self.batch_size = batch_size

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        ds = load_dataset("roneneldan/TinyStories")
        ds["train"]      = ds["train"].select(range(num_train))
        ds["validation"] = ds["validation"].select(range(num_eval))
        ds = ds.map(self._tokenize, batched=True, remove_columns=["text"])
        ds.set_format(type="torch", columns=["input_ids"])

        self.train_loader = DataLoader(ds["train"], batch_size=batch_size, shuffle=True)
        self.valid_loader = DataLoader(ds["validation"], batch_size=batch_size)

    def _tokenize(self, batch):
        return self.tokenizer(batch["text"], truncation=True, padding="max_length", max_length=self.max_length)
    
    def pad_token(self):
       return self.tokenizer.pad_token_id

    def get_data(self):
        return self.train_loader, self.valid_loader
    



class CombinedBinDataLoader:
    def __init__(self, filename, B, SL, config, split='train'):
        self.B = B
        self.SL = SL
        self.config = config
        
        # Memory-map the binary file (stays on disk, essentially 0 RAM usage)
        self.data = np.memmap(filename, dtype=np.uint16, mode='r')
        
        # Calculate the 90% split index
        n = int(0.9 * len(self.data))
        
        if split == 'train':
            self.split_data = self.data[:n]
        else:
            self.split_data = self.data[n:]
            
        self.cursor = 0
        print(f"Initialized {split} loader with {len(self.split_data):,} tokens.")

    def get_data(self):
        # 1. Fetch the continuous chunk from disk and convert to torch.long (int64)
        # chunk = self.split_data[self.cursor : self.cursor + self.B * self.SL + 1]
        # buf = torch.from_numpy(chunk.astype(np.int64))
        buf = torch.from_numpy(self.split_data[self.cursor : self.cursor + self.B * self.SL + 1].astype(np.int32))
        # 2. Standard autoregressive shifting
        x = buf[:-1].view(self.B, self.SL)
        y = buf[1:].view(self.B, self.SL)
        
        # 3. --- Y2 LAG BEHIND LOGIC ---
        k = self.config.lag_behind
        shift = k 

        # Fill with -100 (PyTorch's default ignore_index for Cross Entropy)
        y2 = torch.full_like(x, -100) 

        # # Shift the sequence to the right by 'shift' steps
        # if shift > 0:
        #     y2[:, shift:] = x[:, :-shift]
        # elif shift == 0:
        #     y2 = x.clone()

        # # 4. Advance the cursor for the next batch
        # self.cursor += self.B * self.SL
        
        # # Reset cursor if we hit the end of our split
        # if self.cursor + (self.B * self.SL + 1) > len(self.split_data):
        #     self.cursor = 0
            
        return x, y, y2