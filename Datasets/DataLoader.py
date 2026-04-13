import tiktoken
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
    def __init__(self, tokenizer_name="gpt2", max_length=512, batch_size=32):
        self.max_length = max_length
        self.batch_size = batch_size

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        ds = load_dataset("roneneldan/TinyStories")
        ds["train"]      = ds["train"].select(range(100000))
        ds["validation"] = ds["validation"].select(range(10000))
        ds = ds.map(self._tokenize, batched=True, remove_columns=["text"])
        ds.set_format(type="torch", columns=["input_ids", "attention_mask"])

        self.train_loader = DataLoader(ds["train"], batch_size=batch_size, shuffle=True)
        self.valid_loader = DataLoader(ds["validation"], batch_size=batch_size)

    def _tokenize(self, batch):
        return self.tokenizer(batch["text"], truncation=True, padding="max_length", max_length=self.max_length)

    def get_data(self):
        return self.train_loader, self.valid_loader