import tiktoken
import numpy as np
import torch
import threading
from datasets import load_dataset
from tqdm import tqdm



class CombinedBinDataLoader:
    def __init__(self, data, split_starts, B, SL, config, seed=42):
        self.B = B
        self.SL = SL
        self.config = config
        self.chunk_size = B * SL + 1
        self.data = data
        self.split_starts = split_starts
        self.rng = np.random.default_rng(seed)
        self._shuffle()
        self.idx = 0

        self._next = None
        self._thread = None
        self._prefetch()

        print(f"Initialized loader with {len(self.split_starts):,} chunks of size {B * SL + 1}.")
    
    def __iter__(self):
        return self

    def __next__(self):
        x, y = self.get_data()
        return x,y
    
    @staticmethod
    def create_loaders(filename, B, SL, train_config, seed=42):
        data = np.array(np.memmap(filename, dtype=np.uint16, mode='r'))
        #data = np.memmap(filename, dtype=np.uint16, mode='r')
        chunk_size = B * SL + 1

        all_starts = np.arange(0, len(data) - chunk_size, B * SL, dtype=np.int64)

        # One shuffle, one split
        rng = np.random.default_rng(seed)
        shuffled = rng.permutation(all_starts)
        n = int(0.95 * len(shuffled))

        train_loader = CombinedBinDataLoader(data, shuffled[:n], B, SL, train_config, seed=seed)
        val_loader = CombinedBinDataLoader(data, shuffled[n:], B, SL, train_config, seed=seed + 1)
        return train_loader, val_loader
    
    @staticmethod
    def create_test_loader(filename, B, SL, config, seed=42):
        data = np.array(np.memmap(filename, dtype=np.uint16, mode='r'))
        chunk_size = B * SL + 1
        all_starts = np.arange(0, len(data) - chunk_size, B * SL, dtype=np.int64)
        return CombinedBinDataLoader(data, all_starts, B, SL, config, seed=seed)
    

    @staticmethod
    def fetch_dataset(output_file, num_tokens):
        dataset = load_dataset("HuggingFaceFW/fineweb",
                       name="sample-10BT",
                       split="train",
                       streaming=True)

        enc = tiktoken.get_encoding('gpt2')
        tokens_buffer = []
        total_tokens = 0
        target_tokens = num_tokens

        with open(output_file, 'wb') as f:
            for item in tqdm(dataset):
                tokens = enc.encode_ordinary(item['text'])
                tokens.append(50256)
                tokens_buffer.extend(tokens)
                total_tokens += len(tokens)

                if len(tokens_buffer) >= 1_000_000:
                    np_tokens = np.array(tokens_buffer, dtype=np.uint16)
                    f.write(np_tokens.tobytes())
                    tokens_buffer = []

                if total_tokens >= target_tokens:
                    break

            if tokens_buffer:
                np_tokens = np.array(tokens_buffer, dtype=np.uint16)
                f.write(np_tokens.tobytes())

        print(f"Saved {total_tokens:,} tokens to {output_file}")


    def _shuffle(self):
        self.shuffled_starts = self.rng.permutation(self.split_starts)

    def _load(self):
        start = self.shuffled_starts[self.idx]
        buf = torch.from_numpy(self.data[start : start + self.chunk_size].astype(np.int64))
        x = buf[:-1].view(self.B, self.SL)
        y = buf[1:].view(self.B, self.SL)
        self.idx += 1
        if self.idx >= len(self.shuffled_starts):
            self.idx = 0
            self._shuffle()
        self._next = (x, y)

    def _prefetch(self):
        self._thread = threading.Thread(target=self._load, daemon=True)
        self._thread.start()

    def get_data(self):
        self._thread.join()
        x, y = self._next
        self._prefetch()
        return x, y