"""
SimpleWiki data preparation for nanoGPT.

Place this file at: nanoGPT/data/simplewiki/prepare.py
Run: python data/simplewiki/prepare.py

Creates train.bin and val.bin in the same directory.
"""

import os
import re
import bz2
import requests
import numpy as np
import tiktoken
from tqdm import tqdm


DATA_DIR = os.path.dirname(os.path.abspath(__file__))


def download_simplewiki():
    url = "https://dumps.wikimedia.org/simplewiki/latest/simplewiki-latest-pages-articles.xml.bz2"
    filepath = os.path.join(DATA_DIR, "simplewiki.xml.bz2")
    if os.path.exists(filepath):
        print(f"Already downloaded: {filepath}")
        return filepath
    print(f"Downloading SimpleWiki from {url} ...")
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get("content-length", 0))
    with open(filepath, "wb") as f:
        with tqdm(total=total, unit="B", unit_scale=True, desc="Download") as pbar:
            for chunk in resp.iter_content(8192):
                f.write(chunk)
                pbar.update(len(chunk))
    return filepath


def clean_wikitext(text):
    text = re.sub(r"\{\{[^}]*\}\}", "", text)
    text = re.sub(r"<ref[^>]*>.*?</ref>", "", text, flags=re.DOTALL)
    text = re.sub(r"<ref[^/]*/>", "", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\[\[[^|\]]*\|([^\]]*)\]\]", r"\1", text)
    text = re.sub(r"\[\[([^\]]*)\]\]", r"\1", text)
    text = re.sub(r"\[https?://[^\s\]]+\s*([^\]]*)\]", r"\1", text)
    text = re.sub(r"\[\[Category:[^\]]*\]\]", "", text)
    text = re.sub(r"'{2,5}", "", text)
    text = re.sub(r"={2,6}\s*(.*?)\s*={2,6}", r"\n\1\n", text)
    text = re.sub(r"^\*+\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"^#+\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


def extract_articles(filepath):
    print("Extracting articles ...")
    articles = []
    current_text = []
    in_text = False
    with bz2.open(filepath, "rt", encoding="utf-8") as f:
        for line in tqdm(f, desc="Parsing"):
            if "<text" in line:
                in_text = True
                m = re.search(r"<text[^>]*>(.*)", line)
                if m:
                    current_text.append(m.group(1))
            elif "</text>" in line:
                in_text = False
                m = re.search(r"(.*)</text>", line)
                if m:
                    current_text.append(m.group(1))
                raw = "\n".join(current_text).strip()
                if len(raw) > 100:
                    cleaned = clean_wikitext(raw)
                    if len(cleaned) > 100:
                        articles.append(cleaned)
                current_text = []
            elif in_text:
                current_text.append(line.strip())
    print(f"Extracted {len(articles)} articles")
    return articles


def main():
    filepath = download_simplewiki()
    articles = extract_articles(filepath)

    enc = tiktoken.get_encoding("gpt2")
    print("Tokenizing ...")
    all_tokens = []
    for art in tqdm(articles, desc="Tokenize"):
        tokens = enc.encode_ordinary(art)
        tokens.append(enc.eot_token)  # <|endoftext|> between articles
        all_tokens.extend(tokens)

    all_tokens = np.array(all_tokens, dtype=np.uint16)
    print(f"Total tokens: {len(all_tokens):,}")

    # 90/10 split
    n = len(all_tokens)
    n_train = int(n * 0.9)
    train_tokens = all_tokens[:n_train]
    val_tokens = all_tokens[n_train:]

    train_tokens.tofile(os.path.join(DATA_DIR, "train.bin"))
    val_tokens.tofile(os.path.join(DATA_DIR, "val.bin"))
    print(f"Saved train.bin ({len(train_tokens):,} tokens) and val.bin ({len(val_tokens):,} tokens)")


if __name__ == "__main__":
    main()
