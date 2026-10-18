"""Build a fixed token corpus from the Pile, the data Pythia was trained on."""

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

from recgen.config import CONTEXT_LEN, DATA_DIR, MODEL, N_WINDOWS, SEED

CORPUS_PATH = DATA_DIR / "corpus.npy"


def build_corpus() -> np.ndarray:
    """Pack documents end to end with EOS separators, as in Pythia's training,
    cut the stream into fixed windows, and keep a seeded random subset."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    docs = list(load_dataset("NeelNanda/pile-10k", split="train")["text"])
    stream = []
    for ids in tokenizer(docs)["input_ids"]:
        stream.extend(ids)
        stream.append(tokenizer.eos_token_id)
    n_full = len(stream) // CONTEXT_LEN
    windows = np.array(stream[: n_full * CONTEXT_LEN], dtype=np.int32).reshape(n_full, CONTEXT_LEN)
    keep = np.sort(np.random.default_rng(SEED).choice(n_full, size=N_WINDOWS, replace=False))
    return windows[keep]


def load_corpus() -> np.ndarray:
    if not CORPUS_PATH.exists():
        DATA_DIR.mkdir(exist_ok=True)
        np.save(CORPUS_PATH, build_corpus())
    return np.load(CORPUS_PATH)


if __name__ == "__main__":
    corpus = load_corpus()
    print(f"{corpus.shape[0]} windows x {corpus.shape[1]} tokens = {corpus.size:,} tokens")
