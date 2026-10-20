"""Build a fixed token corpus from the Pile, the data Pythia was trained on."""

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

from recgen.config import CONTEXT_LEN, DATA_DIR, MODEL, N_WINDOWS, SEED

CORPUS_PATH = DATA_DIR / "corpus.npy"


def packed_stream(tokenizer, docs: list[str] | None = None) -> np.ndarray:
    """Documents end to end with EOS separators, as in Pythia's training. docs defaults to the Pile sample."""
    if docs is None:
        docs = list(load_dataset("NeelNanda/pile-10k", split="train")["text"])
    stream = []
    for ids in tokenizer(docs)["input_ids"]:
        stream.extend(ids)
        stream.append(tokenizer.eos_token_id)
    return np.array(stream, dtype=np.int32)


def window_starts(n_tokens: int) -> np.ndarray:
    """Where each corpus window starts in the stream: a seeded random subset of its full windows, in order."""
    n_full = n_tokens // CONTEXT_LEN
    return np.sort(np.random.default_rng(SEED).choice(n_full, size=N_WINDOWS, replace=False)) * CONTEXT_LEN


def build_corpus() -> np.ndarray:
    """Cut the packed stream into fixed windows and keep a seeded random subset."""
    stream = packed_stream(AutoTokenizer.from_pretrained(MODEL))
    return np.stack([stream[start:start + CONTEXT_LEN] for start in window_starts(len(stream))])


def load_corpus() -> np.ndarray:
    if not CORPUS_PATH.exists():
        DATA_DIR.mkdir(exist_ok=True)
        np.save(CORPUS_PATH, build_corpus())
    return np.load(CORPUS_PATH)


if __name__ == "__main__":
    corpus = load_corpus()
    print(f"{corpus.shape[0]} windows x {corpus.shape[1]} tokens = {corpus.size:,} tokens")
