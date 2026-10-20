import numpy as np
import pytest

from recgen.config import CONTEXT_LEN, N_WINDOWS
from recgen.corpus import packed_stream, window_starts
from recgen.gpt2_v22 import windows


class CharTokenizer:
    """A stand-in tokenizer: one token per character, ids are code points, end of text is 0."""

    eos_token_id = 0

    def __call__(self, texts):
        return {"input_ids": [[ord(c) for c in t] for t in texts]}


def test_packed_stream_ends_each_given_document_with_eos():
    assert packed_stream(CharTokenizer(), ["ab", "c"]).tolist() == [97, 98, 0, 99, 0]


def test_windows_are_v1s_seeded_full_windows_of_the_stream():
    stream = np.arange(2 * N_WINDOWS * CONTEXT_LEN + 100, dtype=np.int32)
    out = windows(stream)
    starts = window_starts(len(stream))
    assert out.shape == (N_WINDOWS, CONTEXT_LEN) and np.all(np.diff(starts) > 0) and np.all(starts % CONTEXT_LEN == 0)
    np.testing.assert_array_equal(out[:, 0], starts)
    np.testing.assert_array_equal(out[7], stream[starts[7]:starts[7] + CONTEXT_LEN])


def test_windows_refuse_a_stream_too_short_for_the_corpus():
    with pytest.raises(AssertionError):
        windows(np.zeros((N_WINDOWS - 1) * CONTEXT_LEN, dtype=np.int32))
