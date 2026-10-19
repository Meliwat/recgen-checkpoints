"""Output-side measurements for v3: what each traced neuron's firing does to next-token predictions.

For each context (window w, position p), the window's first p tokens run once and their
attention keys and values are cached. Then tokens p to p + LATER_PREDICTIONS run as a batch of
variants, each setting one neuron at p (only at p) to a given value, or changing nothing.
Attention is causal, so this equals running the window with that one change.

Run as a script at the final checkpoint, for every traced neuron:
- its facet examples: as is, the neuron zero-ablated, and mean-ablated. Every other traced
  neuron is also patched there to its facet-typical activation and to zero (negative control);
- N_NULL random contexts: the neuron patched to its facet-typical activation, to zero and to
  its mean. The mean effect is its generic push.
Exploratory, added after the logit-space results: how much probability each facet example's
ablation moves (record_probabilities).
"""

import copy
import json

import numpy as np
import torch

from recgen.activations import MIN_CONTEXT, checkpoint
from recgen.config import CONTEXT_LEN, DATA_DIR, FINAL_STEP, LAYER, RESULTS_DIR, SEED
from recgen.corpus import load_corpus
from recgen.facets import tokenizer
from recgen.facets_v3 import FACETS_V3_PATH
from recgen.record import neuron_activations_path
from recgen.rubric_v3 import LATER_PREDICTIONS, N_NULL

OUTPUTS_PATH = DATA_DIR / f"step{FINAL_STEP}_v3_outputs.npz"
EFFECTS_PATH = {kind: DATA_DIR / f"step{FINAL_STEP}_v3_effects_{kind}.npy" for kind in ("zero", "mean")}
PROBABILITY_PATH = DATA_DIR / f"step{FINAL_STEP}_v3_probability.npz"
DELTA_P_PATH = DATA_DIR / f"step{FINAL_STEP}_v3_delta_p.npy"
N_PROBABILITY_NULL = 128  # random contexts for the exploratory probability check
CHUNK = LATER_PREDICTIONS + 1  # tokens run per variant; their predictions are p+1 to p+CHUNK
GROUP = 24  # variants per forward pass; each holds a copy of the prefix's keys and values


class NeuronSetter:
    """Forward hook on a layer's MLP activation that sets chosen (row, neuron) entries at the
    first position of the batch. Inactive while rows is None."""

    def __init__(self, model, layer: int = LAYER):
        self.rows = None
        self.handle = model.gpt_neox.layers[layer].mlp.act.register_forward_hook(self)

    def __call__(self, _module, _inputs, output):
        if self.rows is not None:
            output[self.rows, 0, self.neurons] = self.values
        return output


def variant_logprobs(model, setter: NeuronSetter, window: np.ndarray, pos: int, variants: list,
                     chunk: int = CHUNK, group: int = GROUP) -> torch.Tensor:
    """Log-probs [len(variants), chunk, vocab] predicted at positions pos to pos + chunk - 1.
    Each variant is (neuron, value), setting that neuron at pos, or None for no change."""
    device = next(model.parameters()).device
    tokens = torch.from_numpy(np.asarray(window)).long().to(device)
    setter.rows = None
    with torch.no_grad():
        cache = model.gpt_neox(input_ids=tokens[None, :pos], use_cache=True).past_key_values
        out = []
        for start in range(0, len(variants), group):
            batch = variants[start:start + group]
            expanded = copy.deepcopy(cache)
            expanded.batch_repeat_interleave(len(batch))
            changed = [(row, v) for row, v in enumerate(batch) if v is not None]
            setter.rows = torch.tensor([row for row, _ in changed], device=device, dtype=torch.long)
            setter.neurons = torch.tensor([n for _, (n, _) in changed], device=device, dtype=torch.long)
            setter.values = torch.tensor([x for _, (_, x) in changed], device=device, dtype=torch.float32)
            ids = tokens[None, pos:pos + chunk].expand(len(batch), -1)
            out.append(torch.log_softmax(model(input_ids=ids, past_key_values=expanded).logits.float(), dim=-1))
            setter.rows = None
    return torch.cat(out)


def centered(x: torch.Tensor) -> torch.Tensor:
    return x - x.mean(dim=-1, keepdim=True)


def null_contexts(candidates: list[dict], n_windows: int, n: int, rng) -> list[tuple[int, int]]:
    """One random position in each of n random windows that hold no v1 facet example, with
    room for the next CHUNK tokens."""
    facet_windows = {w for c in candidates for facet in c["facets"] for w, _, _ in facet}
    windows = np.sort(rng.choice([w for w in range(n_windows) if w not in facet_windows], size=n, replace=False))
    return [(int(w), int(rng.integers(MIN_CONTEXT, CONTEXT_LEN - CHUNK))) for w in windows]


def facet_examples(facets_v3: list[dict]) -> list[tuple[int, int, int, int]]:
    """(neuron index, facet, window, position) for every facet example, in a fixed order."""
    return [(k, f, e[0], e[1]) for k, r in enumerate(facets_v3) for f, facet in enumerate(r["facets"]) for e in facet]


def mean_activations(neurons: list[int]) -> np.ndarray:
    """Each neuron's mean activation over the corpus (positions with full context) at the final step."""
    data = np.load(neuron_activations_path(FINAL_STEP))
    order = [data["neurons"].tolist().index(n) for n in neurons]
    acts = data["acts"][:, MIN_CONTEXT:, order].astype(np.float64)
    return acts.reshape(-1, len(neurons)).mean(axis=0)


def record(model, corpus, facets_v3, contexts, typical, means, vocab: int):
    neurons = [r["neuron"] for r in facets_v3]
    examples = facet_examples(facets_v3)
    n_facets = [len(r["facets"]) for r in facets_v3]
    facet_index = {(k, f): i for i, (k, f) in enumerate((k, f) for k, n in enumerate(n_facets) for f in range(n))}
    device = next(model.parameters()).device
    setter = NeuronSetter(model)
    effects = {kind: np.lib.format.open_memmap(path, mode="w+", dtype=np.float16, shape=(len(examples), vocab))
               for kind, path in EFFECTS_PATH.items()}
    own_actual = np.zeros((len(examples), 3, CHUNK), np.float32)  # as is, zero-ablated, mean-ablated
    # Negative control: each neuron patched into other neurons' facet examples, summed per (facet, half).
    control_sum = torch.zeros((len(neurons), len(facet_index), 2, vocab), device=device)
    control_count = np.zeros((len(facet_index), 2), int)
    steps = torch.arange(CHUNK, device=device)
    seen = {}
    for i, (k, f, w, p) in enumerate(examples):
        half = seen.setdefault((k, f), 0) % 2
        seen[k, f] += 1
        others = [j for j in range(len(neurons)) if j != k]
        variants = [None, (neurons[k], 0.0), (neurons[k], float(means[k]))]
        variants += [v for j in others for v in ((neurons[j], float(typical[j])), (neurons[j], 0.0))]
        lp = variant_logprobs(model, setter, corpus[w], p, variants)
        actual = torch.from_numpy(corpus[w, p + 1:p + 1 + CHUNK]).long().to(device)
        own_actual[i] = lp[:3, steps, actual].cpu().numpy()
        nxt = lp[:, 0, :vocab]
        effects["zero"][i] = centered(nxt[0] - nxt[1]).cpu().numpy()
        effects["mean"][i] = centered(nxt[0] - nxt[2]).cpu().numpy()
        patched = centered(nxt[3::2] - nxt[4::2])
        control_sum[others, facet_index[k, f], half] += patched
        control_count[facet_index[k, f], half] += 1
        if i % 200 == 0:
            print(f"facet examples: {i}/{len(examples)}", flush=True)

    generic = torch.zeros((len(neurons), 2, vocab), device=device)  # zero- and mean-ablation pushes
    null_actual = np.zeros((len(contexts), len(neurons), 3, CHUNK), np.float32)  # patched, zero, mean
    for r, (w, p) in enumerate(contexts):
        variants = [v for j, n in enumerate(neurons) for v in ((n, float(typical[j])), (n, 0.0), (n, float(means[j])))]
        lp = variant_logprobs(model, setter, corpus[w], p, variants)
        actual = torch.from_numpy(corpus[w, p + 1:p + 1 + CHUNK]).long().to(device)
        null_actual[r] = lp[:, steps, actual].reshape(len(neurons), 3, CHUNK).cpu().numpy()
        nxt = lp[:, 0, :vocab].reshape(len(neurons), 3, vocab)
        generic[:, 0] += centered(nxt[:, 0] - nxt[:, 1])
        generic[:, 1] += centered(nxt[:, 0] - nxt[:, 2])
        if r % 200 == 0:
            print(f"null contexts: {r}/{len(contexts)}", flush=True)
    setter.handle.remove()
    for array in effects.values():
        array.flush()
    generic = (generic / len(contexts)).cpu().numpy()
    return examples, own_actual, null_actual, generic, control_summary(effects["zero"], examples, control_sum,
                                                                       control_count, generic[:, 0])


def control_summary(effects_zero, examples, control_sum, control_count, generic_zero) -> dict:
    """Dot products the negative control needs, so the half means needn't be stored:
    each example with its own neuron's half means of every facet, the half means with each
    other, and with the neuron's generic push."""
    halves = control_sum / torch.from_numpy(np.maximum(control_count, 1)).to(control_sum)[None, :, :, None]
    dots = np.zeros((len(examples),) + control_count.shape, np.float32)
    rows_by_neuron = {}
    for i, (k, *_) in enumerate(examples):
        rows_by_neuron.setdefault(k, []).append(i)
    gram = np.zeros((halves.shape[0], halves.shape[1], 2, 2), np.float32)
    with_generic = np.zeros(halves.shape[:3], np.float32)
    for k, rows in rows_by_neuron.items():
        own = torch.from_numpy(effects_zero[rows].astype(np.float32)).to(halves)
        dots[rows] = torch.einsum("ev,fhv->efh", own, halves[k]).cpu().numpy()
        gram[k] = torch.einsum("fhv,fgv->fhg", halves[k], halves[k]).cpu().numpy()
        with_generic[k] = (halves[k] @ torch.from_numpy(generic_zero[k]).to(halves)).cpu().numpy()
    return {"control_dots": dots, "control_gram": gram, "control_generic": with_generic, "control_count": control_count}


def prediction_change(lp: torch.Tensor, vocab: int) -> dict[str, torch.Tensor]:
    """How the next-token prediction changes from lp[1] (neuron changed) to lp[0]: probability
    moved, its total variation and KL divergence, and the cosine between the logit change and the
    prediction's own centered log-probs (positive: the neuron sharpens what is already predicted)."""
    p = lp.exp()
    delta = centered(lp[0, :vocab] - lp[1, :vocab])
    own = centered(lp[0, :vocab])
    return {"delta_p": (p[0] - p[1])[:vocab], "tv": 0.5 * (p[0] - p[1]).abs().sum(),
            "kl": (p[0] * (lp[0] - lp[1])).sum(), "sharpen": delta @ own / (delta.norm() * own.norm())}


def record_probabilities(model, corpus, facets_v3, contexts, typical, vocab: int) -> None:
    """Exploratory, added after the logit-space results: the change in next-token probabilities
    at each facet example (as is minus zero-ablated), and at random contexts (patched to the
    facet-typical activation minus zero) for reference."""
    neurons = [r["neuron"] for r in facets_v3]
    examples = facet_examples(facets_v3)
    setter = NeuronSetter(model)
    delta_p = np.lib.format.open_memmap(DELTA_P_PATH, mode="w+", dtype=np.float16, shape=(len(examples), vocab))
    own = np.zeros((len(examples), 3), np.float32)  # tv, kl, sharpen
    for i, (k, _, w, p) in enumerate(examples):
        lp = variant_logprobs(model, setter, corpus[w], p, [None, (neurons[k], 0.0)], chunk=1)[:, 0]
        change = prediction_change(lp, vocab)
        delta_p[i] = change.pop("delta_p").cpu().numpy()
        own[i] = [v.item() for v in change.values()]
    null = np.zeros((len(contexts), len(neurons), 3), np.float32)
    for r, (w, p) in enumerate(contexts):
        variants = [v for j, n in enumerate(neurons) for v in ((n, float(typical[j])), (n, 0.0))]
        lp = variant_logprobs(model, setter, corpus[w], p, variants, chunk=1)[:, 0]
        for j in range(len(neurons)):
            change = prediction_change(lp[2 * j:2 * j + 2], vocab)
            change.pop("delta_p")
            null[r, j] = [v.item() for v in change.values()]
    setter.handle.remove()
    delta_p.flush()
    np.savez(PROBABILITY_PATH, own=own, null=null)


def main() -> None:
    need_outputs, need_probabilities = not OUTPUTS_PATH.exists(), not PROBABILITY_PATH.exists()
    if not (need_outputs or need_probabilities):
        print(f"{OUTPUTS_PATH.name}, {PROBABILITY_PATH.name}: already recorded")
        return
    corpus = load_corpus()
    candidates = json.loads((RESULTS_DIR / "candidates.json").read_text())
    facets_v3 = json.loads(FACETS_V3_PATH.read_text())
    neurons = [r["neuron"] for r in facets_v3]
    typical = np.array([np.median([e[2] for facet in r["facets"] for e in facet]) for r in facets_v3])
    contexts = null_contexts(candidates, len(corpus), N_NULL, np.random.default_rng(SEED))
    vocab = len(tokenizer())
    with checkpoint(FINAL_STEP) as model:
        if need_outputs:
            means = mean_activations(neurons)
            examples, own_actual, null_actual, generic, control = record(
                model, corpus, facets_v3, contexts, typical, means, vocab)
            np.savez(OUTPUTS_PATH, examples=np.array(examples), neurons=np.array(neurons), typical=typical,
                     means=means, contexts=np.array(contexts), own_actual=own_actual, null_actual=null_actual,
                     generic=generic, **control)
            print(f"{len(examples)} facet examples and {len(contexts)} null contexts recorded")
        if need_probabilities:
            record_probabilities(model, corpus, facets_v3, contexts[:N_PROBABILITY_NULL], typical, vocab)
            print(f"{PROBABILITY_PATH.name}: recorded")


if __name__ == "__main__":
    main()
