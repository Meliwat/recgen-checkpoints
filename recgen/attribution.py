"""Direct attribution of the traced neurons' pre-activations to every upstream MLP neuron.

Pythia's blocks are parallel: attention and MLP both read the residual stream entering the
block. So the MLP at LAYER reads LayerNorm(h), where h is the sum of the token embedding and
every earlier attention and MLP output. Freezing the LayerNorm scale sigma at its value on
each example (the usual linearization), a neuron's pre-activation splits exactly into one
term per upstream component c:

    pre = sum_c (center(c) * gamma) . w / sigma  +  beta . w + b

Upstream MLP neuron i of layer l writes act_i * W_out_l[:, i], so its term is
act_i * D[l, i] / sigma with D[l, i] = (center(W_out_l[:, i]) * gamma) . w.

Run as a script: record everything the decomposition needs at the final checkpoint, on the
traced neurons' facet examples and on random background text.
"""

import json
from dataclasses import dataclass

import numpy as np
import torch

from recgen.activations import BATCH_SIZE, MIN_CONTEXT, checkpoint
from recgen.config import CONTEXT_LEN, DATA_DIR, FINAL_STEP, LAYER, RESULTS_DIR, SEED
from recgen.corpus import load_corpus
from recgen.record import selected_neurons

ATTRIBUTION_PATH = DATA_DIR / f"step{FINAL_STEP}_attribution.npz"
N_BACKGROUND = 1024  # random windows that hold no facet example; one position from each is the baseline


@dataclass
class Capture:
    acts: np.ndarray  # [E, layer, d_mlp] upstream MLP activations at each example's position
    attn: np.ndarray  # [E, layer, d_model] attention outputs
    embed: np.ndarray  # [E, d_model] token embedding
    sigma: np.ndarray  # [E] LayerNorm scale of the residual stream entering the layer
    pre: np.ndarray  # [E, len(neurons)] the neurons' actual pre-activations


class _StopForward(Exception):
    pass


def capture(model, corpus: np.ndarray, examples: list[tuple[int, int]], neurons: list[int],
            layer: int = LAYER) -> Capture:
    """Run each example's window up to `layer` and record, at the example's position,
    every upstream component and the neurons' pre-activations. Examples keep their order."""
    assert model.config.use_parallel_residual, "the decomposition assumes parallel attention and MLP"
    blocks = model.gpt_neox.layers
    device = next(model.parameters()).device
    neuron_index = torch.tensor(neurons, device=device)
    eps = blocks[layer].post_attention_layernorm.eps
    batch, out = {}, {}

    def gather(x):  # [B, T, ...] -> [examples in this batch, ...]
        return x[batch["rows"], batch["pos"]].float().cpu()

    def save(name):
        def hook(_module, _inputs, output):
            out.setdefault(name, []).append(gather(output[0] if isinstance(output, tuple) else output))
        return hook

    def save_sigma(_module, inputs):
        h = gather(inputs[0])
        out.setdefault("sigma", []).append(torch.sqrt(h.var(dim=-1, unbiased=False) + eps))

    def save_pre_and_stop(_module, _inputs, output):
        out.setdefault("pre", []).append(gather(output.index_select(-1, neuron_index)))
        raise _StopForward

    handles = [model.gpt_neox.embed_in.register_forward_hook(save("embed")),
               blocks[layer].post_attention_layernorm.register_forward_pre_hook(save_sigma),
               blocks[layer].mlp.dense_h_to_4h.register_forward_hook(save_pre_and_stop)]
    for l in range(layer):
        handles.append(blocks[l].attention.register_forward_hook(save(f"attn{l}")))
        handles.append(blocks[l].mlp.act.register_forward_hook(save(f"acts{l}")))

    by_window = {}
    for i, (w, p) in enumerate(examples):
        by_window.setdefault(w, []).append((i, p))
    windows = sorted(by_window)
    order = []
    try:
        with torch.no_grad():
            for start in range(0, len(windows), BATCH_SIZE):
                chunk = windows[start:start + BATCH_SIZE]
                rows, pos = [], []
                for row, w in enumerate(chunk):
                    for i, p in by_window[w]:
                        order.append(i)
                        rows.append(row)
                        pos.append(p)
                batch["rows"] = torch.tensor(rows, device=device)
                batch["pos"] = torch.tensor(pos, device=device)
                ids = torch.from_numpy(corpus[chunk]).long().to(device)
                try:
                    model(input_ids=ids)
                except _StopForward:
                    pass
    finally:
        for h in handles:
            h.remove()

    restore = np.argsort(order)
    cat = {name: torch.cat(parts).numpy()[restore] for name, parts in out.items()}
    return Capture(acts=np.stack([cat[f"acts{l}"] for l in range(layer)], axis=1),
                   attn=np.stack([cat[f"attn{l}"] for l in range(layer)], axis=1),
                   embed=cat["embed"], sigma=cat["sigma"], pre=cat["pre"])


def _center(x: torch.Tensor) -> torch.Tensor:
    return x - x.mean(dim=-1, keepdim=True)


def _readout(model, neurons: list[int], layer: int):
    """Each neuron's input weights folded with LayerNorm's gain [d_model, N], and its constant term [N]."""
    norm = model.gpt_neox.layers[layer].post_attention_layernorm
    fc = model.gpt_neox.layers[layer].mlp.dense_h_to_4h
    w, gamma, beta = (x.detach().float().cpu() for x in (fc.weight[neurons], norm.weight, norm.bias))
    return (w * gamma).T, w @ beta + fc.bias[neurons].detach().float().cpu()


def mlp_coefficients(model, neurons: list[int], layer: int = LAYER) -> np.ndarray:
    """D[l, i, n]: neuron n's pre-activation per unit of upstream neuron i's activation, times sigma."""
    readout, _ = _readout(model, neurons, layer)
    w_out = [block.mlp.dense_4h_to_h.weight.detach().float().cpu().T for block in model.gpt_neox.layers[:layer]]
    return torch.stack([_center(w) @ readout for w in w_out]).numpy()


def components(model, cap: Capture, neurons: list[int], layer: int = LAYER) -> dict[str, np.ndarray]:
    """Each neuron's pre-activation on each example [E, N], split by source. The parts sum to cap.pre."""
    readout, constant = _readout(model, neurons, layer)
    blocks = model.gpt_neox.layers[:layer]
    sigma = torch.from_numpy(cap.sigma)[:, None]
    mlp_bias = sum(_center(b.mlp.dense_4h_to_h.bias.detach().float().cpu()) @ readout for b in blocks)
    mlp = np.einsum("eli,lin->en", cap.acts, mlp_coefficients(model, neurons, layer))
    return {
        "mlp neurons": mlp / cap.sigma[:, None],
        "mlp biases": (mlp_bias / sigma).numpy(),
        "attention": (_center(torch.from_numpy(cap.attn)).sum(dim=1) @ readout / sigma).numpy(),
        "embedding": (_center(torch.from_numpy(cap.embed)) @ readout / sigma).numpy(),
        "constant": np.broadcast_to(constant.numpy(), cap.pre.shape),
    }


def background_examples(candidates: list[dict], n_corpus_windows: int, n: int, rng) -> list[tuple[int, int]]:
    """One random position (with full context) in each of n random windows that hold no facet example."""
    facet_windows = {w for c in candidates for facet in c["facets"] for w, _, _ in facet}
    windows = np.sort(rng.choice([w for w in range(n_corpus_windows) if w not in facet_windows], size=n, replace=False))
    return [(int(w), int(rng.integers(MIN_CONTEXT, CONTEXT_LEN))) for w in windows]


def main() -> None:
    corpus = load_corpus()
    candidates = json.loads((RESULTS_DIR / "candidates.json").read_text())
    neurons = selected_neurons()
    facet_examples = sorted({(w, p) for c in candidates for facet in c["facets"] for w, p, _ in facet})
    background = background_examples(candidates, len(corpus), N_BACKGROUND, np.random.default_rng(SEED))
    examples = facet_examples + background
    with checkpoint(FINAL_STEP) as model:
        cap = capture(model, corpus, examples, neurons)
        parts = components(model, cap, neurons)
        coefficients = mlp_coefficients(model, neurons)
    error = np.abs(sum(parts.values()) - cap.pre).max()
    print(f"{len(examples)} examples; largest reconstruction error {error:.2e} "
          f"(pre-activations span {cap.pre.min():.1f} to {cap.pre.max():.1f})")
    np.savez(ATTRIBUTION_PATH, examples=np.array(examples), n_facet_examples=len(facet_examples),
             neurons=np.array(neurons), acts=cap.acts.astype(np.float16), sigma=cap.sigma, pre=cap.pre,
             coefficients=coefficients, **{f"part_{k.replace(' ', '_')}": v for k, v in parts.items()})


if __name__ == "__main__":
    main()
