"""v4 recording: v2's upstream attribution at every v4 checkpoint, with attention split by head.

A head's direct term on a neuron is its slice of the attention output, z_h @ W_O_h^T, read
through the neuron's LayerNorm-folded input weights and divided by the frozen LayerNorm scale,
like every other term in recgen.attribution. The terms add back up to the pre-activation.

Run as a script: for each checkpoint in rubric_v4.STEPS not yet cached, record the v1 facet
examples and v2's background text (same seed, so the same positions).
"""

import json

import numpy as np
import torch

from recgen.activations import BATCH_SIZE, checkpoint
from recgen.attribution import N_BACKGROUND, _center, _readout, background_examples, mlp_coefficients
from recgen.config import DATA_DIR, LAYER, RESULTS_DIR, SEED
from recgen.corpus import load_corpus
from recgen.record import selected_neurons
from recgen.rubric_v4 import STEPS

FORMATION_DIR = DATA_DIR / "v4"


def formation_path(step: int):
    return FORMATION_DIR / f"step{step}.npz"


class _StopForward(Exception):
    pass


def capture(model, corpus: np.ndarray, examples: list[tuple[int, int]], neurons: list[int],
            layer: int = LAYER) -> dict[str, np.ndarray]:
    """Run each example's window up to `layer` and record at the example's position: upstream
    MLP activations, the LayerNorm scale, the neurons' pre-activations, and every direct term
    of those pre-activations, with attention split into heads. Examples keep their order."""
    assert model.config.use_parallel_residual, "the decomposition assumes parallel attention and MLP"
    blocks = model.gpt_neox.layers
    device = next(model.parameters()).device
    n_heads = model.config.num_attention_heads
    readout, constant = _readout(model, neurons, layer)
    readout_on_device = readout.to(device)
    neuron_index = torch.tensor(neurons, device=device)
    eps = blocks[layer].post_attention_layernorm.eps
    batch, out = {}, {}

    def gather(x):  # [B, T, ...] -> [examples in this batch, ...]
        return x[batch["rows"], batch["pos"]].float()

    def keep(name, value):
        out.setdefault(name, []).append(value.cpu())

    def save_heads(l):
        weight = blocks[l].attention.dense.weight.float()  # [d_model, d_model], input columns grouped by head

        def hook(_module, inputs):
            z = gather(inputs[0]).view(-1, n_heads, weight.shape[1] // n_heads)
            per_head = torch.einsum("ehk,dhk->ehd", z, weight.view(weight.shape[0], n_heads, -1))
            keep(f"heads{l}", _center(per_head) @ readout_on_device)  # [e, heads, neurons]
        return hook

    def save_acts(l):
        return lambda _module, _inputs, output: keep(f"acts{l}", gather(output))

    def save_embed(_module, _inputs, output):
        keep("embed", _center(gather(output)) @ readout_on_device)

    def save_sigma(_module, inputs):
        h = gather(inputs[0])
        keep("sigma", torch.sqrt(h.var(dim=-1, unbiased=False) + eps))

    def save_pre_and_stop(_module, _inputs, output):
        keep("pre", gather(output.index_select(-1, neuron_index)))
        raise _StopForward

    handles = [model.gpt_neox.embed_in.register_forward_hook(save_embed),
               blocks[layer].post_attention_layernorm.register_forward_pre_hook(save_sigma),
               blocks[layer].mlp.dense_h_to_4h.register_forward_hook(save_pre_and_stop)]
    for l in range(layer):
        handles.append(blocks[l].attention.dense.register_forward_pre_hook(save_heads(l)))
        handles.append(blocks[l].mlp.act.register_forward_hook(save_acts(l)))

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
                try:
                    model(input_ids=torch.from_numpy(corpus[chunk]).long().to(device))
                except _StopForward:
                    pass
    finally:
        for h in handles:
            h.remove()

    restore = np.argsort(order)
    cat = {name: torch.cat(parts).numpy()[restore] for name, parts in out.items()}
    sigma = cat["sigma"][:, None]
    mlp_bias = sum(_center(b.mlp.dense_4h_to_h.bias.detach().float().cpu()) @ readout for b in blocks[:layer])
    attention_bias = sum(_center(b.attention.dense.bias.detach().float().cpu()) @ readout for b in blocks[:layer])
    acts = np.stack([cat[f"acts{l}"] for l in range(layer)], axis=1)
    heads = np.stack([cat[f"heads{l}"] for l in range(layer)], axis=1) / sigma[:, :, None, None]
    return {
        "acts": acts,  # [E, layer, d_mlp]
        "sigma": cat["sigma"],  # [E]
        "pre": cat["pre"],  # [E, N]
        "heads": heads,  # [E, layer, heads, N]: each head's direct term
        "part_mlp_neurons": np.einsum("eli,lin->en", acts, mlp_coefficients(model, neurons, layer)) / sigma,
        "part_mlp_biases": mlp_bias.numpy() / sigma,
        "part_attention_heads": heads.sum(axis=(1, 2)),
        "part_attention_biases": attention_bias.numpy() / sigma,
        "part_embedding": cat["embed"] / sigma,
        "part_constant": np.broadcast_to(constant.numpy(), cat["pre"].shape),
    }


def record(step: int, corpus: np.ndarray, examples: list[tuple[int, int]], n_facet: int, neurons: list[int]) -> None:
    with checkpoint(step) as model:
        cap = capture(model, corpus, examples, neurons)
        coefficients = mlp_coefficients(model, neurons)
    parts = [v for k, v in cap.items() if k.startswith("part_")]
    error = float(np.abs(sum(parts) - cap["pre"]).max())
    np.savez(formation_path(step), examples=np.array(examples), n_facet_examples=n_facet, neurons=np.array(neurons),
             coefficients=coefficients, reconstruction_error=error,
             **{k: (v.astype(np.float16) if k == "acts" else v) for k, v in cap.items()})
    print(f"step {step}: recorded, largest reconstruction error {error:.1e}", flush=True)


def main() -> None:
    corpus = load_corpus()
    candidates = json.loads((RESULTS_DIR / "candidates.json").read_text())
    neurons = selected_neurons()
    facet_examples = sorted({(w, p) for c in candidates for facet in c["facets"] for w, p, _ in facet})
    background = background_examples(candidates, len(corpus), N_BACKGROUND, np.random.default_rng(SEED))
    FORMATION_DIR.mkdir(exist_ok=True)
    for step in STEPS:
        if formation_path(step).exists():
            print(f"step {step}: already recorded")
            continue
        record(step, corpus, facet_examples + background, len(facet_examples), neurons)


if __name__ == "__main__":
    main()
