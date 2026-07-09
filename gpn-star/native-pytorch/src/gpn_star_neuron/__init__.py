"""gpn_star_neuron — GPN-Star's OWN definition, on Trainium (DEV mode).

The porting product is GPN-Star's own, editable, composable definition — the full
`GPNStarForMaskedLM` (phylogeny-aware MSA encoder + masked-LM head), **vendored** in `modeling.py`
(from `gpn/star/model.py`, MIT; see VENDORED.md) and made to run on Trainium — so you can
subclass it, swap the axial-attention / FIRE-bias modules, and build/train new genomic models from it.
Vendoring the definition also drops the `gpn`-package dependency: `modeling.py` needs only transformers +
networkx (+ torch/numpy).

GPN-Star (`songlab/gpn-star-tair10-b18-25m`, `GPNStarConfig(RoFormerConfig)`, ~25M, 512-hidden/8-layer)
is a phylogeny-aware, MSA-input genomic LM: axial attention over a whole-genome alignment (row
self-attention along the sequence + column/cross attention across aligned species/clades), a FIRE
relative-position + phylo-distance bias, and RoFormer rotary positions. Its forward takes MSA-style
tensors, NOT a single DNA string:
    input_ids       (B, L, T)   token ids for the T target rows (T=1 = reference genome)
    source_ids      (B, L, N)   token ids for the full N-species alignment column (N=18 for tair10-b18)
    target_species  (B, T)      index (into the N species) of each target row

    import gpn_star_neuron as G
    model = G.load(device="neuron")                        # full GPNStarForMaskedLM, on Trainium
    N = G.num_species()
    input_ids, source_ids, target_species = G.build_synthetic_msa(N)   # or your real tokenized MSA
    out = model(input_ids=input_ids.to("neuron"), source_ids=source_ids.to("neuron"),
                target_species=target_species.to("neuron"), output_hidden_states=True)
    logits = out.logits                                    # (B, L, T, vocab)  MLM head
    parts  = G.submodules(model)                           # encoder / cls head — compose freely

Neuron patches: NONE on the compute path (axial attention is matmul/softmax + rotary — lowers cleanly
forward + backward). Two LOAD-TIME shims: (1) construct on the real device (transformers-5.13 meta-init
crashes GPN-Star's tensor-computing __init__), (2) `_tied_weights_keys` list→{} in modeling.py + tie the
one absent `decoder.bias`. Runs in the `gpn-vep` env (networkx present); see README.
"""
import os

import numpy as np
import torch

from .modeling import GPNStarConfig, GPNStarForMaskedLM, GPNStarModel  # vendored definition

MODEL_NAME = "songlab/gpn-star-tair10-b18-25m"   # arabidopsis, 25M params, 18-species alignment


def _snapshot(model_name: str = MODEL_NAME) -> str:
    from huggingface_hub import snapshot_download
    return snapshot_download(model_name)


def _config(model_name: str = MODEL_NAME):
    """Build the vendored GPNStarConfig from config.json, repairing `phylo_dist_path` to the snapshot's
    bundled `phylo_dist/` dir (the checkpoint records a training-time path that doesn't exist at inference)."""
    path = _snapshot(model_name)
    cfg = GPNStarConfig.from_pretrained(path)
    if not os.path.exists(cfg.phylo_dist_path):
        cfg.phylo_dist_path = os.path.join(path, "phylo_dist")
    return cfg, path


def num_species(model_name: str = MODEL_NAME) -> int:
    """N — number of aligned species in the checkpoint's phylogeny (18 for tair10-b18)."""
    path = _snapshot(model_name)
    return int(np.load(os.path.join(path, "phylo_dist", "pairwise.npy")).shape[0])


def load(model_name: str = MODEL_NAME, device="cpu", dtype=torch.float32) -> GPNStarForMaskedLM:
    """Load the FULL GPN-Star (`GPNStarForMaskedLM`) on `device`, fp32, from the VENDORED definition.

    Constructs on the real device (bypassing transformers-5.13 meta-init, which GPN-Star's __init__ can't
    survive) + `load_state_dict`, then ties the one absent `decoder.bias` (standard MLM head)."""
    from safetensors.torch import load_file
    cfg, path = _config(model_name)
    torch.manual_seed(0)                                    # deterministic pre-load init (weights overwritten)
    model = GPNStarForMaskedLM(cfg).eval().to(dtype)
    missing, unexpected = model.load_state_dict(load_file(os.path.join(path, "model.safetensors")), strict=False)
    assert missing == ["cls.predictions.decoder.bias"] and not unexpected, \
        f"unexpected weight mismatch: missing={missing} unexpected={unexpected}"
    model.cls.predictions.decoder.bias = model.cls.predictions.bias   # tie the one absent bias
    return model.to(torch.device(device))


def submodules(model: GPNStarForMaskedLM) -> dict:
    """Named handles to GPN-Star's own building blocks, for composing into new models."""
    return {
        "encoder": model.model,                          # GPNStarModel (axial attention + FIRE phylo-bias)
        "transformer": model.model.encoder,              # the axial-attention layer stack
        "target_embedding": model.model.target_embedding,
        "source_embedding": model.model.source_embedding,
        "cls": model.cls,                                # masked-LM head
    }


def build_synthetic_msa(n_species: int, batch_size: int = 1, seq_len: int = 64,
                        n_targets: int = 1, vocab_size: int = 6, seed: int = 1234):
    """Deterministic, VALID-SHAPED synthetic MSA inputs (correct dtypes/shapes/vocab ranges) for parity.
    Returns (input_ids (B,L,T), source_ids (B,L,N), target_species (B,T))."""
    g = torch.Generator().manual_seed(seed)
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len, n_targets), generator=g)
    source_ids = torch.randint(0, vocab_size, (batch_size, seq_len, n_species), generator=g)
    target_species = torch.zeros((batch_size, n_targets), dtype=torch.long)
    return input_ids, source_ids, target_species


def freeze(module: torch.nn.Module):
    for p in module.parameters():
        p.requires_grad_(False)
    return module


def unfreeze(module: torch.nn.Module):
    for p in module.parameters():
        p.requires_grad_(True)
    return module
