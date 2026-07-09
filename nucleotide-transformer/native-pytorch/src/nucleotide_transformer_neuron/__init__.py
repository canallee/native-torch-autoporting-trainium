"""nucleotide_transformer_neuron — Nucleotide Transformer's OWN definition, on Trainium (DEV mode).

The porting product is the model's **own, editable, composable definition** — the full
`EsmForMaskedLM` (encoder + MLM head), vendored in `modeling_esm.py` / `esm_config.py` and made to run
on Trainium — so you can subclass it, swap modules, and build/train new DNA models from its pieces.

Target: `InstaDeepAI/nucleotide-transformer-v2-50m-multi-species` — a DNA foundation model, ESM-family
architecture with ROTARY position embeddings, 512-d, 12 layers, 6-mer DNA tokenizer (vocab 4107).

Vendored + patched (see VENDORED.md): the model ships as `trust_remote_code` written against transformers
4.32 and won't import/build on the Beta-3 stack (transformers 5.13). Three tiny compat shims restore
removed transformers-4.x helpers — NO architecture/math change. On device the active attention path is
plain matmul/softmax + rotary, so inference AND backprop lower cleanly with no Neuron op-rewrite.

    import nucleotide_transformer_neuron as NT
    model = NT.load(device="neuron")               # full EsmForMaskedLM, on Trainium
    ids, mask = NT.tokenize(["ACGT...", "GGCC..."])
    out = model(input_ids=ids.to("neuron"), attention_mask=mask.to("neuron"),
                output_hidden_states=True)
    logits = out.logits                            # (B, L, vocab)  MLM head
    hidden = out.hidden_states[-1]                 # (B, L, 512)     per-token features

    parts = NT.submodules(model)                   # encoder / embeddings / lm_head — compose freely
    NT.freeze(parts["encoder"])                    # e.g. frozen backbone + a new head
"""
import json
import os

import torch

from .esm_config import EsmConfig
from .modeling_esm import EsmForMaskedLM

MODEL_ID = "InstaDeepAI/nucleotide-transformer-v2-50m-multi-species"


def _resolve_config(model_id: str = MODEL_ID) -> EsmConfig:
    """Build the vendored EsmConfig from the repo's config.json (local path or HF hub id), stripping
    `auto_map`/`architectures` so no `trust_remote_code` prompt fires — we use the VENDORED definition."""
    if os.path.isdir(model_id):
        cfg_path = os.path.join(model_id, "config.json")
    else:
        from huggingface_hub import hf_hub_download
        cfg_path = hf_hub_download(model_id, "config.json")
    with open(cfg_path) as f:
        d = json.load(f)
    d.pop("auto_map", None)
    d.pop("architectures", None)
    return EsmConfig(**d)


def _weights_path(model_id: str = MODEL_ID) -> str:
    if os.path.isdir(model_id):
        return os.path.join(model_id, "model.safetensors")
    from huggingface_hub import hf_hub_download
    return hf_hub_download(model_id, "model.safetensors")


def load(model_id: str = MODEL_ID, device="cpu", dtype=torch.float32) -> EsmForMaskedLM:
    """Load the FULL Nucleotide Transformer (`EsmForMaskedLM`, encoder + MLM head) on `device`, fp32,
    from the VENDORED Beta-3-compatible definition. `model_id` = HF hub id (default) or a local repo path.

    Constructs the model directly + `load_state_dict` (NOT `from_pretrained`): transformers-5.13's
    from_pretrained tied-weights machinery chokes on this 4.x-authored MLM head; direct construction on
    the real device sidesteps it (the checkpoint stores `lm_head.decoder.weight`, so no auto-tie needed)."""
    from safetensors.torch import load_file
    config = _resolve_config(model_id)
    torch.manual_seed(0)                                  # deterministic pre-load init (all weights overwritten)
    model = EsmForMaskedLM(config).eval().to(dtype)
    missing, unexpected = model.load_state_dict(load_file(_weights_path(model_id)), strict=False)
    # tolerated: buffers like `*.position_ids` / a contact-head not in the checkpoint (unused on the fwd path)
    assert not [k for k in missing if "position_ids" not in k and "contact" not in k], f"missing weights: {missing}"
    return model.to(torch.device(device))


def submodules(model: EsmForMaskedLM) -> dict:
    """Named handles to the model's own building blocks, for composing into new models."""
    return {
        "encoder": model.esm,                    # EsmModel (embeddings + transformer stack)
        "embeddings": model.esm.embeddings,
        "transformer": model.esm.encoder,
        "lm_head": model.lm_head,
    }


def get_tokenizer(model_id: str = MODEL_ID):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model_id)


def tokenize(sequences, model_id: str = MODEL_ID, max_length: int = 64):
    """DNA strings -> fixed-shape (input_ids, attention_mask). Static shape for Neuron.
    The tokenizer segments DNA into 6-mers; keep sequences short for fast compiles."""
    tok = get_tokenizer(model_id)
    enc = tok(list(sequences), return_tensors="pt", padding="max_length",
              truncation=True, max_length=max_length)
    return enc["input_ids"], enc["attention_mask"]


def freeze(module: torch.nn.Module):
    for p in module.parameters():
        p.requires_grad_(False)
    return module


def unfreeze(module: torch.nn.Module):
    for p in module.parameters():
        p.requires_grad_(True)
    return module
