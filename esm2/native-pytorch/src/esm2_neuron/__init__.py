"""esm2_neuron — ESM-2 as a Trainium feature extractor (INTEGRATION mode).

Scenario: use ESM-2's per-residue embeddings (NO amino-acid MLM head) as inputs to a NEW model,
on Trainium, with correct inference AND gradient flow (so the new model can train / the backbone
can be finetuned).

Integration-mode choices (vs dev-mode / CLIP):
- **Sealed, installed definition.** We use HuggingFace `transformers.EsmModel` as-is — we are not
  modifying ESM's architecture, so the definition is NOT vendored (its version is recorded instead).
  `EsmModel` (not `EsmForMaskedLM`) already **drops the MLM head** — exactly the used sub-graph.
- **Feature interface**, not the full model: `embed()` (per-residue) / `embed_pooled()` (masked mean),
  plus `freeze`/`unfreeze` for frozen-backbone vs finetune.
- Verified on the Beta-3 stack: embeddings match CPU at cosine 1.0, and gradients flow through the
  backbone into a downstream head at cosine 1.0 — with NO Neuron patches.
"""
import torch

MODEL_NAME = "facebook/esm2_t6_8M_UR50D"   # 8M, 320-d, 6 layers; swap for t12_35M / t30_150M / t33_650M


def load(model_name: str = MODEL_NAME, device="cpu", dtype=torch.float32):
    """Load ESM-2 as an encoder (MLM head dropped, no pooler) on `device`, fp32."""
    from transformers import EsmModel
    model = EsmModel.from_pretrained(model_name, add_pooling_layer=False).eval().to(dtype)
    return model.to(torch.device(device))


def get_tokenizer(model_name: str = MODEL_NAME):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model_name)


def tokenize(sequences, model_name: str = MODEL_NAME, max_length: int = 64):
    """Protein strings -> fixed-shape (input_ids, attention_mask). Static shape for Neuron."""
    tok = get_tokenizer(model_name)
    enc = tok(list(sequences), return_tensors="pt", padding="max_length",
              truncation=True, max_length=max_length)
    return enc["input_ids"], enc["attention_mask"]


def embed(model, input_ids, attention_mask):
    """Per-residue embeddings (B, L, D) — the general feature for a downstream model."""
    return model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state


def embed_pooled(model, input_ids, attention_mask):
    """Masked-mean per-sequence embedding (B, D)."""
    h = embed(model, input_ids, attention_mask)
    m = attention_mask.unsqueeze(-1).to(h.dtype)
    return (h * m).sum(1) / m.sum(1).clamp_min(1.0)


def freeze(module: torch.nn.Module):
    for p in module.parameters():
        p.requires_grad_(False)
    return module


def unfreeze(module: torch.nn.Module):
    for p in module.parameters():
        p.requires_grad_(True)
    return module
