# Vendored source

`modeling.py` is GPN-Star's model definition, copied **verbatim** from the `gpn` package's
`gpn/star/model.py` ([songlab-cal/gpn](https://github.com/songlab-cal/gpn), MIT) — the
`GPNStarConfig` + `GPNStarModel` (phylogeny-aware axial-attention MSA encoder) + `GPNStarForMaskedLM`
(masked-LM head). It is self-contained: needs only transformers + networkx (+ torch/numpy), so vendoring
it drops the `gpn`-package dependency for the port.

## Neuron patches (compute path): NONE
The active path (row self-attention + column/cross attention, FIRE relative-position/phylo bias, RoFormer
rotary, MLM head) is plain matmul / softmax / linear — it lowers cleanly on Neuron for both **inference and
backprop** (verified: logits cosine 1.000000, all 240 param gradients match CPU on the Trainium core).

## Head compat shim (1, inline in `modeling.py`)
- **`GPNStarForMaskedLM._tied_weights_keys` list → `{}`** (`NEURON PATCH`). transformers 4.x used a LIST;
  5.x `post_init`/`get_expanded_tied_weights_keys` expects a DICT (a list raises `'list' object has no
  attribute 'keys'`). The checkpoint stores `cls.predictions.decoder.weight`; only `decoder.bias` is absent
  and is tied to `cls.predictions.bias` by the loader. No architecture/math change.

## Load-time shim (in `__init__.load`, not a source edit)
- **Construct on the real device + `load_state_dict`**, not `from_pretrained`. GPN-Star's `__init__` does
  real tensor work (loads `phylo_dist/` numpy, clusters clades via networkx, `.item()`/`.max()`), but
  transformers 5.13 builds every `from_pretrained` model under `torch.device("meta")` → `Tensor.item()
  cannot be called on meta tensors`. Direct construction avoids the meta context. See `loading_and_deps.md`.

Runs in the **`gpn-vep`** env (networkx present).
