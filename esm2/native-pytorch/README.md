# ESM-2 on AWS Trainium — feature extractor (INTEGRATION mode)

Reference example of **integration mode**: use ESM-2's per-residue embeddings (the amino-acid MLM head
**dropped**) as inputs to a **new model** on Trainium, with correct inference **and** gradient flow — so you
can train the new model / finetune the backbone. You are *not* modifying ESM's architecture, so its
definition is used **sealed** (installed HuggingFace `transformers.EsmModel`), not vendored.

> **Status:** ✅ Embeddings match CPU at cosine 1.000000. Gradient flow (backbone + a new downstream head)
> matches CPU 101/101 tensors. **No Neuron patches needed.**

## Use it as a building block

```python
import sys; sys.path.insert(0, "src")
import esm2_neuron as E

esm = E.load(device="neuron")                     # ESM-2 encoder (no MLM head), on Trainium
ids, mask = E.tokenize(["MKTAYIAK...", "MSTNPKPQ..."])
emb = E.embed(esm, ids.to("neuron"), mask.to("neuron"))        # (B, L, 320) per-residue features

# compose into a new model (finetune or freeze the backbone):
E.freeze(esm)                                     # frozen feature-extractor, or skip to finetune
class MyModel(torch.nn.Module):
    def __init__(self, esm): ...                  # esm backbone + your head
```
See `src/esm2_reference.py` `Composite` for the backbone+head pattern used in the parity checks.

## Environment
```bash
conda activate torch-neuron
pip install -r requirements.txt          # transformers (ESM-2 definition is the installed HF EsmModel)
```

## The parity notebooks (the deliverable)
```bash
cd notebooks
NEURON_RT_VISIBLE_CORES=0 jupyter nbconvert --to notebook --execute --inplace 01_inference_parity.ipynb
NEURON_RT_VISIBLE_CORES=0 jupyter nbconvert --to notebook --execute --inplace 02_backprop_parity.ipynb
NEURON_RT_VISIBLE_CORES=0 jupyter nbconvert --to notebook --execute --inplace 03_training_parity.ipynb
```
- **01_inference_parity** — CPU/CUDA/Trainium; asserts the extracted **embeddings** match (cosine 1.000000).
- **02_backprop_parity** — the integration-critical test: one backward through a **composite (ESM-2 +
  new head)** on each device; asserts **gradients flow through the backbone AND into the new head** match
  (101/101). CUDA auto-skips on the trn box.
- **03_training_parity** — the **dev-mode training-parity check** (a real multi-step Adam training loop,
  backbone + new head; asserts the **loss trajectory + final weights match** CPU, max per-step diff 4e-6,
  and the loss decreases). It's a *dev-mode* deliverable — demonstrated here on ESM-2 as the smallest/fastest
  model for the on-device training smoke test — proving you can *train* on-device, not just run one backward.

## Why this works
`EsmModel` (not `EsmForMaskedLM`) is exactly the used sub-graph — the MLM head is simply never
instantiated, so any Neuron-hostile op it might contain is irrelevant (the integration-mode payoff). ESM's
attention is explicit (linear/matmul/softmax), so both forward and backward lower cleanly with no patch.

## Notes & limitations
- Validated: `esm2_t6_8M_UR50D`, fp32, seqlen 64 — embeddings + gradient flow into a downstream head.
- Swap `E.MODEL_NAME` for larger tiers (t12_35M / t30_150M / t33_650M) — same recipe, single core to ~650M.
- Frozen-backbone vs finetune both supported (`R.load(..., finetune_backbone=False)`); a full training loop
  is the training phase (M5).

## Credits & license
ESM-2 © Meta AI, via HuggingFace `transformers` (`facebook/esm2_t6_8M_UR50D`). Definition used as-is (sealed).
