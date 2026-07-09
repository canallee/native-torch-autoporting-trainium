# GPN-Star on Trainium — Result (DEV mode)

**Status: full GPN-Star (`GPNStarForMaskedLM`) FUNCTIONAL on Trainium** — the vendored, editable,
composable definition (MSA encoder + MLM head). Inference: logits cosine **1.000000** (max-abs 1.1e-5).
Backprop: **all 240 parameter gradients** match CPU, 0 real disagreements. Training: 5-step Adam on
GPN-Star's own masked-LM loss, loss decreases + trajectory/weights match CPU↔Trainium. No Neuron op patches
(one head compat shim + one load-time shim). Real-data variant-effect validation: `example/vep/`.

## Environment
- Instance: trn2.3xlarge, 1 NeuronCore (`NEURON_RT_VISIBLE_CORES=0`), eager `device="neuron"` path.
- Software: native-PyTorch Neuron **Beta 3** (torch 2.11.0+cpu, torch-neuronx 2.11.3.0.1278,
  neuronxcc 2.25.1280, transformers 5.13.0). Runs in the **`gpn-vep`** env (a clone of `torch-neuron` that
  adds `networkx` — the vendored definition's only extra dep — and `gpn`, used by the VEP example).
  See `references/environment.md`.
- Model definition: **VENDORED** — `GPNStarConfig` + `GPNStarModel` + `GPNStarForMaskedLM` copied verbatim
  from the `gpn` package's `gpn/star/model.py` (MIT) into `src/gpn_star_neuron/modeling.py`; see
  `VENDORED.md`. Vendoring drops the `gpn`-package import (needs only transformers + networkx). Weights
  `songlab/gpn-star-tair10-b18-25m` via `huggingface_hub.snapshot_download`.

## Phase-A reference (oracle)
- Baseline env: torch 2.11.0 (CPU), captured directly in the Beta-3 stack (GPN-Star runs as-is — only a
  load-time shim, no extra/conflicting deps → no separate reference env).
- Manifest: `baselines/gpn-star/MANIFEST.json` — 1 output tensor (logits `(1,64,1,6)`, the full MLM head),
  **240 grad tensors**, seed 1234, `adapter.loss_fn` = -1745.146.
- Inputs: deterministic **synthetic** MSA — `input_ids (1,64,1)`, `source_ids (1,64,18)`,
  `target_species (1,1)` (valid dtypes/shapes/vocab range; parity needs identical inputs, not real data).

## What worked (Phase B recipe, dev mode)
1. **Vendored the repo's own definition (R1).** The full `GPNStarForMaskedLM` (axial-attention MSA encoder +
   FIRE phylo-bias + MLM head) is the product — editable/composable. Vendored (not sealed) because the
   used path needs load-time shims and the definition is worth owning; drops the `gpn`-package dependency.
2. **Neutralizations (R2).** None required. GPN-Star's attention already uses
   `F.scaled_dot_product_attention(enable_math=True)`; `use_cache`/decoder paths are off; shapes are static.
3. **Load-time shim (loading_and_deps).** Custom loader instead of `from_pretrained`: transformers-5.13
   builds models under `torch.device("meta")`, and GPN-Star's `__init__` does real tensor work (phylo numpy
   load + networkx clade clustering + `.item()`), which raises `Tensor.item() cannot be called on meta
   tensors`. Fix: construct `GPNStarForMaskedLM(config)` on the real device and `load_state_dict`. Plus one
   head compat shim: `_tied_weights_keys` list→{} and tie the one absent `decoder.bias`. No architecture edit.
4. **Op rewrites (R5).** None. Both forward and backward lower cleanly on the encoder AND the MLM head.
5. **Precision (R6).** fp32 (default). No collapse observed.

## Validation (R7)
- **Inference** (`01_inference_parity.ipynb`) — Neuron vs CPU at runtime: logits `(1,64,1,6)` **cosine
  1.000000, max-abs 1.144e-05**. PASS (≥ 0.99).
- **Backprop** (`02_backprop_parity.ipynb`) — one backward through the full model on each device,
  magnitude-aware gate: **240/240** grad tensors matched, global |grad| = 6.102e+02, **0 real
  disagreements** (cosine < 0.99 AND max-abs/scale > 1e-3). PASS.
- **Training** (`03_training_parity.ipynb`, dev-mode: on-device training) — self-compare CPU↔Trainium at
  runtime. 5-step Adam loop on GPN-Star's own **masked-token CE** (MLM: 24/64 target positions masked with a
  fixed in-vocab id, predict the originals), ALL ~25M params trainable, `eval()` (dropout off) for a
  deterministic step. No frozen oracle. Loss decreases **7.603 → 1.853**; trajectory matches to **6.90e-7
  relative**, final weights to **2.18e-4**. Gate is RELATIVE (`|Δ|/|loss| ≤ 1e-2`), weights absolute. PASS.
- Compile: first on-device forward/backward ~1–2 min (short seq, single core); warm runs fast.

## Files (src/)
- `gpn_star_neuron/__init__.py` — the dev-mode interface: `load()` (full `GPNStarForMaskedLM` via the custom
  loader), `submodules()` (encoder / cls head — compose freely), `num_species()`, `build_synthetic_msa()`,
  `freeze`/`unfreeze`.
- `gpn_star_neuron/modeling.py` + `VENDORED.md` — the vendored definition + the two inline shims.
- `gpn_star_reference.py` — shared full-model wrapper (`GPNStarWrapper`), `build_inputs`, `loss_fn`;
  used by the oracle capture AND all three notebooks (byte-identical model + inputs).
- No `*_patch.py` — no Neuron op patches were needed.

## Open / next
- Larger GPN-Star tiers (85M / 200M) and longer windows; multi-target-row (`T>1`) alignments.
- Real tokenized MSA inputs (via `gpn.star.data.GenomeMSA`) in place of the synthetic tensors.
- A full multi-epoch / distributed finetune — training phase.
