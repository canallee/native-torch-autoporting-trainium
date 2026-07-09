# ESM-2 on Trainium — Result (INTEGRATION mode)

**Status: ESM-2 FUNCTIONAL on Trainium as a feature extractor.** Reference example of **integration
mode**: extract per-residue embeddings (MLM head dropped) and feed a new downstream model, with verified
inference and gradient flow. No Neuron patches.

## Environment
- Instance: `trn2.3xlarge`, 1 NeuronCore (`NEURON_RT_VISIBLE_CORES=0`).
- Software: native-PyTorch Neuron **Beta 3** (`torch-neuron`; torch 2.11.0, torch-neuronx 2.11.3, neuronxcc 2.25.1280).
- Model definition: **installed HF `transformers.EsmModel`** (sealed, NOT vendored — integration mode),
  `facebook/esm2_t6_8M_UR50D` (320-d, 6 layers).

## Mode: integration (vs CLIP's dev mode)
- Used sub-graph = encoder → `last_hidden_state`; the amino-acid MLM head is **never instantiated**
  (`EsmModel`, not `EsmForMaskedLM`).
- Definition sealed/installed (no architecture edits), version recorded — not vendored.
- Deliverable interface: `esm2_neuron.embed()` / `embed_pooled()` + `freeze`/`unfreeze`.

## Phase-A reference (oracle) — the integration path
- `baselines/esm2/MANIFEST.json`: outputs = embeddings (2, 64, 320) + downstream logits (2, 2);
  **101 grad tensors** (99 backbone + 2 new head), seed 1234.
- Composite = ESM-2 backbone + a toy `Linear(320→2)` head on masked-mean pooling (the integration use).

## Validation
**Inference** (`01_inference_parity.ipynb`): embeddings cosine **1.000000** (max-abs 4.3e-6); logits cosine 1.000000.

**Gradient flow** (`02_backprop_parity.ipynb`) — the integration-critical test: one backward through the
composite on each device.
- **101/101 grad tensors match** (99 backbone + 2 new head), magnitude-aware gate. global |grad| scale 1.97e3.
- Confirms gradients flow through the **sealed ESM-2 backbone** *and* into the **new downstream head** on
  Trainium — i.e. a new model built on ESM-2 embeddings trains correctly, and the backbone can be finetuned.

Both notebooks run CPU/CUDA/Trainium (CUDA auto-skips on the trn box). Gate: **PASS** on both.

## Files
- `src/esm2_neuron/` — the integration feature-extractor interface over the sealed HF definition.
- `src/esm2_reference.py` — shared harness (`Composite` backbone+head, deterministic inputs, loss).
- `notebooks/01_inference_parity.ipynb`, `02_backprop_parity.ipynb`, `03_training_parity.ipynb`.

## Training (M5) — on-device training loop ✅
`03_training_parity.ipynb`: 8-step Adam training of the composite (ESM-2 backbone + new head), CPU vs
Trainium. **Loss trajectory identical** (max per-step diff 4.05e-6), **final weights match** (1.25e-3), and
the loss decreases 0.685→0.016. Proves `optimizer.step()`, Adam optimizer state, and multi-step convergence
work on-device.

## Open / next
- Larger tiers (t12_35M … t33_650M): same recipe, single core.
- Training beyond the smoke test: multi-batch datasets, LR schedules, checkpointing, multi-core/FSDP.
- Frozen-backbone variant: `esm2_reference.load(..., finetune_backbone=False)`.
