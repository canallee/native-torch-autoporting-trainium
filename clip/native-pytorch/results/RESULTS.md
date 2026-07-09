# CLIP on Trainium — Result

**Status: CLIP FUNCTIONAL on Trainium (definition-as-product).** openai CLIP's own definition, vendored
and composable, verified for inference, one-step backprop, AND multi-step on-device training vs CPU.
Reference implementation of the "definition-as-product" pattern (M1, rebuilt).

## Environment
- Instance: `trn2.3xlarge`, 1 NeuronCore (`NEURON_RT_VISIBLE_CORES=0`).
- Software: native-PyTorch Neuron **Beta 3** (`torch-neuron`; torch 2.11.0, torch-neuronx 2.11.3, neuronxcc 2.25.1280).
- Model definition: **openai/CLIP `clip/model.py` vendored** in `src/clip_neuron/` (MIT), ViT-B/32.

## Phase-A reference (oracle) — captured from the SHIPPED definition
- `baselines/clip/MANIFEST.json`: 3 output tensors (image_features [2,512], text_features [2,512],
  logits_per_image [2,2]) + **302 grad tensors**, seed 1234, loss 105.8954 (sum-of-outputs).
- Baseline env: torch 2.11.0 CPU (apples-to-apples with the Trainium run).

## What worked (Phase B)
1. **Vendored the repo's own definition** (not an HF equivalent) — importable/composable `clip_neuron`.
2. **Inference: runs on `device="neuron"` unmodified.**
3. **Backprop needed one inline patch (R5).** `nn.MultiheadAttention`'s fused backward with a causal mask
   fails to compile on Neuron (`aten::add.out`, text tower); vision tower (mask-free) trains fine.
   Rewrote `ResidualAttentionBlock.attention` to manual QKV (linear/bmm/softmax) from the module's own
   params — same math, checkpoint-compatible, backward lowers cleanly. **CPU-validated vs the original
   before recompiling** (forward ~5e-7, grads ~5e-5).
4. fp32 (correctness); no other issues.

## Validation
**Inference** (`01_inference_parity.ipynb` / `port_clip.py`):
| Output | shape | cosine | max-abs |
|---|---|---|---|
| image_features | (2,512) | 1.000000 | 1.2e-05 |
| text_features | (2,512) | 1.000000 | 4.3e-06 |
| logits_per_image | (2,2) | 1.000000 | 1.7e-05 |

**Backprop** (`02_backprop_parity.ipynb` / `port_clip.py --grad`):
- CPU loss 105.895393 vs Trainium 105.895309; **302/302 grad tensors match**, 0 real disagreements
  (magnitude-aware gate — see `knowledge_base/validation_metrics.md`).

**Training** (`03_training_parity.ipynb`, dev-mode: on-device training) — 5-step Adam loop on CLIP's own
**symmetric InfoNCE** contrastive loss (the scaled cosine-similarity matrix's diagonal = matched image/text
pairs; CE over rows AND columns vs `arange(N)`), ALL 302 params trainable (both encoders + `logit_scale`),
`eval()` (dropout off) for a deterministic step. Self-compares CPU↔Trainium at runtime (no frozen oracle).
Loss decreases **0.710 → 0.350**; trajectory matches to **1.62e-4 relative**, final weights to **4.12e-5**.
Gate is RELATIVE (`|Δ|/|loss| ≤ 1e-2`), weights absolute. Two details: this trains through the text tower,
so it exercises the R5 manual-QKV attention rewrite on the backward path; and the loss is written with
`log_softmax` over rows+columns rather than two `F.cross_entropy(logits, ·)` + `logits.t()` calls — the
latter aliases storage and trips Neuron's in-place `cross_entropy` op ("variable modified by an inplace
operation" on the shared logits tensor's double backward).

All three notebooks run CPU/CUDA/Trainium (CUDA auto-skips on the trn box). Gate: **PASS** on all three.

## Files
- `src/clip_neuron/` — the vendored, Trainium-ready, composable **model definition** (the product).
- `src/clip_reference.py` — shared harness (model wrapper + deterministic inputs + loss).
- `notebooks/01_inference_parity.ipynb`, `02_backprop_parity.ipynb`, `03_training_parity.ipynb` — CPU/CUDA/Trainium parity.
- `src/port_clip.py` — headless CLI running the inference + backprop checks.

## Open / next
- Full multi-epoch / distributed contrastive training on a real image–text corpus — training phase (M5).
- ViT-L/14 — same `clip_neuron.load("ViT-L/14")`.
