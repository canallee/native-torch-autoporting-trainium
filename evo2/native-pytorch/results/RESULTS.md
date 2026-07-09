# Evo 2 on Trainium — Result (DEV mode, FFT→conv1d)

**Status: Evo2-1B FUNCTIONAL on Trainium.** The headline M2 result — the unsupported-op debug loop:
complex FFT (`NCC_EVRF004`) → real `conv1d`, CPU-validated then device-validated.

## Environment
- Instance: `trn2.3xlarge`, 1 NeuronCore (`NEURON_RT_VISIBLE_CORES=0`). Compile+run ~27s.
- Software: native-PyTorch Neuron **Beta 3** (`torch-neuron`; torch 2.11.0, torch-neuronx 2.11.3, neuronxcc 2.25.1280).
- Model definition: **vendored** HF port `Taykhoom/Evo2-1B-8K` (StripedHyena2, 1B, pure PyTorch) in
  `src/evo2_neuron/`. Single env — the port runs under our transformers 5.13 (no isolated env needed; the
  tokenizer just takes a `str`, not a `list`).

## What worked (Phase B, dev mode)
1. **Vendored the repo's own definition** (9 modeling files) — composable `evo2_neuron.load()`.
2. **R5 op rewrite — FFT → conv1d (the real work).** Two Hyena FFT sites in `engine.py`
   (`fftconv_func`, `parallel_iir` modal-FFT branch) → flipped depthwise causal `conv1d`, inline patch
   (`NEURON PATCH`). **CPU-validated vs the FFT originals first** (logits/hidden cosine 1.000000, top-1
   100%), then run on device.
3. **R2/R6 recipe:** `use_fp8_input_projections=False`, `attn_implementation="eager"`, **fp32** (bf16
   collapses the norm on ~1e16 activations), `use_cache=False`.

## Phase-A reference (oracle)
- `baselines/evo2/MANIFEST.json`: outputs = hidden (1,32,1920) + logits (1,32,512); **265 grad tensors**,
  seed 1234. Oracle captured from the **original FFT model** (ground truth) on CPU.

## Validation
**Inference** (`01_inference_parity.ipynb`) — port (conv1d) vs original FFT model:
| device | hidden cosine | logits cosine | top-1 next-byte |
|---|---|---|---|
| CPU (rewrite proof) | 1.000001 | 1.000000 | 100% |
| Trainium | 1.000001 (max-abs 1.6e-2) | 1.000000 (max-abs 1.0e-1) | 100% |

**Backprop** (`02_backprop_parity.ipynb`): **265/265 grad tensors match** CPU↔Trainium (magnitude-aware).
Logits max-abs is larger (pre-softmax, large scale) but cosine 1.0 + top-1 100% = functionally exact.

**Training** (`03_training_parity.ipynb`, dev-mode: on-device training) — 5-step Adam loop on Evo2's own
next-token CE (shift logits/labels) over the fixed DNA batch, ALL 1B params trainable, `eval()` (dropout
off) for a deterministic step. Self-compares CPU↔Trainium at runtime (no frozen oracle). Loss decreases
**0.951 → 0.198** (below uniform ln 512 ≈ 6.24 → genuinely learning the periodic input); trajectory matches
to **2.31e-4 relative**, final weights to **5.18e-5**. Gate is RELATIVE (`|Δ|/|loss| ≤ 1e-2`) — an absolute
gate false-FAILs large-loss models. `LR=1e-5`: the pretrained model already fits this in-distribution input,
so a larger LR overshoots into a steep, near-saturated basin where CE is fp-noisy (weights still agree, but
the loss-trajectory gate flakes); a gentle LR keeps the descent monotonic and the comparison meaningful.

Gate: **PASS** on all three.

## Files
- `src/evo2_neuron/` — vendored Evo2 definition + inline FFT→conv1d patch (`engine.py`), `load()`/`tokenize()`.
- `src/evo2_reference.py` — port vs FFT-reference harness.
- `notebooks/01_inference_parity.ipynb`, `02_backprop_parity.ipynb`, `03_training_parity.ipynb`.

## Open / next
- Decode/generation (KV-cache + Hyena `step_fir`/`step_iir`) and long context (8K+): not validated.
- 7B/40B: same recipe + sharding (training phase / future).
