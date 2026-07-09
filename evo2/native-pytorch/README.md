# Evo 2 on AWS Trainium — native PyTorch (definition-as-product, DEV mode)

Runs **Evo 2** (Arc Institute's StripedHyena2 DNA language model, 1B) on AWS Trainium. The product is
Evo2's **own model definition**, vendored in [`src/evo2_neuron/`](src/evo2_neuron/) with one inline Neuron
patch (FFT→conv1d) — importable & composable, so you can build/train new DNA models on its Hyena blocks.

> **Status:** ✅ Evo2-1B on Trainium — logits/hidden cosine **1.000000**, next-byte top-1 **100%** vs the
> original FFT model; backprop 265/265 gradients match. Single core, compile+run ~27s.

## Use it as a building block
```python
import sys; sys.path.insert(0, "src")
import evo2_neuron as E
model = E.load(device="neuron")                      # Evo2-1B, vendored + patched, fp32
ids = E.tokenize("ACGTACGTACGT").to("neuron")        # pass a STRING (byte tokenizer)
out = model(input_ids=ids, use_cache=False, output_hidden_states=True)
dna_embeddings = out.hidden_states[-1]               # (1, T, 1920)
next_byte_logits = out.logits                        # (1, T, 512)
```

## Environment
```bash
conda activate torch-neuron
pip install -r requirements.txt        # transformers, einops (definition is vendored)
```

## The three parity notebooks (the deliverable)
```bash
cd notebooks
NEURON_RT_VISIBLE_CORES=0 jupyter nbconvert --to notebook --execute --inplace 01_inference_parity.ipynb
NEURON_RT_VISIBLE_CORES=0 jupyter nbconvert --to notebook --execute --inplace 02_backprop_parity.ipynb
NEURON_RT_VISIBLE_CORES=0 jupyter nbconvert --to notebook --execute --inplace 03_training_parity.ipynb
```
- **01_inference_parity** — the port (conv1d) on CPU/CUDA/Trainium vs the **original FFT model**: proves
  the rewrite (CPU cosine 1.0) *and* device parity (Trainium cosine 1.0, top-1 100%).
- **02_backprop_parity** — one backward through the port on each device; gradients match (265/265).
- **03_training_parity** — 5-step Adam loop on Evo2's own next-token CE loss; asserts the loss decreases
  and the loss trajectory + final weights match CPU↔Trainium (on-device training).

## The one Neuron patch (the M2 headline)
neuronx-cc can't compile complex/FFT ops (`NCC_EVRF004`). StripedHyena2's Hyena long convolutions use
`torch.fft` (`fftconv_func` FIR path + `parallel_iir` modal-FFT IIR path). Each convolves a real signal
with a real filter, so it equals a **flipped depthwise causal `conv1d`** — the FFT is just an O(L log L)
form of the same linear convolution. The rewrite is inline in `src/evo2_neuron/engine.py` (marked
`NEURON PATCH`), CPU-validated against the FFT originals (cosine 1.000000) before running on device.
(The model's own `long_fir_threshold` conv branch is the *wrong*, no-flip one — not used.)

## Why the rest just works
`use_fp8_input_projections=False` (no TransformerEngine), `attn_implementation="eager"` (no flash-attn),
**fp32** (layers ~24–25 emit ~1e16 activations the final norm absorbs; bf16 collapses them to 0 on Neuron),
`use_cache=False` (static prefill). All applied by `evo2_neuron.load()`.

## Notes & limitations
- Validated: Evo2-1B, prefill, fp32 — inference (cosine 1.0, top-1 100%) + one backward (265/265 grads).
- Decode/generation (KV-cache + Hyena recurrence) and long context (8K+) not validated here.
- 7B/40B: same recipe; 40B needs sharding across cores (training phase / future work).

## Credits & license
HF port [`Taykhoom/Evo2-1B-8K`](https://huggingface.co/Taykhoom/Evo2-1B-8K) (Apache-2.0), a pure-PyTorch
re-implementation of [arcinstitute/evo2](https://github.com/ArcInstitute/evo2). Definition vendored — see
[`src/evo2_neuron/VENDORED.md`](src/evo2_neuron/VENDORED.md).
