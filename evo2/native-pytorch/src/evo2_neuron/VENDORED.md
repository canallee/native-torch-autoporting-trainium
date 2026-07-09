# Vendored source

Evo 2 (StripedHyena2) model definition, copied **verbatim** from the HuggingFace port
[`Taykhoom/Evo2-1B-8K`](https://huggingface.co/Taykhoom/Evo2-1B-8K) (a pure-PyTorch re-implementation of
the Arc Institute model; **not** `vortex`/TransformerEngine). Files: `modeling_evo2.py`, `hyena.py`,
`engine.py`, `attention.py`, `rotary.py`, `layers.py`, `configuration_evo2.py`, `cache.py`,
`tokenization_evo2.py`. License: Apache-2.0.

## Neuron patches (1)
- **`engine.py` — FFT → conv1d** (marked `NEURON PATCH` at end of file). neuronx-cc cannot compile
  complex/FFT ops (`NCC_EVRF004`). The two Hyena FFT sites (`fftconv_func` FIR path and
  `parallel_iir`'s modal-FFT IIR branch) each convolve a real signal with a real finite filter, so each
  equals a **flipped depthwise causal `conv1d`**. Overridden for the prefill path (`inference_params=None`).
  Validated equal to the FFT originals on CPU: logits/hidden cosine **1.000000**, top-1 **100%**; and on
  Trainium vs the CPU FFT reference: cosine **1.000000**, top-1 **100%** (compile+run ~27s, single core).

Everything else is unmodified. Recipe (applied by `__init__.load`): `use_fp8_input_projections=False`,
`attn_implementation="eager"`, fp32, `use_cache=False`.
