# Precision pitfalls & source triage

## Precision (R6)
- **Default fp32 for correctness.** bf16 collapses in known spots:
  - evo2: layers 24–25 emit ~1.8e16 activations the final RMSNorm absorbs; in bf16 on Neuron the
    norm collapses to exactly 0. `model.float()` fixes it.
  - FLUX / high-res diffusion: residual/softmax reductions past ~1 MP need fp32 or the image goes gray.
- **Mixed precision is the optimization, not the default.** Once correct in fp32, isolate fp32 to just
  the norm / massive-activation layers and run the rest in bf16 (~2× on the matrix engines, and the
  path to fitting bigger models). Re-validate against the oracle after any precision change.
- **Match the reference's accumulation.** Op-rewrites compute in fp32 then cast back, matching the
  original's fp32 FFT/accumulation (evo2). A head that's slightly off (alphagenome `contact_maps` ~0.4%)
  is usually fp32-accumulation drift — force that head's einsums to fp32 if it matters.
- **⚠️ A definition may force its OWN dtype in `__init__`.** Some models call a `force_dtype()`/`half()` in
  their constructor (evo2 → bf16-except-poles) — so `from_pretrained(..., dtype=torch.float32)` does NOT
  yield fp32. Apply **`.float()` AFTER construction** (or after `load()`), and verify a param's dtype.

## Source triage (R1 / Phase A2)
- **Pick the pure-PyTorch entry point.** HF `transformers` class or a `trust_remote_code` port over the
  CUDA-native original.
  - evo2 → `Taykhoom/Evo2-1B-8K` HF port, NOT `vortex`/TransformerEngine.
  - bacformer → plain `EsmModel` for Stage-1, NOT `faesm` (CUDA flash-attn).
- **Red-flag deps** that can't run on Neuron: `flash-attn`, `transformer_engine`/FP8, `vortex`, `faesm`,
  custom CUDA kernels, `torch.fft`/complex, `sort`/top-k over long sequences.
- **Size → instance:** single NeuronCore for <~1B; `trn2.48xlarge` + tensor/sequence sharding for 40B+.
- **Two-stage models** (bacformer: ESM-2 embed → genome transformer): port/validate each stage separately.
