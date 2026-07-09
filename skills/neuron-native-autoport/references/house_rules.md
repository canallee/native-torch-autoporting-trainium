# House Rules — hard-won lessons (from native-PyTorch reference ports)

Operating principles distilled from prior native-PyTorch Trainium ports. Apply them by default;
they encode debugging hours.

- **Native PyTorch first.** On the Beta-3 stack that means eager **`torch.device("neuron")`** or
  **`torch.compile(backend="neuron")`** — there is **no `torch_neuronx.trace`** here (that's the public
  SDK; see `environment.md`). Frameworks that hide the model are a last resort.
- **Verify on CPU before you burn a 15-minute compile.** Every op-rewrite patch must be
  validated against the original on CPU *first* (evo2: patched-conv1d vs FFT cosine 0.999999
  before ever touching Neuron). Weight-split / equivalence checks save more hours than they cost.
- **bf16 is great until it isn't.** Some models collapse in half precision — evo2's RMSNorm
  zeroes out on ~1e16 activations; FLUX goes gray past ~1 MP. Default to fp32 for correctness,
  then isolate fp32 to just the norm/massive-activation layers as an optimization.
- **Synthetic inputs are fine for parity.** The oracle only needs CPU and Trainium to see *identical,
  valid-shaped* inputs — not biologically/semantically real ones. When real data is impractical (e.g. an
  MSA model needing aligned genomes), use **deterministic random inputs of the correct shape/dtype/vocab
  range** (seed them). Real weights + synthetic inputs is a valid parity port; note it in RESULTS. (A
  real-data validation pass is a nice follow-up, not a requirement.)
- **Static shapes.** Neuron compiles a fixed graph — pad inputs to a fixed length, `use_cache=False`.
  Mixing lengths means recompiles; pick one length and keep the compile cache.
- **Tensor parallelism shards weights — not the sequence.** If it still OOMs at more cores, you're
  not sharding what's actually big. Split fused projections (QKV, SwiGLU) into separately-shardable
  linears so the *whole* model shards, not just attention heads.
- **Keep names 1:1 with HuggingFace.** When you port a component, keep its HF name. If there's no
  clean 1:1 mapping, append **`_u`** to flag it.
- **Use the pure-PyTorch source, not the CUDA original.** HF `trust_remote_code` port over
  `vortex`/`faesm`/TransformerEngine-bound code every time.
- **If a researcher can read the code and change it, we did our job.** The deliverable is a
  copy-paste README + a clean runner, not a framework.
