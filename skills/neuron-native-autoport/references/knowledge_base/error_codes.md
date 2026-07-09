# Compiler / runtime error → fix index

Seeded from the 5 studied ports. Grow this table at M2/M3 as new errors surface.

| Symptom / code | Root cause | Fix |
|---|---|---|
| `NCC_EVRF004` — complex data types not supported **OR** `RuntimeError: aten::_fft_c2r ... complex dtypes are not supported natively on Neuron` (eager torch fallback — grep for BOTH strings) | `torch.fft` / complex64 ops (Hyena long-conv) | Rewrite to a mathematically-equivalent **real** op — see `op_rewrites.md` (FFT-conv → depthwise causal `conv1d`, filter flipped). CPU-validate vs original first. |
| `sort is not supported` / `NCC_EVRF029` | HLO `sort` / top-k over the sequence | Skip that head / compute it on CPU (alphagenome `splice_junctions`); or lower to a supported TopK / small NKI kernel. |
| `NCC_ITIN902` / `AffineIV` at long sequence lengths | compiler polyhedral (array-index) bug at default optlevel | `export NEURON_CC_FLAGS=--optlevel=1` **before** the compiler is imported (no change to the math). Set it per-module if only some modules need it. |
| `Got a cached failed neff` / `[NLA001]` JSON parse / `FileNotFoundError` on neff paths | interrupted or stale compile cache | `rm -rf /var/tmp/neuron-compile-cache` and rerun. |
| Output all zeros / garbage | running in bf16 where the model needs fp32 | `model.float()` — see `precision.md`. |
| `FP8 requires Transformer Engine` | config requests FP8 input projections (needs Hopper) | `config.use_fp8_input_projections=False` → pure-PyTorch fallback. |
| `ModuleNotFoundError: torch_xla` | env / wrong path | `conda activate torch-neuron`; the xla path also needs `torch_xla` installed (it is, as of 2026-07-08). |
| `Logical Neuron Core(s) not available Requested:4 Available:N` | PJRT grabs all cores; another proc holds some | pin `NEURON_RT_VISIBLE_CORES=<free core>` for single-core; see `references/environment.md`. |
| torch_xla silently on CPU | `PJRT_DEVICE` unset | `export PJRT_DEVICE=NEURON`. |
| `AttributeError: module 'torch_neuronx' has no attribute 'trace'` | using the public-SDK API on Beta 3 | Beta 3 has **no** `torch_neuronx.trace`. Use eager `device="neuron"` or `torch.compile(backend="neuron")` (see `environment.md`). |
| `WARNING:Neuron:TP degree (XX) and KV heads (YY) not divisible ... CONVERT_TO_MHA` | benign | ignore. |
| EFA/NCCL/OFI init failures on startup | no multi-node fabric | benign for single-core inference; ignore. |
| `FutureWarning: torch.backends.cuda.sdp_kernel(...)` on a CPU/Neuron box | model wraps SDPA in a `torch.backends.cuda.*` context | **benign** — it runs and lowers cleanly on Neuron. Do NOT preemptively "patch" it. |
