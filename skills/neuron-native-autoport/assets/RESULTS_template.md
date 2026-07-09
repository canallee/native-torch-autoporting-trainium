# {MODEL} on Trainium — Result

**Status: {MODEL} {FUNCTIONAL / PARTIAL} on Trainium — {headline metric vs Phase-A oracle}.**

## Environment
- Instance: {trn2.Nxlarge}, {N} NeuronCore(s).
- Software: native-PyTorch Neuron **Beta 3** (`torch-neuron` conda env; torch 2.11.0,
  torch-neuronx 2.11.3, neuronxcc 2.25.1280). See `references/environment.md`.
- Source: {HF port / repo used, NOT the CUDA original}.

## Phase-A reference (oracle)
- Baseline env: torch 2.11.0 (CPU){, or repo-native + caveat}.
- Manifest: `baselines/{model}/MANIFEST.json` — {N} output tensors, {N} grad tensors, seed {seed}.
- Loss used for backprop reference: {adapter.loss_fn / default sum-of-outputs}.

## What worked (Phase B recipe)
1. {source triage — which entry point, which CUDA deps bypassed}
2. {R2 neutralizations — eager attn, FP8 off, static shapes}
3. {R5 op rewrites / compiler flags — the real work, if any}
4. {R6 precision decision}

## Validation (R7)
- Neuron vs Phase-A oracle: **outputs cosine {x}, max-abs {y}**{, top-1 {z}%}.
- Per-tensor: {pass/fail summary; any tensor above tolerance and why}.
- First compile ~{t}; warm run ~{t}.

## Files (src/)
- `{runner}.py` — the deliverable runner.
- `{patch}.py` — Neuron patches (if any), CPU-validated vs original.

## Open / next
- {larger variants / sharding / decode / long context / training}.
