# Nucleotide Transformer on Trainium — Result (DEV mode)

**Status: full Nucleotide Transformer (`EsmForMaskedLM`) FUNCTIONAL on Trainium** — the vendored, editable,
composable definition (rotary ESM encoder + MLM head). Inference: logits cosine **1.000011**, hidden
**1.000001**. Backprop: **all 177 parameter gradients** match CPU. Training: 5-step Adam on NT's own
masked-LM loss, loss decreases + trajectory/weights match CPU↔Trainium. No on-device Neuron op-rewrite;
only transformers-4.x→5.x compatibility patches (2 of them for the MLM head; see VENDORED.md).

## Environment
- Instance: `trn2.3xlarge`, 1 NeuronCore (`NEURON_RT_VISIBLE_CORES=0`).
- Software: native-PyTorch Neuron **Beta 3** (`torch-neuron`; torch 2.11.0, torch-neuronx 2.11.3, neuronxcc 2.25.1280, transformers 5.13.0).
- Model definition: **VENDORED + patched** from `InstaDeepAI/nucleotide-transformer-v2-50m-multi-species`
  trust_remote_code (ESM-family, **rotary** position embeddings, 512-d, 12 layers, 6-mer DNA tokenizer, vocab 4107).

## Mode: dev (full editable definition)
- The "model" under test is the vendored **FULL `EsmForMaskedLM`** (encoder + MLM head), tensor-in /
  tuple-out (`forward(input_ids, attention_mask) -> (logits (B,L,vocab), last_hidden_state)`). Parity
  validates the full model's own outputs and **all 177 of its parameters** — the editable definition
  itself, not a downstream composite.
- Definition **vendored** (not sealed): the trust_remote_code code targets transformers 4.32 and would not
  import/build on the Beta-3 transformers 5.13 — three used-path compatibility patches were required
  (skill R1: "vendor only if a used-path patch is needed"). Patches recorded in `VENDORED.md`; source
  commit `81b29e5786726d891dbf929404ef20adca5b36f1`.
- Deliverable interface: `nucleotide_transformer_neuron.load()` (full `EsmForMaskedLM`) + `freeze`/`unfreeze`
  to subclass / compose / finetune.

## Phase-A reference (oracle)
- Baseline env: torch 2.11.0 (CPU) + transformers 5.13.0 (same as the Beta-3 run; the model needs no
  extra deps, so the oracle is apples-to-apples — only device + compiler differ).
- `baselines/nucleotide-transformer/MANIFEST.json`: outputs = logits (2, 64, 4107) + hidden (2, 64, 512);
  **177 grad tensors**, seed 1234, deterministic DNA inputs (`input_ids`/`attention_mask`, seqlen 64).
- Loss used for the backprop reference: `adapter.loss_fn` (sum of float outputs → exercises every param's grad).

## What worked (Phase B recipe)
1. **Source triage.** Pure-PyTorch trust_remote_code `EsmForMaskedLM`; no flash-attn / FP8 / FFT / sort.
2. **Vendor + 3 compatibility patches** (transformers 4.x→5.x, all on the used path, no math change):
   `find_pruneable_heads_and_indices` import, `is_decoder`/`add_cross_attention` config defaults,
   `get_head_mask` mixin method. See `src/nucleotide_transformer_neuron/VENDORED.md`.
3. **R2 neutralizations:** `use_cache=False` (config default), static/padded shapes (seqlen 64), fp32.
4. **R5 run path:** eager `device="neuron"` (no `torch_neuronx.trace` on Beta 3). No op rewrite needed —
   attention is plain matmul/softmax + rotary, lowers cleanly forward and backward.
5. **R6 precision:** fp32 (default). No precision issue observed.

## Validation (R7)
**Inference** (`01_inference_parity.ipynb`): Neuron vs CPU oracle
- logits (2,64,4107) cosine **1.000011**, max-abs **1.316e-04** — OK
- hidden (2,64,512) cosine **1.000001**, max-abs **1.484e-05** — OK
- Gate (cosine ≥ 0.99): **PASS**.

**Backprop** (`02_backprop_parity.ipynb`): one backward through the full model on each device, magnitude-aware gate.
- **177/177 grad tensors match** CPU↔Trainium. Global |grad| scale 2.295e5, zero real disagreements.

**Training** (`03_training_parity.ipynb`, dev-mode: on-device training) — 5-step Adam loop on NT's own
**masked-token CE** (MLM: mask ~30% of the attended tokens with the real `[MASK]` id, predict the
originals), ALL 177 params trainable, `eval()` (dropout off) for a deterministic step. Self-compares
CPU↔Trainium at runtime (no frozen oracle). Loss decreases **4.885 → 0.103**; trajectory matches to
**6.67e-6 relative**, final weights to **1.83e-4**. Gate is RELATIVE (`|Δ|/|loss| ≤ 1e-2`), weights absolute.

All three notebooks run CPU/CUDA/Trainium (CUDA auto-skips on the trn box). Gate: **PASS** on all three.

## Files (src/)
- `nucleotide_transformer_neuron/` — the dev-mode interface: `load()` (full `EsmForMaskedLM`), `get_tokenizer`,
  `tokenize`, `freeze`/`unfreeze` — over the **vendored** definition (`modeling_esm.py`, `esm_config.py`,
  `VENDORED.md`).
- `nucleotide_transformer_reference.py` — shared harness (full-model wrapper, deterministic inputs, loss);
  used by the oracle capture AND all three notebooks (byte-identical model + inputs).
- `notebooks/01_inference_parity.ipynb`, `notebooks/02_backprop_parity.ipynb`, `notebooks/03_training_parity.ipynb`.

## Open / next
- Larger NT variants (100M / 250M / 500M / 2.5B): same recipe, single core to a few hundred M; sharding for 2.5B.
- Full multi-epoch / distributed finetune on real MSA-free DNA corpora — training phase.
- Downstream task heads (token / sequence classification) composed on the vendored backbone.
