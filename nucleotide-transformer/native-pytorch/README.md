# Nucleotide Transformer on AWS Trainium — native PyTorch (DEV mode)

The porting product is the model's **own, editable, composable definition** — the full
**`EsmForMaskedLM`** (encoder + MLM head) of `InstaDeepAI/nucleotide-transformer-v2-50m-multi-species`
(a DNA foundation model, ESM-family with rotary positions; 512-d, 12 layers, 6-mer DNA tokenizer),
vendored in `src/nucleotide_transformer_neuron/` and made to run on Trainium — so you can subclass it,
swap modules, and build/train new DNA models from its pieces.

> **Status:** ✅ full model on Trainium — outputs match CPU (logits cosine **1.000011**, hidden
> **1.000001**); **all 177 parameter gradients** match CPU. **No on-device Neuron op-rewrite**; the only
> patches are transformers-4.x→5.x compatibility fixes so the trust_remote_code definition loads/builds on
> the Beta-3 stack (see `src/nucleotide_transformer_neuron/VENDORED.md`).

## Use it as a building block

```python
import sys; sys.path.insert(0, "src")
import torch
import nucleotide_transformer_neuron as NT

model = NT.load(device="neuron")                    # full EsmForMaskedLM (encoder + MLM head), on Trainium
ids, mask = NT.tokenize(["ATGGCGCCTAGC...", "GGCCTTAACCGG..."])
out = model(input_ids=ids.to("neuron"), attention_mask=mask.to("neuron"), output_hidden_states=True)
logits = out.logits                                 # (B, L, vocab)  MLM head
hidden = out.hidden_states[-1]                       # (B, L, 512)    per-token DNA features

# subclass / compose / finetune the editable definition:
parts = NT.submodules(model)                        # encoder / embeddings / lm_head — compose freely
NT.freeze(parts["encoder"])                          # e.g. frozen backbone + a new head, or skip to finetune all
```
See `src/nucleotide_transformer_reference.py` for the full-model wrapper + deterministic inputs used in
the parity checks (`notebooks/01`–`03`).

## 1. Instance
`trn2.3xlarge`, a single NeuronCore (this is a 50M-param model — one core is plenty).

## 2. Environment
```bash
conda activate torch-neuron            # the Beta-3 native-PyTorch stack (see references/environment.md)
pip install -r requirements.txt        # transformers 5.13 + huggingface_hub; the NT definition is VENDORED
```

## 3. The three parity notebooks (the deliverable)
```bash
cd notebooks
NEURON_RT_VISIBLE_CORES=0 jupyter nbconvert --to notebook --execute --inplace 01_inference_parity.ipynb
NEURON_RT_VISIBLE_CORES=0 jupyter nbconvert --to notebook --execute --inplace 02_backprop_parity.ipynb
NEURON_RT_VISIBLE_CORES=0 jupyter nbconvert --to notebook --execute --inplace 03_training_parity.ipynb
```
- **01_inference_parity** — CPU/CUDA/Trainium; asserts the full model's **logits** + hidden state match (cosine 1.000011).
- **02_backprop_parity** — one backward through the full model on each device; asserts **all 177 parameter
  gradients** match (magnitude-aware). CUDA auto-skips on the trn box.
- **03_training_parity** — 5-step Adam loop on NT's own masked-LM loss; asserts the loss decreases and the
  loss trajectory + final weights match CPU↔Trainium (on-device training).

## Why this works (what the port handles for you)
1. **The full editable definition.** `NT.load()` builds the full `EsmForMaskedLM` (rotary ESM encoder + MLM
   head) — the product you subclass / compose / finetune, not a headless subgraph. Both encoder and head
   lower cleanly on device (see #3).
2. **Definition vendored + patched for the Beta-3 stack.** The model ships as `trust_remote_code` written
   for transformers 4.32; on transformers 5.13 it wouldn't even import/build. Three tiny compatibility
   patches (restore removed transformers-4.x helpers) — no architecture/math change. See `VENDORED.md`.
3. **No on-device op rewrite.** The active attention path is plain `matmul`+`softmax`+rotary
   (`outer`/`cos`/`sin`/`chunk`/`cat`), all of which lower and backprop cleanly on Trainium. fp32.
4. **Static shapes.** `NT.tokenize(..., max_length=64)` pads to a fixed length; `use_cache=False`.

## Troubleshooting
| Symptom | Fix |
|---|---|
| `neuron-ls` shows nothing | Not on a Trainium instance / wrong AMI. |
| `Got a cached failed neff` | `rm -rf /var/tmp/neuron-compile-cache` and rerun. |
| `Logical Neuron Core(s) not available` | pin `NEURON_RT_VISIBLE_CORES=0` (single core). |
| Output is all zeros / garbage | Precision — keep fp32 (default here). |
| First run is slow | One-time compile; it caches. |
| `find_pruneable_heads_and_indices` / `get_head_mask` / `is_decoder` ImportError/AttributeError | You're using the hub's raw trust_remote_code on transformers 5.x — use this VENDORED definition instead. |

## Ways to optimize / extend
### Easy
1. Keep the compile cache; fix one sequence length to avoid recompiles.
### Medium
2. Mixed precision (bf16 where safe); batch inputs. Re-validate vs the oracle after any precision change.
### Advanced
3. Larger NT variants (100M / 250M / 500M multi-species, or the 2.5B) — same recipe, single core to a few
   hundred M; sharding for the largest.

## Notes & limitations
- Validated: `nucleotide-transformer-v2-50m-multi-species`, fp32, seqlen 64 — full-model logits, all-param
  gradient parity, and a multi-step training loop, all at cosine/parity ~1.0 on Trainium.
- Frozen-backbone (`NT.freeze(...)`) vs full finetune both supported; a full multi-epoch training loop
  (optimizer, multi-step, distributed) is the training phase.

## Credits & license
Nucleotide Transformer © InstaDeep, via HuggingFace `InstaDeepAI/nucleotide-transformer-v2-50m-multi-species`
(**CC-BY-NC-SA-4.0** — non-commercial). Its `trust_remote_code` definition is vendored and minimally patched
for the Beta-3 stack (patches listed in `src/nucleotide_transformer_neuron/VENDORED.md`); the architecture
and weights are unchanged.
