# GPN-Star on AWS Trainium — native PyTorch (DEV mode)

The porting product is GPN-Star's **own, editable, composable definition** — the full
**`GPNStarForMaskedLM`** (phylogeny-aware MSA encoder + masked-LM head), **vendored** in
`src/gpn_star_neuron/modeling.py` (from `gpn/star/model.py`) and made to run on Trainium — so you can
subclass it, swap the axial-attention / FIRE-bias modules, and build/train new genomic models from it.
Vendoring also drops the `gpn`-package dependency: `modeling.py` needs only transformers + networkx.

GPN-Star (`songlab/gpn-star-tair10-b18-25m`, `GPNStarConfig(RoFormerConfig)`, ~25M, 512-hidden / 8-layer)
is a **phylogeny-aware, MSA-input** genomic LM: axial attention over a whole-genome alignment (row
self-attention along the sequence + column / cross attention across aligned species/clades), a **FIRE**
relative-position + phylo-distance bias, and RoFormer rotary positions. Its forward takes **MSA-style
tensors**, not a single DNA string:

| input | shape | meaning |
|---|---|---|
| `input_ids` | `(B, L, T)` | token ids for the `T` target rows (`T=1` = reference genome) |
| `source_ids` | `(B, L, N)` | token ids for the full `N`-species alignment column (`N=18` for tair10-b18) |
| `target_species` | `(B, T)` | index (into the `N` species) of each target row |

The phylogeny (pairwise distances, clade clustering, in-clade distances) is baked into the checkpoint's
`phylo_dist/` numpy files and reconstructed inside the model at construction time.

> **Status:** ✅ full model on Trainium — logits match CPU at **cosine 1.000000** (max-abs 1.1e-5); **all
> 240 parameter gradients** match CPU, 0 real disagreements. **No Neuron op patches needed** (one head compat
> shim + one load-time shim — see VENDORED.md). Real-data variant-effect validation in `example/vep/`.

## Use it as a building block

```python
import sys; sys.path.insert(0, "src")
import gpn_star_neuron as G

gps = G.load(device="neuron")                                   # full GPNStarForMaskedLM, on Trainium
N   = G.num_species()                                           # 18 aligned species
input_ids, source_ids, target_species = G.build_synthetic_msa(N)   # or your real tokenized MSA tensors
out = gps(input_ids=input_ids.to("neuron"), source_ids=source_ids.to("neuron"),
          target_species=target_species.to("neuron"))
logits = out.logits                                             # (B, L, T, vocab)  MLM head

parts = G.submodules(gps)                                       # encoder / cls head — compose freely
G.freeze(parts["encoder"])   # e.g. frozen backbone + a new head, or skip to finetune the whole definition
```
See `src/gpn_star_reference.py` for the full-model wrapper + deterministic inputs used in the parity
checks (`notebooks/01`–`03`).

## Inputs are deterministic synthetic MSAs
GPN-Star consumes aligned-genome MSAs; **no real aligned-genome data is needed for parity.**
`G.build_synthetic_msa(...)` returns deterministic, valid-shaped tensors (correct dtypes / shapes /
vocab range) — CPU-vs-Trainium numerical parity only needs identical, static-shaped inputs across
devices, and the model **weights + real phylogeny** are the genuine checkpoint. Swap in your own
tokenized MSA tensors of the same shapes for real inference.

## Environment
The vendored definition needs only `networkx` on top of the neuron stack; the real-data VEP example
(`example/vep/`) additionally uses the `gpn` package. To keep `torch-neuron` clean for the standard
native-torch ports, GPN-Star runs in a **dedicated `gpn-vep` env** (a clone of `torch-neuron` + `gpn`,
which brings `networkx`):
```bash
conda activate gpn-vep                   # neuron stack + gpn + genomics deps
# (to create it: conda create --clone torch-neuron -n gpn-vep -y ; pip install gpn)
```
The vendored definition itself needs only **networkx** (for clade clustering) on top of the neuron stack;
the `gpn` package is only used by the real-data VEP example (`example/vep/`).

## The three parity notebooks (the deliverable)
```bash
cd notebooks
NEURON_RT_VISIBLE_CORES=0 jupyter nbconvert --to notebook --execute --inplace 01_inference_parity.ipynb
NEURON_RT_VISIBLE_CORES=0 jupyter nbconvert --to notebook --execute --inplace 02_backprop_parity.ipynb
NEURON_RT_VISIBLE_CORES=0 jupyter nbconvert --to notebook --execute --inplace 03_training_parity.ipynb
```
- **01_inference_parity** — CPU/CUDA/Trainium; asserts the full model's **logits** match (cosine 1.000000).
- **02_backprop_parity** — one backward through the full model on each device; asserts **all 240 parameter
  gradients** match (magnitude-aware). CUDA auto-skips on the trn box.
- **03_training_parity** — 5-step Adam loop on GPN-Star's own masked-LM loss; asserts the loss decreases and
  the loss trajectory + final weights match CPU↔Trainium (on-device training).

## Why this works
The product is GPN-Star's **own full definition** (`GPNStarForMaskedLM`), vendored and made to run on
Trainium — MSA encoder + FIRE phylo-bias + MLM head, all editable/composable. GPN-Star's attention is
`F.scaled_dot_product_attention` with `enable_math=True` plus explicit linear/matmul/softmax; both forward
and backward lower cleanly on Beta-3 with **no op rewrite** — for the encoder AND the MLM head.

### The load-time shims (not `from_pretrained`)
GPN-Star's `__init__` does *real* tensor work (loads phylo numpy arrays, clusters clades with `networkx`,
calls `.item()`/`.max()`). Beta-3's **transformers 5.13** constructs every `from_pretrained` model under a
`torch.device("meta")` fast-init context, so those ops raise `Tensor.item() cannot be called on meta
tensors`. This is transformers **4.37→5.13 drift** (the checkpoint was authored on 4.37), **not** a Neuron
issue. `G.load()` constructs `GPNStarForMaskedLM(config)` on the **real** device (no meta context) and
`load_state_dict`s the weights, then ties the one absent `decoder.bias` (a `_tied_weights_keys` list→{}
compat fix). These are load-time shims, not architecture edits — the vendored math is unchanged.

## Notes & limitations
- Validated: `gpn-star-tair10-b18-25m` (arabidopsis, 18-species alignment), fp32, `B=1, L=64, T=1` —
  full-model logits, all-param gradient parity, and a multi-step training loop, all at cosine/parity ~1.0
  on Trainium.
- Swap `G.MODEL_NAME` for other GPN-Star tiers (`ce11-n135-25m`, `dm6-i124-85m`, `hg38-*-200m`, …) — same
  recipe; each ships its own `phylo_dist/`, so `num_species()` / `build_synthetic_msa()` adapt automatically.
- Frozen-backbone (`G.freeze(...)`) vs full finetune both supported; a full multi-epoch training loop is the
  training phase.

## Credits & license
GPN-Star © Ye, Benegas et al. (Song Lab, UC Berkeley), MIT-licensed, vendored from the `gpn` package
`gpn/star/model.py` (`songlab/gpn-star-tair10-b18-25m`). See `src/gpn_star_neuron/VENDORED.md`.
