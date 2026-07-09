# Vendored definition — Nucleotide Transformer (DEV mode)

**Source:** `InstaDeepAI/nucleotide-transformer-v2-50m-multi-species` (HuggingFace), `trust_remote_code`
custom code — commit `81b29e5786726d891dbf929404ef20adca5b36f1`. License: **CC-BY-NC-SA-4.0** (non-commercial).
Files vendored verbatim then patched: `modeling_esm.py`, `esm_config.py`.

## Why vendored (not sealed)
Dev mode vendors the definition so it is editable/composable (the product is the model's own definition).
Here vendoring is doubly forced: the definition ships as `trust_remote_code` code **written against
transformers 4.32**. On the Beta-3 stack (transformers
**5.13**) the module does not even IMPORT / build — three transformers-4.x → 5.x compatibility breaks hit
the *used* encoder path. Per the skill's R1 ("vendor only if a used-path patch is needed") we vendor and
record the patches here. **No architecture or math was changed** — the patches only restore helpers that
transformers 4.x provided and 5.x removed. All patch sites are marked `NEURON PATCH` in the source.

## Patches 1–3 (import + encoder path)

1. **`modeling_esm.py` — `find_pruneable_heads_and_indices` import.**
   `from transformers.pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer`
   fails: `find_pruneable_heads_and_indices` was **removed** from `transformers.pytorch_utils` in 5.x
   (this failure is at module-import time, so the encoder couldn't load at all). Fix: import only
   `prune_linear_layer` (still present) and vendor `find_pruneable_heads_and_indices` verbatim from
   transformers 4.x. It is only used by head-pruning, which is off the embedding path.

2. **`esm_config.py` — `is_decoder` / `add_cross_attention` defaults.**
   transformers 4.x `PretrainedConfig.__init__` populated `is_decoder` and `add_cross_attention` on the
   base config; 5.x no longer does. The (unmodified) `EsmSelfAttention.__init__` reads
   `config.is_decoder`, raising `AttributeError`. Fix: after `super().__init__`, set both to their 4.x
   defaults (`False`) if absent. Both are `False` for this bidirectional DNA encoder — identical behavior.

3. **`modeling_esm.py` — `EsmPreTrainedModel.get_head_mask`.**
   `ModuleUtilsMixin.get_head_mask` (called by `EsmModel.forward`) was **removed** in transformers 5.x.
   Fix: vendor `get_head_mask` + `_convert_head_mask_to_5d` verbatim from transformers 4.x onto
   `EsmPreTrainedModel`. (`get_extended_attention_mask` and `invert_attention_mask` still exist in 5.13,
   so only this one is restored.) With `head_mask=None` it returns `[None] * num_layers` — a no-op.

## Dev-mode head shims (2, for the full `EsmForMaskedLM`)
The dev port ships the full `EsmForMaskedLM` (encoder + MLM head). Beyond the encoder-path fixes above, the
MLM head needs two more transformers-4.x→5.x compat fixes (both marked `NEURON PATCH`, no math change):
4. **`EsmForMaskedLM.__init__`: `init_weights()` → `post_init()`.** 4.x `init_weights` routes into a tie
   step that on 5.x reads `all_tied_weights_keys` before it is set (AttributeError). `post_init` is the 5.x
   equivalent and sets it first.
5. **`EsmForMaskedLM._tied_weights_keys` list → `{}`.** Same list-vs-dict drift as the loader KB entry. The
   checkpoint stores `lm_head.decoder.weight`, so no auto-tie is needed. `load()` builds the model via
   direct construction + `load_state_dict` (not `from_pretrained`, whose 5.13 tie machinery also chokes).

## Not needed on device
No Neuron op-rewrite was required: the active attention path is plain `torch.matmul` + `softmax` + rotary
(`torch.outer`/`cos`/`sin`/`chunk`/`cat`), all of which lower and backprop cleanly on Trainium — for both
the encoder AND the MLM head. The `relative_key` einsum branches and the contact head are never instantiated.
