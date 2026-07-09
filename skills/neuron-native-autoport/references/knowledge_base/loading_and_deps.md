# Loading & dependency issues (BEFORE any Neuron op work)

The KB's other files are about the neuronx-cc *compiler*. This file is about getting the model to
**import, build, and load weights at all** — the gaps that bite before you ever reach the device.
(Surfaced by the Nucleotide-Transformer M3 port; the failures were 100% here, 0% on the device.)

## `trust_remote_code` written against an OLDER transformers than Beta-3's 5.13
Custom-code models (`auto_map` / a repo's own modeling file) are often authored against transformers
**4.x**; on the Beta-3 stack (**transformers 5.13**) they may not even import/build. This is NOT a Neuron
issue and `analyze_hlo.py`/`error_codes.md` won't help — it's Python-level API drift.

**Decision:** this is a "used-path patch" per R1 → **vendor the definition and add minimal compat shims**
(restore removed 4.x helpers; change NO math). Mark each site `NEURON PATCH` and list them in `VENDORED.md`.
It also flips an *integration*-mode port from "sealed/installed" to "vendored" — that's expected and fine.

Concrete 4.x→5.x breaks seen (ESM-family; others will differ but rhyme):
| Symptom | Cause | Minimal shim |
|---|---|---|
| `ImportError: cannot import name 'find_pruneable_heads_and_indices' from 'transformers.pytorch_utils'` | removed in 5.x (import-time crash) | import only what still exists; vendor the removed helper verbatim (used only by head-pruning, off the fwd path) |
| `AttributeError: ... 'is_decoder'` / `add_cross_attention` | 4.x `PretrainedConfig.__init__` defaulted these; 5.x doesn't | in the vendored config, set them to their 4.x defaults (`False`) after `super().__init__` if absent |
| `AttributeError: ... 'get_head_mask'` | `ModuleUtilsMixin.get_head_mask` removed in 5.x | vendor `get_head_mask` + `_convert_head_mask_to_5d` onto the model's PreTrainedModel (no-op when `head_mask=None`) |

Approach: run the CPU load, read the traceback, restore the one removed symbol, repeat. Keep shims tiny and
behavior-identical; verify the CPU oracle is unchanged.

## `Tensor.item() cannot be called on meta tensors` — transformers 5.x meta-device fast-init
transformers **5.x constructs every `from_pretrained` model under `torch.device("meta")`** (fast init,
weights materialized after). Any model whose `__init__` does **real tensor work** — `.item()`/`.max()`,
`np.load` of packaged buffers, `networkx`/clustering, data-derived buffers (e.g. GPN-Star's phylo-distance
reconstruction) — crashes with `RuntimeError: Tensor.item() cannot be called on meta tensors`. This is
version drift (checkpoint authored on an older transformers), NOT a Neuron issue; `low_cpu_mem_usage=False`
does **not** fix it.

**Fix (a LOAD-TIME shim, architecture untouched — see "loader shim" in SKILL integration mode):** don't go
through `from_pretrained`'s meta path. Construct the model class directly on the **real** device from the
config, then load weights yourself:
```python
cfg = <Config>.from_pretrained(path)                 # drop auto_map if trust_remote_code
model = <ModelClass>(cfg)                             # real device, no meta context -> __init__ runs
sd = safetensors.torch.load_file(f"{path}/model.safetensors")
model.load_state_dict({k: v for k, v in sd.items() if not k.startswith("cls.")}, strict=False)  # drop the head
```
This keeps the definition SEALED (no edits) — it's a loader workaround, belongs in the model's `load()` +
a note in README, not in `VENDORED.md`.

## `'list' object has no attribute 'keys'` in `post_init` — `_tied_weights_keys` 4.x list vs 5.x dict
A model with a **tied LM head** (e.g. `*ForMaskedLM`) may declare `_tied_weights_keys` as a **4.x list**
(`["cls.predictions.decoder.weight", ...]`); transformers 5.x `post_init`/`get_expanded_tied_weights_keys`
expects a **dict**, raising `AttributeError: 'list' object has no attribute 'keys'` at construction. Shim
before constructing: `ModelClass._tied_weights_keys = {}` (empty dict = no auto-tie), then after
`load_state_dict(strict=False)` re-tie by hand anything the checkpoint omitted (commonly the decoder bias:
`m.cls.predictions.decoder.bias = m.cls.predictions.bias`). Load-time shim, architecture untouched.
Check `missing_keys` from the `strict=False` load to see exactly what to re-tie — often it is ONLY the
decoder *bias* (the decoder *weight* is present, tied to the input embedding), so a single manual bias
re-tie is sufficient; don't over-engineer a full weight re-tie you don't need. (GPN-Star: missing is exactly
`['cls.predictions.decoder.bias']`.)
*(Only matters when you need the head — e.g. VEP/MLM scoring; a headless-encoder integration port skips it.
Seen on GPN-Star `GPNStarForMaskedLM`; see `gpn-star/native-pytorch/example/vep/run_vep_validation.py`.)*

## Interactive `trust_remote_code` prompt hangs under headless execution
`from_pretrained(..., trust_remote_code=True)` can fire `Do you wish to run the custom code? [y/N]`, which
**EOFs/hangs** under `jupyter nbconvert --execute` / non-TTY. Once you've VENDORED the definition you don't
need dynamic loading anyway: build the config from `config.json` (drop its `auto_map`), instantiate your
**vendored** class, and call `from_pretrained(path, config=cfg, trust_remote_code=False)`. (Or set
`HF_HUB_DISABLE_IMPLICIT_TOKEN=1` and pre-accept, but vendored+`False` is cleaner and reproducible.)

## Weights come down as git-LFS pointer files
`git clone` of an HF repo on a box without `git-lfs` yields ~134-byte **pointer** files for
`*.safetensors`/`*.bin` — the model won't load. **Fetch weights via `huggingface_hub`
(`snapshot_download` / `hf_hub_download`), not git clone.** Clone the repo only to READ code; get weights
through the hub API (or `curl` the `resolve/main/<file>` URL). Prefer `snapshot_download` in `load()`.
