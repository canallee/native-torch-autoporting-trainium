# ---
# jupyter:
#   jupytext:
#     text_representation:
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---
# Phase-A capture notebook TEMPLATE (jupytext "percent" format).
# Convert to the committed deliverable notebooks with:
#     jupytext --to notebook capture_notebook_template.py -o baselines/<model>/inference.ipynb
# Run headless (the agent does this) with:
#     jupyter nbconvert --to notebook --execute baselines/<model>/inference.ipynb
#
# DEVICE POLICY: cpu is the DEFAULT (runs on the trn box, no GPU here). Set DEVICE="cuda"
# only on a GPU host; it auto-falls-back to cpu when cuda is unavailable.

# %% [markdown]
# # {MODEL} — reference capture (inference + one backprop)
# Produces the frozen oracle (`MANIFEST.json` + gitignored `artifacts/*.pt`) that the
# Trainium port (Phase B) is validated against. Baseline env pins torch==2.11.0.

# %%
import os, sys, json
# make the skill's scripts importable, plus this model's capture.py adapter
sys.path.insert(0, os.path.abspath("../../skills/neuron-native-autoport/scripts"))
sys.path.insert(0, os.path.abspath("."))
import torch
import capture as adapter                     # baselines/<model>/capture.py
from capture_reference import run_capture, resolve_device

DEVICE = os.environ.get("CAPTURE_DEVICE", "cpu")   # cpu (default) | cuda | neuron
SEED = 1234
print("torch", torch.__version__, "| requested device:", DEVICE, "| model:", getattr(adapter, "MODEL_ID", "?"))

# %% [markdown]
# ## 1. Inference + 2. one backprop — save artifacts
# %%
outputs, grads, meta = run_capture(adapter, device=DEVICE, seed=SEED, out_dir="artifacts")
print(json.dumps(meta, indent=2))

# %% [markdown]
# ## 3. Freeze the manifest (the committed oracle)
# %%
from build_manifest import _manifest_section
manifest = {"meta": meta,
            "env_lock": "env/requirements.lock" if os.path.exists("env/requirements.lock") else None,
            "outputs": _manifest_section(outputs),
            "grads": _manifest_section(grads)}
with open("MANIFEST.json", "w") as f:
    json.dump(manifest, f, indent=2)
print(f"wrote MANIFEST.json: {len(manifest['outputs'])} outputs, {len(manifest['grads'])} grad tensors")

# %% [markdown]
# The `artifacts/` tensors are gitignored; `MANIFEST.json` is committed and is what
# Phase B R7 compares the Trainium run against (see scripts/compare.py).
