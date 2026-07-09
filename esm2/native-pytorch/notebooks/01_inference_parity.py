# ---
# jupyter:
#   jupytext:
#     text_representation: {format_name: percent}
#   kernelspec: {display_name: Python 3, name: python3}
# ---

# %% [markdown]
# # ESM-2 embedding inference parity — CPU / CUDA / Trainium  (INTEGRATION mode)
# Runs ESM-2 as a **feature extractor** (per-residue embeddings; MLM head dropped) on every available
# device and checks the extracted embeddings match. CPU is the reference; `cuda` auto-skips when absent;
# `neuron` runs on the Trainium core. Pin a free core:
# `NEURON_RT_VISIBLE_CORES=0 jupyter nbconvert --to notebook --execute 01_inference_parity.ipynb`.

# %%
import os
os.environ.setdefault("NEURON_RT_VISIBLE_CORES", "0")
import sys
sys.path.insert(0, os.path.abspath("../src"))
import torch
import torch.nn.functional as F
import esm2_reference as R

def devices():
    devs = ["cpu"]
    if torch.cuda.is_available():
        devs.append("cuda")
    try:
        import torch_neuronx  # noqa: F401
        devs.append("neuron")
    except Exception as e:
        print("neuron unavailable:", e)
    return devs

DEVICES = devices()
print("torch", torch.__version__, "| devices:", DEVICES, "| model:", R.MODEL_NAME)

# %% [markdown]
# ## Run the integration forward (embeddings + downstream logits) on each device
# %%
def run(device):
    model = R.load(device)
    ids, mask = R.build_inputs()
    with torch.no_grad():
        out = model(ids.to(device), mask.to(device))
    if device == "neuron":
        import torch_neuronx; torch_neuronx.synchronize()
    return tuple(t.detach().float().cpu() for t in out)

results = {d: run(d) for d in DEVICES}
for name, t in zip(R.OUTPUT_ORDER, results["cpu"]):
    print(f"cpu {name:12s} shape={tuple(t.shape)}")

# %% [markdown]
# ## Check every device matches CPU (embeddings are the extracted feature)
# %%
def cos(a, b): return F.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()

ref, all_ok = results["cpu"], True
for d in DEVICES:
    if d == "cpu":
        continue
    print(f"\n{d} vs cpu:")
    for name, a, b in zip(R.OUTPUT_ORDER, ref, results[d]):
        c = cos(a, b); mab = (a - b).abs().max().item(); ok = c >= 0.99
        all_ok = all_ok and ok
        print(f"  {name:12s} cosine={c:.6f}  max-abs={mab:.3e}  {'OK' if ok else 'FAIL'}")

print("\nEMBEDDING INFERENCE PARITY:", "PASS" if all_ok else "FAIL")
assert all_ok, "embeddings diverged across devices"
