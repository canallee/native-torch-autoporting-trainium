# ---
# jupyter:
#   jupytext:
#     text_representation: {format_name: percent}
#   kernelspec: {display_name: Python 3, name: python3}
# ---

# %% [markdown]
# # CLIP inference parity — CPU / CUDA / Trainium
# Runs openai CLIP's **own vendored definition** (`clip_neuron`) on every available device and checks
# the outputs match. CPU is the reference; `cuda` is auto-skipped when absent (e.g. on the trn box);
# `neuron` runs on the Trainium core. Pin a free core before running:
# `NEURON_RT_VISIBLE_CORES=0 jupyter nbconvert --to notebook --execute 01_inference_parity.ipynb`.

# %%
import os
os.environ.setdefault("NEURON_RT_VISIBLE_CORES", "0")   # set before the Neuron runtime initializes
import sys
sys.path.insert(0, os.path.abspath("../src"))
import torch
import torch.nn.functional as F
import clip_reference as R

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
print("torch", torch.__version__, "| devices:", DEVICES)

# %% [markdown]
# ## Run inference on each device
# %%
def run_inference(device):
    model = R.load(device)
    px, ids = R.build_inputs()
    with torch.no_grad():
        out = model(px.to(device), ids.to(device))
    if device == "neuron":
        import torch_neuronx; torch_neuronx.synchronize()
    return tuple(t.detach().float().cpu() for t in out)

results = {d: run_inference(d) for d in DEVICES}
for name, t in zip(R.OUTPUT_ORDER, results["cpu"]):
    print(f"cpu {name:18s} shape={tuple(t.shape)}")

# %% [markdown]
# ## Check every non-CPU device matches CPU (cosine + max-abs per output)
# %%
def cos(a, b): return F.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()

ref = results["cpu"]
all_ok = True
for d in DEVICES:
    if d == "cpu":
        continue
    print(f"\n{d} vs cpu:")
    for name, a, b in zip(R.OUTPUT_ORDER, ref, results[d]):
        c = cos(a, b); mab = (a - b).abs().max().item()
        ok = c >= 0.99
        all_ok = all_ok and ok
        print(f"  {name:18s} cosine={c:.6f}  max-abs={mab:.3e}  {'OK' if ok else 'FAIL'}")

print("\nINFERENCE PARITY:", "PASS" if all_ok else "FAIL")
assert all_ok, "inference outputs diverged across devices"
