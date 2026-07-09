# ---
# jupyter:
#   jupytext:
#     text_representation: {format_name: percent}
#   kernelspec: {display_name: Python 3, name: python3}
# ---

# %% [markdown]
# # Evo2 inference parity — CPU / CUDA / Trainium  (DEV mode, FFT→conv1d)
# Runs the **port** (vendored Evo2 with the inline FFT→conv1d Neuron patch) on every device and compares
# to the **original FFT model** (ground-truth oracle, CPU). This proves BOTH: (a) the conv1d rewrite equals
# the FFT original, and (b) CPU/Trainium device parity. `cuda` auto-skips when absent.
# `NEURON_RT_VISIBLE_CORES=0 jupyter nbconvert --to notebook --execute 01_inference_parity.ipynb`.

# %%
import os
os.environ.setdefault("NEURON_RT_VISIBLE_CORES", "0")
import sys
sys.path.insert(0, os.path.abspath("../src"))
import torch
import torch.nn.functional as F
import evo2_reference as R

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
print("torch", torch.__version__, "| devices:", DEVICES, "| model:", R.MODEL_ID)

# %% [markdown]
# ## Ground-truth: original FFT model on CPU
# %%
ids, = R.build_inputs()
with torch.no_grad():
    ref = tuple(t.float() for t in R.load_fft_reference()(ids))   # (hidden, logits), FFT
for name, t in zip(R.OUTPUT_ORDER, ref):
    print(f"FFT-ref {name:7s} shape={tuple(t.shape)}")

# %% [markdown]
# ## Run the port (conv1d) on each device; compare to the FFT reference
# %%
def run(device):
    model = R.load(device)                       # vendored conv1d port
    with torch.no_grad():
        out = model(ids.to(device))
    if device == "neuron":
        import torch_neuronx; torch_neuronx.synchronize()
    return tuple(t.detach().float().cpu() for t in out)

def cos(a, b): return F.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()

all_ok = True
for d in DEVICES:
    out = run(d)
    print(f"\nport@{d} vs FFT reference:")
    for name, r, o in zip(R.OUTPUT_ORDER, ref, out):
        c = cos(r, o); ok = c >= 0.99
        all_ok = all_ok and ok
        print(f"  {name:7s} cosine={c:.6f}  max-abs={(r-o).abs().max():.3e}  {'OK' if ok else 'FAIL'}")
    t1 = (ref[1].argmax(-1) == out[1].argmax(-1)).float().mean().item() * 100
    print(f"  top-1 next-byte agreement={t1:.1f}%")

print("\nINFERENCE PARITY (rewrite + device):", "PASS" if all_ok else "FAIL")
assert all_ok, "port diverged from the FFT reference"
