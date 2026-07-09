# ---
# jupyter:
#   jupytext:
#     text_representation: {format_name: percent}
#   kernelspec: {display_name: Python 3, name: python3}
# ---

# %% [markdown]
# # ESM-2 backprop / gradient-flow parity — CPU / CUDA / Trainium  (INTEGRATION mode)
# The integration-critical test: run one backward through a **composite (ESM-2 backbone + a new
# downstream head)** on every device and check the gradients match — proving that when you use ESM-2
# embeddings as inputs to a new model, gradients flow correctly through the backbone **and** into the
# new head on Trainium (so the new model can train / the backbone can be finetuned).
#
# Pin a free core: `NEURON_RT_VISIBLE_CORES=0 jupyter nbconvert --to notebook --execute 02_backprop_parity.ipynb`.
# Gradient parity is **magnitude-aware** (near-zero grads are fp noise where cosine is meaningless).

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
print("torch", torch.__version__, "| devices:", DEVICES)

# %% [markdown]
# ## One backward per device through the composite; collect grads (backbone + head)
# %%
def run_backprop(device):
    model = R.load(device, finetune_backbone=True)   # backbone trainable -> grads flow all the way
    for p in model.parameters():
        p.requires_grad_(True)
    model.zero_grad(set_to_none=True)
    ids, mask = R.build_inputs()
    out = model(ids.to(device), mask.to(device))
    R.loss_fn(out).backward()
    if device == "neuron":
        import torch_neuronx; torch_neuronx.synchronize()
    grads = {n: p.grad.detach().float().cpu() for n, p in model.named_parameters() if p.grad is not None}
    return grads

results = {d: run_backprop(d) for d in DEVICES}
for d in DEVICES:
    print(f"{d:7s} grad_tensors={len(results[d])}")

# %% [markdown]
# ## Magnitude-aware gradient comparison vs CPU (report head vs backbone separately)
# %%
def compare(ref, test):
    scale = max(g.abs().max().item() for g in ref.values()) or 1.0
    real = []
    for n, gr in ref.items():
        a, b = gr.flatten(), test[n].flatten()
        if F.cosine_similarity(a, b, dim=0).item() < 0.99 and (a - b).abs().max().item() / scale > 1e-3:
            real.append((n, (a - b).abs().max().item()))
    return scale, real

ref, all_ok = results["cpu"], True
for d in DEVICES:
    if d == "cpu":
        continue
    scale, real = compare(ref, results[d])
    n_head = sum(1 for n in ref if n.startswith("head."))
    n_bb = len(ref) - n_head
    print(f"\n{d} vs cpu: {len(ref)} grad tensors ({n_bb} backbone + {n_head} new-head) | global |grad|={scale:.3e}")
    print(f"  matched: {len(ref) - len(real)}/{len(ref)}   (gradient flows through backbone AND into the new head)")
    for n, ad in real:
        print(f"  REAL DIFF {n}  absdiff={ad:.3e}")
    all_ok = all_ok and not real

print("\nGRADIENT-FLOW PARITY:", "PASS" if all_ok else "FAIL")
assert all_ok, "gradients diverged across devices (beyond near-zero fp noise)"
