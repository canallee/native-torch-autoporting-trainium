# ---
# jupyter:
#   jupytext:
#     text_representation: {format_name: percent}
#   kernelspec: {display_name: Python 3, name: python3}
# ---

# %% [markdown]
# # CLIP backprop parity — CPU / CUDA / Trainium
# Runs one backward pass of openai CLIP's **own vendored definition** on every available device and
# checks the **gradients** match. CPU is the reference; `cuda` auto-skips when absent; `neuron` runs on
# the Trainium core. Pin a free core:
# `NEURON_RT_VISIBLE_CORES=0 jupyter nbconvert --to notebook --execute 02_backprop_parity.ipynb`.
#
# ⚠️ Gradient parity is judged **magnitude-aware**, not by raw cosine: gradient sets contain
# analytically-zero tensors (e.g. attention key/`in_proj` bias components) whose cosine is pure fp noise.
# A tensor passes if `cosine >= 0.99` OR its max-abs diff is `<= 1e-3` of the model's global grad scale.

# %%
import os
os.environ.setdefault("NEURON_RT_VISIBLE_CORES", "0")
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
# ## One backprop on each device; collect per-parameter grads
# %%
def run_backprop(device):
    model = R.load(device)
    for p in model.parameters():
        p.requires_grad_(True)
    model.zero_grad(set_to_none=True)
    px, ids = R.build_inputs()
    out = model(px.to(device), ids.to(device))
    loss = R.loss_fn(out)
    loss.backward()
    if device == "neuron":
        import torch_neuronx; torch_neuronx.synchronize()
    grads = {n: p.grad.detach().float().cpu() for n, p in model.named_parameters() if p.grad is not None}
    return float(loss.detach().float().cpu()), grads

results = {d: run_backprop(d) for d in DEVICES}
for d in DEVICES:
    print(f"{d:7s} loss={results[d][0]:.6f}  grad_tensors={len(results[d][1])}")

# %% [markdown]
# ## Magnitude-aware gradient comparison vs CPU
# %%
def compare_grads(ref, test):
    global_scale = max(g.abs().max().item() for g in ref.values()) or 1.0
    real_fail, near_zero_ok = [], 0
    for n, gr in ref.items():
        a, b = gr.flatten(), test[n].flatten()
        c = F.cosine_similarity(a, b, dim=0).item()
        abs_diff = (a - b).abs().max().item()
        if c < 0.99:
            if abs_diff / global_scale <= 1e-3:
                near_zero_ok += 1          # analytically-zero grad; cosine meaningless
            else:
                real_fail.append((n, c, abs_diff))
    return global_scale, near_zero_ok, real_fail

ref = results["cpu"][1]
all_ok = True
for d in DEVICES:
    if d == "cpu":
        continue
    gscale, nz, real = compare_grads(ref, results[d][1])
    print(f"\n{d} vs cpu: {len(ref)} grad tensors | global |grad| scale={gscale:.3e}")
    print(f"  cosine-pass or negligible: {len(ref) - len(real)}/{len(ref)} "
          f"(of which {nz} near-zero judged by abs tolerance)")
    for n, c, ad in real:
        print(f"  REAL DIFF {n}  cos={c:.3f}  absdiff={ad:.3e}")
    all_ok = all_ok and not real

print("\nBACKPROP PARITY:", "PASS" if all_ok else "FAIL")
assert all_ok, "gradients diverged across devices (beyond near-zero fp noise)"
