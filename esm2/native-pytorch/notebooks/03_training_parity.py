# ---
# jupyter:
#   jupytext:
#     text_representation: {format_name: percent}
#   kernelspec: {display_name: Python 3, name: python3}
# ---

# %% [markdown]
# # ESM-2 training parity — CPU / CUDA / Trainium  (M5: on-device training)
# The third parity check (after 01-inference, 02-backprop): a real **multi-step training loop** — forward →
# CrossEntropy loss → backward → **`optimizer.step()`** — on a composite (ESM-2 backbone + a new head), run
# on every device, checking the **loss trajectory** and **final weights** match CPU. This proves that
# `optimizer.step()`, the Adam optimizer state, and multi-step convergence all work on the Trainium core —
# i.e. you can actually *train* new models on these backbones, not just run one backward.
#
# `cuda` auto-skips when absent. Pin a free core:
# `NEURON_RT_VISIBLE_CORES=0 jupyter nbconvert --to notebook --execute 03_training_parity.ipynb`.

# %%
import os
os.environ.setdefault("NEURON_RT_VISIBLE_CORES", "0")
import sys
sys.path.insert(0, os.path.abspath("../src"))
import torch
import torch.nn.functional as F
import esm2_reference as R

K = 8               # training steps
LR = 1e-3
LABELS = torch.tensor([0, 1])       # deterministic binary labels for the fixed batch

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
print("torch", torch.__version__, "| devices:", DEVICES, "| model:", R.MODEL_NAME, f"| {K} steps, Adam lr={LR}")

# %% [markdown]
# ## Train K steps on each device (identical seed / init / batch / optimizer)
# %%
def train(device):
    torch.manual_seed(0)                                  # identical init across devices
    model = R.load(device, finetune_backbone=True)        # ESM-2 backbone + new head, ALL params trainable
    ids, mask = R.build_inputs()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    losses = []
    for _ in range(K):
        opt.zero_grad(set_to_none=True)
        _, logits = model(ids.to(device), mask.to(device))
        loss = F.cross_entropy(logits, LABELS.to(device))
        loss.backward()
        opt.step()
        if device == "neuron":
            import torch_neuronx; torch_neuronx.synchronize()
        losses.append(float(loss.detach().cpu()))
    weights = {n: p.detach().float().cpu() for n, p in model.named_parameters()}
    return losses, weights

results = {d: train(d) for d in DEVICES}
for d in DEVICES:
    print(f"{d:7s} loss: {[round(x, 4) for x in results[d][0]]}")
lc = results["cpu"][0]
print(f"\nloss decreased on CPU: {lc[0]:.4f} -> {lc[-1]:.4f}  ({'learning' if lc[-1] < lc[0] else 'NO decrease'})")

# %% [markdown]
# ## Check the training trajectory + final weights match CPU
# %%
ref_loss, ref_w = results["cpu"]
all_ok = lc[-1] < lc[0]        # must actually learn
for d in DEVICES:
    if d == "cpu":
        continue
    dl, dw = results[d]
    max_loss_diff = max(abs(a - b) for a, b in zip(ref_loss, dl))
    max_w_diff = max((ref_w[n] - dw[n]).abs().max().item() for n in ref_w)
    ok = max_loss_diff < 1e-2 and max_w_diff < 1e-2
    all_ok = all_ok and ok
    print(f"{d} vs cpu: max per-step loss diff={max_loss_diff:.3e}  max final-weight diff={max_w_diff:.3e}  {'OK' if ok else 'FAIL'}")

print("\nTRAINING PARITY:", "PASS" if all_ok else "FAIL")
assert all_ok, "training trajectory / final weights diverged across devices, or the loss did not decrease"
