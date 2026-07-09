# ---
# jupyter:
#   jupytext:
#     text_representation: {format_name: percent}
#   kernelspec: {display_name: Python 3, name: python3}
# ---

# %% [markdown]
# # GPN-Star training parity — CPU / CUDA / Trainium  (dev-mode: on-device training)
# The third parity check (after 01-inference, 02-backprop): a real **multi-step training loop** —
# forward -> masked-token **CrossEntropy** loss -> backward -> **`optimizer.step()`** (Adam) — on
# GPN-Star's **own vendored definition** (`gpn_star_neuron`, the full `GPNStarForMaskedLM`: axial-
# attention MSA encoder + FIRE phylo-bias + MLM head), run on every device, checking the **loss
# trajectory** and **final weights** match CPU and that the loss **decreases**. This proves
# `optimizer.step()`, the Adam optimizer state, and multi-step convergence all lower on the Trainium
# core — i.e. you can actually *train / fine-tune* GPN-Star on device, not just run one backward.
# Since dev mode owns and trains the architecture, ALL ~25M parameters are trainable here.
#
# GPN-Star is an **MLM**, so we train on its native objective: mask a deterministic subset of the
# target positions and predict the original tokens (masked-token CE). Unlike 01/02 there is **no
# frozen oracle** — 03 self-compares CPU-vs-device at runtime; that IS the check. The model runs in
# `eval()` (dropout off) so CPU and device draw no divergent RNG; the optimizer.step()/Adam-state/
# multi-step-convergence path is still fully exercised. Static shapes so the train step compiles once.
#
# Pin a free core: `NEURON_RT_VISIBLE_CORES=0 jupyter nbconvert --to notebook --execute 03_training_parity.ipynb`.

# %%
import os
os.environ.setdefault("NEURON_RT_VISIBLE_CORES", "0")
import sys
sys.path.insert(0, os.path.abspath("../src"))
import torch
import torch.nn.functional as F
import gpn_star_reference as R

K = 5               # training steps
LR = 1e-4           # Adam
MASK_ID = 5         # fixed in-vocab token used to mask targets (vocab_size=6); deterministic, not the answer
MASK_FRAC = 0.30    # fraction of target positions masked (deterministic mask, fixed seed)

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
print("torch", torch.__version__, "| devices:", DEVICES, "| model:", R.MODEL_NAME,
      f"| {K} steps, Adam lr={LR}")

# %% [markdown]
# ## Build a fixed masked-LM batch (deterministic mask), then train K steps on each device
# input_ids/source_ids/target_species come from the shared harness; we mask a fixed subset of the
# target rows and keep the originals as labels. Identical seed / init / batch / mask / optimizer on
# every device. logits are (B, L, T, vocab); labels are (B, L, T).
# %%
def masked_batch():
    input_ids, source_ids, target_species = R.build_inputs()
    labels = input_ids.clone()
    g = torch.Generator().manual_seed(0)                              # deterministic mask across devices
    mask = torch.rand(input_ids.shape, generator=g) < MASK_FRAC
    if not bool(mask.any()):
        mask[..., 0] = True                                           # guarantee >=1 masked position
    masked_input = input_ids.masked_fill(mask, MASK_ID)
    return masked_input, source_ids, target_species, labels, mask

MASKED_INPUT, SOURCE_IDS, TARGET_SPECIES, LABELS, MASK = masked_batch()
print(f"masked {int(MASK.sum())}/{MASK.numel()} target positions")

def ce_masked(logits, labels, mask):
    V = logits.shape[-1]
    sel = mask.reshape(-1)
    return F.cross_entropy(logits.reshape(-1, V)[sel], labels.reshape(-1)[sel])

def train(device):
    torch.manual_seed(0)                                  # identical init across devices
    model = R.load(device)                                # full GPNStarForMaskedLM, fp32, ALL params trainable
    mi, si, ts, lab, mk = (t.to(device) for t in (MASKED_INPUT, SOURCE_IDS, TARGET_SPECIES, LABELS, MASK))
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    losses = []
    for _ in range(K):
        opt.zero_grad(set_to_none=True)
        logits = model(mi, si, ts)[0]                     # (logits,) -> (B, L, T, vocab)
        loss = ce_masked(logits, lab, mk)
        loss.backward()
        opt.step()
        if device == "neuron":
            import torch_neuronx; torch_neuronx.synchronize()
        losses.append(float(loss.detach().float().cpu()))
    weights = {n: p.detach().float().cpu() for n, p in model.named_parameters()}
    return losses, weights

results = {d: train(d) for d in DEVICES}
for d in DEVICES:
    print(f"{d:7s} loss: {[round(x, 4) for x in results[d][0]]}")
lc = results["cpu"][0]
print(f"\nloss decreased on CPU: {lc[0]:.4f} -> {lc[-1]:.4f}  "
      f"({'learning' if lc[-1] < lc[0] else 'NO decrease'})")

# %% [markdown]
# ## Check the training trajectory + final weights match CPU
# RELATIVE loss gate (loss scales vary — use relative, never an absolute `<1e-2` gate).
# Weights: magnitude-aware absolute diff.
# %%
ref_loss, ref_w = results["cpu"]
all_ok = lc[-1] < lc[0]        # must actually learn
for d in DEVICES:
    if d == "cpu":
        continue
    dl, dw = results[d]
    max_loss_diff = max(abs(a - b) for a, b in zip(ref_loss, dl))
    max_w_diff = max((ref_w[n] - dw[n]).abs().max().item() for n in ref_w)
    rel_loss = max_loss_diff / max(abs(x) for x in ref_loss)
    ok = rel_loss < 5e-3 and max_w_diff < 1e-2
    all_ok = all_ok and ok
    print(f"{d} vs cpu: max per-step loss diff={max_loss_diff:.3e} ({rel_loss:.2e} rel)  "
          f"max final-weight diff={max_w_diff:.3e}  {'OK' if ok else 'FAIL'}")

print("\nTRAINING PARITY:", "PASS" if all_ok else "FAIL")
assert all_ok, "training trajectory / final weights diverged across devices, or the loss did not decrease"
