# ---
# jupyter:
#   jupytext:
#     text_representation: {format_name: percent}
#   kernelspec: {display_name: Python 3, name: python3}
# ---

# %% [markdown]
# # Evo2 training parity — CPU / CUDA / Trainium  (dev-mode: on-device training)
# The third parity check (after 01-inference, 02-backprop): a real **multi-step training loop** —
# forward -> next-token **CrossEntropy** loss -> backward -> **`optimizer.step()`** (Adam) — on
# Evo2's **own vendored definition** (`evo2_neuron`, StripedHyena2 with the FFT->conv1d op-rewrite),
# run on every device, checking the **loss trajectory** and **final weights** match CPU and that the
# loss **decreases**. This proves `optimizer.step()`, the Adam optimizer state, and multi-step
# convergence all lower on the Trainium core — i.e. you can actually *train / fine-tune* Evo2 on
# device, not just run one backward. Since dev mode owns and trains the architecture, ALL 1B
# parameters are trainable here.
#
# Unlike 01/02 there is **no frozen oracle**: 03 self-compares CPU-vs-device at runtime — that IS the
# check. The model runs in `eval()` (dropout off) so CPU and device draw no divergent RNG; the
# optimizer.step()/Adam-state/multi-step-convergence path is still fully exercised. Static shapes
# (fixed batch + seq len) so the train step compiles once and reuses. Neuron auto-casts int64->int32
# for `input_ids` (benign).
#
# Pin a free core: `NEURON_RT_VISIBLE_CORES=0 jupyter nbconvert --to notebook --execute 03_training_parity.ipynb`.

# %%
import os
os.environ.setdefault("NEURON_RT_VISIBLE_CORES", "0")
import sys
sys.path.insert(0, os.path.abspath("../src"))
import torch
import torch.nn.functional as F
import evo2_reference as R

K = 5               # training steps
LR = 1e-5           # Adam — gentle enough for a monotonic descent (the pretrained Evo2 already models the
                    # periodic input, so a large LR overshoots into a steep, fp-noisy near-saturated basin)

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
print("torch", torch.__version__, "| devices:", DEVICES, "| model:", R.MODEL_ID,
      f"| {K} steps, Adam lr={LR}")

# %% [markdown]
# ## Train K steps on each device (identical seed / init / batch / optimizer)
# Loss = next-token cross-entropy over the fixed input batch (a genuine causal-DNA-LM objective:
# shift logits/labels). `R.load` returns the wrapper `(last_hidden_state, logits)` — logits is [1].
# %%
def ce_next_token(logits, ids):
    V = logits.shape[-1]
    shift_logits = logits[:, :-1, :].reshape(-1, V)
    shift_labels = ids[:, 1:].reshape(-1)
    return F.cross_entropy(shift_logits, shift_labels)

def train(device):
    torch.manual_seed(0)                                  # identical init across devices
    model = R.load(device)                                # Evo2 (conv1d rewrite), fp32, ALL params trainable
    (ids,) = R.build_inputs()
    ids = ids.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    losses = []
    for _ in range(K):
        opt.zero_grad(set_to_none=True)
        logits = model(ids)[1]                            # (hidden, logits) -> logits
        loss = ce_next_token(logits, ids)
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
# RELATIVE loss gate (evo2's first-step loss is ~40 — an absolute `<1e-2` gate would false-FAIL).
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
    # magnitude-aware: loss scale varies wildly across models (evo2's first-step CE can be ~40) — use a
    # RELATIVE loss-trajectory gate (|Δ|/|loss| ≤ 1e-2), NEVER an absolute one. Weights: absolute diff.
    rel_loss = max_loss_diff / max(abs(x) for x in ref_loss)
    ok = rel_loss < 1e-2 and max_w_diff < 1e-2
    all_ok = all_ok and ok
    print(f"{d} vs cpu: max per-step loss diff={max_loss_diff:.3e} ({rel_loss:.2e} rel)  "
          f"max final-weight diff={max_w_diff:.3e}  {'OK' if ok else 'FAIL'}")

print("\nTRAINING PARITY:", "PASS" if all_ok else "FAIL")
assert all_ok, "training trajectory / final weights diverged across devices, or the loss did not decrease"
