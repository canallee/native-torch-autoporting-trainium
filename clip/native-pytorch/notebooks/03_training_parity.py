# ---
# jupyter:
#   jupytext:
#     text_representation: {format_name: percent}
#   kernelspec: {display_name: Python 3, name: python3}
# ---

# %% [markdown]
# # CLIP training parity — CPU / CUDA / Trainium  (dev-mode: on-device training)
# The third parity check (after 01-inference, 02-backprop): a real **multi-step training loop** —
# forward -> **contrastive** loss -> backward -> **`optimizer.step()`** (Adam) — on CLIP's **own
# vendored definition** (`clip_neuron`, openai ViT-B/32 image encoder + text transformer + learnable
# `logit_scale`), run on every device, checking the **loss trajectory** and **final weights** match
# CPU and that the loss **decreases**. This proves `optimizer.step()`, the Adam optimizer state, and
# multi-step convergence all lower on the Trainium core — i.e. you can actually *train / fine-tune*
# CLIP on device, not just run one backward. Since dev mode owns and trains the architecture, ALL
# params (both encoders + `logit_scale`) are trainable here.
#
# CLIP's OWN objective is the **symmetric InfoNCE** contrastive loss: over the (image, text) batch,
# the scaled cosine-similarity matrix `logits_per_image` (N×N) should be largest on its diagonal
# (matched pairs) — cross-entropy of rows AND columns against `arange(N)`. Unlike 01/02 there is **no
# frozen oracle** — 03 self-compares CPU-vs-device at runtime; that IS the check. The model runs in
# `eval()` (dropout off) so CPU and device draw no divergent RNG; the optimizer.step()/Adam-state/
# multi-step-convergence path is still fully exercised. Static shapes compile once.
#
# Pin a free core: `NEURON_RT_VISIBLE_CORES=0 jupyter nbconvert --to notebook --execute 03_training_parity.ipynb`.

# %%
import os
os.environ.setdefault("NEURON_RT_VISIBLE_CORES", "0")
import sys
sys.path.insert(0, os.path.abspath("../src"))
import torch
import torch.nn.functional as F
import clip_reference as R

K = 5               # training steps
LR = 1e-5           # Adam — gentle: pretrained CLIP + scaled logits (logit_scale.exp()~100) give a steep
                    # contrastive landscape; a large LR overshoots into an fp-noisy near-saturated basin

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
# ## Train K steps on each device (identical seed / init / batch / optimizer)
# Loss = symmetric contrastive CE over the fixed (image, text) batch. `R.load` returns the wrapper
# `(image_features, text_features, logits_per_image)` — logits_per_image (N×N) is [2].
# %%
def clip_contrastive(logits_per_image):
    # symmetric InfoNCE via log_softmax over rows (each image vs texts) AND columns (each text vs
    # images) — mathematically identical to CE-of-rows + CE-of-cols vs arange(N), but avoids the
    # transpose-view aliasing + neuron's in-place cross_entropy op (which else bumps the shared
    # logits tensor's version and breaks the double backward).
    n = logits_per_image.shape[0]
    idx = torch.arange(n, device=logits_per_image.device)
    log_p_img = F.log_softmax(logits_per_image, dim=1)   # each image over the texts
    log_p_txt = F.log_softmax(logits_per_image, dim=0)   # each text over the images (column-wise)
    return -0.5 * (log_p_img[idx, idx].mean() + log_p_txt[idx, idx].mean())

def train(device):
    torch.manual_seed(0)                                  # identical init across devices
    model = R.load(device)                                # openai CLIP, fp32, ALL params trainable
    pixel_values, input_ids = (t.to(device) for t in R.build_inputs())
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    losses = []
    for _ in range(K):
        opt.zero_grad(set_to_none=True)
        logits_per_image = model(pixel_values, input_ids)[2]
        loss = clip_contrastive(logits_per_image)
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
    ok = rel_loss < 1e-2 and max_w_diff < 1e-2
    all_ok = all_ok and ok
    print(f"{d} vs cpu: max per-step loss diff={max_loss_diff:.3e} ({rel_loss:.2e} rel)  "
          f"max final-weight diff={max_w_diff:.3e}  {'OK' if ok else 'FAIL'}")

print("\nTRAINING PARITY:", "PASS" if all_ok else "FAIL")
assert all_ok, "training trajectory / final weights diverged across devices, or the loss did not decrease"
