#!/usr/bin/env python
"""Headless CLI: validate the CLIP port (openai's own vendored definition) on Trainium.

The notebooks (../notebooks/01_inference_parity.ipynb, 02_backprop_parity.ipynb) are the primary,
human-facing deliverable — CPU/CUDA/Trainium parity for inference and backprop. This CLI runs the
same two checks headlessly (for CI / quick confirmation) against the frozen Phase-A oracle.

Run on the Trainium box:
    conda activate torch-neuron
    NEURON_RT_VISIBLE_CORES=0 python port_clip.py [--grad]
"""
import argparse
import json
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import clip_reference as R  # shared harness over the vendored clip_neuron definition


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="../../../baselines/clip/MANIFEST.json")
    ap.add_argument("--grad", action="store_true", help="also check gradient parity vs the oracle")
    args = ap.parse_args()
    here = os.path.dirname(os.path.abspath(__file__))

    print("[clip] loading vendored openai CLIP on device='neuron' (first call compiles ~minutes)")
    model = R.load("neuron")
    px, ids = R.build_inputs()
    dev_inputs = (px.to("neuron"), ids.to("neuron"))

    import torch_neuronx
    with torch.no_grad():
        out = model(*dev_inputs)
    torch_neuronx.synchronize()
    out = tuple(t.detach().float().cpu() for t in out)

    ref = json.load(open(os.path.join(here, args.manifest)))["outputs"]
    ok = True
    print("[clip] inference vs Phase-A oracle:")
    # Compare in manifest key order ([0],[1],[2]) against our tuple order.
    keys = sorted(ref, key=lambda k: int(k.strip("[]")) if k.strip("[]").isdigit() else k)
    for key, t in zip(keys, out):
        a = torch.as_tensor(ref[key]["slice"])
        b = t.flatten()[: a.numel()]
        cos = F.cosine_similarity(a, b, dim=0).item()
        mab = (a - b).abs().max().item()
        ok = ok and cos >= 0.99
        print(f"    {key} cos={cos:.6f} max-abs={mab:.3e}")

    grad_ok = check_grad_parity(model, dev_inputs) if args.grad else True
    print("[clip] PASS" if (ok and grad_ok) else "[clip] FAIL")
    sys.exit(0 if (ok and grad_ok) else 1)


def check_grad_parity(model, dev_inputs, grad_oracle="../../../baselines/clip/artifacts/grads.pt"):
    """One on-device backprop vs the CPU grad oracle. MAGNITUDE-AWARE gate (see
    knowledge_base/validation_metrics.md): pass if cosine>=0.99 OR max-abs diff is <=1e-3 of the
    global grad scale (near-zero grads are fp-noise where cosine is meaningless)."""
    import torch_neuronx
    here = os.path.dirname(os.path.abspath(__file__))
    cpu_grads = torch.load(os.path.join(here, grad_oracle))
    for p in model.parameters():
        p.requires_grad_(True)
    model.zero_grad(set_to_none=True)
    R.loss_fn(model(*dev_inputs)).backward()
    torch_neuronx.synchronize()
    neu = {n: p.grad.detach().float().cpu() for n, p in model.named_parameters() if p.grad is not None}
    scale = max(g.abs().max().item() for g in cpu_grads.values()) or 1.0
    real = []
    for n, gr in cpu_grads.items():
        a, b = gr.flatten().float(), neu[n].flatten()
        if F.cosine_similarity(a, b, dim=0).item() < 0.99 and (a - b).abs().max().item() / scale > 1e-3:
            real.append(n)
    print(f"[clip] grad parity: {len(cpu_grads)} tensors, real disagreements={len(real)}")
    for n in real:
        print("    REAL DIFF", n)
    return not real


if __name__ == "__main__":
    main()
