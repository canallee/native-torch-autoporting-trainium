"""Phase A (A4/A5): capture the original model's reference outputs + gradients.

Model-agnostic engine. Each `baselines/<model>/capture.py` supplies a small adapter
with these callables; this runner orchestrates forward + one backprop and saves
artifacts + summary stats for the MANIFEST.

Adapter contract (all in baselines/<model>/capture.py):
    load_model(device: torch.device) -> torch.nn.Module        # eval() model on `device`
    build_inputs(device: torch.device) -> dict | tuple         # deterministic, seed-stable
    loss_fn(outputs) -> torch.Tensor (scalar)   [optional]      # default: sum of float outputs
    MODEL_ID: str                                [optional]     # for the manifest

Device policy (per design): CPU is the DEFAULT (works on the trn box, no GPU here).
`cuda` is auto-detected and gracefully skipped when unavailable — the notebook/CLI
pass device="cuda" only on a GPU host. Reference envs pin torch==2.11.0 so the oracle
is apples-to-apples with the Trainium run.
"""
import os
import torch


def _to_device(inputs, device):
    if torch.is_tensor(inputs):
        return inputs.to(device)
    if isinstance(inputs, dict):
        return {k: _to_device(v, device) for k, v in inputs.items()}
    if isinstance(inputs, (list, tuple)):
        return type(inputs)(_to_device(v, device) for v in inputs)
    return inputs


def _call(model, inputs):
    if isinstance(inputs, dict):
        return model(**inputs)
    if isinstance(inputs, (list, tuple)):
        return model(*inputs)
    return model(inputs)


def _default_loss(outputs):
    """Sum of all floating-point output tensors — a simple scalar to get gradients
    when the model defines no loss. The manifest records that this default was used."""
    from compare import _flatten_named
    total = None
    for t in _flatten_named(outputs).values():
        if torch.is_floating_point(t):
            s = t.float().sum()
            total = s if total is None else total + s
    if total is None:
        raise RuntimeError("no floating-point outputs to build a default loss from")
    return total


def resolve_device(name: str) -> torch.device:
    """cpu (default) | cuda (auto-skip if absent) | neuron."""
    name = (name or "cpu").lower()
    if name == "cuda" and not torch.cuda.is_available():
        print("[capture] cuda requested but unavailable — falling back to cpu")
        return torch.device("cpu")
    return torch.device(name)


def run_capture(adapter, device="cpu", seed=1234, out_dir=None, dtype=None):
    """Run forward + one backprop; return (outputs, grads, meta). Saves .pt if out_dir set."""
    dev = resolve_device(device)
    torch.manual_seed(seed)

    model = adapter.load_model(dev)
    model.eval()
    if dtype is not None:
        model = model.to(dtype)
    inputs = _to_device(adapter.build_inputs(dev), dev)
    loss_fn = getattr(adapter, "loss_fn", _default_loss)
    used_default_loss = not hasattr(adapter, "loss_fn")

    # (1) Inference — no grad.
    with torch.no_grad():
        outputs = _call(model, inputs)

    # (2) One backprop step.
    model.zero_grad(set_to_none=True)
    for p in model.parameters():
        p.requires_grad_(True)
    outputs_grad = _call(model, inputs)
    loss = loss_fn(outputs_grad)
    loss.backward()
    grads = {n: p.grad.detach().cpu() for n, p in model.named_parameters() if p.grad is not None}

    meta = {
        "device": str(dev),
        "seed": seed,
        "dtype": str(dtype) if dtype else "model-default",
        "loss": "default(sum-of-float-outputs)" if used_default_loss else "adapter.loss_fn",
        "loss_value": float(loss.detach().cpu()),
        "model_id": getattr(adapter, "MODEL_ID", None),
        "torch_version": torch.__version__,
        "num_grad_tensors": len(grads),
    }

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        torch.save(outputs, os.path.join(out_dir, "outputs.pt"))
        torch.save(grads, os.path.join(out_dir, "grads.pt"))
        print(f"[capture] saved outputs.pt + grads.pt -> {out_dir}")
    print(f"[capture] device={dev} loss={meta['loss_value']:.6e} grad_tensors={len(grads)}")
    return outputs, grads, meta


def _load_adapter(path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("capture_adapter", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


if __name__ == "__main__":
    # One-command Phase-A driver: capture the oracle AND write MANIFEST.json (no manual meta copy).
    # Run from baselines/<model>/ :  python <skill>/scripts/capture_reference.py --adapter capture.py
    import argparse
    import json
    import os
    from build_manifest import _manifest_section

    ap = argparse.ArgumentParser(description="Phase-A: capture reference oracle + write MANIFEST.json")
    ap.add_argument("--adapter", default="capture.py", help="path to this model's capture.py adapter")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out-dir", default="artifacts")
    ap.add_argument("--manifest", default="MANIFEST.json")
    ap.add_argument("--env-lock", default="env/versions.txt")
    args = ap.parse_args()

    adapter = _load_adapter(args.adapter)
    outputs, grads, meta = run_capture(adapter, device=args.device, seed=args.seed, out_dir=args.out_dir)
    manifest = {
        "meta": meta,
        "env_lock": args.env_lock if os.path.exists(args.env_lock) else None,
        "outputs": _manifest_section(outputs),
        "grads": _manifest_section(grads),
    }
    json.dump(manifest, open(args.manifest, "w"), indent=2)
    print(f"[capture] wrote {args.manifest}: "
          f"{len(manifest['outputs'])} outputs, {len(manifest['grads'])} grad tensors")
