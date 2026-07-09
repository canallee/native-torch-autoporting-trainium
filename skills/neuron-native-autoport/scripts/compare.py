"""Numerical comparison for Phase B (R7): a Neuron run vs the frozen Phase-A oracle.

Two entry points:
  summarize(named_tensors) -> dict   # compact per-tensor stats + a flat slice (for a MANIFEST)
  compare(ref_stats, test_stats)     # ref = manifest stats, test = summarize() of the Neuron run

Generalized summarize/compare over a flat {name: tensor} mapping so it works for any model.
"""
import torch
import torch.nn.functional as F


def _flatten_named(obj, prefix=""):
    """Walk a tensor / dict / list structure into a flat {name: tensor} mapping."""
    out = {}
    if torch.is_tensor(obj):
        out[prefix or "out"] = obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            out.update(_flatten_named(v, f"{prefix}.{k}" if prefix else str(k)))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            out.update(_flatten_named(v, f"{prefix}[{i}]" if prefix else f"[{i}]"))
    return out


def summarize(outputs, slice_n=64):
    """Reduce arbitrary tensor output structure to comparable per-tensor stats.

    Returns {name: {shape, dtype, mean, std, absmax, slice}} where slice is a small
    flattened fp32 sample for direct numerical comparison.
    """
    stats = {}
    for name, t in _flatten_named(outputs).items():
        t = t.detach().to(torch.float32).cpu()
        flat = t.flatten()
        stats[name] = {
            "shape": tuple(t.shape),
            "dtype": str(t.dtype),
            "mean": flat.mean().item(),
            "std": flat.std(unbiased=False).item(),
            "absmax": flat.abs().max().item(),
            "slice": flat[:slice_n].clone(),
        }
    return stats


def compare(ref, test, atol=1e-2, rtol=1e-2):
    """Compare two summarize() dicts (ref = oracle). Returns (all_ok, rows).

    A tensor passes if max relative diff on the sampled slice <= rtol OR max abs
    diff <= atol. Also reports cosine over the slice.
    """
    rows = []
    all_ok = True
    for key in sorted(ref):
        if key not in test:
            rows.append((key, "MISSING", "", "", ""))
            all_ok = False
            continue
        r, t = ref[key], test[key]
        if tuple(r["shape"]) != tuple(t["shape"]):
            rows.append((key, "SHAPE", str(r["shape"]), str(t["shape"]), ""))
            all_ok = False
            continue
        a = torch.as_tensor(r["slice"]).flatten()
        b = torch.as_tensor(t["slice"]).flatten()
        denom = a.abs().clamp_min(1e-6)
        max_rel = ((a - b).abs() / denom).max().item()
        max_abs = (a - b).abs().max().item()
        cos = F.cosine_similarity(a, b, dim=0).item()
        ok = max_rel <= rtol or max_abs <= atol
        all_ok = all_ok and ok
        rows.append((key, "OK" if ok else "DIFF", f"cos={cos:.6f}",
                     f"maxrel={max_rel:.4f}", f"maxabs={max_abs:.3e}"))
    return all_ok, rows


def compare_to_manifest(manifest, run_outputs, section="outputs", atol=1e-2, rtol=1e-2):
    """R7 convenience: compare a Neuron run against the frozen Phase-A MANIFEST.json.

    manifest: path to MANIFEST.json OR the loaded dict. run_outputs: the model's output
    structure (tensor/tuple/dict) — summarize()'d here. section: 'outputs' or 'grads'.

    ⚠️ KEY ALIGNMENT: keys must match between the manifest and `run_outputs`. `summarize()`
    keys a dict/`ModelOutput` by NAME (`logits`) and a tuple/list POSITIONALLY (`[0]`). Grads
    align automatically (both keyed by param name). OUTPUTS only align if the Phase-A capture
    and this Neuron run produced the SAME structure — the shipped ports ensure that by running
    the SAME shared `<model>_reference` wrapper on both sides (capture_adapter_template note).
    If your manifest keyed outputs by NAME (raw-model capture) but the port wrapper returns a
    TUPLE, re-key the run first: `compare_to_manifest(m, {"logits": logits})`. A silent key
    mismatch reports every tensor as `MISSING` — a bogus FAIL, not a real numerical one.
    The manifest's list-valued slices are `as_tensor`-compatible. Returns (all_ok, rows).
    """
    import json as _json
    if isinstance(manifest, str):
        with open(manifest) as f:
            manifest = _json.load(f)
    ref = manifest.get(section, manifest)
    return compare(ref, summarize(run_outputs), atol=atol, rtol=rtol)


def print_rows(rows):
    for key, status, c, mr, ma in rows:
        print(f"    {key:36s} {status:7s} {c:18s} {mr:16s} {ma}")
