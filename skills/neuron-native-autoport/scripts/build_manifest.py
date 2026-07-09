"""Phase A (A5): freeze the reference oracle into a committed MANIFEST.json.

The manifest is the ground truth Phase B validates against. It is small and
diff-friendly: per-tensor shape/dtype/stats + a short flat slice + a sha256 hash of
the full tensor bytes (so drift is detectable even beyond the sampled slice). Raw
tensors (outputs.pt / grads.pt) stay as gitignored artifacts; only this JSON is committed.

Usage (from a baselines/<model>/ dir, after capture_reference saved artifacts):
    python build_manifest.py --artifacts artifacts/ --out MANIFEST.json \
        --meta-json '{"device":"cpu","seed":1234,...}'
"""
import argparse, hashlib, json, os
import torch
from compare import summarize


def _hash_tensor(t: torch.Tensor) -> str:
    return hashlib.sha256(t.detach().to(torch.float32).cpu().numpy().tobytes()).hexdigest()[:16]


def _manifest_section(named):
    """summarize() + a per-tensor hash, JSON-safe (slice -> list)."""
    from compare import _flatten_named
    flat = _flatten_named(named)
    stats = summarize(named)
    section = {}
    for name, s in stats.items():
        section[name] = {
            "shape": list(s["shape"]),
            "dtype": s["dtype"],
            "mean": s["mean"], "std": s["std"], "absmax": s["absmax"],
            "slice": s["slice"].tolist(),
            "sha256_16": _hash_tensor(flat[name]),
        }
    return section


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", default="artifacts", help="dir with outputs.pt + grads.pt")
    ap.add_argument("--out", default="MANIFEST.json")
    ap.add_argument("--meta-json", default="{}", help="capture meta dict as JSON")
    ap.add_argument("--env-lock", default="env/versions.txt", help="path to recorded env lockfile/versions")
    args = ap.parse_args()

    outputs = torch.load(os.path.join(args.artifacts, "outputs.pt"))
    grads = torch.load(os.path.join(args.artifacts, "grads.pt"))
    meta = json.loads(args.meta_json)

    manifest = {
        "meta": meta,
        "env_lock": args.env_lock if os.path.exists(args.env_lock) else None,
        "outputs": _manifest_section(outputs),
        "grads": _manifest_section(grads),
    }
    with open(args.out, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[manifest] wrote {args.out}: "
          f"{len(manifest['outputs'])} output tensors, {len(manifest['grads'])} grad tensors")


if __name__ == "__main__":
    main()
