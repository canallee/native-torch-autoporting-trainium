#!/usr/bin/env python
"""Real-data validation of the Trainium-ported GPN-Star, on Arabidopsis variant-effect prediction.

Scores real 1001 Genomes variants with GPN-Star (masked-LM log-likelihood-ratio, alt vs ref, fwd/rev
averaged) on CPU and on the Trainium NeuronCore, then compares against the authors' published reference
predictions (`songlab/1001gp` predictions/GPN-Star.parquet).

Three checks:
  (a) PORT PARITY on real data   — our Trainium scores vs our CPU scores  (the port claim)
  (b) FIDELITY to the reference  — our CPU scores vs the published GPN-Star scores
      (raw LLR, and after the pentanucleotide-context calibration the authors apply)
  (c) SCIENCE reproduces         — AUROC(score, label) for ours vs the published reference

Notes:
- VEP needs the masked-LM head, so this loads the FULL GPNStarForMaskedLM (the integration port shipped
  the headless encoder). Same sealed loader-shim (construct on real device, bypassing transformers-5.13
  meta-init) + two head-specific shims (4.x->5.x): `_tied_weights_keys` list->{} and tie decoder.bias.
- Real aligned-genome inputs come from the `songlab/tair10_multiz18way` 18-species MSA (data/).
Run:  NEURON_RT_VISIBLE_CORES=0 python run_vep_validation.py --n 256
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "../../src"))
import gpn_star_neuron as G          # reuse the port's config resolver (phylo_dist fix) + snapshot
import gpn.star.model as gsm
from gpn.star.vep import VEPInference
from gpn.star.data import GenomeMSA
from gpn.data import Tokenizer, ReverseComplementer

ZARR = os.path.join(HERE, "data/results/msa/tair10_multiz18way.zarr")
WINDOW = 128
ID2CHAR = {1: "A", 2: "C", 3: "G", 4: "T"}


def load_masked_lm(device):
    """Full GPNStarForMaskedLM on `device` (fp32), via the sealed loader-shim + head shims."""
    cfg, path = G._config()
    gsm.GPNStarForMaskedLM._tied_weights_keys = {}          # 4.x list -> 5.x dict (empty = no auto-tie)
    torch.manual_seed(0)
    model = gsm.GPNStarForMaskedLM(cfg).eval().float()
    model.load_state_dict(load_file(os.path.join(path, "model.safetensors")), strict=False)
    model.cls.predictions.decoder.bias = model.cls.predictions.bias   # tie the one missing bias
    return model.to(torch.device(device))


def make_vep():
    gm = GenomeMSA(ZARR, n_species=18, in_memory=False)
    vep = VEPInference.__new__(VEPInference)      # skip __init__ (it builds a crashing AutoModel)
    vep.genome_msa_list = [gm]
    vep.window_size = WINDOW
    vep.disable_aux_features = False
    vep.reverse_complementer = ReverseComplementer()
    vep.tokenizer = Tokenizer()
    return vep


def _llr(model, iid, sid, tsp, pos, ref, alt, device):
    logits = model(
        input_ids=torch.as_tensor(iid).to(device),
        source_ids=torch.as_tensor(sid).to(device),
        target_species=torch.as_tensor(tsp).to(device),
    ).logits
    logits = logits[torch.arange(len(pos)), torch.as_tensor(pos).to(device), 0]   # (B, vocab)
    ref, alt = torch.as_tensor(ref).to(device), torch.as_tensor(alt).to(device)
    return (logits[torch.arange(len(alt)), alt] - logits[torch.arange(len(ref)), ref])


def score(model, res, device, batch=16):
    """Batched, static-shape LLR = (fwd + rev)/2 over all variants. Returns a numpy vector."""
    n = res["input_ids_fwd"].shape[0]
    out = np.zeros(n, dtype=np.float64)
    for s in range(0, n, batch):
        e = min(s + batch, n)
        idx = slice(s, e)
        pad = batch - (e - s)

        def grab(key, pad_to_batch=True):
            a = res[key][idx]
            if pad and pad_to_batch:                              # pad last batch -> static shape
                a = np.concatenate([a, np.repeat(a[-1:], pad, axis=0)], axis=0)
            return a

        with torch.no_grad():
            lf = _llr(model, grab("input_ids_fwd"), grab("source_ids_fwd"), grab("target_species"),
                      grab("pos_fwd"), grab("ref_fwd"), grab("alt_fwd"), device)
            lr = _llr(model, grab("input_ids_rev"), grab("source_ids_rev"), grab("target_species"),
                      grab("pos_rev"), grab("ref_rev"), grab("alt_rev"), device)
        if device == "neuron":
            import torch_neuronx; torch_neuronx.synchronize()
        llr = ((lf + lr) / 2).float().cpu().numpy()
        out[s:e] = llr[: e - s]
    return out


def pentanuc_mut(res, V):
    """5-mer ref context + alt allele per variant, e.g. 'AAAAA_C' — the calibration key (fwd strand)."""
    pos = WINDOW // 2
    ids = res["input_ids_fwd"][:, pos - 2: pos + 3, 0].copy()     # (N,5); center is masked
    ids[:, 2] = res["ref_fwd"]                                    # restore center = ref token
    keys = []
    for row, alt in zip(ids, V["alt"]):
        five = "".join(ID2CHAR.get(int(t), "N") for t in row)
        keys.append(f"{five}_{alt}")
    return np.array(keys)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=256, help="subset size (stratified by label)")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    # --- sample a label-stratified subset of real variants + the published reference scores ---
    tv = pd.read_parquet(hf_hub_download("songlab/1001gp", "test.parquet", repo_type="dataset"))
    ref = pd.read_parquet(hf_hub_download("songlab/1001gp", "predictions/GPN-Star.parquet",
                                          repo_type="dataset"))["score"]
    tv = tv.assign(_ref=ref.values, _i=np.arange(len(tv)))
    half = args.n // 2
    rng = np.random.default_rng(args.seed)
    picks = pd.concat([
        tv[tv.label].sample(half, random_state=args.seed),
        tv[~tv.label].sample(args.n - half, random_state=args.seed),
    ]).sample(frac=1, random_state=args.seed).reset_index(drop=True)
    print(f"[vep] {len(picks)} variants ({int(picks.label.sum())} label=True / {int((~picks.label).sum())} False)")

    V = {"chrom": picks["chrom"].astype(str).values, "pos": picks["pos"].values,
         "ref": picks["ref"].values, "alt": picks["alt"].values}
    reference = picks["_ref"].values.astype(np.float64)
    label = picks["label"].values.astype(int)

    # --- build real inputs once (CPU/numpy), + the calibration ---
    vep = make_vep()
    res = vep.tokenize_function(V)
    keys = pentanuc_mut(res, V)
    ctab = pd.read_parquet(hf_hub_download("songlab/gpn-star-tair10-b18-25m",
                                           "calibration_table/llr.parquet")).set_index("pentanuc_mut")["llr_neutral_mean"]
    neutral = np.array([ctab.get(k, np.nan) for k in keys])
    print(f"[vep] calibration context found for {np.isfinite(neutral).sum()}/{len(keys)} variants")

    # --- score on CPU and on Trainium ---
    scores = {}
    for dev in ["cpu", "neuron"]:
        print(f"[vep] scoring on {dev} (first Neuron call compiles ~min)...")
        model = load_masked_lm(dev)
        scores[dev] = score(model, res, dev, batch=args.batch)
        del model

    raw_cpu, raw_neu = scores["cpu"], scores["neuron"]
    cal_cpu = raw_cpu - neutral        # authors' pentanucleotide-context calibration
    cal_neu = raw_neu - neutral

    # --- (a) PORT PARITY: Trainium vs CPU on the actual predictions ---
    def cos(a, b): return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    print("\n=== (a) PORT PARITY  (Trainium vs CPU, real variants) ===")
    print(f"  raw LLR       : cosine={cos(raw_cpu, raw_neu):.6f}  max-abs={np.abs(raw_cpu-raw_neu).max():.3e}")
    m = np.isfinite(cal_cpu)
    print(f"  calibrated    : cosine={cos(cal_cpu[m], cal_neu[m]):.6f}  max-abs={np.abs(cal_cpu[m]-cal_neu[m]).max():.3e}")

    # --- (b) FIDELITY vs published GPN-Star.parquet ---
    print("\n=== (b) FIDELITY  (our CPU vs published GPN-Star reference) ===")
    print(f"  raw LLR    : pearson={pearsonr(raw_cpu, reference)[0]:.4f}  spearman={spearmanr(raw_cpu, reference).correlation:.4f}")
    print(f"  calibrated : pearson={pearsonr(cal_cpu[m], reference[m])[0]:.4f}  spearman={spearmanr(cal_cpu[m], reference[m]).correlation:.4f}"
          f"  max-abs={np.abs(cal_cpu[m]-reference[m]).max():.4e}")

    # --- (c) SCIENCE: AUROC(score, label) ---
    print("\n=== (c) SCIENCE  (AUROC vs label — higher score = putatively neutral/common) ===")
    for name, sc in [("published ref", reference), ("our raw LLR", raw_cpu),
                     ("our calibrated", cal_cpu), ("our Trainium raw", raw_neu)]:
        s2, l2 = (sc[m], label[m]) if "calibrated" in name else (sc, label)
        print(f"  {name:18s} AUROC={roc_auc_score(l2, s2):.4f}")

    # --- save artifacts ---
    out = picks[["chrom", "pos", "ref", "alt", "AF", "label"]].copy()
    out["reference"] = reference
    out["our_llr_cpu"], out["our_llr_neuron"] = raw_cpu, raw_neu
    out["our_calibrated_cpu"], out["our_calibrated_neuron"] = cal_cpu, cal_neu
    out.to_parquet(os.path.join(HERE, "results_scores.parquet"))
    print(f"\n[vep] wrote {os.path.join(HERE, 'results_scores.parquet')} ({len(out)} variants)")


if __name__ == "__main__":
    main()
