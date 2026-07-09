# Running {MODEL} on AWS Trainium — Step-by-Step Guide

Copy-paste guide to run **{MODEL}** ({one-line what it does}) on **AWS Trainium** in
native PyTorch. **You do not need to know anything about Trainium or Neuron.** Every
command is copy-paste.

> **Status:** {✅ working / 🟡 partial} — {metric, e.g. cosine 1.0000 vs CPU reference}.

---

## 1. Instance
{Which trn2 slice. Single NeuronCore for <~1B; trn2.48xlarge + sharding for large models.}

## 2. Environment
```bash
conda activate torch-neuron            # the Beta-3 native-PyTorch stack (see references/environment.md)
pip install -r requirements.txt
```

## 3. Run
```bash
python src/{runner}.py {args}
```

## What the script does
1. {load model + tokenizer/processor}
2. {build fixed-shape inputs — Neuron needs static shapes}
3. {Beta-3 run path: eager `device="neuron"` (default) OR `torch.compile(backend="neuron")`}
4. {compare against the Phase-A oracle: cosine / max-abs / top-1}

## Why this just works (what the wrapper handles for you)
{Enumerate each Neuron-specific fix applied automatically — eager attention, FP8 off,
op rewrites, precision, optlevel flag, skipped heads. One numbered item each.}

## Troubleshooting
| Symptom | Fix |
|---|---|
| `neuron-ls` shows nothing | Not on a Trainium instance / wrong AMI. |
| `Got a cached failed neff` | `rm -rf /var/tmp/neuron-compile-cache` and rerun. |
| Output is all zeros / garbage | Precision — try fp32 (`model.float()`). |
| First run is slow | One-time compile; it caches. |
| {model-specific compile error} | {fix} |

## Ways to optimize / extend
### Easy
1. Keep the compile cache; fix one sequence length to avoid recompiles.
### Medium
2. Mixed precision (bf16 where safe); batch inputs.
### Advanced
3. Multi-core / tensor parallel for larger variants.

## Notes & limitations
{What is and isn't validated. Decode/long-context/training caveats.}

## Credits & license
{Upstream HF port + original model, with licenses.}
