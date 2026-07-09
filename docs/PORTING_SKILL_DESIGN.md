# Native-Torch Trainium Auto-Port — Architecture & Roadmap (draft v1)

> Goal: an agentic workflow, packaged as a deployable **agent skill**, that ingests
> an existing **CUDA-based model repo** and produces a **native PyTorch (`torch_neuronx`)
> port** on AWS Trainium — the `<model>/native-pytorch/` deliverable, with inference, backprop,
> and (dev mode) multi-step on-device training all validated vs a CPU oracle.
>
> *Note: a private collection of hand-written native-PyTorch Trainium ports was studied during
> development to derive the recipe below. It is a reference only — NOT required to run the skill,
> and not referenced by any shipped skill file.*

---

## 0. Where this fits (and what it is *not*)

| | Existing `neuron-framework-autoport` | **This skill (new)** |
|---|---|---|
| Target stack | NeuronX Distributed Inference (NxDI) framework | **Native `torch_neuronx` / `torch_xla`** |
| Model source | HF `*ForCausalLM` only | Any CUDA/PyTorch repo (LLM, DiT, encoder, bio) |
| Output | NxDI reimplementation | **`native-pytorch/` folder** (src + README + RESULTS) |
| Validation | ≥95% greedy token match | **cosine ≥0.99 vs CPU oracle** (target 1.0000) |
| Scope | inference | inference now, training later |

`neuron-agentic-development/` is used **only** as the packaging blueprint (SKILL.md
frontmatter, `references/knowledge_base/`, `scripts/`, `assets/`, agent `.md` +
`.agent-spec.json`, pip deployer). We do **not** build on NxDI.

## 1. Environment (confirmed on this box)

- Host is a live **`trn2.3xlarge`** — `neuron-ls` shows 1 device / 4 cores / 96 GB.
  **The skill can be developed *and* validated here** (not dry-run only).
- **Toolchain: the `torch-neuron` conda env** (the AWS native-PyTorch Beta-3 stack) — activate with:
  ```bash
  conda activate torch-neuron
  ```
  Confirmed contents (Python 3.12.13):

  | Package | Version | Notes |
  |---|---|---|
  | `torch` | 2.11.0+cpu | CPU wheel — device execution goes through `torch_neuronx` / the Neuron runtime |
  | `torch-neuronx` | 2.11.3.0.1278 | ✅ imports — **`torch_neuronx.trace()` path is available here** |
  | `neuronx-cc` | 2.25.1280.0 | compiler present |
  | `transformers` | 5.13.0 | |
  | `torch-xla` | **2.11.0** | ✅ installed 2026-07-08 — **`xm.xla_device()` + `mark_step()` path now works on-device** |
  | `libneuronxla` | 3.0.3854.0 | installed alongside torch-xla (pulls boto3/botocore) |

- **Both compile paths (§2 R5) now run on-device in this env**, verified 2026-07-08:
  - **`torch_neuronx.trace` → TorchScript** (clip / esm2 / bacformer): works. M1 target class.
  - **`xm.xla_device()` + `mark_step()`** native torch_xla lazy path (**evo2, alphagenome**): unblocked —
    a 128×128 matmul compiled + executed on the Trainium device (correct result).
- **⚠️ Required env vars for the torch_xla / device path:**
  ```bash
  export PJRT_DEVICE=NEURON               # else torch_xla silently defaults to CPU
  export NEURON_RT_VISIBLE_CORES=0        # pin ONE logical core (see below); use a free core index
  ```
  - Without `PJRT_DEVICE=NEURON`, torch_xla runs on CPU-backed XLA (wrong hardware, still "works").
  - **Core contention:** a stray orphaned `multiprocessing-fork` worker can hold part of the device, so
    `neuron-ls` may show only ~1 logical NeuronCore free
    (`logical-neuroncore-config: 2` ⇒ cores allocate in pairs). PJRT defaults to grabbing **all 4** cores
    and fails (`Requested:4 Available:3`). Pinning `NEURON_RT_VISIBLE_CORES` to a single core sidesteps it —
    both core 0 and core 1 verified working — and single-core is all our inference ports need anyway. If we
    later want multi-core (TP for 40B), that orphan must be reaped first (ask the user; don't kill blindly).
  - EFA/NCCL/OFI init warnings on startup are **benign** for single-core inference (no multi-node collectives).
- The base `python3` outside conda is CPU `torch 2.11.0+cpu` with no Neuron packages — always run inside `torch-neuron`.
- **Beta-3 parity — RESOLVED.** All the reference native-PyTorch ports were done on the **native-PyTorch Neuron beta (Beta 3)**,
  and this skill was developed on that same stack: the env was built from the AWS native-PyTorch Beta-3
  setup recipe, producing exactly these versions. No version gap to evo2/alphagenome. **Beta-3 access is
  gated by the AWS Neuron team** (see `references/environment.md`, which has the full context + gotchas and
  is the canonical reminder).
- **Three device execution APIs** are available (skill picks per model, §2 R5): (1) native eager
  `torch.device("neuron")` — the Beta-3 headline, the reference ports' stated default; (2) `torch_neuronx.trace()` →
  TorchScript; (3) `xm.xla_device()` + `mark_step()` lazy torch_xla. Compiler imports as **`neuronxcc`**.

---

## 2. The porting recipe

Two phases. **Phase A (Baseline & Reference Capture) is MANDATORY and runs first, on CPU/CUDA — no
Neuron.** It produces the frozen reference oracle that Phase B validates against. **Phase B (Neuron Port)**
is the R1–R8 pipeline extracted from esm2 / clip / bacformer / evo2 / alphagenome.

### Two modes (skill arg `mode: dev | integration`)

Both modes share everything below (Phase A oracle, torch-2.11 env, device API, KB, compare/manifest, and
the **01-inference + 02-backprop parity notebooks**). **Dev mode additionally ships a third,
`03_training_parity`** (a real multi-step `optimizer.step()` loop, CPU vs Trainium — because dev mode owns
and *trains* the architecture). They differ only in **how much of the model is ported/validated and what
interface ships**:

| | **dev** (full definition) | **integration** (component) |
|---|---|---|
| intent | own & modify the architecture | use part/whole as a building block; no architecture edits |
| ported/validated surface | the **whole** model; every path must run | only the **used sub-graph** (e.g. encoder→embeddings) |
| unsupported op in an **unused** head | must patch (R5) | **drop it** — not ported |
| definition handling | **vendor** an editable copy into `src/<model>_neuron/` | use the **installed/sealed** definition (record version); vendor only if a used-path patch is needed |
| exposed interface | full composable definition + all submodules | a **feature interface** (`embed(x)` at a chosen layer) + `freeze`/finetune |
| inference parity checks | full model outputs | the extraction output (e.g. embeddings) |
| backprop parity checks | all params | grads on the **used+trainable** path **+ gradient flow into a downstream stub** (backbone+toy-head) |
| **training parity** (`03`) | **required** — multi-step Adam loop, loss trajectory + final weights match CPU | not required (mode doesn't train the model itself) |
| parity notebooks shipped | **3** (inference, backprop, training) | **2** (inference, backprop) |
| reference example | **CLIP, evo2, NT, GPN-Star** (dev) | **ESM-2** (integration) |

Integration mode is usually cheaper and can unblock a model whose only Neuron-hostile op lives in a head
you don't use. Declare frozen-vs-finetuned; validate grads for what you'll actually train.

---

### Phase A — Baseline & Reference Capture (mandatory, CPU-default, no Neuron)

Goal: get the *original* model running faithfully and freeze its outputs + gradients as the oracle. This
protects the Beta-3 `torch-neuron` env (we never install target deps into it) and gives Phase B ground truth.

- **A1 — Clone target repo(s).** Into the designated targets dir **`../port-targets/<repo>/`** (sibling to
  this repo, **gitignored**; holds many repos). Cloning the target repo here is a required first step.
- **A2 — Inspect.** Agent reads the repo: entry points, model class, expected inputs/shapes, dependency
  list, and CUDA-only red flags (`flash-attn`, TransformerEngine/FP8, `vortex`, `faesm`, custom CUDA).
  Feeds R1 triage.
- **A3 — Build a per-repo reference env.** **Policy: pin `torch==2.11.0` (CPU), layer the repo's other
  requirements around that pin** — so the oracle is apples-to-apples with Trainium (only device+compiler
  differ). If a repo is irreconcilable with torch 2.11 / transformers 5.13, fall back to a repo-native env
  and **record the torch-version caveat in the manifest**. Never touch the `torch-neuron` env.
- **A4 — Author capture notebooks** under **`baselines/<model>/`** (inside this repo). Two notebooks,
  **device-parametric (`cpu` default, `cuda` auto-detected & gracefully skipped on the trn box)**, each
  headless-runnable (`nbconvert --execute` / papermill) with the real logic in an importable
  `capture.py` (so it's diffable/testable, not notebook-only):
  1. **inference** — build proper inputs, run forward, save outputs.
  2. **one backprop** — a single backward pass; if the model defines no loss, synthesize a simple scalar
     loss (e.g. `out.sum()`) and **record which loss** was used. Save gradients.
- **A5 — Freeze the oracle.** Run on CPU (default). Save raw outputs + gradients as **artifacts**
  (`baselines/<model>/artifacts/*.pt`, **gitignored**) and commit a compact **`MANIFEST.json`**:
  per-tensor shape/dtype/stats/flat-slice/hash, RNG seed, the env lockfile, and the loss definition. This
  manifest — not an ad-hoc re-run — is what Phase B R4/R7 compare against.

---

### Phase B — Neuron Port (R1–R8)

### R1 — Definition triage (port the REPO'S OWN definition)
**The product is the ingested repo's own model definition, made Trainium-runnable + composable — NOT a
substituted equivalent.** (Decision 2026-07-08, for AI+Bio foundational-model development: bio models
usually have no clean HF equivalent, and you need editable, faithful module source to build/train from.)
- **Vendor the repo's definition** (its `nn.Module` source) into `src/<model>_neuron/` and make *it* run.
  Substitute a clean equivalent (e.g. HF) ONLY when the repo's code genuinely can't be ported, and
  document the substitution.
- **Red-flag ops/deps** don't force a substitution — you **patch around them inline** (R5) in the vendored
  definition: `flash-attn`, `transformer_engine`/FP8, `vortex`, `faesm`, custom CUDA kernels,
  `torch.fft`/complex, `sort`/top-k, and backward-only failures (e.g. `nn.MultiheadAttention`+mask, see KB).
- Size the model → recommend instance (single core for <~1B; 48xlarge + sharding for 40B+).

### R2 — Neutralize GPU-only paths (config-level, usually free)
- `attn_implementation="eager"` (no flash-attn).
- Disable FP8 / TransformerEngine (`use_fp8_input_projections=False` → pure-PyTorch fallback).
- Install model deps with `--no-deps` so they don't clobber the neuron `torch`.
- `use_cache=False` and **static shapes** — pad inputs to fixed length (Neuron needs static graphs);
  respect model constraints (AlphaGenome seq_len multiple of 128).

### R3 — Trace-friendly wrapper
- Wrap the HF model in a small `nn.Module`: **tensor inputs in, tuple-of-tensors out**
  (no dicts/kwargs/`return_dict`). See `ClipWrapper`, `BacformerEncoderWrapper`.

### R4 — Load the Phase-A oracle (ground truth)
- Do **not** recompute a CPU reference ad hoc — load the frozen **`baselines/<model>/MANIFEST.json`**
  (+ artifacts) from Phase A. Reuse its exact deterministic inputs and RNG seed so the Neuron run is
  directly comparable. (If Phase A recorded a torch-version caveat, carry it into the tolerance decision.)

### R5 — Compile + unsupported-op debug loop (the real work)
Two compile paths (skill picks per model):
- **`torch_neuronx.trace(wrapper, example)` → TorchScript** (esm2, clip, bacformer). Simplest; save `.pt`.
- **`xm.xla_device()` + `mark_step()`** native torch_xla lazy path (evo2, alphagenome). For
  models you drive programmatically / can't cleanly trace.

On compile failure, map the error code → fix (this is the seed knowledge base):

| Symptom / code | Root cause | Fix pattern |
|---|---|---|
| `NCC_EVRF004` complex dtype | `torch.fft`/complex ops | Rewrite to mathematically-equivalent real op (evo2 **FFT-conv → depthwise causal `conv1d`**, filter flipped) |
| `sort not supported` (`NCC_EVRF029`) | HLO `sort`/top-k | Skip that head / run it on CPU (alphagenome `splice_junctions`), or a TopK/NKI kernel |
| `NCC_ITIN902` / `AffineIV` at long seq | compiler polyhedral bug | `NEURON_CC_FLAGS=--optlevel=1` (set **before** compiler import) |
| `Got a cached failed neff` | interrupted compile | `rm -rf /var/tmp/neuron-compile-cache` |
- Localize unsupported ops to **PyTorch source lines** with `analyze_hlo.py` (HLO opcode
  histogram + suspect-op list; needs `XLA_IR_DEBUG=1` at trace time).
- **Every op-rewrite patch is CPU-validated against the original before recompiling** (evo2:
  patched-conv1d vs original-FFT cosine 0.999999 *first*, then touch Neuron).

### R6 — Precision
- bf16 collapses in known spots (evo2 norm-collapse on ~1e16 activations; FLUX "gray" past 1MP).
  Default to **fp32** for correctness; offer mixed-precision as an optimization (isolate fp32 to
  the norm/massive-activation layers to fit bigger models).

### R7 — Numerical validation gate
- Neuron vs CPU oracle: **cosine ≥ 0.99** (target 1.0000), report `max_abs_diff`, and for LMs
  **top-1 token agreement**. Per-head for multi-output models (`common.compare`, atol/rtol).
- Fail → iterate R5/R6. This is the "do not declare success until it passes" gate.

### R8 — Package the deliverable (definition-as-product)
```
<model>/native-pytorch/
  src/<model>_neuron/   the VENDORED, Trainium-ready, COMPOSABLE model definition (the product):
                        modeling.py (repo's nn.Module source, Neuron patches inline+marked),
                        __init__.py (load(device=...), submodule accessors, freeze()/unfreeze()),
                        VENDORED.md (source + license + list of Neuron patches)
  src/<model>_reference.py   shared harness: tuple-out wrapper + deterministic inputs + loss
  src/port_<model>.py        headless CLI running the two parity checks
  notebooks/
    01_inference_parity.ipynb   CPU/CUDA/Trainium inference, asserts outputs match (cosine ≥ 0.99)
    02_backprop_parity.ipynb    CPU/CUDA/Trainium one-backprop, asserts gradients match (magnitude-aware)
    03_training_parity.ipynb    DEV ONLY: multi-step Adam loop, asserts loss trajectory + final weights match
  README.md · results/RESULTS.md · requirements.txt
```
**Parity notebooks are REQUIRED deliverables** (device-parametric: `cpu` default, `cuda` auto-skips,
`neuron` on the core; headless-runnable via `nbconvert --execute`) — **integration ships 01+02, dev ships
01+02+03** (`03_training_parity`, since dev owns/trains the architecture). The definition must be
**importable and composable** — submodule access + freeze/unfreeze — so it can be a building block in new models.
README tone is fixed: *"You do not need to know anything about Trainium."* + symptom→fix table +
easy/medium/advanced optimization ladder.

---

## 3. Skill package layout (follows neuron-agentic-development conventions)

```
neuron-native-autoport/                     # repo root (skill name TBD)
  README.md  .gitignore
  docs/PORTING_SKILL_DESIGN.md              # this doc
  skills/neuron-native-autoport/            # the deployable skill
    SKILL.md                                # frontmatter + the Phase A / Phase B (R1–R8) workflow; READ references/environment.md first
    references/
      environment.md                        # ✅ Beta-3 provenance + env gotchas (canonical reminder)
      house_rules.md                        # House rules (bf16→gray, verify-on-CPU-first, TP shards weights)
      knowledge_base/                       # error-code→fix, op-rewrite patterns, precision pitfalls, source-triage rules
    scripts/                                # reusable, model-agnostic
      setup_reference_env.sh                # Phase A3: torch-2.11-pinned per-repo env
      capture_reference.py                  # Phase A4/A5: run fwd + 1 backprop, save artifacts
      build_manifest.py                     # Phase A5: freeze MANIFEST.json (shapes/stats/slices/hashes)
      compare.py                            # R7: Neuron vs manifest (cosine / maxrel / top-1)
      analyze_hlo.py                        # R5: localize unsupported HLO ops → source lines
    assets/
      capture_notebook_template.py          # device-parametric capture notebook (jupytext percent)
      wrapper_template.py  README_template.md  RESULTS_template.md
  baselines/                                # Phase A frozen oracle, per model (committed: MANIFEST)
    <model>/ capture.py  MANIFEST.json  env/  artifacts/(gitignored)
  <model>/native-pytorch/                   # the PRODUCT (see R8): src/<model>_neuron/ (composable def),
                                            # notebooks/{01_inference,02_backprop}_parity.ipynb, README, RESULTS
  agents/
    neuron-native-autoport-agent.md + .agent-spec.json   # autonomous driver
../port-targets/                            # cloned target repos — OUTSIDE the repo, gitignored
```
A parallel **native-equivalence** capability can reuse `neuron-framework-equivalence`'s
CPU/device numerical-comparison machinery (it's framework-agnostic).

---

## 4. Roadmap

- **M0 — Scaffold + KB seed.** ✅ **Done (2026-07-08).** Skill package, `SKILL.md`, `references/`
  (environment + house_rules + KB: error_codes / op_rewrites / precision_and_triage), generic scripts
  (setup_reference_env / capture_reference / build_manifest / compare / analyze_hlo), asset templates
  (adapter, notebook, wrapper, README, RESULTS), and the driver agent (md + spec). Engine smoke-tested
  green on a toy module (forward + backprop + manifest + compare round-trip). No model run yet.
  <br>Original M0 scope: Stand up the skill package; write `SKILL.md` (Phase A / Phase B, points
  at `references/environment.md` first); author the generic Phase-A machinery (`setup_reference_env.sh`,
  `capture_reference.py`, `build_manifest.py`, notebook template) and lift `compare.py` / `analyze_hlo.py`;
  seed `knowledge_base/` from the 5 studied ports (error table, FFT→conv1d, sort→CPU, optlevel, precision)
  + `house_rules.md`. **No model run yet** (templates + scripts only).
- **M1 — Definition-as-product end-to-end.** ✅ **Done (2026-07-08) — CLIP, rebuilt under the pivot.**
  Vendored openai CLIP's **own** `model.py` into `clip/native-pytorch/src/clip_neuron/` (composable:
  `load(device)`, submodule accessors, freeze). Phase A oracle captured from that same definition
  (3 outputs + **302 grad tensors**). Deliverables include the **two required parity notebooks**
  (inference + backprop, CPU/CUDA/Trainium). **Results: inference cosine 1.000000; backprop 302/302
  gradients match.** Two findings baked into the skill:
  - Beta 3 has **no `torch_neuronx.trace`** (public-SDK only) → use eager `device="neuron"` /
    `torch.compile(backend="neuron")`.
  - `nn.MultiheadAttention` **backward with a causal mask fails to compile on Neuron** (inference is fine)
    → manual-QKV attention patch, CPU-validated (KB `op_rewrites.md`). The canonical training-only patch.
- **M1b — Integration mode + ESM-2.** ✅ **Done (2026-07-08).** Added the `dev|integration` mode split
  (see §2). Built **ESM-2 as the integration reference**: sealed HF `EsmModel` (MLM head dropped),
  `esm2_neuron.embed()` feature interface, oracle from the integration path. Two parity notebooks PASS:
  embeddings cosine 1.000000; **gradient-flow 101/101** (99 backbone + 2 new head) — grads flow through the
  sealed backbone into a new downstream head on-device. No patches. Deliverables: `esm2/native-pytorch/`.
- **M2 — Unsupported-op debug loop.** 🟡 **evo2 done (2026-07-08); alphagenome pending.**
  - **evo2 ✅ (dev mode):** vendored the 1B StripedHyena2 definition; the real work = **FFT→conv1d** inline
    patch in `engine.py` (complex ops `NCC_EVRF004`). CPU-validated vs FFT (cosine 1.000000, top-1 100%),
    then on Trainium (cosine 1.000000, top-1 100%, ~27s); backprop 265/265. **No isolated env needed** —
    the port runs under our transformers 5.13. Deliverables in `evo2/native-pytorch/`.
  - **alphagenome ⏳:** sort-skip (drop `splice_junctions` head) + `NEURON_CC_FLAGS=--optlevel=1`; torch_xla
    path. Next.
- **M3 — Held-out hardening.** 🟢 **In progress (2026-07-09).** Handed the skill to an autonomous agent
  (skill-only, no hand-holding) to port **Nucleotide-Transformer v2 50M** (integration mode). **Result: it
  succeeded fully autonomously** — both parity notebooks PASS on-device (inference cosine 1.0, grad-flow
  173/173) with no Neuron op-rewrite needed. The agent returned a 7-item skill-gaps list; **all real ones
  hardened**: new KB `loading_and_deps.md` (trust_remote_code vs transformers-version drift — the whole
  difficulty; + headless prompt + git-lfs weights), fixed `house_rules.md` (stale `torch_neuronx.trace`),
  reconciled A3 (capture in torch-neuron when no extra deps) / A4 (parity notebooks live in the product, not
  baselines), `build_manifest.py` env-lock default.
  <br>**2nd held-out: GPN-Star** (`songlab/gpn-star-tair10-b18-25m`, MSA transformer gLM, integration) — also
  **succeeded autonomously** on the hardened skill: both notebooks PASS (inference cosine 1.0, grad-flow
  236/236), one load-time shim, no op-rewrite. 2nd gap round hardened: KB `loading_and_deps.md` gained the
  **transformers-5.x meta-device init crash** (construct-on-real-device + load_state_dict); `house_rules.md`
  now sanctions **synthetic valid-shaped inputs** for parity; integration mode names the **three
  intervention levels** (sealed → loader-shim → vendor+patch); `capture_reference.py` is now a one-command
  CLI (A4 fixed); benign `sdp_kernel` warning noted. Two clean autonomous ports ⇒ the skill is
  self-sufficient for the encoder/feature-extractor common case.
  <br>*(Both were later converted to **dev mode** — the full editable definition, `EsmForMaskedLM` /
  `GPNStarForMaskedLM` — and gained the `03_training_parity` notebook in M5; the numbers above are the
  original integration-mode held-out run.)*
- **M4 — Packaging + deploy.** pip-installable deployer to Claude Code / Kiro, like
  neuron-agentic-development.
- **M5 — Training.** 🟢 **On-device training parity DONE for all 5 gold ports (2026-07-09)** — 4 dev-mode
  (train the model's own definition) + ESM-2 (integration mode, trains the backbone+head composite). Added a
  **third parity check** `03_training_parity` (after 01-inference, 02-backprop): a real multi-step loop —
  forward → loss → backward → `optimizer.step()` (Adam) — on each model's OWN objective, CPU vs Trainium,
  asserting the loss decreases AND the trajectory + final weights match (RELATIVE loss gate, magnitude-aware
  weights, `eval()` for a deterministic step). Proves optimizer.step + Adam state + multi-step convergence
  work on-device. Coverage:
  - **esm2** — classification composite (backbone + head + CE on seeded labels), loss 0.685→0.016.
  - **evo2** — next-token CE (causal DNA LM), loss 0.951→0.198 (`LR=1e-5`: the pretrained model overshoots
    a trivial input at higher LR — see the SKILL training-parity note).
  - **nucleotide-transformer** / **gpn-star** — masked-token CE (MLM), loss 4.885→0.103 / 7.603→1.853.
  - **clip** — symmetric InfoNCE contrastive, loss 0.710→0.350 (written with `log_softmax`, not double
    `cross_entropy`, to dodge Neuron's in-place CE op — see the SKILL note).

  All self-compare CPU↔Trainium at runtime (no frozen oracle). Next: multi-batch datasets, LR schedules,
  checkpointing, larger models, multi-core/distributed (FSDP per the beta guide).

The 5 gold ports are our **regression fixtures** — the skill must reproduce them.

---

## 5. Open questions for you

1. **Skill name** — `neuron-native-autoport`? something else?
2. **First target for M1** — reproduce an existing reference port (CLIP/bacformer) as the harness test,
   or go straight at a brand-new model?
3. ~~torch_xla install / Beta-3 parity~~ ✅ **Resolved** — `torch-xla 2.11.0` + `libneuronxla 3.0.3854`
   installed; both compile paths verified on-device; and the dev box *is* the Beta-3 stack (built from the
   AWS Beta-3 setup recipe), so there's no version gap to evo2/alphagenome. Remaining nit: a stray orphaned
   worker can hold a core — reap it (frees the full 4-core device for TP) or stay single-core?
4. **Deliverable coupling** — should the skill write directly into the standard `native-pytorch/`
   repo layout, or a standalone output dir?
