---
name: neuron-native-autoport
description: |
  Port a CUDA/GPU PyTorch model to a native torch_neuronx inference port on AWS Trainium,
  in the native-PyTorch ("native-pytorch") style. Use when the user says "port to trainium",
  "native torch neuron port", "autoport native", or invokes /neuron-native-autoport.
  Two phases: Phase A captures a frozen CPU/CUDA reference oracle (inference + one backprop);
  Phase B ports to torch_neuronx, compiles, and validates the Trainium run against that oracle
  (cosine >= 0.99). Inference first; training is a later phase.
argument-hint: "targetRepoURLorPath, ModelName (optional), huggingFaceModelID (optional)"
---

# Native-Torch Trainium Auto-Port

## READ FIRST
- **`references/environment.md`** — the box is the AWS native-PyTorch **Beta 3** stack (`torch-neuron`
  conda env); it documents the three device APIs, required env vars (`PJRT_DEVICE`,
  `NEURON_RT_VISIBLE_CORES`), the `neuronxcc` import name, and the core-contention gotcha. Load it before doing anything.
- **`references/house_rules.md`** — operating principles (verify-on-CPU-first, fp32-by-default, static shapes).
- **`references/knowledge_base/`** — `loading_and_deps.md` (getting the model to import/load — read when a
  `trust_remote_code` model or version drift blocks you *before* the device), `error_codes.md`,
  `op_rewrites.md`, `precision_and_triage.md`, `validation_metrics.md`. Consult when a compile fails, and
  `loading_and_deps.md` when import/load fails.

Do not stop until Phase B validation passes (>= 0.99 cosine vs the Phase-A oracle) or you have a
documented blocker.

---

## Phase A — Baseline & Reference Capture (MANDATORY, CPU-default, no Neuron)

Never install target deps into `torch-neuron`. Produce a frozen oracle first.

1. **A1 Get the target.** Clone the repo into `../port-targets/<repo>/` (gitignored sibling) to READ code.
   **Fetch weights via `huggingface_hub` (`snapshot_download`), NOT git clone** — without `git-lfs`, cloned
   `*.safetensors`/`*.bin` are useless pointer files. See `knowledge_base/loading_and_deps.md`.
2. **A2 Inspect** — entry points, model class, expected inputs/shapes, deps, CUDA red flags
   (`flash-attn`, TransformerEngine/FP8, `vortex`, `faesm`, custom CUDA). See `precision_and_triage.md`.
3. **A3 Reference env — only if the target needs deps beyond the Beta-3 stack.** If the model runs on what
   `torch-neuron` already has (torch 2.11 + transformers 5.13, etc.), **capture the CPU oracle directly in
   `torch-neuron`** (record `env/versions.txt`) — no separate env needed. Only when the repo has extra or
   *conflicting* deps, run `scripts/setup_reference_env.sh <env> ../port-targets/<repo>` (pins torch 2.11,
   layers the repo's requirements); if irreconcilable, fall back to a repo-native env and record a
   `torch_version_caveat`. ⚠️ `trust_remote_code` code written against an older transformers may fail to
   *import* on 5.13 — that's a used-path patch (vendor + shim); see `loading_and_deps.md`.
4. **A4 Capture adapter** — copy `assets/capture_adapter_template.py` → `baselines/<model>/capture.py`
   (fill `load_model`/`build_inputs`/optional `loss_fn`), record `env/versions.txt`, then from
   `baselines/<model>/` run the one-command driver — it captures AND writes the manifest in one shot:
   `python <skill>/scripts/capture_reference.py --adapter capture.py` (→ `artifacts/*.pt` + `MANIFEST.json`).
   (The device-parametric *parity* notebooks — the shipped
   deliverable — live in the PRODUCT at `<model>/native-pytorch/notebooks/`, authored in R8, NOT here;
   `baselines/<model>/` holds only `capture.py` + `MANIFEST.json` + `env/`.)
5. **A5 Freeze the oracle** — `scripts/capture_reference.py` saves `artifacts/*.pt` (gitignored) and
   `scripts/build_manifest.py` writes the committed **`MANIFEST.json`** (per-tensor shape/dtype/stats/
   slice/sha256, seed, env lock, loss def). This is the ground truth.

**GATE:** do not start Phase B until `baselines/<model>/MANIFEST.json` exists and the model ran cleanly on CPU.

---

## Mode: `dev` | `integration` (default `dev`)
Both modes do Phase A + Phase B and ship the **two parity notebooks**; they differ in scope:
- **dev** — own/modify/train the architecture: vendor the **whole** editable definition; the **full model**
  runs (all outputs + all params validated); patch any unsupported op; ships **three** parity notebooks
  (inference, backprop, **training** — see "Training parity" below). (References: CLIP, evo2,
  Nucleotide-Transformer, GPN-Star.)
- **integration** — use part/whole as a building block, no architecture edits: port only the **used
  sub-graph** (e.g. encoder→embeddings), **drop unused heads** (their unsupported ops don't matter), and
  ship a **feature interface** (`embed(x)` + `freeze`/finetune). Backprop check adds a **gradient-flow-into-
  a-downstream-stub** test. (Reference: ESM-2.)
  <br>Integration has **three escalating levels of intervention** — use the least that works:
  1. **Sealed** — installed definition loads & runs as-is (record its version). *ESM-2.*
  2. **Loader shim** — architecture UNTOUCHED, but `load()` sidesteps a construction/loading problem
     (transformers meta-init crash → construct on real device + `load_state_dict`; or vendor + minimal
     compat shims for 4.x→5.x import drift). Document in `load()`/README (or `VENDORED.md` if you vendored).
     *GPN-Star (meta-init), Nucleotide-Transformer (import shims). See `loading_and_deps.md`.*
  3. **Vendor + op patch** — a used-path op won't lower on Neuron; edit the math (inline, `NEURON PATCH`).

## Phase B — Neuron Port (R1–R8)

- **R1 Definition triage** — *(dev)* **vendor the repo's OWN model definition** into `src/<model>_neuron/`; *(integration)* select the used sub-graph, drop unused heads, wrap the installed definition. Make
  *it* run on Trainium (the product is the definition, composable — not a substitute). Substitute a clean
  equivalent only if the repo's code truly can't be ported, and document it. (`precision_and_triage.md`)
- **R2 Neutralize GPU paths** — patch inline in the vendored definition: `attn_implementation="eager"`,
  FP8 off, `use_cache=False`, static/padded shapes.
- **R3 Device-friendly wrapper** — `assets/wrapper_template.py`: tensor in, tuple-of-tensors out (avoids
  dict/None branches that trip graph capture).
- **R4 Load the Phase-A oracle** — `baselines/<model>/MANIFEST.json` + its exact seeded inputs. Do NOT
  recompute an ad-hoc reference.
- **R5 Run + debug loop** — pick a Beta-3 device API (`environment.md`): **eager `device="neuron"`**
  (simplest, start here — M1 CLIP passed this way) or **`torch.compile(model, backend="neuron")`**.
  ⚠️ `torch_neuronx.trace` does NOT exist on Beta 3 (public-SDK only). Pin `NEURON_RT_VISIBLE_CORES`.
  On failure: map the error via
  `knowledge_base/error_codes.md`, localize with `scripts/analyze_hlo.py` (+`XLA_IR_DEBUG=1`), rewrite
  per `op_rewrites.md`, **CPU-validate the patch vs original**, clear the compile cache, recompile.
- **R6 Precision** — default fp32; mixed precision as an optimization (`precision_and_triage.md`).
- **R7 Validate** — `scripts/compare.py`: `summarize()` the Neuron run, `compare()` vs the manifest.
  **Gate: outputs cosine >= 0.99** (target ~1.0), report max-abs and top-1 for LMs, per-tensor.
  For **gradient** parity use a MAGNITUDE-AWARE gate (near-zero grads like attention `k_proj.bias` are
  analytically 0 — cosine on their fp-noise is meaningless). See `knowledge_base/validation_metrics.md`.
- **R8 Package (definition-as-product)** — emit `<model>/native-pytorch/`: `src/<model>_neuron/`
  (vendored composable definition — `load(device)`, submodule accessors, freeze/unfreeze, `VENDORED.md`
  listing patches), `src/<model>_reference.py` (shared wrapper+inputs+loss), and the parity notebooks
  `notebooks/{01_inference,02_backprop}_parity.ipynb` (CPU/CUDA/Trainium, assert match) — plus, in **dev
  mode**, `03_training_parity.ipynb` (see "Training parity") — plus README (from template) +
  results/RESULTS.md + requirements.txt. So: **dev ships 3 notebooks; integration ships 2** (03 optional —
  ESM-2, the integration reference, ships one to demonstrate composite training). See CLIP as the reference example.
- **R9 Source + self cross-check (do this before declaring done).** Every deliverable doc/comment must match
  (a) the actual source repo in `../port-targets/<repo>/` and (b) the code + baked-in notebook outputs you
  just produced. This is where a mode conversion (integration→dev, or vice-versa) leaks: updating the header
  but not the body leaves the doc lying. Grep the WHOLE port tree (`README.md`, `results/RESULTS.md`, `src/**`
  incl. `VENDORED.md` + module docstrings, `example/**`) and reconcile:
  - **License** — read it from the source (`../port-targets/<repo>/LICENSE`, `setup.py`, or the HF
    `README.md` frontmatter `license:`), never guess. Real examples that bit: **gpn = MIT** (not Apache),
    **Nucleotide-Transformer = CC-BY-NC-SA-4.0** (non-commercial!), CLIP = MIT, Evo2 = Apache-2.0. Put the
    correct license in `VENDORED.md` + README credits.
  - **Mode words** — a **dev** port ships the full editable definition, so it must NOT describe itself with
    integration language ("used sub-graph", "head never instantiated / dropped", "sealed", `embed()`/
    `embed_pooled()`, a `Composite` wrapper, "feature-extractor") — and an **integration** port must not
    claim to own/train the full definition. Both `README` and `RESULTS` bodies, not just the title.
  - **Counts, shapes, API, notebook list** — grad-tensor count, output tensor shapes, the 2-vs-3 notebook
    list, and EVERY copy-pasteable code example (the `load(...)` signature, method names like `submodules`
    vs a nonexistent `embed`, kwargs like `finetune_backbone=`) must match the shipped module and the
    notebook outputs. A broken README example (a method/kwarg that doesn't exist) is a correctness bug.

## Training parity — a DEV-MODE deliverable (third parity notebook)
Because **dev mode owns and trains the architecture**, a dev-mode port also ships a
**`03_training_parity` notebook**: a real multi-step loop (forward → loss → backward →
`optimizer.step()`, Adam) CPU vs Trainium, asserting the **loss trajectory + final weights match** and the
loss decreases. This proves optimizer.step + optimizer state + multi-step convergence lower on-device.
- **Loss / task per model type:** ESM-2's reference uses a classification **composite (backbone + head +
  CrossEntropy on seeded labels)** — that fits models with a task head. For a **bare generative/MLM model**
  (no head/labels), train the **full model on its OWN LM loss** over a fixed synthetic batch: **next-token
  CE** for a causal LM (shift logits/labels), **masked-token CE** for an MLM. (evo2 reference: 5-step Adam,
  next-token CE, loss ~40→6.)
- **Gate must be RELATIVE, not absolute.** Loss scales vary wildly (evo2's first-step loss ~40, a ~0.13
  step-2 CPU/device transient). Use a **relative** loss-trajectory tolerance (e.g. `|Δ|/|loss| ≤ 1e-2`) —
  an absolute `< 1e-2` gate false-FAILs large-loss models. Weights: magnitude-aware, same as 02.
- **No frozen oracle:** unlike 01/02, `03` has no Phase-A artifact (capture freezes only inference + one
  backprop) — it **self-compares CPU-vs-device at runtime**. That IS the check; no golden trajectory needed.
- **Make the step DETERMINISTIC (dropout).** Run the loop in `model.eval()` (or otherwise disable ALL
  dropout) so CPU and device draw no divergent RNG — otherwise the trajectories desync at step 0 and the
  relative gate false-FAILs. Watch for **inline** dropout: `F.scaled_dot_product_attention(dropout_p=... if
  self.training else 0.0)` is NOT an `nn.Dropout` module, so zeroing `nn.Dropout` layers does not disable
  it — only `eval()` (gating on `self.training`) does. `eval()` still fully exercises optimizer.step()/Adam
  state + multi-step convergence; parity needs a deterministic step, not stochastic regularization.
- **A PRETRAINED model on a trivial input can false-FAIL the loss gate — tune the LR down.** When the
  loop trains loaded weights on an easy/in-distribution batch (e.g. evo2 next-token CE on periodic DNA),
  the model already fits it, so an aggressive LR overshoots into a steep, near-saturated basin where the
  loss is a few×0.1 and CE is hypersensitive to fp-noise in the logits — the CPU/device *weights* still
  agree to ~1e-4 but the *loss trajectory* diverges by percent (relative gate flakes). Fix: a **gentle LR**
  (evo2 needed `1e-5`, not `1e-4`) for a monotonic descent that keeps the loss out of saturation, so the
  trajectory comparison is meaningful. Watch for a loss that *bounces up* between steps — that's the
  overshoot signal. (Contrast: a from-scratch/large-loss objective is well-conditioned at any sane LR.)
- **Don't feed a tensor AND its transpose-view into two `F.cross_entropy` calls — Neuron's CE op mutates
  its input in place.** The textbook symmetric-InfoNCE `0.5*(CE(logits, y) + CE(logits.t(), y))` (CLIP-style
  contrastive) aliases storage between the two calls; on device the second call bumps the shared logits
  tensor's version and the double backward dies with *"one of the variables needed for gradient computation
  has been modified by an inplace operation"* (`MmBackward0` output at version 1). Write it with
  `log_softmax` over rows AND columns instead — `-0.5*(log_softmax(L,dim=1)[i,i] + log_softmax(L,dim=0)[i,i]).mean()`
  — mathematically identical, no aliasing, no custom CE op. (Same class of bug can bite any loss that reuses
  one logits tensor through two in-place-mutating device ops; a `.contiguous()` copy also breaks the alias.)
- Notes: Neuron auto-casts int64→int32 (benign); keep static shapes so the train step compiles once.
Reference `03`s (validated on-device, `*/native-pytorch/notebooks/03_training_parity.ipynb`): four dev-mode
— `evo2` (next-token CE), `gpn-star` / `nucleotide-transformer` (masked-token CE / MLM), `clip` (symmetric
InfoNCE contrastive) — plus `esm2`, the **integration-mode** classification-composite reference (backbone +
new head + CE on seeded labels). Integration mode doesn't *require* `03` (its `02` already covers
gradient-flow into a downstream head), but `esm2` ships one to demonstrate composite training on-device.

## Parameters
- `targetRepoURLorPath` (required) — git URL or local path to the CUDA model repo.
- `mode` (optional, default `dev`) — `dev` (full editable definition) or `integration` (component/feature
  extractor). For `integration`, also take `usedSubgraph`/`outputLayer` (what to extract) and
  `frozen|finetune`.
- `ModelName`, `huggingFaceModelID` (optional) — inferred from the repo if omitted; confirm with the user.
  For a **multi-size model family**, pick the **smallest** checkpoint for the parity port (fewer
  params/species → faster compile + iteration); the architecture is identical, so the port recipe carries to
  the larger variants unchanged.
