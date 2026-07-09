# neuron-native-autoport

An agentic workflow, packaged as a deployable **agent skill**, that ingests an existing
**CUDA-based model repo** and produces a **native PyTorch (`torch_neuronx`) port** on AWS
Trainium — the `<model>/native-pytorch/` deliverable (the model's own editable definition +
Neuron patches + CPU-oracle harness + copy-paste README + validated RESULTS), with
**inference, backprop, AND multi-step on-device training** all validated vs a CPU oracle.

> ⚠️ **Access requirement.** This skill targets the **AWS native-PyTorch Neuron beta ("Beta 3")**, not the
> public Neuron SDK. To install and run it you must first obtain Beta-3 access **from the AWS Neuron team**
> (the beta DLC / wheels are access-gated — your account must be whitelisted). The AWS native-PyTorch
> Beta-3 user guide and DLC are private-beta material under the beta terms: do not redistribute them. See
> [`skills/neuron-native-autoport/references/environment.md`](skills/neuron-native-autoport/references/environment.md)
> for the exact stack + setup.

**Status:** M1–M3 + M5 landed. **Five gold ports** validated on-device, each shipping all three parity
notebooks (`01_inference`, `02_backprop`, `03_training`). Four are **dev mode** (the model's own editable
definition — CLIP, evo2, Nucleotide-Transformer, GPN-Star); **ESM-2** is the **integration-mode** example
(a feature-extractor subgraph + a new downstream head). Integration mode doesn't *require* `03`, but ESM-2
ships one as the composite-training reference.

## Layout

```
docs/                                   design docs & decisions
  PORTING_SKILL_DESIGN.md               architecture, Phase A + Phase B (R1–R8) recipe, roadmap
skills/neuron-native-autoport/          the skill (deploys to Claude Code / Kiro)
  SKILL.md                              Phase A / Phase B workflow the agent follows
  references/
    environment.md                      Beta-3 provenance + env gotchas (READ FIRST)
    house_rules.md                      hard-won lessons (bf16→gray, verify-on-CPU-first, ...)
    knowledge_base/                     error-code→fix, op-rewrite patterns, precision pitfalls
  scripts/                              reusable tooling (reference capture / manifest / compare / analyze_hlo)
  assets/                               notebook + wrapper + README + RESULTS templates
baselines/<model>/                      Phase-A outputs: capture notebooks + committed MANIFEST.json
                                        (raw output/gradient tensors under artifacts/ are gitignored)
agents/                                 autonomous driver agent (.md + .agent-spec.json)
../port-targets/                        cloned CUDA target repos — OUTSIDE this repo, gitignored
```

## Workflow (two phases)

1. **Phase A — Baseline & Reference Capture** (mandatory, CPU-default, no Neuron): clone the target repo
   into `../port-targets/`, build a **torch-2.11-pinned** per-repo env, and capture inference + one
   backprop in device-parametric notebooks. Freeze outputs/gradients into `baselines/<model>/MANIFEST.json`
   — the oracle everything downstream is validated against.
2. **Phase B — Neuron Port** (R1–R8): port to native `torch_neuronx`, compile, and validate the Trainium
   run against the frozen Phase-A oracle (cosine ≥ 0.99).

## Reference material (private, optional — NOT required to run the skill)

The skill is fully self-contained. During development we studied two private reference repos that live
outside this directory; neither is needed to run or deploy the skill:

- A private collection of hand-written native-PyTorch Trainium ports — informed the recipe and serves as
  optional regression fixtures.
- `../neuron-agentic-development/` — AWS's deployable skill package; **packaging blueprint only**.

See [docs/PORTING_SKILL_DESIGN.md](docs/PORTING_SKILL_DESIGN.md) for the full plan.
