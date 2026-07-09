# Environment & Provenance — READ FIRST

> **Context/reminder for this skill.** The target output style is a **native-PyTorch (`torch_neuronx`)
> port** validated against a CPU reference oracle.
>
> - ⚠️ **Access requirement — read this first.** This skill targets the **AWS native-PyTorch Neuron beta
>   ("Beta 3")**, NOT the public Neuron SDK. To install and run it you must first obtain the beta **from the
>   AWS Neuron team**: the Beta-3 DLC / wheels are access-gated (your account must be whitelisted). Without
>   Beta-3 access the device APIs this skill relies on (eager `device="neuron"`,
>   `torch.compile(backend="neuron")`) are unavailable. Reproduce the exact stack below by following the
>   Beta-3 setup recipe (`native-pytorch-setup-guide.md`) that AWS provides with the beta.
>
> The native-PyTorch **reference ports** that informed this skill's recipe (evo2, alphagenome, clip, esm2,
> …) were all produced on Beta 3 — that is the target toolchain, not the public SDK. Those ports live in a
> separate, private reference collection that is NOT required to run this skill; everything the skill needs
> is self-contained here.

## Beta-3 parity: RESOLVED

The reference ports (esp. evo2, alphagenome) explicitly required "Beta 3" and warned the public SDK
is insufficient. **This skill was developed on that Beta-3 stack** — the AWS native-PyTorch Beta-3 setup
recipe produces exactly these versions, so there is no version gap to worry about:

| Component | Version | Import name |
|---|---|---|
| torch | `2.11.0+cpu` | `torch` |
| torch-neuronx | `2.11.3.0.1278` | `torch_neuronx` |
| neuronx-cc (compiler) | `2.25.1280.0` | **`neuronxcc`** (no underscore!) |
| torch-xla | `2.11.0` | `torch_xla` (added 2026-07-08 — see below) |
| libneuronxla | `3.0.3854.0` | — |
| nki | `0.4.0` | `nki` |
| transformers | `5.13.0` | `transformers` |
| Python / Neuron Runtime | `3.12.13` / `2.32.19` | — |

Toolchain lives in the **`torch-neuron` conda env**: `conda activate torch-neuron`.

## Ways to run on the device — Beta-3 API (VERIFIED at M1)

⚠️ **`torch_neuronx.trace` does NOT exist in this Beta-3 stack** — that is the *public-SDK* API the
public-SDK examples describe. Beta 3 exposes `neuron` as a **first-class PyTorch device backend**.
`dir(torch_neuronx)` here has device/memory/stream management + `neuron_dynamo_backend`, no `trace`.
The two real paths, both confirmed on-device:

1. **Native eager `device="neuron"`** ✅ **M1: CLIP matched CPU oracle at cosine 1.000000.** Move
   model+inputs to the device and run; `.cpu()` (or `torch_neuronx.synchronize()`) forces execution.
   The Beta-3 headline (`torch.ones(4, device="neuron").sum() == 4.0`). Simplest — start here.
   ```python
   dev = torch.device("neuron")
   model = model.eval().to(dev)
   out = model(*[t.to(dev) for t in inputs]); torch_neuronx.synchronize()
   out = [t.cpu() for t in out]
   ```
2. **`torch.compile(model, backend="neuron")`** — the graph-optimizer path (backend `"neuron"` is
   registered; verify with `torch._dynamo.list_backends()`). Use when eager needs fusing/optimizing.
3. *(torch_xla lazy `xm.xla_device()` + `mark_step()` also imports — see below — but it is the older
   XLA path and needs `PJRT_DEVICE=NEURON`. Prefer paths 1–2 on Beta 3.)*

## Env-var gotchas (bite hard)

- **`NEURON_RT_VISIBLE_CORES=<free core>`** — applies to ALL device paths (eager, compile, xla). M1 CLIP
  ran with `NEURON_RT_VISIBLE_CORES=0`. Pin ONE logical core for single-core inference. A stray orphaned
  `multiprocessing-fork` worker can hold part of the device, leaving ~1 free logical core
  (`logical-neuroncore-config: 2` ⇒ cores allocate in pairs); PJRT otherwise tries to grab all 4 and fails
  (`Requested:4 Available:3`). Reap any orphaned worker before multi-core work.
- **`PJRT_DEVICE=NEURON`** — ONLY for the torch_xla path (3); without it torch_xla silently runs
  CPU-backed XLA. Not needed for eager `device="neuron"` or `torch.compile(backend="neuron")`.
- **Multi-core (TP/FSDP, e.g. 40B):** `NEURON_RT_VIRTUAL_CORE_SIZE=2 NEURON_RT_NUM_CORES=<N> torchrun …`
  (see the beta guide's "Multi-core training with FSDP").
- Compiler cache wedged (`Got a cached failed neff`): `rm -rf /var/tmp/neuron-compile-cache`.
- EFA/NCCL/OFI init warnings on startup are **benign** for single-core (no multi-node collectives).

## Setup gotchas worth remembering (from the Beta-3 setup recipe)

- Compiler imports as **`neuronxcc`**, not `neuronx_cc` (the pip/package name is `neuronx-cc`).
- Beta wheels come from the **AWS Beta-3 DLC** (`torch_neuron_eager`, `neuronx_cc_wheels`, `nki_wheels`),
  not a public GitHub clone. Your account must be **whitelisted by the AWS Neuron team** to pull the DLC.
- conda env built from **conda-forge** (`--override-channels`); install `pip`/`uv` into it first.
- The Beta-3 DLC/recipe ships on-instance examples worth mining for API patterns (fp8 probes, GEMM/MLP
  TP-FSDP benchmarks, a `gpt2-train-loop`) — use them to confirm the eager/compile device idioms.

## Confidentiality

The AWS native-PyTorch Beta-3 user guide and DLC are **AWS private-beta material** under the beta terms —
do NOT copy them into this skill or redistribute them. Obtain them from the AWS Neuron team; reference by
name only.
