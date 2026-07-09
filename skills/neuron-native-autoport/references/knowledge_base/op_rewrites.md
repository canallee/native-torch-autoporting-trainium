# Unsupported-op rewrite patterns

When `analyze_hlo.py` localizes an unsupported op to a source line, rewrite it to a
Neuron-supported op that is *mathematically identical*, and **validate the rewrite against
the original on CPU before recompiling** (house rule).

## FFT / complex convolution → real depthwise causal `conv1d` (evo2)
StripedHyena2's Hyena operators do long convolutions via FFT (complex64) — `NCC_EVRF004`.
A linear convolution of a signal with a **real, finite** filter equals a real depthwise causal
`F.conv1d` (the FFT is just an O(L log L) implementation of the same math):
- `weight = k.flip(-1).reshape(C, 1, Lk)` — **flip the filter** (conv1d cross-correlates), depthwise via `groups=C`.
- **Normalization invariant:** `rfft(k)/fft_size` + `irfft(..., norm="forward")` (and a mixed `rfft`/`fft`
  in `parallel_iir`) collapse to a **plain, unscaled** linear convolution — so the `conv1d` needs **no 1/N
  factor**. Confirm this for your model's exact FFT calls; a stray norm factor is a silent scale bug.
- compute in fp32 to match the FFT path, then cast back.
- covers the non-bidirectional, `k_rev=None` prefill path only.
- ⚠️ a model's own `long_fir_threshold` conv branch may be buggy (correlates without flipping); replace
  both FFT sites (`fftconv_func` FIR + `parallel_iir` modal-FFT branch) rather than trusting it.
- The modal-FFT side effects (`prefill_via_modal_fft`) are skipped when `inference_params=None` (prefill),
  so the conv1d replacement is clean for that path.
Validated equal to the FFT originals on CPU (logits/hidden cosine 1.000000, top-1 100%) then on device.
Worked reference implementation: `evo2/native-pytorch/src/evo2_neuron/engine.py` (`NEURON PATCH`).

## `sort` / top-k → skip-on-device or CPU (alphagenome)
`sort` doesn't lower (`NCC_EVRF029`). If it's confined to one output head (alphagenome
`splice_junctions`), skip that head on Neuron and compute it on CPU. Broader fix: rewrite the
selection to a supported TopK op or a small NKI kernel.

## Attention → eager
`attn_implementation="eager"` replaces flash-attn with pure-PyTorch attention that traces.
Models already using `F.scaled_dot_product_attention` (bacformer Stage-2) trace directly.

## `nn.MultiheadAttention` backward with a causal mask → manual QKV attention (CLIP M1)
`nn.MultiheadAttention`'s **fused backward fails to compile on Neuron when an additive attn_mask is
present** (`RuntimeError: Compilation error ... aten::add.out ... [3,77,2,512]` — the packed-QKV grad).
Symptom is **backward-only**: inference and mask-free attention (e.g. a ViT tower) work; a masked text
tower errors at `.backward()`/synchronize. Fix — compute the SAME attention manually from the module's own
parameters so the backward is plain ops:
```python
qkv = F.linear(x, attn.in_proj_weight, attn.in_proj_bias); q,k,v = qkv.chunk(3, -1)
# reshape each to [N*H, L, Hd]
scores = torch.bmm(q, k.transpose(1,2)) / (Hd**0.5)
if mask is not None: scores = scores + mask
out = torch.bmm(scores.softmax(-1), v)              # -> [N*H, L, Hd] -> [L, N, E]
return F.linear(out, attn.out_proj.weight, attn.out_proj.bias)
```
Weight layout is unchanged (checkpoints load as-is). Validate equal to `nn.MultiheadAttention` on CPU
(fwd + grads) before recompiling. Reference: `clip/native-pytorch/src/clip_neuron/modeling.py` (`NEURON PATCH`).
This is the canonical example of a patch needed only for **training/backprop**, not inference.

## ⚠️ Validating a rewrite needs BOTH paths live in one process — use a per-INSTANCE selector
The house rule "CPU-validate the patch vs the original" requires the **original op AND the rewritten op to
coexist in one process** (build `m_original` and `m_patched`, compare their outputs). If you select the
path with a **module-global flag** read at forward time, loading the second model **flips the flag for
BOTH** → the gate silently compares patched-vs-patched and reports a bogus **max-abs 0.000 "PASS"**. Guard
against it: make the selector **per-instance** (e.g. `self._use_conv`, or two separate module classes), or
run the two models in **separate processes**. A suspiciously *perfect* diff (exactly 0) means the gate
no-op'd — treat 0.000 as a red flag, not a win.

## Add patterns here as M2/M3 discovers them (name → cause → equivalent op → validation).
