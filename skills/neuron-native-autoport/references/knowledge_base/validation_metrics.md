# Validation metrics — comparing a Neuron run to the CPU oracle (R7)

## Comparison must be MAGNITUDE-AWARE, not pure cosine
A tensor passes if **cosine ≥ 0.99 OR max-abs diff is negligible vs a sensible scale**
(`scripts/compare.py` already does `max_rel <= rtol OR max_abs <= atol`). Pure cosine is the
wrong gate for **near-zero tensors** — two vectors of floating-point noise around zero have a
meaningless cosine (anywhere in [-1, 1]).

## This bites HARD on gradients
Gradient sets always contain **analytically-zero** components. The canonical example (seen on CLIP M1):
- **`self_attn.k_proj.bias` gradients are ≈ 0.** The key-projection bias adds a constant to *every*
  key; attention softmax is shift-invariant across keys, so ∂loss/∂k_proj.bias vanishes analytically.
  What's left is fp noise (~1e-7). On CLIP, all 24 "cosine failures" were exactly these — every
  `k_proj.bias` in both towers — while the true max-abs grad diff across the whole model was ~1e-8 of
  the global grad scale.
- **Gate for gradients:** per-tensor, pass if `cosine ≥ 0.99` OR
  `max_abs_diff / max(|grad| over the whole model) <= 1e-3`. Report the count of *real* disagreements
  (those failing BOTH), not raw cosine failures. See `clip/native-pytorch/src/port_clip.py:check_grad_parity`.
- **Weight-diff gate (03 training) — beware a single large scalar param inflating the GLOBAL scale.** A
  `1e-3 · max(|weight|)` tolerance is loosened if one big scalar dominates the max (e.g. GPN-Star's
  `FIRETimeBias.c = 100` → tolerance 0.1, far looser than the real per-tensor weights need). Harmless when
  the device matches tightly anyway, but for a strict check prefer **per-tensor relative norms**
  (`‖Δw‖/‖w‖` per parameter) over one global scale so small weights aren't graded against a large one.

## Recommended reporting
- Inference: per-output cosine + max-abs (+ top-1 token agreement for LMs).
- Gradients: loss value CPU vs Neuron, max-abs grad diff vs global grad scale, and the count of real
  disagreements — never a bare cosine pass-rate.
- Always compare against the FROZEN Phase-A manifest/artifacts, on the same seeded inputs.
