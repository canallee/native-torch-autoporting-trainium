"""Shared GPN-Star DEV harness — used by the oracle capture, the CLI, and all three notebooks so they run
byte-identical models + inputs (required for CPU/CUDA/Trainium parity).

Dev mode: the "model" under test is the vendored FULL model (`GPNStarForMaskedLM`, MSA encoder + MLM
head), tensor-in / tuple-out. Parity validates the full model's own outputs (logits) and all of its
parameters' gradients. Inputs are deterministic synthetic MSA tensors (see gpn_star_neuron.build_synthetic_msa).
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gpn_star_neuron as G

MODEL_NAME = G.MODEL_NAME
MAXLEN = 64
N_TARGETS = 1
SEED = 1234
OUTPUT_ORDER = ("logits",)


class GPNStarWrapper(torch.nn.Module):
    """tensor-in, tuple-out over the full GPNStarForMaskedLM.
    forward(input_ids, source_ids, target_species) -> (logits,)  # (B, L, T, vocab) MLM head."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, source_ids, target_species):
        out = self.model(input_ids=input_ids, source_ids=source_ids, target_species=target_species)
        return (out.logits,)


def load(device="cpu") -> GPNStarWrapper:
    """Full vendored GPN-Star wrapped for tuple output, on `device`, fp32."""
    model = G.load(MODEL_NAME, device=device, dtype=torch.float32)
    return GPNStarWrapper(model).eval().to(torch.device(device))


def build_inputs():
    """Deterministic synthetic MSA (input_ids, source_ids, target_species) — positional tuple."""
    n = G.num_species(MODEL_NAME)
    return G.build_synthetic_msa(n, batch_size=1, seq_len=MAXLEN, n_targets=N_TARGETS, seed=SEED)


def loss_fn(outputs):
    """Scalar loss for the backprop reference: sum of all float output tensors (deterministic)."""
    return sum(t.float().sum() for t in outputs)
