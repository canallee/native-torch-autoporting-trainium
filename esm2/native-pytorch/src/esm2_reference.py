"""Shared ESM-2 integration harness — used by the oracle capture, the CLI, and BOTH notebooks so
they run byte-identical models + inputs (required for CPU/CUDA/Trainium parity).

Integration mode: the "model" under test is a COMPOSITE (ESM-2 backbone + a small downstream head) —
this is the real use case and it exercises the integration-critical property: gradients must flow
correctly through the sealed backbone into the new head on-device.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import esm2_neuron as E

MODEL_NAME = E.MODEL_NAME
MAXLEN = 64
N_OUT = 2                      # toy downstream task (e.g. binary property)
# deterministic protein inputs
SEQS = ["MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQ",
        "MSTNPKPQRKTKRNTNRRPQDVKFPGGGQIVGGVYLLPRRGPRLGVRATRKTSERSQPRGRRQPIP"]
OUTPUT_ORDER = ("embeddings", "logits")


class Composite(torch.nn.Module):
    """ESM-2 backbone (feature extractor) + a new downstream head.
    forward(input_ids, attention_mask) -> (embeddings (B,L,D), logits (B,N_OUT))."""
    def __init__(self, backbone, hidden_size, n_out=N_OUT):
        super().__init__()
        self.backbone = backbone
        self.head = torch.nn.Linear(hidden_size, n_out)

    def forward(self, input_ids, attention_mask):
        embeddings = E.embed(self.backbone, input_ids, attention_mask)     # (B, L, D)
        m = attention_mask.unsqueeze(-1).to(embeddings.dtype)
        pooled = (embeddings * m).sum(1) / m.sum(1).clamp_min(1.0)          # (B, D)
        logits = self.head(pooled)                                         # (B, N_OUT)
        return embeddings, logits


def load(device="cpu", finetune_backbone: bool = True) -> Composite:
    """Composite on `device`, fp32. Head init is seeded so all devices are identical.
    finetune_backbone=False freezes ESM-2 (the frozen-feature-extractor case)."""
    backbone = E.load(MODEL_NAME, device="cpu", dtype=torch.float32)
    torch.manual_seed(0)                          # deterministic head init across devices
    model = Composite(backbone, backbone.config.hidden_size).eval()
    if not finetune_backbone:
        E.freeze(model.backbone)
    return model.to(torch.device(device))


def build_inputs():
    """Deterministic fixed-shape (input_ids, attention_mask)."""
    return E.tokenize(SEQS, MODEL_NAME, MAXLEN)


def loss_fn(outputs):
    """Sum of float outputs (default-compatible, deterministic). Flows through both the
    embeddings (into the backbone) and the logits (through the new head)."""
    return sum(t.float().sum() for t in outputs)
