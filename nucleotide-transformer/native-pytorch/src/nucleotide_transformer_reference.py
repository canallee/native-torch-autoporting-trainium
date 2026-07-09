"""Shared Nucleotide-Transformer DEV harness — used by the oracle capture, the CLI, and all three notebooks
so they run byte-identical models + inputs (required for CPU/CUDA/Trainium parity).

Dev mode: the "model" under test is the vendored FULL model (`EsmForMaskedLM`, encoder + MLM head),
returned tensor-in / tuple-out. Parity validates the full model's own outputs and all of its parameters'
gradients — the editable definition itself, not a downstream composite.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nucleotide_transformer_neuron as NT

MODEL_ID = NT.MODEL_ID
MAXLEN = 64
# deterministic DNA inputs (multi-species-style sequences of A/C/G/T)
SEQS = ["ATGGCGCCTAGCTAGCTAGGCTAACGTACGTTAGCTAGCATCGATCGTAGCTAGCTAGCTAGCTAAGGCTA",
        "GGCCTTAACCGGTTAACCGGTTAACCGGATCGATCGGCTAGCTAGCATCGGGCCTTAACGTACGTAGCTAG"]
OUTPUT_ORDER = ("logits", "hidden")


class NTWrapper(torch.nn.Module):
    """tensor-in, tuple-out over the full EsmForMaskedLM.
    forward(input_ids, attention_mask) -> (logits (B,L,vocab), last_hidden_state (B,L,D))."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        out = self.model(input_ids=input_ids, attention_mask=attention_mask,
                         output_hidden_states=True)
        return out.logits, out.hidden_states[-1]


def load(device="cpu", model_id: str = MODEL_ID) -> NTWrapper:
    """Full vendored Nucleotide Transformer wrapped for tuple output, on `device`, fp32."""
    model = NT.load(model_id, device=device, dtype=torch.float32)
    return NTWrapper(model).eval().to(torch.device(device))


def build_inputs(model_id: str = MODEL_ID):
    """Deterministic fixed-shape (input_ids, attention_mask)."""
    return NT.tokenize(SEQS, model_id, MAXLEN)


def loss_fn(outputs):
    """Scalar loss for the backprop reference: sum of all float output tensors (deterministic)."""
    return sum(t.float().sum() for t in outputs)
