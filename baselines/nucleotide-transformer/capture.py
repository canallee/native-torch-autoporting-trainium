"""Phase-A capture adapter for Nucleotide Transformer (DEV mode).

Captures the oracle from the SHIPPED full model (`EsmForMaskedLM`, encoder + MLM head) via
nucleotide-transformer/native-pytorch/src/nucleotide_transformer_reference.py — so the frozen oracle
covers the full-model outputs (logits + hidden) and ALL of the model's parameter gradients.
"""
import os
import sys

_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "../../nucleotide-transformer/native-pytorch/src")
sys.path.insert(0, os.path.abspath(_SRC))
import nucleotide_transformer_reference as R

MODEL_ID = "InstaDeepAI/nucleotide-transformer-v2-50m-multi-species (dev: full EsmForMaskedLM)"


def load_model(device):
    return R.load(str(device))


def build_inputs(device):
    return R.build_inputs()


def loss_fn(outputs):
    return R.loss_fn(outputs)
