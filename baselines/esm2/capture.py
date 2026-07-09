"""Phase-A capture adapter for ESM-2 (INTEGRATION mode).

Captures the oracle from the SHIPPED integration path: the Composite (ESM-2 backbone + downstream
head) from esm2/native-pytorch/src/esm2_reference.py. So the frozen oracle covers both the extracted
embeddings (inference) and gradient flow through the backbone into the head (backprop).
"""
import os
import sys

_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../esm2/native-pytorch/src")
sys.path.insert(0, os.path.abspath(_SRC))
import esm2_reference as R

MODEL_ID = "facebook/esm2_t6_8M_UR50D (integration: encoder embeddings, MLM head dropped)"


def load_model(device):
    return R.load(str(device), finetune_backbone=True)


def build_inputs(device):
    return R.build_inputs()


def loss_fn(outputs):
    return R.loss_fn(outputs)
