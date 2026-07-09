"""Phase-A capture adapter for GPN-Star (DEV mode).

Captures the oracle from the SHIPPED full model (`GPNStarForMaskedLM`, MSA encoder + MLM head) via the
VENDORED definition in gpn-star/native-pytorch/src/gpn_star_reference.py — so the frozen oracle covers the
full-model outputs (logits) and ALL of the model's parameter gradients.

Inputs are deterministic, valid-shaped SYNTHETIC MSA tensors (GPN-Star takes aligned-genome MSAs; for
CPU-vs-Trainium parity only identical static-shaped inputs are needed — see gpn_star_neuron).
"""
import os
import sys

_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../gpn-star/native-pytorch/src")
sys.path.insert(0, os.path.abspath(_SRC))
import gpn_star_reference as R

MODEL_ID = "songlab/gpn-star-tair10-b18-25m (dev: full GPNStarForMaskedLM, vendored)"


def load_model(device):
    return R.load(str(device))


def build_inputs(device):
    return R.build_inputs()


def loss_fn(outputs):
    return R.loss_fn(outputs)
