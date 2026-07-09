"""Phase-A capture adapter for CLIP — uses the SHIPPED own-definition (clip_neuron, vendored
openai CLIP), so the frozen oracle matches exactly what Phase B validates and what the notebooks run.

Delegates to the shared harness clip/native-pytorch/src/clip_reference.py.
"""
import os
import sys

_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../clip/native-pytorch/src")
sys.path.insert(0, os.path.abspath(_SRC))
import clip_reference as R

MODEL_ID = "openai/CLIP ViT-B/32 (vendored own-definition)"


def load_model(device):
    return R.load(str(device))


def build_inputs(device):
    return R.build_inputs()


def loss_fn(outputs):
    return R.loss_fn(outputs)
