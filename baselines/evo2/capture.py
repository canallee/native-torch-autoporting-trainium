"""Phase-A capture adapter for Evo2 (DEV mode).

Oracle = the ORIGINAL FFT model on CPU (ground truth), captured via evo2_reference.load_fft_reference().
The port (vendored conv1d) is validated against this in Phase B / the notebooks.
"""
import os
import sys

_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../evo2/native-pytorch/src")
sys.path.insert(0, os.path.abspath(_SRC))
import evo2_reference as R

MODEL_ID = "Taykhoom/Evo2-1B-8K (FFT reference / ground truth)"


def load_model(device):
    # Oracle is the FFT original on CPU regardless of requested device.
    return R.load_fft_reference()


def build_inputs(device):
    return R.build_inputs()


def loss_fn(outputs):
    return R.loss_fn(outputs)
