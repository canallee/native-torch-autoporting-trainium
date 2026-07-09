"""Shared Evo2 harness — used by the oracle capture, the CLI, and BOTH notebooks so they run
byte-identical models + inputs.

`load()` returns the VENDORED (conv1d-patched) model = the port. `load_fft_reference()` returns the
ORIGINAL FFT model (CPU) = the ground-truth oracle. The notebooks compare the port (on each device) to
the FFT reference — proving BOTH the FFT->conv1d rewrite and CPU/Trainium device parity.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import evo2_neuron as E

MODEL_ID = E.MODEL_ID
DNA = "ACGTACGTACGTACGTACGTACGTACGTACGT"      # deterministic fixed-length input
OUTPUT_ORDER = ("hidden", "logits")


class Evo2Wrapper(torch.nn.Module):
    """tensor-in, tuple-out. forward(input_ids) -> (last_hidden_state, logits)."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids):
        out = self.model(input_ids=input_ids, use_cache=False, output_hidden_states=True)
        return out.hidden_states[-1], out.logits


def load(device="cpu") -> Evo2Wrapper:
    """The PORT: vendored conv1d-patched Evo2 on `device`, fp32."""
    return Evo2Wrapper(E.load(MODEL_ID, device=device, dtype=torch.float32)).eval()


def load_fft_reference() -> Evo2Wrapper:
    """The ORACLE: the ORIGINAL FFT model (trust_remote_code) on CPU, fp32 — ground truth."""
    from transformers import AutoConfig, AutoModelForCausalLM
    path = E._resolve(MODEL_ID)
    cfg = AutoConfig.from_pretrained(path, trust_remote_code=True)
    cfg.use_fp8_input_projections = False
    m = AutoModelForCausalLM.from_pretrained(
        path, trust_remote_code=True, config=cfg, attn_implementation="eager"
    ).eval().float()
    return Evo2Wrapper(m).eval()


def build_inputs():
    """Deterministic (1, T) byte token ids. Positional tuple to match model(*inputs)."""
    return (E.tokenize(DNA, MODEL_ID),)


def loss_fn(outputs):
    return sum(t.float().sum() for t in outputs)
