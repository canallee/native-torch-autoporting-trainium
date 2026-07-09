"""Per-model capture adapter — copy to baselines/<model>/capture.py and fill in.

The generic engine (skills/.../scripts/capture_reference.py) imports these callables.
Keep inputs DETERMINISTIC and seed-stable so the Phase-A oracle and the Phase-B Neuron
run see identical inputs.

⚠️ OUTPUT-KEY ALIGNMENT — avoids a SILENT bogus R7 FAIL.
`compare_to_manifest` matches the Neuron run to the manifest by the keys `summarize()`
emits: a dict/`ModelOutput` keys by NAME (`logits`, ...), a tuple/list POSITIONALLY
(`[0]`, `[1]`). So the oracle captured here and the Phase-B run MUST produce the SAME
output structure — else every tensor reports `MISSING` and R7 false-FAILs. The shipped
ports guarantee this the clean way: `load_model` here delegates to the SAME shared
`<model>_reference` wrapper the notebooks use (see esm2/evo2/nucleotide-transformer
`baselines/<model>/capture.py` → `R.load(...)`), so both sides key identically. Returning a
raw `AutoModel` here instead (its `ModelOutput` keys by NAME) will NOT match a tuple-returning
port wrapper — if you do that, re-key the Neuron run to the manifest keys before comparing
(pass `{"logits": logits}`), per `scripts/compare.py`.
"""
import torch

MODEL_ID = "org/model-id"          # HF id or repo tag; recorded in the manifest


def load_model(device: torch.device) -> torch.nn.Module:
    """Return the ORIGINAL model (eval, on `device`). PREFER delegating to the shared
    `<model>_reference` wrapper the Phase-B notebooks use, so outputs key identically (see
    the output-key note above):  `import <model>_reference as R; return R.load(str(device))`.
    Use the pure-PyTorch / HF entry point, not the CUDA-native original (house_rules / R1).
    The raw-AutoModel form below is the fallback — if you use it, heed the key-alignment note."""
    from transformers import AutoModel
    model = AutoModel.from_pretrained(MODEL_ID).eval().to(device)
    return model


def build_inputs(device: torch.device):
    """Return deterministic, fixed-shape inputs (dict of kwargs OR positional tuple).
    Neuron needs static shapes — pad/crop to a fixed length here so Phase A and Phase B
    match exactly.

    Inputs need only be VALID-SHAPED, not semantically real: for parity the oracle just needs
    CPU and Trainium to see identical tensors. When real data is impractical (e.g. an MSA model
    needing aligned genomes), use SEEDED random ints/floats of the right shape/dtype/vocab range."""
    torch.manual_seed(0)
    # Example: return {"input_ids": ..., "attention_mask": ...}
    raise NotImplementedError("fill in build_inputs for this model")


# Optional: define a task-appropriate scalar loss for the backprop reference.
# If omitted, the engine uses sum-of-float-outputs and records that in the manifest.
# def loss_fn(outputs) -> torch.Tensor:
#     return outputs.logits.float().pow(2).mean()
