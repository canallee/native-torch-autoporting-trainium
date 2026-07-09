"""evo2_neuron — Evo 2 (StripedHyena2 DNA LM) OWN definition, made available on AWS Trainium (DEV mode).

This is the definition-as-product: the HuggingFace Evo2 port's own modeling files
(`modeling_evo2.py`, `hyena.py`, `engine.py`, `attention.py`, `rotary.py`, `layers.py`,
`configuration_evo2.py`, `cache.py`, `tokenization_evo2.py`) are vendored here (Apache-2.0; see
VENDORED.md), with ONE Neuron patch applied inline in `engine.py` (FFT → conv1d; complex ops don't
lower). Importable & composable — build/train new DNA models from its Hyena/attention blocks.

    import evo2_neuron as E
    model = E.load(device="neuron")            # Evo2-1B on Trainium, fp32
    ids = E.tokenize("ACGTACGT").to("neuron")
    out = model(input_ids=ids, use_cache=False, output_hidden_states=True)
    hidden = out.hidden_states[-1]             # (1, T, 1920) DNA embeddings

Beta-3 recipe baked into load(): use_fp8_input_projections=False (no TransformerEngine),
attn_implementation="eager" (no flash-attn), fp32 (a couple of layers emit ~1e16 activations the
final norm absorbs; bf16 collapses them to 0 on Neuron), use_cache=False (static prefill graph).
"""
import torch

MODEL_ID = "Taykhoom/Evo2-1B-8K"   # HF port (pure PyTorch, no vortex/CUDA/TransformerEngine)


def _resolve(path_or_id: str) -> str:
    import os
    if os.path.isdir(path_or_id):
        return path_or_id
    from huggingface_hub import snapshot_download
    return snapshot_download(path_or_id)


def load(path_or_id: str = MODEL_ID, device="cpu", dtype=torch.float32):
    """Build Evo2 from the VENDORED definition (patch already inline in engine.py) + HF weights."""
    from .configuration_evo2 import Evo2Config
    from .modeling_evo2 import Evo2ForCausalLM
    path = _resolve(path_or_id)
    cfg = Evo2Config.from_pretrained(path)
    cfg.use_fp8_input_projections = False                 # pure-PyTorch projections
    model = Evo2ForCausalLM.from_pretrained(
        path, config=cfg, attn_implementation="eager"
    ).eval()
    model = model.float() if dtype == torch.float32 else model.to(dtype)
    return model.to(torch.device(device))


def get_tokenizer(path_or_id: str = MODEL_ID):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(_resolve(path_or_id), trust_remote_code=True)


def tokenize(dna: str, path_or_id: str = MODEL_ID):
    """DNA string (A/C/G/T) -> (1, T) byte token ids. NOTE: pass a str, not a list."""
    return get_tokenizer(path_or_id)(dna, return_tensors="pt")["input_ids"]
