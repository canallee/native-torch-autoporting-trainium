"""clip_neuron — openai CLIP's OWN model definition, made available on AWS Trainium.

This is the *definition-as-product*: `modeling.py` is openai's `clip/model.py`, vendored
verbatim (MIT; see VENDORED.md) and validated to run on `device="neuron"` with NO patches
(nn.MultiheadAttention, QuickGELU, the ViT + text Transformer all lower cleanly; encode_image /
encode_text match a CPU reference at cosine 1.0).

Use it as a composable building block for new models:

    import clip_neuron as C
    model = C.load("ViT-B/32", device="neuron")          # openai CLIP, on Trainium
    img_feat = model.encode_image(C.preprocess(pil_img).unsqueeze(0).to("neuron"))
    txt_feat = model.encode_text(C.tokenize(["a dog"]).to("neuron"))

    # compose CLIP's own submodules into a new model:
    vit  = model.visual                 # VisionTransformer backbone
    text = model.transformer            # text Transformer stack
    C.freeze(vit)                       # use as a frozen feature extractor
"""
import gzip  # noqa: F401  (used indirectly by simple_tokenizer)
import hashlib
import os
import urllib.request

import numpy as np
import torch
from PIL import Image

from .modeling import build_model, CLIP  # noqa: F401  (CLIP re-exported for typing/subclassing)
from .simple_tokenizer import SimpleTokenizer

# openai CLIP checkpoints (JIT archives; we extract the state_dict and rebuild the nn.Module).
_MODELS = {
    "ViT-B/32": ("https://openaipublic.azureedge.net/clip/models/"
                 "40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
                 "40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af"),
    "ViT-L/14": ("https://openaipublic.azureedge.net/clip/models/"
                 "b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt",
                 "b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836"),
}

# CLIP image normalization.
_MEAN = (0.48145466, 0.4578275, 0.40821073)
_STD = (0.26862954, 0.26130258, 0.27577711)

_tokenizer = SimpleTokenizer()


def _download(name: str, root: str = None) -> str:
    root = root or os.path.expanduser("~/.cache/clip_neuron")
    os.makedirs(root, exist_ok=True)
    url, sha = _MODELS[name]
    target = os.path.join(root, os.path.basename(url))
    if os.path.exists(target):
        if hashlib.sha256(open(target, "rb").read()).hexdigest() == sha:
            return target
    print(f"[clip_neuron] downloading {name} ...")
    urllib.request.urlretrieve(url, target)
    return target


def available_models():
    return list(_MODELS)


def load(name: str = "ViT-B/32", device="cpu", dtype=torch.float32) -> CLIP:
    """Build openai's CLIP nn.Module from its checkpoint and move it to `device`.

    device: "cpu" (default) | "cuda" | "neuron". Kept in fp32 by default (correctness);
    openai's build_model would otherwise convert to fp16.
    """
    path = _download(name)
    state_dict = torch.jit.load(path, map_location="cpu").eval().state_dict()
    model = build_model(state_dict).eval()
    if dtype == torch.float32:
        model = model.float()          # undo build_model's fp16 conversion
    else:
        model = model.to(dtype)
    return model.to(torch.device(device))


def tokenize(texts, context_length: int = 77, truncate: bool = True):
    """Text -> (N, context_length) int token ids, using CLIP's BPE tokenizer."""
    if isinstance(texts, str):
        texts = [texts]
    sot, eot = _tokenizer.encoder["<|startoftext|>"], _tokenizer.encoder["<|endoftext|>"]
    result = torch.zeros(len(texts), context_length, dtype=torch.long)
    for i, text in enumerate(texts):
        tokens = [sot] + _tokenizer.encode(text) + [eot]
        if len(tokens) > context_length:
            if not truncate:
                raise RuntimeError(f"input {text!r} too long for context {context_length}")
            tokens = tokens[:context_length]
            tokens[-1] = eot
        result[i, : len(tokens)] = torch.tensor(tokens)
    return result


def preprocess(image: Image.Image, n_px: int = 224) -> torch.Tensor:
    """PIL image -> (3, n_px, n_px) normalized tensor. Minimal, torchvision-free:
    resize shortest side to n_px (bicubic), center-crop, scale to [0,1], normalize."""
    image = image.convert("RGB")
    w, h = image.size
    s = n_px / min(w, h)
    image = image.resize((round(w * s), round(h * s)), Image.BICUBIC)
    w, h = image.size
    left, top = (w - n_px) // 2, (h - n_px) // 2
    image = image.crop((left, top, left + n_px, top + n_px))
    arr = torch.from_numpy(np.asarray(image, dtype=np.float32) / 255.0).permute(2, 0, 1)
    mean = torch.tensor(_MEAN).view(3, 1, 1)
    std = torch.tensor(_STD).view(3, 1, 1)
    return (arr - mean) / std


# ---- composability helpers ----
def freeze(module: torch.nn.Module):
    """Freeze a submodule (use CLIP pieces as a fixed feature extractor)."""
    for p in module.parameters():
        p.requires_grad_(False)
    return module


def unfreeze(module: torch.nn.Module):
    for p in module.parameters():
        p.requires_grad_(True)
    return module


def submodules(model: CLIP) -> dict:
    """Named handles to CLIP's own building blocks, for composing into new models."""
    return {
        "visual": model.visual,                     # image encoder (ViT or ModifiedResNet)
        "transformer": model.transformer,           # text Transformer stack
        "token_embedding": model.token_embedding,
        "ln_final": model.ln_final,
        "positional_embedding": model.positional_embedding,
        "text_projection": model.text_projection,
        "logit_scale": model.logit_scale,
    }
