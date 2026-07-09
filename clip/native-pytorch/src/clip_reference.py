"""Shared CLIP reference harness — used by the oracle capture, the CLI, and BOTH notebooks
so they run the byte-identical model + inputs (required for meaningful CPU/CUDA/Trainium parity).

Built on the vendored openai definition in `clip_neuron/` (the definition-as-product).
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import clip_neuron as C

MODEL_NAME = "ViT-B/32"
TEXTS = ["a photo of a cat", "a photo of a dog"]
IMG_SEED = 0
OUTPUT_ORDER = ("image_features", "text_features", "logits_per_image")


class ClipWrapper(torch.nn.Module):
    """tensor-in, tuple-out over openai CLIP's own definition.
    forward(pixel_values, input_ids) -> (image_features, text_features, logits_per_image)."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pixel_values, input_ids):
        image_features = self.model.encode_image(pixel_values)
        text_features = self.model.encode_text(input_ids)
        # cosine-normalized, then scaled logits (openai CLIP.forward math)
        i = image_features / image_features.norm(dim=1, keepdim=True)
        t = text_features / text_features.norm(dim=1, keepdim=True)
        logits_per_image = self.model.logit_scale.exp() * i @ t.t()
        return image_features, text_features, logits_per_image


def load(device="cpu") -> ClipWrapper:
    """openai CLIP (vendored) wrapped for tuple output, on `device` (cpu|cuda|neuron), fp32."""
    model = C.load(MODEL_NAME, device=device, dtype=torch.float32)
    return ClipWrapper(model).eval().to(torch.device(device))


def build_inputs():
    """Deterministic fixed-shape inputs (built on CPU; caller moves to device).
    Returns (pixel_values (N,3,224,224), input_ids (N,77))."""
    from PIL import Image
    rng = np.random.default_rng(IMG_SEED)
    imgs = [Image.fromarray(rng.integers(0, 255, (256, 256, 3), dtype=np.uint8)) for _ in TEXTS]
    pixel_values = torch.stack([C.preprocess(im) for im in imgs])
    input_ids = C.tokenize(TEXTS)
    return (pixel_values, input_ids)


def loss_fn(outputs):
    """Scalar loss for the backprop reference: sum of all float output tensors
    (matches capture_reference's default; deterministic)."""
    return sum(t.float().sum() for t in outputs)
