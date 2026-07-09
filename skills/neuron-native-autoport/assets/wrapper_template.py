"""Phase B (R3): device-friendly wrapper template.

The Beta-3 `device="neuron"` / `torch.compile(backend="neuron")` paths capture a graph best from a
module with TENSOR inputs and a TUPLE-OF-TENSORS output — no dicts, no kwargs leaking out, no
`return_dict` object, no data-dependent None branches. Copy this per model; keep output names 1:1
with HuggingFace where possible (append `_u` to any name with no clean HF mapping — see house_rules).

Usage (Beta-3, eager path — verified on CLIP at M1):
    dev = torch.device("neuron")
    wrapper = PortWrapper(model).eval().to(dev)
    out = wrapper(*[t.to(dev) for t in inputs]); torch_neuronx.synchronize()
    out = tuple(t.cpu() for t in out)
(NOTE: no `torch_neuronx.trace` on Beta 3 — that is the public-SDK API.)
"""
import torch


class PortWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask, pixel_values=None):
        # Call the underlying model with explicit kwargs; return a flat tuple of tensors.
        out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            return_dict=True,       # unpack here; trace sees only the returned tuple
        )
        # Return exactly the tensors you validate against the Phase-A manifest, in a
        # STABLE order. Document the order — the manifest keys must line up.
        return out.logits_per_image, out.image_embeds, out.text_embeds
