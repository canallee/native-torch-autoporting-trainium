# CLIP on AWS Trainium — native PyTorch (definition-as-product)

The porting **product is the model definition itself**: openai CLIP's own `model.py` is vendored in
[`src/clip_neuron/`](src/clip_neuron/) and made to run on Trainium, **importable and composable** so you
can build/train new models from its modules — not a black-box inference wrapper.

> **Status:** ✅ Inference **and** one-step backprop verified on Trainium vs CPU. Inference: all outputs
> cosine 1.000000. Backprop: 302/302 gradient tensors match. One inline Neuron patch (attention backward).

## Use it as a building block

```python
import sys; sys.path.insert(0, "src")
import clip_neuron as C

model = C.load("ViT-B/32", device="neuron")          # openai CLIP, on Trainium, fp32
img = C.preprocess(pil_image).unsqueeze(0).to("neuron")
txt = C.tokenize(["a photo of a cat"]).to("neuron")
image_features = model.encode_image(img)
text_features  = model.encode_text(txt)

# compose CLIP's own submodules into a new model:
parts = C.submodules(model)          # visual, transformer, token_embedding, ...
backbone = C.freeze(parts["visual"]) # ViT as a frozen feature extractor
```

## Environment
```bash
conda activate torch-neuron          # Beta-3 native-PyTorch stack (see references/environment.md)
pip install -r requirements.txt      # ftfy, regex, pillow, numpy  (model def is vendored)
```

## The three parity notebooks (the deliverable)
All run CPU / CUDA / Trainium and assert the results match (`cuda` auto-skips when absent):

```bash
cd notebooks
NEURON_RT_VISIBLE_CORES=0 jupyter nbconvert --to notebook --execute --inplace 01_inference_parity.ipynb
NEURON_RT_VISIBLE_CORES=0 jupyter nbconvert --to notebook --execute --inplace 02_backprop_parity.ipynb
NEURON_RT_VISIBLE_CORES=0 jupyter nbconvert --to notebook --execute --inplace 03_training_parity.ipynb
```
- **01_inference_parity** — forward on each device; per-output cosine + max-abs. Result: cosine 1.000000.
- **02_backprop_parity** — one backward on each device; magnitude-aware gradient check. Result: 302/302 match.
- **03_training_parity** — 5-step Adam loop on CLIP's own symmetric InfoNCE contrastive loss; asserts the
  loss decreases and the loss trajectory + final weights match CPU↔Trainium (on-device training).

Headless one-shot equivalent: `NEURON_RT_VISIBLE_CORES=0 python src/port_clip.py --grad`.

## Why this works / the one Neuron patch
openai CLIP is a clean-trace model — the ViT + text Transformer run on `device="neuron"` unmodified for
**inference**. The single change (`src/clip_neuron/modeling.py`, marked `NEURON PATCH`): the text tower's
**backward** through `nn.MultiheadAttention` with a causal mask fails to compile on Neuron, so attention is
computed manually from the module's own parameters (`in_proj_weight/bias`, `out_proj`) with
linear/bmm/softmax — same math, checkpoint-compatible, and its backward lowers cleanly. Validated equal to
`nn.MultiheadAttention` on CPU (forward ~5e-7, grads ~5e-5) before running on device.

## Troubleshooting
| Symptom | Fix |
|---|---|
| `neuron-ls` shows nothing | Not on a Trainium instance / wrong AMI. |
| `Logical Neuron Core(s) not available Requested:4` | Pin `NEURON_RT_VISIBLE_CORES=0` (or another free core). |
| `Compilation error ... aten::add.out` in text backward | The attention patch isn't applied — see `src/clip_neuron/modeling.py`. |
| `Got a cached failed neff` | `rm -rf /var/tmp/neuron-compile-cache` and rerun. |

## Notes & limitations
- Validated: ViT-B/32, fp32 — inference (cosine 1.0), one backward step (302/302 grads), and a multi-step
  on-device training loop (`03_training_parity`: 5-step Adam on CLIP's contrastive loss).
- A full multi-epoch / distributed contrastive training run on a real image–text corpus is the training
  phase; the composable definition + verified backward + training-parity loop are the foundation for it.

## Credits & license
openai CLIP definition vendored under MIT — see [`src/clip_neuron/VENDORED.md`](src/clip_neuron/VENDORED.md).
Weights: openai ViT-B/32 checkpoint (downloaded on first `load`).
