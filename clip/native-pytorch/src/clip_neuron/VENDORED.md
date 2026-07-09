# Vendored source

- `modeling.py` — openai CLIP model definition, copied **verbatim** from
  [openai/CLIP](https://github.com/openai/CLIP) `clip/model.py` (commit in `../../../../port-targets/CLIP`).
- `simple_tokenizer.py` + `bpe_simple_vocab_16e6.txt.gz` — openai CLIP BPE tokenizer, verbatim.

License: MIT (openai/CLIP). This is the **ingested repo's own model definition**, made available on
Trainium as the porting product.

**Neuron patches applied: NONE.** Verified on the AWS native-PyTorch Beta-3 stack that the definition runs
on `device="neuron"` unmodified — `encode_image` / `encode_text` match a CPU reference at cosine 1.0
(inference) and gradients match (backprop). If a future variant needs an op rewrite, apply it inline here
and record it in this file (per `knowledge_base/op_rewrites.md`).
