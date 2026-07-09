# baselines/

Phase-A outputs, one folder per model: the capture adapter (`capture.py`), the
`inference.ipynb` / `backprop.ipynb` notebooks, the committed **`MANIFEST.json`** oracle,
and an `env/requirements.lock`. Raw output/gradient tensors under `<model>/artifacts/` are
gitignored — only the manifest is committed.
