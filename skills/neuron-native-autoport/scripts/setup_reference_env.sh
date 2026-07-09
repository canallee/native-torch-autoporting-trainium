#!/usr/bin/env bash
# Phase A (A3): build a per-repo REFERENCE env for capturing the oracle.
#
# Policy (per design decision): pin torch==2.11.0 (CPU) and layer the target repo's
# OTHER requirements around that pin, so the oracle is apples-to-apples with the
# Trainium run (only device + compiler differ). NEVER install target deps into the
# Beta-3 `torch-neuron` env — this creates a separate, disposable env instead.
#
# If the repo is irreconcilable with torch 2.11 / transformers 5.13, the operator
# should fall back to a repo-native env and RECORD the torch-version caveat in the
# manifest meta (`"torch_version_caveat": "..."`).
#
# Usage:
#   setup_reference_env.sh <env-name> <target-repo-dir> [extra pip args...]
# Example:
#   setup_reference_env.sh ref-clip ../port-targets/CLIP
set -euo pipefail

ENV_NAME="${1:?usage: setup_reference_env.sh <env-name> <target-repo-dir> [extra pip args]}"
REPO_DIR="${2:?need target repo dir}"
shift 2 || true

TORCH_PIN="torch==2.11.0"

source "$(conda info --base)/etc/profile.d/conda.sh"

if ! conda env list | grep -qE "^${ENV_NAME}\s"; then
  echo "[env] creating conda env '${ENV_NAME}' (python 3.12, conda-forge)"
  conda create -n "${ENV_NAME}" -c conda-forge --override-channels python=3.12 -y
  conda run -n "${ENV_NAME}" conda install -c conda-forge --override-channels -y pip uv
fi

echo "[env] pinning ${TORCH_PIN} (CPU) — oracle must match the Trainium torch version"
conda run -n "${ENV_NAME}" pip install "${TORCH_PIN}" --index-url https://download.pytorch.org/whl/cpu

# Install the repo's own requirements AROUND the torch pin. A constraints file keeps
# transitive deps from silently bumping torch off 2.11.
CONSTRAINTS="$(mktemp)"; echo "${TORCH_PIN}" > "${CONSTRAINTS}"
if [ -f "${REPO_DIR}/requirements.txt" ]; then
  echo "[env] installing ${REPO_DIR}/requirements.txt (torch pinned via constraints)"
  conda run -n "${ENV_NAME}" pip install -c "${CONSTRAINTS}" -r "${REPO_DIR}/requirements.txt" "$@" || {
    echo "[env] WARNING: repo requirements conflict with torch 2.11."
    echo "      Consider a repo-native env and record a torch_version_caveat in the manifest."
  }
elif [ -f "${REPO_DIR}/pyproject.toml" ] || [ -f "${REPO_DIR}/setup.py" ]; then
  echo "[env] installing ${REPO_DIR} as a package (--no-deps to protect the torch pin; add deps explicitly)"
  conda run -n "${ENV_NAME}" pip install -c "${CONSTRAINTS}" --no-deps -e "${REPO_DIR}" "$@"
fi
rm -f "${CONSTRAINTS}"

echo "[env] recording lockfile -> env/requirements.lock"
mkdir -p env
conda run -n "${ENV_NAME}" pip freeze > env/requirements.lock

echo "[env] verify torch pin held:"
conda run -n "${ENV_NAME}" python -c "import torch; print('  torch', torch.__version__); assert torch.__version__.startswith('2.11'), 'torch pin broken — see caveat note'"
echo "[env] done: conda activate ${ENV_NAME}"
