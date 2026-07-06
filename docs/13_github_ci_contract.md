# GitHub CI Contract

## Scope

This GitHub-hosted workflow validates the public source package on an
Ubuntu CPU runner with Python 3.13.

## What CI verifies

- package installation from `pyproject.toml`;
- training and development optional Python dependencies;
- a CPU PyTorch dependency for unit tests;
- the repository's `python -m pytest -q` test suite.

## What CI does not run

- raw ESCI data download;
- Hugging Face model download;
- CUDA or BF16 runtime verification;
- GPU training;
- development-model fitting;
- one-shot official test evaluation;
- regeneration or modification of evidence artifacts.

The workflow is source-level regression coverage only. It does not replace
the pinned local GPU evidence or the one-shot official-test receipt.
