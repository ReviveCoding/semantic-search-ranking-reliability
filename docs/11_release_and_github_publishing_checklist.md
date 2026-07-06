# GitHub Publishing Checklist

## Before publishing

- [ ] Confirm `git status --short` is empty.
- [ ] Keep `data/raw/`, `.venv/`, model artifacts, Hugging Face cache, and local artifact runs untracked.
- [ ] Remove local usernames, absolute paths, tokens, and credentials from public files.
- [ ] Keep the official-test receipt unchanged.
- [ ] Do not re-run official test evaluation.
- [ ] Verify README rendering on GitHub.
- [ ] Choose a license before public release.

## Public claim

Built a GPU-accelerated four-class semantic query-product reranker on the official Amazon ESCI benchmark using a pinned DeBERTa-v3 cross-encoder and gain-aligned ranking score. In a one-shot held-out official test evaluation, improved nDCG@10 from 0.686 to 0.804, increased Top-1 Exact-or-Substitute precision by +7.57pp, and reduced irrelevant Top-1 selections by 4.42 percentage points versus a frozen lexical baseline.

## Do not claim

- Production deployment
- CTR or conversion lift
- Advertising causality
- End-to-end retrieval quality
