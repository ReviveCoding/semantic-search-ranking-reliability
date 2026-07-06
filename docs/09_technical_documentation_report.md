# Technical Documentation Report

## Executive summary

Built a GPU-accelerated four-class semantic query-product reranker on the official Amazon ESCI benchmark using a pinned DeBERTa-v3 cross-encoder and gain-aligned ranking score. In a one-shot held-out official test evaluation, improved nDCG@10 from 0.686 to 0.804, increased Top-1 Exact-or-Substitute precision by +7.57pp, and reduced irrelevant Top-1 selections by 4.42 percentage points versus a frozen lexical baseline.

## Evaluation design

Development used only official ESCI train rows under a query-disjoint split. The lexical comparator was frozen before GPU semantic training. The selected semantic configuration had to pass pre-registered development gates before exactly one official held-out evaluation.

| Metric | Development lexical | Development semantic | Official-test lexical | Official-test semantic |
| --- | --- | --- | --- | --- |
| nDCG@10 | 0.678238 | 0.801477 | 0.685705 | 0.804019 |
| Top-1 E/S precision | 84.38% | 92.21% | 85.14% | 92.71% |
| Top-1 irrelevant rate | 10.23% | 6.12% | 9.99% | 5.57% |

## Official held-out result

| Metric | Frozen lexical | Semantic reranker | Change |
| --- | --- | --- | --- |
| nDCG@10 | 0.685705 | 0.804019 | +0.118314 |
| Top-1 E/S precision | 85.14% | 92.71% | +7.57pp |
| Top-1 irrelevant rate | 9.99% | 5.57% | -4.42pp |
| Four-class macro F1 | N/A | 0.548303 | N/A |

## Paired-query uncertainty evidence

| Metric | Point estimate | 95% paired-query bootstrap CI |
| --- | --- | --- |
| nDCG@10 | +0.118314 | [+0.113828, +0.122670] |
| Top-1 E/S precision | +7.57pp | [+6.74pp, +8.40pp] |
| Top-1 irrelevant rate | -4.42pp | [-5.09pp, -3.83pp] |

## Model and runtime

- Backbone: `microsoft/deberta-v3-base`
- Revision: `8ccc9b6f36199bec6961081d44eb72fb3f7353f3`
- Inputs: query, title, brand, color, and bullet points
- Objective: four-class cross entropy
- Ranking score: expected ESCI gain from class probabilities
- GPU: `NVIDIA GeForce RTX 4090 Laptop GPU`
- BF16 supported: `True`
- Peak allocated GPU memory: `5.138 GB`
- Final training runtime: `10.15 hours`

## Dataset lineage

- ESCI source commit: `7916cdf6ab75a462e77f20ab40428a10923998d5`
- Train candidate rows: `419653`
- Test candidate rows: `181701`
- Train queries: `20888`
- Test queries: `8956`
- Train/test query overlap: `False`

## Frozen-lexical-anchor support diagnostics

Support is a training-only product appearance count, used here only for post-hoc diagnostics. It was not a model input or selection criterion.

| Frozen lexical support bucket | Queries | Lexical E/S | Semantic E/S | E/S change | Lexical I | Semantic I | I-rate change |
| --- | --- | --- | --- | --- | --- | --- | --- |
| UNSEEN_IN_TRAIN | 7102 | 85.54% | 92.54% | +7.00pp | 9.52% | 5.62% | -3.90pp |
| SINGLETON_IN_TRAIN | 1049 | 83.89% | 94.18% | +10.30pp | 11.15% | 4.29% | -6.86pp |
| LOW_SUPPORT_2_TO_3 | 504 | 83.33% | 91.87% | +8.53pp | 12.10% | 6.94% | -5.16pp |
| MEDIUM_SUPPORT_4_TO_7 | 213 | 83.10% | 93.43% | +10.33pp | 12.68% | 6.10% | -6.57pp |
| HIGH_SUPPORT_8_PLUS | 88 | 82.95% | 92.05% | +9.09pp | 15.91% | 7.95% | -7.95pp |

## Top-1 transition taxonomy

| Top-1 transition | Queries | Share |
| --- | --- | --- |
| I_TO_E_OR_S | 649 | 7.25% |
| C_TO_E_OR_S | 353 | 3.94% |
| I_TO_C | 10 | 0.11% |
| E_OR_S_TO_E_OR_S | 7301 | 81.52% |
| C_TO_C | 73 | 0.82% |
| I_TO_I | 236 | 2.64% |
| E_OR_S_TO_C | 71 | 0.79% |
| E_OR_S_TO_I | 253 | 2.82% |
| C_TO_I | 10 | 0.11% |

## Reproducibility anchors

- Source receipt SHA-256: `0cbe82cf5439020e23a58448dc2f421cca43b1fd36c0a49549127aa1c8f0e6e6`
- Frozen lexical receipt SHA-256: `c8a4a1e5f11e7919dea6344143002a2539ab037aeb6a189dda0b2a5c1be64aeb`
- Development receipt SHA-256: `e7d8f17ce602345bb86c0bc0c978c5d32499bcd8dbea0ad3b0bd4fbe86c3bd5b`
- Official-test receipt SHA-256: `19b714a0a716ba346baa6a7159e235098620494e79ebde5d16dedaa399514fb3`
- Read-only report summary SHA-256: `1212d77a76fbd87189540aac8d67a75661bc8bbe8990383861b41fc5fe89c25f`
- Final model SHA-256: `af4c95979eeed6932dcc2d90725ff853b0e03a1b36c04d5b4272a2902e3d80ad`

## Claim boundary

Supported: offline candidate-set semantic reranking on the official ESCI relevance benchmark.

Not supported: production deployment, CTR or conversion lift, advertising causality, end-to-end retrieval quality, or production cold-start claims.
