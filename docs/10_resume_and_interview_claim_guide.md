# Resume and Interview Claim Guide

## Primary resume bullet

Built a GPU-accelerated four-class semantic query-product reranker on the official Amazon ESCI benchmark using a pinned DeBERTa-v3 cross-encoder and gain-aligned ranking score. In a one-shot held-out official test evaluation, improved nDCG@10 from 0.686 to 0.804, increased Top-1 Exact-or-Substitute precision by +7.57pp, and reduced irrelevant Top-1 selections by 4.42 percentage points versus a frozen lexical baseline.

## Search / ranking bullet

- Built a reliability-first semantic product-search reranker with a pinned DeBERTa-v3 cross-encoder; improved nDCG@10 from `0.686` to `0.804` and reduced irrelevant Top-1 selections by `4.42 percentage points` on a one-shot official ESCI held-out test.

## Applied Scientist bullet

- Designed a query-disjoint development and one-shot held-out evaluation protocol for GPU semantic ranking; increased Top-1 Exact-or-Substitute precision by `+7.57pp` with paired-query bootstrap confidence intervals supporting all pre-registered gates.

## 30-second explanation

I built an offline semantic candidate reranker for product search. I froze a transparent lexical baseline, trained a four-class DeBERTa-v3 cross-encoder on query and product metadata, and ranked candidates by expected relevance gain. The reliability work was query-disjoint development, a pinned backbone revision, a pre-registered development gate, and one official test evaluation. The final held-out result improved nDCG@10 from `0.686` to `0.804` while reducing irrelevant Top-1 selections by `4.42 percentage points`.

## Limitations to state

- Offline candidate-set reranking, not full retrieval.
- Static relevance labels, not clicks or conversion logs.
- No production latency or deployment claim.
- No causal advertising or marketplace business-impact claim.
