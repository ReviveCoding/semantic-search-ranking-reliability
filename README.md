# Reliability-First Semantic Search Ranking

GPU-accelerated semantic query-product reranking on the official Amazon Shopping Queries ESCI benchmark.

## Research question

Can a four-class semantic query-product cross-encoder improve held-out ranking quality and reduce irrelevant Top-1 selections relative to a frozen lexical comparator?

## Scope

- Existing candidate-set reranking, not end-to-end retrieval.
- Official ESCI US Task 1 relevance benchmark.
- Query-disjoint development evaluation.
- One official held-out test evaluation after development decisions are frozen.

## Model contract

- Backbone: `microsoft/deberta-v3-base`
- Input: query paired with product title, brand, color, and bullet points
- Labels: Exact, Substitute, Complement, Irrelevant
- Ranking score: expected ESCI gain from class probabilities

## Reliability boundaries

- No production-click, conversion, CTR, or causal ads claims.
- No product ID, query ID, support frequency, or held-out label used as model features.
- Support is used only for post-hoc robustness slices.
- No model ensemble, pseudo-labeling, or test-set model selection.