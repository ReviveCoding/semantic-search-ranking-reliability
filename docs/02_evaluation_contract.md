# Evaluation Contract

## Registered ESCI gain map

- Exact: 1.00
- Substitute: 0.10
- Complement: 0.01
- Irrelevant: 0.00

## Metrics

Primary:

- nDCG@10 using the registered gain map

Secondary:

- Top-1 Exact-or-Substitute precision
- Top-1 Irrelevant rate
- Macro F1 for four-class relevance classification
- Support-stratified reliability slices

## Official-test gate

Official test evaluation is authorized only when query-disjoint development results show:

1. higher nDCG@10 than the frozen lexical comparator;
2. higher Top-1 Exact-or-Substitute precision; and
3. no increase in Top-1 Irrelevant rate.