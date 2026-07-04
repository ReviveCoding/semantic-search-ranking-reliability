# Research Contract

## Primary hypothesis

A GPU semantic cross-encoder can rank query-product candidates more accurately than a frozen lexical comparator on held-out queries.

## Development protocol

- Only official ESCI US Task 1 training rows may be used for model development.
- Every candidate belonging to a query remains in one development partition.
- Model architecture, input fields, epochs, optimization settings, and gain mapping are frozen before official test evaluation.
- Official test rows are used once after the development freeze.

## Allowed runtime inputs

- query
- product title
- product brand
- product color
- product bullet point

## Forbidden runtime inputs

- ESCI labels
- product IDs
- query IDs
- support-frequency features
- test aggregates
- post-ranking behavior data

## Claim boundary

This project measures static offline relevance. It does not claim deployment, user behavior impact, click-through-rate lift, conversion lift, or advertising causality.