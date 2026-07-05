# Read-Only Reliability Report Contract

## Purpose

This report documents the already-completed one-shot official ESCI
test evaluation. It is not a new model experiment.

## Prohibited actions

The report generator must not:

- load or train a model;
- alter model weights;
- alter input fields, epochs, learning rate, loss, or ranking gain map;
- evaluate a second official test run;
- use test metrics for model selection;
- create an additional official-test evaluation receipt.

## Permitted inputs

The report generator may read:

- the official source-contract receipt;
- the frozen lexical baseline receipt;
- the development metrics receipt;
- the one-shot official-test receipt;
- already-written lexical and semantic Top-1 CSV outputs;
- official ESCI training rows only for train-only product-support diagnostics.

## Support slice definition

Support slices are anchored to the frozen lexical comparator's selected
Top-1 product. Product support is the count of that product's appearances
in official ESCI US Task 1 training candidate rows.

This is a post-hoc reliability diagnostic only. Support is not a model
feature and is not used to select or tune any model.

## Outputs

- final reliability report in Markdown;
- machine-readable report summary;
- development-versus-test consistency table;
- frozen-lexical-anchored support slice table;
- Top-1 semantic-versus-lexical change taxonomy;
- deterministic representative examples;
- SHA-256 manifest of generated report artifacts.

## Claim boundary

The report supports an offline candidate-set semantic reranking claim on
the official ESCI relevance benchmark. It does not support production,
CTR, conversion, causal advertising, or end-to-end retrieval claims.
