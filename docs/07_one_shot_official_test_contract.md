# One-Shot Official Test Evaluation Contract

## Authorization

The official test evaluation is authorized only because the frozen
query-disjoint development run passed all pre-registered gates:

1. nDCG@10 improvement confidence interval lower bound above zero.
2. Top-1 Exact-or-Substitute improvement confidence interval lower bound above zero.
3. Top-1 Irrelevant-rate confidence interval upper bound at or below zero.

## Final model

- Backbone: `microsoft/deberta-v3-base`
- Revision: `8ccc9b6f36199bec6961081d44eb72fb3f7353f3`
- Runtime inputs:
  - query
  - product title
  - product brand
  - product color
  - product bullet point
- Loss: four-class cross entropy
- Training rows: all official ESCI US Task 1 train rows
- Epochs: pre-registered `final_epochs: 2`
- Mixed precision: BF16 autocast with FP32 trainable weights

## Evaluation

The frozen lexical comparator and final semantic model are both evaluated
once on official ESCI US Task 1 test rows.

Test labels are used only to calculate metrics after rankings are produced.
They are not passed to the model, tokenizer, optimization loop, or ranking score.

## Metrics

- nDCG@10 with registered ESCI gains
- Top-1 Exact-or-Substitute precision
- Top-1 Irrelevant rate
- Four-class macro F1
- paired query-level bootstrap confidence intervals

## One-shot rule

This repository refuses to create a second
`official_test_evaluation_receipt.json`.

No feature, parameter, model, epoch count, or score-map change may be made
after this receipt is created.