# GPU Cross-Encoder Development Contract

## Model

The development model is a four-class query-product cross-encoder.

- Backbone: `microsoft/deberta-v3-base`
- Runtime input pair:
  - sequence A: query
  - sequence B: product title, brand, color, bullet points
- Labels:
  - Exact: `E`
  - Substitute: `S`
  - Complement: `C`
  - Irrelevant: `I`

## Text boundary

The model cannot receive:

- product ID
- query ID
- ESCI label
- product support frequency
- held-out label aggregates
- behavioral outcomes

Missing catalog fields are omitted. They must never become the literal text token `nan`.

## Training objective

Standard four-class cross-entropy on development-fit rows only.

## Ranking score

For each candidate pair:

`P(E) × 1.00 + P(S) × 0.10 + P(C) × 0.01 + P(I) × 0.00`

## Development validation

- Official ESCI training split only
- Query-disjoint validation partitions 8 and 9
- Frozen lexical comparator receipt is revalidated before semantic scoring
- Official ESCI test rows are not read

## Smoke run

The first run downloads the model at an exact Hub revision, executes a small GPU forward/backward workload, and verifies validation inference.

A smoke result is integration evidence only. It cannot authorize official test evaluation or claim ranking improvement.

## Mixed-Precision Runtime Contract

Development training uses:

- FP32 trainable model parameters
- BF16 autocast on CUDA
- no `GradScaler`
- gradient clipping after backward propagation and before `optimizer.step()`

The driver refuses to train if CUDA BF16 support is unavailable or if any trainable parameter is not FP32.
