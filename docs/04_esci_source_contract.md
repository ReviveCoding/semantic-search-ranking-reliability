# ESCI Source Contract

## Source

The only admitted raw data source is the official
`amazon-science/esci-data` repository.

## Required raw files

- `shopping_queries_dataset_examples.parquet`
- `shopping_queries_dataset_products.parquet`
- `shopping_queries_dataset_sources.csv`

## Intake requirements

The source receipt must record:

- source repository commit
- SHA-256 of every admitted raw file
- file size
- required schema columns
- US Task 1 train/test counts
- ESCI label distribution

## Boundary

The raw checkout is ignored by Git. The repository stores only
code, contracts, tests, and reproducible receipts.

No prior-project artifact, transformed label, model output, or
threshold may become an input to this project.