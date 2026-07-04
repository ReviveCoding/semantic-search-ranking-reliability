# Development Dataset and Frozen Lexical Comparator

## Development data boundary

Only official ESCI US Task 1 rows with `split == train` are used.

Official test rows are not read by this development dataset builder or lexical comparator.

## Query-disjoint split

Each `query_id` receives a stable SHA-256-derived partition value from 0 through 9.

- Partitions 0 through 7: development fit
- Partitions 8 and 9: development validation

Every candidate row for one query belongs to exactly one partition.

## Runtime text fields

The future semantic model may use:

- query
- product title
- product brand
- product color
- product bullet point

The frozen lexical comparator uses only query text and product title.

## Frozen lexical comparator

For each validation query, rank candidate products by:

1. descending unique query-token coverage in product title;
2. ascending product ID for deterministic score ties.

Lexical score:

`|unique(query tokens) ∩ unique(title tokens)| / |unique(query tokens)|`

The comparator uses no ESCI label, product frequency, query ID, product ID feature, test aggregate, or behavioral data.

## Reported metrics

- nDCG@10 using registered ESCI gains
- Top-1 Exact-or-Substitute precision
- Top-1 Irrelevant rate
- selected Top-1 label distribution

The comparator is frozen after this receipt and becomes the required development baseline for GPU semantic training.