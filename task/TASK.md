# Task: `orders_dedupe` CLI

Build a Python 3.11 command-line tool that reads a CSV of orders and flags duplicate orders.
Runtime code uses the standard library only. Write unit tests with pytest in `tests/`.

## Layout

- The package lives in `orders_dedupe/` at the project root (not under `src/`), so that
  `python -m orders_dedupe` works when run from the project root.
- Tests live in `tests/` and must pass with `python -m pytest` from the project root.

## Input

A UTF-8 CSV file with a header row. Required columns: `order_id`, `customer_email`, `sku`,
`quantity`, `order_date`. Any other columns are allowed and must be preserved.

## Duplicate rules

Process rows in file order. A row is a duplicate of the earliest previous row it matches, where
two rows match if either:

1. their `order_id` values are equal after stripping surrounding whitespace; or
2. all of the following are equal:
   - `customer_email`, compared case-insensitively after stripping whitespace
   - `sku`, compared case-insensitively after stripping whitespace
   - `quantity`, compared as integers (`"2"`, `" 2 "` and `"02"` are equal)
   - `order_date`, compared after stripping whitespace

The first occurrence is never a duplicate.

## CLI

```
python -m orders_dedupe INPUT_CSV [--output OUTPUT_CSV]
```

- Writes a CSV (to stdout, or to `OUTPUT_CSV` if given) containing every input row in the
  original order, with all original columns in their original order, followed by two new columns:
  - `is_duplicate`: `true` or `false`
  - `duplicate_of`: the `order_id` (stripped) of the row it duplicates, or empty
- Prints a summary line to stderr exactly in the form `N rows, M duplicates`.
- Exit code 0 on success.
- If a required column is missing: print an error to stderr containing `missing column` and the
  column name, and exit with code 2.
- If the input file does not exist: print an error to stderr and exit with code 2.
- A file with only a header row is valid: output the header only, summary `0 rows, 0 duplicates`.
