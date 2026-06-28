# BEANS-CBI Splits

These files define one fixed low-resource BEANS-CBI trial for the release.
Training subsets are class-stratified subsets of the official BEANS-CBI
training split at 10%, 25%, and 50% label budgets. The official validation and
test splits are kept unchanged.

The CSV files store parquet shard names, row-group indices, sample IDs, labels,
and label indices. They do not contain raw audio.
