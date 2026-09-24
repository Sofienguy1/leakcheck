# leakcheck

**Catch data leakage before it ruins your model.**

A model that scores 99% in testing and fails in production usually has a leak: test rows that were also in training, a feature that secretly contains the answer, or a time series split in the wrong order. `leakcheck` is a linter for your train/test split. Run it before you train.

```
$ leakcheck train.csv test.csv --target price_nok

leakcheck  train: 1,600 rows × 7 cols   test: 460 rows × 7 cols

  ✗ 60 test rows (13.0%) also appear in train
  ✗ Column 'final_price_eur' has correlation 1.000 with target 'price_nok'
  ⚠ Column 'listing_id' looks like an ID (unique per row) — drop it before training
  ⚠ 100.0% of test rows on 'listed_date' are not after the latest train date (2024-05-14)

2 problem(s), 2 warning(s) — likely leakage
```

## Install

```bash
pip install git+https://github.com/Sofienguy1/leakcheck
```

## Usage

```bash
leakcheck train.csv test.csv --target label             # basic
leakcheck train.parquet test.parquet -t label           # parquet (pip install "leakcheck[parquet]")
leakcheck train.csv test.csv -t label --time-col date   # enforce a time-based split
leakcheck train.csv test.csv -t label --json            # machine-readable output
leakcheck train.csv test.csv -t label --strict          # fail on warnings too
```

The exit code is `1` when problems are found, so you can drop it into CI and block a pipeline that would train on leaky data.

## What it checks

| Check | What it catches | How |
|---|---|---|
| **Duplicates** | Test rows the model already saw in training | Each row is hashed to 64 bits; overlap is a set lookup, so it runs in O(n + m) instead of comparing every pair of rows |
| **Target leakage** | Features that are the target in disguise (e.g. price in another currency, a status set after the outcome) | Pearson correlation for numeric features; *predictive purity* for categorical ones, meaning how much better than the majority-class baseline a feature predicts the target on its own |
| **ID columns** | Unique-per-row identifiers the model can memorise | Uniqueness + dtype + naming heuristics |
| **Temporal leakage** | Randomly split time series, where the model trains on the future | Compares the test dates against the latest training date; date columns are detected automatically |

## Try it

```bash
python examples/make_demo.py       # house-price data with 4 planted leaks
leakcheck examples/train.csv examples/test.csv --target price_nok
```

## Development

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

## Roadmap

- [ ] Near-duplicate detection with MinHash + LSH (rows that are *almost* identical)
- [ ] Group leakage (same patient/user in both train and test)
- [ ] GitHub Action
- [ ] Publish to PyPI
- [ ] HTML report

## License

MIT
