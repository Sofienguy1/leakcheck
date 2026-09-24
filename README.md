# leakcheck

[![tests](https://github.com/Sofienguy1/leakcheck/actions/workflows/tests.yml/badge.svg)](https://github.com/Sofienguy1/leakcheck/actions/workflows/tests.yml) [![PyPI](https://img.shields.io/pypi/v/leakcheck-ml)](https://pypi.org/project/leakcheck-ml/) [![Python](https://img.shields.io/pypi/pyversions/leakcheck-ml)](https://pypi.org/project/leakcheck-ml/)

**Catch data leakage before it ruins your model.**

> Machine-learning data leakage, not security leaks: `leakcheck` finds problems in train/test splits that make a model look better than it really is.

A model that scores 99% in testing and fails in production usually has a leak: test rows that were also in training, the same patient on both sides of the split, a feature that secretly contains the answer, or a time series split in the wrong order. `leakcheck` is a linter for your train/test split. Run it before you train.

```
$ leakcheck train.csv test.csv --target price_nok

leakcheck  train: 1,600 rows × 7 cols   test: 500 rows × 7 cols

  ✗ 60 test rows (12.0%) also appear in train
  ✗ 62 test rows (12.4%) are near-duplicates of train rows (≥75% of values match) — about 30 more than the 32 expected by chance
  ✗ Column 'final_price_eur' has correlation 1.000 with target 'price_nok'
  ⚠ Column 'listing_id' looks like an ID (unique per row) — drop it before training
  ⚠ 100.0% of test rows on 'listed_date' are not after the latest train date (2024-05-14)

3 problem(s), 2 warning(s) — likely leakage
```

## Install

```bash
pip install leakcheck-ml
```

The package is `leakcheck-ml` on PyPI; the command and the import are both `leakcheck`.

## Usage

```bash
leakcheck train.csv test.csv --target label             # basic
leakcheck train.parquet test.parquet -t label           # parquet (pip install "leakcheck[parquet]")
leakcheck train.csv test.csv -t label --time-col date   # enforce a time-based split
leakcheck train.csv test.csv -t label -g patient_id     # enforce a group split (no patient on both sides)
leakcheck train.csv test.csv -t label --similarity 0.9 # stricter near-duplicate matching
leakcheck train.csv test.csv -t label --json            # machine-readable output
leakcheck train.csv test.csv -t label --strict          # fail on warnings too
```

The exit code is `1` when problems are found, so you can drop it into CI and block a pipeline that would train on leaky data.

## What it checks

| Check | What it catches | How |
|---|---|---|
| **Duplicates** | Test rows the model already saw in training | Each row is hashed to 64 bits; overlap is a set lookup, so it runs in O(n + m) instead of comparing every pair of rows |
| **Near-duplicates** | Copies that were slightly changed: re-rounded numbers, new IDs, different capitalisation or spacing | MinHash + locality-sensitive hashing find candidate pairs without comparing every row to every other row; candidates are then scored exactly. Numbers are bucketed on two offset grids so tiny changes don't break a match. Only flagged when the count is significantly above a **chance baseline** (train matched against itself, z ≥ 3), because similar rows are normal in low-dimensional data |
| **Group leakage** | The same patient, user or customer in both train and test, so the model learns to recognise the *person* instead of the pattern | Checks which test rows belong to groups already seen in train. Pass `--group-col`, or it auto-detects entity-like columns (`user_id`, `patient`…) whose values repeat. Row numbers that restart in each file are ignored because they don't repeat |
| **Target leakage** | Features that are the target in disguise (e.g. price in another currency, a status set after the outcome) | Pearson correlation for numeric features; *predictive purity* for categorical ones, meaning how much better than the majority-class baseline a feature predicts the target on its own |
| **ID columns** | Unique-per-row identifiers the model can memorise | Uniqueness + dtype + naming heuristics |
| **Temporal leakage** | Randomly split time series, where the model trains on the future | Compares the test dates against the latest training date; date columns are detected automatically |

## Try it

```bash
python examples/make_demo.py       # two demo datasets with planted leaks
leakcheck examples/train.csv examples/test.csv --target price_nok
leakcheck examples/patients_train.csv examples/patients_test.csv --target diagnosis
```

```
  ✗ 176 of 183 'patient_id' groups (263 test rows, 93.9%) also appear in train — split by group instead of by row
```

## Development

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

## Roadmap

- [x] Near-duplicate detection with MinHash + LSH (rows that are *almost* identical)
- [x] Group leakage (same patient/user in both train and test)
- [ ] GitHub Action
- [x] Publish to PyPI
- [ ] HTML report

## License

MIT
