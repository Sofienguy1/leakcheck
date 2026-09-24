"""Leakage checks. Each check takes train/test DataFrames and returns a list of Findings."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

PASS, WARN, FAIL = "pass", "warn", "fail"


@dataclass
class Finding:
    check: str
    status: str  # pass | warn | fail
    message: str
    details: dict = field(default_factory=dict)


def _shared_feature_columns(train: pd.DataFrame, test: pd.DataFrame, target: str | None) -> list[str]:
    return [c for c in train.columns if c in test.columns and c != target]


def _row_hashes(df: pd.DataFrame) -> pd.Series:
    # One 64-bit hash per row, so comparing rows becomes a set lookup: O(n + m) instead of O(n * m).
    return pd.util.hash_pandas_object(df, index=False)


def check_duplicates(train: pd.DataFrame, test: pd.DataFrame, target: str | None = None) -> list[Finding]:
    """Rows in the test set that also appear (exactly) in the training set."""
    cols = _shared_feature_columns(train, test, target)
    if not cols:
        return [Finding("duplicates", WARN, "No shared feature columns between train and test")]

    train_hashes = set(_row_hashes(train[cols]))
    in_train = _row_hashes(test[cols]).isin(train_hashes)
    n, pct = int(in_train.sum()), float(in_train.mean() * 100) if len(test) else 0.0

    if n == 0:
        return [Finding("duplicates", PASS, "No test rows are duplicated in train")]
    status = FAIL if pct >= 1 else WARN
    return [Finding(
        "duplicates", status,
        f"{n} test rows ({pct:.1f}%) also appear in train",
        {"count": n, "percent": round(pct, 2), "example_test_rows": test.index[in_train][:5].tolist()},
    )]


def check_target_leakage(train: pd.DataFrame, target: str | None, fail_at: float = 0.97,
                         warn_at: float = 0.90) -> list[Finding]:
    """Features that are suspiciously good predictors of the target on their own.

    Strong real features often reach 0.8+; leaked ones are near-perfect, so thresholds sit high
    to avoid flagging legitimate predictors."""
    if not target:
        return []
    if target not in train.columns:
        return [Finding("target-leakage", FAIL, f"Target column '{target}' not found in train")]

    y = train[target]
    findings = []
    for col in train.columns:
        if col == target:
            continue
        x = train[col]

        if x.equals(y):
            findings.append(Finding("target-leakage", FAIL, f"Column '{col}' is an exact copy of the target",
                                    {"column": col}))
            continue

        score, kind = _association(x, y)
        if score is None:
            continue
        if score >= fail_at:
            status = FAIL
        elif score >= warn_at:
            status = WARN
        else:
            continue
        findings.append(Finding("target-leakage", status,
                                f"Column '{col}' has {kind} {score:.3f} with target '{target}'",
                                {"column": col, "score": round(score, 4), "measure": kind}))

    if not findings:
        findings.append(Finding("target-leakage", PASS, "No features look like proxies for the target"))
    return findings


def _association(x: pd.Series, y: pd.Series) -> tuple[float | None, str]:
    """How well x predicts y, in [0, 1]. Pearson for numeric pairs, otherwise 'purity':
    the share of rows whose target is the majority target for their x-value."""
    both = pd.DataFrame({"x": x.to_numpy(), "y": y.to_numpy()}).dropna()
    if len(both) < 10:
        return None, ""
    x, y = both["x"], both["y"]

    if pd.api.types.is_numeric_dtype(x) and pd.api.types.is_numeric_dtype(y) and y.nunique() > 2:
        if x.std() == 0 or y.std() == 0:
            return None, ""
        return abs(float(np.corrcoef(x, y)[0, 1])), "correlation"

    # Purity is trivially 1.0 when every x-value is unique (e.g. IDs) — that's the ID check's job.
    if x.nunique() > 0.5 * len(x) or y.nunique() > 50:
        return None, ""
    majority = both.groupby("x")["y"].agg(lambda s: s.value_counts().iloc[0])
    baseline = y.value_counts().iloc[0] / len(y)
    purity = majority.sum() / len(y)
    if baseline >= 0.99:
        return None, ""
    # Rescale so "no better than always guessing the majority class" is 0.
    return max(0.0, (purity - baseline) / (1 - baseline)), "predictive purity"


def check_id_columns(train: pd.DataFrame, test: pd.DataFrame, target: str | None = None) -> list[Finding]:
    """Columns that look like row identifiers. Models can memorise these instead of learning."""
    findings = [Finding("id-columns", WARN,
                        f"Column '{col}' looks like an ID (unique per row) — drop it before training",
                        {"column": col})
                for col in _id_like_columns(train, test, target)]
    if not findings:
        findings.append(Finding("id-columns", PASS, "No ID-like columns found"))
    return findings


def _id_like_columns(train: pd.DataFrame, test: pd.DataFrame, target: str | None) -> list[str]:
    found = []
    for col in _shared_feature_columns(train, test, target):
        s = train[col]
        name_hint = col.lower() in {"id", "index", "uuid", "key"} or col.lower().endswith(("_id", "id"))
        unique = len(s) > 0 and s.nunique() == len(s)
        integer_or_text = pd.api.types.is_integer_dtype(s) or pd.api.types.is_string_dtype(s) or s.dtype == object
        if unique and integer_or_text and (name_hint or len(s) >= 50):
            found.append(col)
    return found


# --- Near-duplicates: MinHash + locality-sensitive hashing ---------------------------------------------------
#
# Each row becomes a set of tokens like "size_m2≈bin 41" or "district=fana". Two rows' similarity is the
# Jaccard index of their token sets: |A ∩ B| / |A ∪ B|. Comparing every test row to every train row is
# O(n · m), too slow for big data. Instead:
#   1. MinHash compresses each row to a signature of P numbers. For a random hash function h, the chance that
#      min(h(A)) == min(h(B)) is exactly Jaccard(A, B), so the share of matching signature slots estimates it.
#   2. LSH splits signatures into bands. Rows that match on a whole band land in the same bucket and become
#      candidates, so only a handful of likely pairs are ever compared.
#   3. Candidates are then scored exactly, so the MinHash estimate's noise never decides the verdict.

_PRIME = (1 << 31) - 1  # Mersenne prime; a * x stays below 2^62, so uint64 arithmetic never overflows
_NUM_PERM, _BANDS = 128, 32  # 32 bands × 4 rows: pairs above ~0.45 similarity almost always become candidates


def _is_number(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)


def _tokenize(ref: pd.DataFrame, query: pd.DataFrame, cols: list[str],
              scale: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """(rows × tokens) arrays of token hashes. Text is trimmed and lower-cased. Numbers are bucketed so tiny
    changes map to the same token — on two grids offset by half a bucket, so a value sitting on a bucket edge
    loses at most one of its two tokens. Buckets are sized from `scale` so both sets share one scale."""
    out = []
    for df in (ref, query):
        slots = []
        for col in cols:
            s = df[col]
            if _is_number(s):
                std = scale[col].std()
                width = std * 0.02 if pd.notna(std) and std > 0 else 1.0
                x = s.to_numpy(dtype=float) / width
                slots += [np.floor(x), np.floor(x + 0.5)]
            else:
                slots.append(s.astype(str).str.strip().str.lower().to_numpy(dtype=object))
        tokens = np.empty((len(df), len(slots)), dtype=np.uint64)
        for j, values in enumerate(slots):
            # Mix in the slot position so "rooms=3" and "floor=3" are different tokens.
            tokens[:, j] = pd.util.hash_array(values) ^ np.uint64((j + 1) * 0x9E3779B97F4A7C15 % 2**64)
        out.append(tokens % np.uint64(_PRIME))
    return out[0], out[1]


def _minhash(tokens: np.ndarray, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    a = rng.integers(1, _PRIME, _NUM_PERM, dtype=np.uint64)
    b = rng.integers(0, _PRIME, _NUM_PERM, dtype=np.uint64)
    sig = np.empty((len(tokens), _NUM_PERM), dtype=np.uint64)
    for i in range(_NUM_PERM):
        sig[:, i] = ((a[i] * tokens + b[i]) % np.uint64(_PRIME)).min(axis=1)
    return sig


def _candidate_pairs(sig_train: np.ndarray, sig_test: np.ndarray) -> np.ndarray:
    """(test_row, train_row) pairs that share at least one LSH bucket."""
    rows = _NUM_PERM // _BANDS
    pairs = []
    for band in range(_BANDS):
        cols = slice(band * rows, (band + 1) * rows)
        train_keys = pd.util.hash_pandas_object(pd.DataFrame(sig_train[:, cols]), index=False).to_numpy()
        test_keys = pd.util.hash_pandas_object(pd.DataFrame(sig_test[:, cols]), index=False).to_numpy()
        # One representative train row per bucket keeps the candidate count linear in the test size.
        bucket_keys, first_row = np.unique(train_keys, return_index=True)
        pos = np.searchsorted(bucket_keys, test_keys).clip(max=len(bucket_keys) - 1)
        hit = np.nonzero(bucket_keys[pos] == test_keys)[0]
        pairs.append(np.column_stack([hit, first_row[pos[hit]]]))
    return np.unique(np.vstack(pairs), axis=0) if pairs else np.empty((0, 2), dtype=int)


def check_near_duplicates(train: pd.DataFrame, test: pd.DataFrame, target: str | None = None,
                          threshold: float = 0.75) -> list[Finding]:
    """Test rows that are almost the same as a train row: rounded numbers, new IDs, changed whitespace.

    `threshold` is the share of a row's values (tokens) that must match."""
    ids = set(_id_like_columns(train, test, target))
    cols = [c for c in _shared_feature_columns(train, test, target) if c not in ids]
    if len(cols) < 3 or train.empty or test.empty:
        return []  # with only a couple of columns, "similar rows" is normal, not suspicious

    shared = _shared_feature_columns(train, test, target)
    best = _near_duplicate_matches(train, test, cols, shared, threshold, scale=train)
    n = len(best)
    pct = n / len(test) * 100
    chance = _chance_rate(train, cols, shared, threshold) * 100
    details = {"count": n, "percent": round(pct, 2), "expected_by_chance_percent": round(chance, 2),
               "ignored_id_columns": sorted(ids),
               "examples": [{"test_row": test.index[r.test], "train_row": train.index[r.train],
                             "similarity": round(float(r.similarity), 2)} for r in best.head(5).itertuples()]}

    if n == 0:
        return [Finding("near-duplicates", PASS, "No near-duplicate rows between train and test", details)]
    # Similar rows are normal in low-dimensional data, so only flag counts that chance can't explain.
    # Under the baseline, the count is ~Binomial(m, p): z measures how many standard deviations above it we are.
    m, p = len(test), chance / 100
    expected = m * p
    z = (n - expected) / np.sqrt(max(m * p * (1 - p), 1.0))
    details |= {"expected_by_chance": round(expected, 1), "z_score": round(float(z), 1)}
    excess = pct - chance
    if z < 3 or excess < 0.5:
        return [Finding("near-duplicates", PASS,
                        f"{n} test rows ({pct:.1f}%) have near-duplicates in train, in line with the "
                        f"{chance:.1f}% expected by chance for this data", details)]
    return [Finding(
        "near-duplicates", FAIL if excess >= 5 else WARN,
        f"{n} test rows ({pct:.1f}%) are near-duplicates of train rows (≥{threshold:.0%} of values match) "
        f"— about {n - expected:.0f} more than the {expected:.0f} expected by chance",
        details,
    )]


def _near_duplicate_matches(ref: pd.DataFrame, query: pd.DataFrame, cols: list[str], exact_cols: list[str],
                            threshold: float, scale: pd.DataFrame) -> pd.DataFrame:
    """Best match in `ref` for each `query` row at or above `threshold`, compared on `cols`: columns test, train,
    similarity (positional row numbers). Rows identical on `exact_cols` are left to the duplicates check."""
    tok_ref, tok_query = _tokenize(ref, query, cols, scale)
    sig_ref, sig_query = _minhash(tok_ref), _minhash(tok_query)
    pairs = _candidate_pairs(sig_ref, sig_query)
    empty = pd.DataFrame({"test": [], "train": [], "similarity": []})
    if len(pairs) == 0:
        return empty

    # LSH only proposes candidates; score them exactly. Tokens are position-mixed, so a slot can only ever
    # match the same slot in the other row, and the share of equal slots is the share of matching values.
    similarity = (tok_query[pairs[:, 0]] == tok_ref[pairs[:, 1]]).mean(axis=1)
    matches = pd.DataFrame({"test": pairs[:, 0], "train": pairs[:, 1], "similarity": similarity})
    matches = matches[matches["similarity"] >= threshold]

    # Exact copies are already reported by the duplicates check.
    exact = _row_hashes(query[exact_cols]).isin(set(_row_hashes(ref[exact_cols]))).to_numpy()
    matches = matches[~exact[matches["test"].to_numpy()]]
    return matches.sort_values("similarity", ascending=False).drop_duplicates("test")


def _chance_rate(train: pd.DataFrame, cols: list[str], exact_cols: list[str], threshold: float,
                 seed: int = 0) -> float:
    """Share of rows expected to have a near-duplicate in train by coincidence, measured by matching one random
    half of train against the other. Doubled because the real reference (all of train) is twice as large."""
    if len(train) < 100:
        return 0.0
    order = np.random.default_rng(seed).permutation(len(train))
    half_a, half_b = train.iloc[order[: len(train) // 2]], train.iloc[order[len(train) // 2:]]
    rate = len(_near_duplicate_matches(half_a, half_b, cols, exact_cols, threshold, scale=train)) / len(half_b)
    return min(1.0, 2 * rate)


def check_temporal(train: pd.DataFrame, test: pd.DataFrame, time_col: str | None = None) -> list[Finding]:
    """For time-based data the test set should come strictly after the training set."""
    cols = [time_col] if time_col else _detect_time_columns(train, test)
    findings = []
    for col in cols:
        if col not in train.columns or col not in test.columns:
            findings.append(Finding("temporal", FAIL, f"Time column '{col}' missing from train or test"))
            continue
        tr = pd.to_datetime(train[col], errors="coerce").dropna()
        te = pd.to_datetime(test[col], errors="coerce").dropna()
        if tr.empty or te.empty:
            continue
        overlap = float((te <= tr.max()).mean() * 100)
        if overlap == 0:
            findings.append(Finding("temporal", PASS, f"Test set is strictly after train on '{col}'"))
        else:
            findings.append(Finding(
                "temporal", FAIL if time_col else WARN,
                f"{overlap:.1f}% of test rows on '{col}' are not after the latest train date ({tr.max().date()})",
                {"column": col, "percent_overlap": round(overlap, 2),
                 "train_max": str(tr.max()), "test_min": str(te.min())},
            ))
    return findings


def _detect_time_columns(train: pd.DataFrame, test: pd.DataFrame) -> list[str]:
    found = []
    for col in train.columns:
        if col not in test.columns:
            continue
        s = train[col]
        if pd.api.types.is_datetime64_any_dtype(s):
            found.append(col)
        elif (pd.api.types.is_string_dtype(s) or s.dtype == object) and any(
                k in col.lower() for k in ("date", "time", "timestamp")):
            if pd.to_datetime(s.head(50), errors="coerce").notna().mean() > 0.9:
                found.append(col)
    return found


def run_all(train: pd.DataFrame, test: pd.DataFrame, target: str | None = None,
            time_col: str | None = None, similarity: float = 0.75) -> list[Finding]:
    return [
        *check_duplicates(train, test, target),
        *check_near_duplicates(train, test, target, similarity),
        *check_target_leakage(train, target),
        *check_id_columns(train, test, target),
        *check_temporal(train, test, time_col),
    ]
