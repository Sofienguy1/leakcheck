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
    findings = []
    for col in _shared_feature_columns(train, test, target):
        s = train[col]
        name_hint = col.lower() in {"id", "index", "uuid", "key"} or col.lower().endswith(("_id", "id"))
        unique = len(s) > 0 and s.nunique() == len(s)
        integer_or_text = pd.api.types.is_integer_dtype(s) or pd.api.types.is_string_dtype(s) or s.dtype == object
        if unique and integer_or_text and (name_hint or len(s) >= 50):
            findings.append(Finding("id-columns", WARN,
                                    f"Column '{col}' looks like an ID (unique per row) — drop it before training",
                                    {"column": col}))
    if not findings:
        findings.append(Finding("id-columns", PASS, "No ID-like columns found"))
    return findings


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
            time_col: str | None = None) -> list[Finding]:
    return [
        *check_duplicates(train, test, target),
        *check_target_leakage(train, target),
        *check_id_columns(train, test, target),
        *check_temporal(train, test, time_col),
    ]
