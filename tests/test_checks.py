import sys
from pathlib import Path

import numpy as np
import pandas as pd

from leakcheck.checks import (FAIL, PASS, WARN, check_duplicates, check_id_columns, check_target_leakage,
                              check_temporal)
from leakcheck.cli import main

sys.path.insert(0, str(Path(__file__).parents[1] / "examples"))
from make_demo import make  # noqa: E402


def clean_data(n=500, seed=1):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"a": rng.normal(size=n), "b": rng.integers(0, 5, n), "y": rng.normal(size=n)})
    return df.iloc[:400], df.iloc[400:]


def statuses(findings):
    return {f.status for f in findings}


def test_no_duplicates_in_clean_split():
    train, test = clean_data()
    assert statuses(check_duplicates(train, test, "y")) == {PASS}


def test_detects_duplicates():
    train, test = clean_data()
    test = pd.concat([test, train.head(20)])
    [f] = check_duplicates(train, test, "y")
    assert f.status == FAIL and f.details["count"] == 20


def test_duplicates_ignore_target_column():
    # Same features, different label: still the same input the model already saw.
    train, test = clean_data()
    dup = train.head(5).assign(y=999.0)
    [f] = check_duplicates(train, pd.concat([test, dup]), "y")
    assert f.details["count"] == 5


def test_detects_correlated_proxy():
    train, _ = clean_data()
    train = train.assign(leak=train["y"] * 3 + 0.001)
    found = check_target_leakage(train, "y")
    assert any(f.status == FAIL and f.details["column"] == "leak" for f in found)


def test_detects_categorical_proxy():
    rng = np.random.default_rng(0)
    label = rng.choice(["yes", "no"], 300)
    train = pd.DataFrame({"label": label, "status_code": np.where(label == "yes", "A", "B"),
                          "noise": rng.choice(["x", "y", "z"], 300)})
    found = {f.details.get("column"): f.status for f in check_target_leakage(train, "label")}
    assert found.get("status_code") == FAIL
    assert "noise" not in found


def test_clean_features_pass_target_check():
    train, _ = clean_data()
    assert statuses(check_target_leakage(train, "y")) == {PASS}


def test_id_column_warns():
    train, test = clean_data()
    train = train.assign(user_id=range(len(train)))
    test = test.assign(user_id=range(1000, 1000 + len(test)))
    assert any(f.status == WARN and f.details["column"] == "user_id" for f in check_id_columns(train, test, "y"))


def test_temporal_split_ok_and_overlap():
    dates = pd.date_range("2024-01-01", periods=100).strftime("%Y-%m-%d")
    df = pd.DataFrame({"date": dates, "x": range(100)})
    assert statuses(check_temporal(df.iloc[:80], df.iloc[80:], "date")) == {PASS}
    shuffled = df.sample(frac=1, random_state=0)
    assert statuses(check_temporal(shuffled.iloc[:80], shuffled.iloc[80:], "date")) == {FAIL}


def test_demo_dataset_catches_every_planted_leak():
    train, test = make()
    msgs = " ".join(f.message for f in [*check_duplicates(train, test, "price_nok"),
                                        *check_target_leakage(train, "price_nok"),
                                        *check_id_columns(train, test, "price_nok"),
                                        *check_temporal(train, test)])
    assert "60 test rows" in msgs
    assert "final_price_eur" in msgs
    assert "listing_id" in msgs
    assert "listed_date" in msgs
    # Legit features must not be flagged as target leakage.
    leaks = [f.details["column"] for f in check_target_leakage(train, "price_nok") if f.status != PASS]
    assert leaks == ["final_price_eur"]


def test_cli_exit_codes(tmp_path):
    train, test = clean_data()
    train.to_csv(tmp_path / "train.csv", index=False)
    test.to_csv(tmp_path / "test.csv", index=False)
    assert main([str(tmp_path / "train.csv"), str(tmp_path / "test.csv"), "-t", "y"]) == 0

    pd.concat([test, train.head(30)]).to_csv(tmp_path / "leaky.csv", index=False)
    assert main([str(tmp_path / "train.csv"), str(tmp_path / "leaky.csv"), "-t", "y"]) == 1
