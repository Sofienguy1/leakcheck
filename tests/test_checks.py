import sys
from pathlib import Path

import numpy as np
import pandas as pd

from leakcheck.checks import (FAIL, PASS, WARN, check_duplicates, check_id_columns, check_near_duplicates,
                              check_target_leakage, check_temporal)
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


def wide_data(n=2000, seed=2):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({f"f{i}": rng.normal(size=n) for i in range(6)})
    df["city"] = rng.choice(["Bergen", "Oslo", "Trondheim"], n)
    return df.iloc[: int(n * 0.8)], df.iloc[int(n * 0.8):]


def test_clean_data_has_no_near_duplicates():
    train, test = wide_data()
    assert statuses(check_near_duplicates(train, test)) == {PASS}


def test_detects_slightly_changed_copies():
    train, test = wide_data()
    copies = train.sample(50, random_state=0).copy()
    copies["f0"] = copies["f0"] + 0.0001        # tiny numeric change
    copies["city"] = " " + copies["city"].str.upper() + " "  # formatting change
    copies["f5"] = 99.0                          # one column completely different
    [f] = check_near_duplicates(train, pd.concat([test, copies]))
    assert f.status == FAIL and f.details["count"] == 50


def test_near_duplicates_ignore_ids_and_skip_exact_copies():
    train, test = wide_data()
    train = train.assign(row_id=range(len(train)))
    test = test.assign(row_id=range(10_000, 10_000 + len(test)))
    exact = train.head(10)
    new_id = train.iloc[10:30].assign(row_id=range(50_000, 50_020))
    [f] = check_near_duplicates(train, pd.concat([test, exact, new_id]))
    assert f.details["count"] == 20  # exact copies belong to the duplicates check
    assert f.details["ignored_id_columns"] == ["row_id"]


def test_near_duplicates_scale_to_large_data():
    import time
    train, test = wide_data(n=200_000)
    start = time.perf_counter()
    check_near_duplicates(train, test)
    assert time.perf_counter() - start < 30


def test_demo_catches_relisted_rows():
    train, test = make()
    [f] = check_near_duplicates(train, test, "price_nok")
    # 40 planted re-listings on top of a few coincidental look-alikes; flagged because it beats the baseline.
    assert f.status == FAIL
    assert f.details["count"] >= 40
    assert f.details["z_score"] >= 3  # far more look-alikes than chance explains


def test_lookalikes_at_chance_level_pass():
    # Low-dimensional data where many rows naturally resemble each other: no leak, so no alarm.
    rng = np.random.default_rng(3)
    df = pd.DataFrame({"sex": rng.choice(["m", "f"], 3000), "cls": rng.integers(1, 4, 3000),
                       "port": rng.choice(["S", "C", "Q"], 3000), "sibsp": rng.integers(0, 3, 3000)})
    [f] = check_near_duplicates(df.iloc[:2400], df.iloc[2400:])
    assert f.status == PASS
