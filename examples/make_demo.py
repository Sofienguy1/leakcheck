"""Generate a house-price dataset with deliberately planted leakage, for demos and tests.

    python examples/make_demo.py   ->  examples/train.csv, examples/test.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd


def make(n: int = 2000, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    size = rng.normal(90, 30, n).clip(20, 300).round()
    rooms = (size / 25 + rng.normal(0, 1, n)).clip(1, 10).round()
    district = rng.choice(["Sentrum", "Fana", "Åsane", "Laksevåg", "Arna"], n)
    premium = pd.Series(district).map({"Sentrum": 1.4, "Fana": 1.2, "Åsane": 1.0, "Laksevåg": 0.95, "Arna": 0.85})
    price = (size * 55_000 * premium + rng.normal(0, 400_000, n)).round(-3)

    df = pd.DataFrame({
        "listing_id": np.arange(100_000, 100_000 + n),                              # leak: unique ID
        "listed_date": pd.date_range("2023-01-01", periods=n, freq="6h").strftime("%Y-%m-%d"),
        "district": district,
        "size_m2": size,
        "rooms": rooms,
        "final_price_eur": (price / 11.5 + rng.normal(0, 50, n)).round(),           # leak: target in disguise
        "price_nok": price,
    })

    # Random split (not by time) -> temporal leakage.
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    train, test = df.iloc[: int(n * 0.8)], df.iloc[int(n * 0.8):].copy()

    copied = train.sample(100, random_state=seed)
    # 60 exact copies of training rows -> duplicate leakage.
    exact = copied.iloc[:60]
    # 40 re-listed copies: new ID, slightly different euro price -> near-duplicate leakage.
    relisted = copied.iloc[60:].assign(
        listing_id=np.arange(900_000, 900_040),
        final_price_eur=lambda d: (d["final_price_eur"] + rng.normal(0, 30, len(d))).round(),
    )
    test = pd.concat([test, exact, relisted], ignore_index=True)
    return train, test


if __name__ == "__main__":
    out = Path(__file__).parent
    train, test = make()
    train.to_csv(out / "train.csv", index=False)
    test.to_csv(out / "test.csv", index=False)
    print(f"wrote {out / 'train.csv'} ({len(train)} rows) and {out / 'test.csv'} ({len(test)} rows)")
