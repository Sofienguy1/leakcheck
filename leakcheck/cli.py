"""Command-line entry point: `leakcheck train.csv test.csv --target price`."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from . import __version__
from .checks import FAIL, WARN, run_all
from .report import to_json, to_text


def load(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        sys.exit(f"leakcheck: file not found: {path}")
    suffix = p.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(p)
    if suffix in {".tsv", ".tab"}:
        return pd.read_csv(p, sep="\t")
    return pd.read_csv(p)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="leakcheck",
                                     description="Detect train/test data leakage before you train a model.")
    parser.add_argument("train", help="training data (.csv, .tsv or .parquet)")
    parser.add_argument("test", help="test data (.csv, .tsv or .parquet)")
    parser.add_argument("-t", "--target", help="name of the target (label) column")
    parser.add_argument("--time-col", help="time column; test rows must come after all train rows")
    parser.add_argument("--similarity", type=float, default=0.75, metavar="0-1",
                        help="share of values two rows must share to count as near-duplicates (default: 0.75)")
    parser.add_argument("--json", action="store_true", help="output findings as JSON")
    parser.add_argument("--strict", action="store_true", help="exit with code 1 on warnings too")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    args = parser.parse_args(argv)

    train, test = load(args.train), load(args.test)
    if not 0 < args.similarity <= 1:
        parser.error("--similarity must be between 0 and 1")
    findings = run_all(train, test, target=args.target, time_col=args.time_col, similarity=args.similarity)

    print(to_json(findings) if args.json else to_text(findings, train.shape, test.shape))

    # Non-zero exit lets CI pipelines fail the build when leakage is found.
    bad = {FAIL, WARN} if args.strict else {FAIL}
    return 1 if any(f.status in bad for f in findings) else 0


if __name__ == "__main__":
    raise SystemExit(main())
