"""Regenerate the estimator report, optionally using a verified final-DSM supplement."""

import argparse
from pathlib import Path

from recovery_estimators.records import load_results
from recovery_estimators.report import render


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--supplement", type=Path)
    parser.add_argument("--out", type=Path, default=Path("docs/estimators.md"))
    args = parser.parse_args()
    records, manifest, notes = load_results(args.results, args.supplement)
    args.out.write_text(render(records, manifest, notes))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
