from __future__ import annotations

import argparse
from pathlib import Path
import sys

from swlp.report import print_report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Print a summary report from a SWLP benchmark file.")
    parser.add_argument("path", type=Path, help="Path to a benchmark JSON or CSV file")
    args = parser.parse_args(argv)
    print_report(args.path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
