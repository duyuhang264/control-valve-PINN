from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from pgm3f.data import build_manifest, summarize_manifest


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build DAMADICS manifest and data quality report")
    p.add_argument("--data-root", required=True)
    p.add_argument("--output-dir", default="outputs")
    p.add_argument("--default-duration-s", type=float, default=200.0)
    p.add_argument("--expected-faults", type=int, default=20)
    p.add_argument("--expected-conditions", type=int, default=10)
    p.add_argument("--expected-openings", type=int, default=20)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    quality_path = out / "data_quality_report.csv"
    manifest_path = out / "manifest.csv"

    manifest = build_manifest(
        args.data_root,
        report_path=str(quality_path),
        default_duration_s=args.default_duration_s,
        expected_faults=args.expected_faults,
        expected_conditions=args.expected_conditions,
        expected_openings=args.expected_openings,
    )
    manifest.to_csv(manifest_path, index=False, encoding="utf-8")

    print("Manifest:", manifest_path)
    print("Quality report:", quality_path)
    print("Summary:", summarize_manifest(manifest))


if __name__ == "__main__":
    main()
