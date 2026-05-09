"""Data preparation dispatcher — routes to synthetic or CWRU based on params."""

import argparse
import sys
import subprocess
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    parser = argparse.ArgumentParser(description="Prepare data for the pipeline")
    parser.add_argument("--output", default="data/raw", help="Output directory")
    parser.add_argument("--params", default="params.yaml", help="Params file")
    args = parser.parse_args()

    with open(args.params) as f:
        params = yaml.safe_load(f)

    data_source = params.get("data_source", {}).get("type", "synthetic")

    if data_source == "synthetic":
        print("Using synthetic data source")
        result = subprocess.run(
            [sys.executable, "src/generate_synthetic.py", "--output", args.output],
            check=True,
        )
    elif data_source == "cwru":
        print("Using CWRU Bearing Dataset")
        result = subprocess.run(
            [sys.executable, "src/download_cwru.py", "--output", args.output],
            check=True,
        )
    else:
        raise ValueError(f"Unknown data_source.type: '{data_source}'. Use 'synthetic' or 'cwru'.")


if __name__ == "__main__":
    main()