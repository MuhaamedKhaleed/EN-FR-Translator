"""Explicit tiny real English-French CPU training, never import-time execution."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Prepared pairs.json")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import torch

    from enfr_translator.model import train

    torch.set_num_threads(1)
    rows = json.loads(args.data.read_text())
    if len(rows) != 12 or any(not r.get("credit") for r in rows):
        raise ValueError("Expected12 attributed pairs from prepare_french.py")
    pairs = [(r["source"], r["target"]) for r in rows]
    scores = train(
        args.model,
        pairs[:8],
        pairs[8:],
        args.output,
        epochs=1,
        batch_size=1,
        max_length=16,
    )
    (args.output / "source-attribution.json").write_text(json.dumps(rows, indent=2))
    (args.output / "real-data-report.json").write_text(json.dumps(scores, indent=2))
    print(json.dumps(scores))


if __name__ == "__main__":
    main()
