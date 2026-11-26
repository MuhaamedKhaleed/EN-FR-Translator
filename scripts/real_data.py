"""Explicit real French/English CPU recipe from trusted local sources."""

import argparse
import json
import zipfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument(
        "--data", required=True, type=Path, help="ManyThings fra-eng.zip"
    )
    parser.add_argument(
        "--output", required=True, type=Path, help="New output directory"
    )
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output exists; use a new destination.")
    import torch

    from enfr_translator.model import train

    torch.set_num_threads(1)
    with zipfile.ZipFile(args.data) as archive:
        lines = archive.read("fra.txt").decode("utf-8").splitlines()
    pairs, attribution, seen = [], [], set()
    for line in lines:
        source, target, credit = line.split("\t", 2)
        key = source.casefold().strip()
        if key in seen:
            continue
        seen.add(key)
        pairs.append((source, target))
        attribution.append(credit)
        if len(pairs) == 12:
            break
    del lines
    if len(pairs) != 12:
        raise ValueError("Expected at least twelve unique-source bilingual pairs.")
    scores = train(
        args.model,
        pairs[:8],
        pairs[8:],
        args.output,
        epochs=1,
        batch_size=1,
        max_length=16,
    )
    report = {
        "real_data": True,
        "language_direction": "English to French",
        "training": scores,
        "inference": "Run translate in a separate process after this exits.",
        "limits": "8train/4development path check, not benchmark quality",
    }
    (args.output / "real-data-report.json").write_text(json.dumps(report, indent=2))
    (args.output / "attribution.txt").write_text("\n".join(attribution))
    print(json.dumps(report))


if __name__ == "__main__":
    main()
