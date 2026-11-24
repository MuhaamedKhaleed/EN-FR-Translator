"""Prepare a tiny attributed French/English development slice from official exports."""

import bz2
import csv
import hashlib
import json
from pathlib import Path


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    pairs = []
    seen = set()
    with bz2.open(args.data / "fra-eng_links.tsv.bz2", "rt") as source:
        for line in source:
            de, en = line.strip().split("\t")
            if de not in seen and en not in seen:
                pairs.append((de, en))
                seen.update((de, en))
            if len(pairs) == 12:
                break
    wanted = {k for pair in pairs for k in pair}
    records = {}
    for lang in ["fra", "eng"]:
        with bz2.open(
            args.data / (lang + "_sentences_detailed.tsv.bz2"), "rt"
        ) as source:
            for row in csv.reader(source, delimiter="\t", quoting=csv.QUOTE_NONE):
                if row[0] in wanted:
                    records[row[0]] = row
                    if wanted <= records.keys():
                        break
    if wanted - records.keys():
        raise ValueError("Missing linked sentence records")
    out = []
    for de, en in pairs:
        d, e = records[de], records[en]
        if d[3] == r"\N" or e[3] == r"\N":
            raise ValueError("Unattributed sentence in selected slice")
        out.append(
            {
                "source": e[2],
                "target": d[2],
                "credit": {
                    "license": "CC BY2.0 France",
                    "fr_id": de,
                    "fr_author": d[3],
                    "en_id": en,
                    "en_author": e[3],
                },
            }
        )
    (args.output / "pairs.json").write_text(json.dumps(out, indent=2))
    hashes = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in args.data.glob("*.bz2")
    }
    (args.output / "source-hashes.json").write_text(json.dumps(hashes, indent=2))


if __name__ == "__main__":
    main()
