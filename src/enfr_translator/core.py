"""Validate parallel CSV and split by normalized source to prevent overlap."""

from __future__ import annotations

import csv
import math
import random
from pathlib import Path

Pair = tuple[str, str]


def read_pairs(path: Path) -> list[Pair]:
    """Read nonempty UTF-8 source/target CSV, reject duplicate source sentences.

    Case/whitespace-normalized repeated sources are rejected, including those
    with different targets. No silent dropping or auto language detection.
    """
    pairs: list[Pair] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8", newline="") as stream:
        rows = csv.DictReader(stream)
        headers = rows.fieldnames or []
        if len(set(headers)) != len(headers) or not {"source", "target"} <= set(
            headers
        ):
            raise ValueError("CSV requires unique source and target headers.")
        for i, row in enumerate(rows, 2):
            if None in row or any(v is None for v in row.values()):
                raise ValueError(f"Wrong CSV field count at row {i}.")
            source, target = row["source"].strip(), row["target"].strip()
            if not source or not target:
                raise ValueError(f"Empty source or target at row {i}.")
            key = " ".join(source.casefold().split())
            if key in seen:
                raise ValueError(
                    f"Duplicate normalized source at row {i}, review data."
                )
            seen.add(key)
            pairs.append((source, target))
    if not pairs:
        raise ValueError("CSV has no parallel pairs.")
    return pairs


def split_pairs(
    pairs: list[Pair], fraction: float = 0.2, seed: int = 42
) -> tuple[list[Pair], list[Pair]]:
    """Seeded pair split, disjoint normalized source sentences."""
    if len(pairs) < 2 or not math.isfinite(fraction) or not 0 < fraction < 1:
        raise ValueError("Need at least two pairs and fraction in (0,1).")
    keys = [" ".join(p[0].casefold().split()) for p in pairs]
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate source sentences before splitting.")
    data = list(pairs)
    random.Random(seed).shuffle(data)
    n = max(1, min(len(data) - 1, math.ceil(len(data) * fraction)))
    return data[n:], data[:n]
