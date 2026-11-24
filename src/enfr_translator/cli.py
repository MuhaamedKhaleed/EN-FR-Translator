"""Local-only data checks, Marian translation and deliberate CPU fine-tuning."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import typer
from dotenv import dotenv_values

from .core import read_pairs, split_pairs

app = typer.Typer(
    help="Local-only Marian translation. No automatic downloads or training."
)
logger = logging.getLogger(__name__)


def run(action: Callable[[], Any]) -> None:
    """Emit JSON only on stdout, expected input failures on stderr with exit2."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        typer.echo(json.dumps(action()))
    except (OSError, ValueError, TypeError, KeyError) as exc:
        logger.error("%s", exc)
        raise typer.Exit(2) from exc


def resolve(data: Path | None, config: Path | None) -> Path:
    """Explicit CSV wins over an opt-in dotenv PARALLEL_CSV setting."""
    if data:
        return data
    if config:
        if not config.is_file():
            raise ValueError("Config missing. Supply existing dotenv.")
        value = dotenv_values(config).get("PARALLEL_CSV")
        if value:
            return Path(value)
    raise ValueError("Supply --data or --config containing PARALLEL_CSV.")


@app.command()
def check_data(
    data: Path | None = None,
    config: Path | None = None,
    validation_fraction: float = 0.2,
    seed: int = 42,
) -> None:
    """Validate parallel CSV and disjoint-source split without ML imports."""

    def action() -> dict[str, int]:
        pairs = read_pairs(resolve(data, config))
        training, validation = split_pairs(pairs, validation_fraction, seed)
        return {
            "pairs": len(pairs),
            "train_pairs": len(training),
            "validation_pairs": len(validation),
        }

    run(action)


@app.command()
def translate(
    model: Path, texts: list[str], max_length: int = 128, beams: int = 4
) -> None:
    """Translate via an explicitly supplied trusted local safetensors directory."""

    def action() -> dict[str, list[str]]:
        from .model import translate as infer

        return {"translations": infer(model, texts, max_length, beams)}

    run(action)


@app.command()
def train(
    model: Path,
    data: Path,
    output: Path,
    epochs: int = 3,
    batch_size: int = 16,
    max_length: int = 128,
    validation_fraction: float = 0.2,
    seed: int = 42,
) -> None:
    """Start local CPU fine-tuning explicitly. Large data can take a long time."""

    def action() -> dict[str, int | float]:
        from .model import train as fit

        training, validation = split_pairs(read_pairs(data), validation_fraction, seed)
        return fit(
            model, training, validation, output, epochs, batch_size, max_length, seed
        )

    run(action)
