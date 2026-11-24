"""Local-only Marian adapters, no network or automatic model choice."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from .core import Pair


def backend() -> tuple[Any, Any, Any]:
    """Import the optional torch/transformers backend only when needed."""
    try:
        import torch
        from transformers import MarianMTModel, MarianTokenizer
    except ImportError as exc:
        raise ValueError("ML backend missing. Install uv sync --extra ml.") from exc
    return torch, MarianMTModel, MarianTokenizer


def load(directory: Path) -> tuple[Any, Any, Any]:
    """Load local safetensors and tokenizer only, never pickle or remote code."""
    if not directory.is_dir():
        raise ValueError("Model must be an existing trusted local directory.")
    if (
        not (directory / "model.safetensors").is_file()
        and not (directory / "model.safetensors.index.json").is_file()
    ):
        raise ValueError("Local safetensors weights required, pickle weights refused.")
    torch, model_cls, tokenizer_cls = backend()
    tokenizer = tokenizer_cls.from_pretrained(str(directory), local_files_only=True)
    model = model_cls.from_pretrained(
        str(directory), local_files_only=True, use_safetensors=True
    ).to("cpu")
    return torch, model, tokenizer


def encode(pairs: list[Pair], tokenizer: Any, max_length: int = 128) -> Any:
    """Use target tokenizer mode and ignore target padding in loss."""
    if not pairs or not 2 <= max_length <= 512:
        raise ValueError("Supply pairs and max length in [2,512].")
    encoded = tokenizer(
        [p[0] for p in pairs],
        text_target=[p[1] for p in pairs],
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    encoded["labels"][encoded["labels"] == tokenizer.pad_token_id] = -100
    return encoded


def translate(
    directory: Path, texts: list[str], max_length: int = 128, beams: int = 4
) -> list[str]:
    """Run deterministic CPU inference on 1..32 strings, no model download."""
    if not 1 <= len(texts) <= 32 or any(not t.strip() or len(t) > 10000 for t in texts):
        raise ValueError("Supply 1..32 nonempty texts, at most10000 characters each.")
    if not 2 <= max_length <= 512 or not 1 <= beams <= 8:
        raise ValueError("Max length must be [2,512], beams [1,8].")
    torch, model, tokenizer = load(directory)
    model.eval()
    inputs = tokenizer(
        texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length
    )
    with torch.inference_mode():
        output = model.generate(
            **inputs, max_new_tokens=max_length, num_beams=beams, do_sample=False
        )
    return list(tokenizer.batch_decode(output, skip_special_tokens=True))


def train(
    directory: Path,
    training: list[Pair],
    validation: list[Pair],
    output: Path,
    epochs: int = 3,
    batch_size: int = 16,
    max_length: int = 128,
    seed: int = 42,
) -> dict[str, int | float]:
    """Explicit local CPU fine-tuning and validation token-weighted loss.

    Loss is teacher-forced, not translation quality. A separate untouched test
    and generated-text metrics are needed before any quality claim.
    """
    if output.exists():
        raise ValueError("Output exists, choose a new directory; no overwrite.")
    if (
        epochs < 1
        or batch_size < 1
        or not 2 <= max_length <= 512
        or not training
        or not validation
    ):
        raise ValueError(
            "Positive epochs/batch, nonempty splits, length [2,512] required."
        )

    def normal(text: str) -> str:
        return " ".join(text.casefold().split())

    if {normal(p[0]) for p in training} & {normal(p[0]) for p in validation}:
        raise ValueError("Training/validation normalized source overlap.")
    torch, model, tokenizer = load(directory)
    torch.manual_seed(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
    for _ in range(epochs):
        model.train()
        for start in range(0, len(training), batch_size):
            batch = encode(training[start : start + batch_size], tokenizer, max_length)
            optimizer.zero_grad()
            loss = model(**batch).loss
            if not bool(torch.isfinite(loss)):
                raise ValueError("Nonfinite training loss; no final artifact saved.")
            loss.backward()
            optimizer.step()
    model.eval()
    weighted = 0.0
    tokens = 0
    with torch.inference_mode():
        for start in range(0, len(validation), batch_size):
            batch = encode(
                validation[start : start + batch_size], tokenizer, max_length
            )
            count = int((batch["labels"] != -100).sum())
            value = float(model(**batch).loss)
            if not math.isfinite(value) or count < 1:
                raise ValueError(
                    "Invalid validation loss/token count; no artifact saved."
                )
            weighted += value * count
            tokens += count
    scores = {
        "validation_loss": weighted / tokens,
        "validation_target_tokens": tokens,
        "validation_pairs": len(validation),
    }
    output.mkdir(parents=True)
    model.save_pretrained(str(output), safe_serialization=True)
    tokenizer.save_pretrained(str(output))
    import json

    (output / "validation.json").write_text(json.dumps(scores), encoding="utf-8")
    return scores
