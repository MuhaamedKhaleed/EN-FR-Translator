"""Explicit source download. No import-time network or data redistribution."""

import argparse
import hashlib
import json
import shutil
import urllib.request
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", required=True, type=Path, help="New source directory"
    )
    parser.add_argument(
        "--cached", type=Path, help="Use already downloaded files, without network"
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = []
    for filename, url, sha in SOURCES:
        target = args.output / filename
        if args.cached:
            shutil.copyfile(args.cached / filename, target)
        else:
            with (
                urllib.request.urlopen(url, timeout=120) as source,
                target.open("wb") as out,
            ):
                shutil.copyfileobj(source, out)
        actual = hashlib.sha256(target.read_bytes()).hexdigest()
        if actual != sha:
            target.unlink()
            raise ValueError("Source checksum mismatch: " + filename)
        manifest.append({"file": filename, "url": url, "sha256": actual})
    (args.output / "source-manifest.json").write_text(json.dumps(manifest, indent=2))
    import torch
    from transformers import MarianConfig, MarianMTModel

    torch.set_num_threads(1)
    # Exact official pinned bytes only. Never load arbitrary pickle weights.
    state = torch.load(
        args.output / "pytorch_model.bin", map_location="cpu", weights_only=True
    )
    model = MarianMTModel(
        MarianConfig.from_pretrained(args.output, local_files_only=True)
    )
    missing, unexpected = model.load_state_dict(state, strict=False)
    if set(missing) - {"lm_head.weight"} or unexpected:
        raise ValueError("Unexpected official state dictionary keys")
    del state
    model.tie_weights()
    model.save_pretrained(args.output, safe_serialization=True)
    (args.output / "pytorch_model.bin").unlink()


SOURCES = [
    (
        "config.json",
        "https://huggingface.co/Helsinki-NLP/opus-mt-en-fr/resolve/dd7f6540a7a48a7f4db59e5c0b9c42c8eea67f18/config.json",
        "62eb1ac08f8c7b047fad7ac3a41e769f375bdfa5c08cfb25ed20b7956284b415",
    ),
    (
        "generation_config.json",
        "https://huggingface.co/Helsinki-NLP/opus-mt-en-fr/resolve/dd7f6540a7a48a7f4db59e5c0b9c42c8eea67f18/generation_config.json",
        "33090cbaa9f66b89b0332b8583ff8ddb1fee6d15b401041a5fbb7d022f2d0c96",
    ),
    (
        "pytorch_model.bin",
        "https://huggingface.co/Helsinki-NLP/opus-mt-en-fr/resolve/dd7f6540a7a48a7f4db59e5c0b9c42c8eea67f18/pytorch_model.bin",
        "cc1de10b49342ad2f33e06bc4474ddd6eaca278474903c4a8636ce15680d64de",
    ),
    (
        "source.spm",
        "https://huggingface.co/Helsinki-NLP/opus-mt-en-fr/resolve/dd7f6540a7a48a7f4db59e5c0b9c42c8eea67f18/source.spm",
        "173e9f493a668fe396d599e28d414a201193094e6ffd7a4678e5aab0f6d3d838",
    ),
    (
        "target.spm",
        "https://huggingface.co/Helsinki-NLP/opus-mt-en-fr/resolve/dd7f6540a7a48a7f4db59e5c0b9c42c8eea67f18/target.spm",
        "78d0e717c77053f1c4b856d8661d9cb87c64f083a35418c087b9146300e4f585",
    ),
    (
        "tokenizer_config.json",
        "https://huggingface.co/Helsinki-NLP/opus-mt-en-fr/resolve/dd7f6540a7a48a7f4db59e5c0b9c42c8eea67f18/tokenizer_config.json",
        "3492a8555368d21fc116cac84bdf551aee16783413be3afbfe4823de045960cf",
    ),
    (
        "vocab.json",
        "https://huggingface.co/Helsinki-NLP/opus-mt-en-fr/resolve/dd7f6540a7a48a7f4db59e5c0b9c42c8eea67f18/vocab.json",
        "945c604346ce15ce4aff9001001e7f925e336d942c4087017f191871162cbdc4",
    ),
    (
        "fra-eng.zip",
        "https://www.manythings.org/anki/fra-eng.zip",
        "9e35994767f7307dc1fa9d5efcc36681f3756a1bdaaa02c1381b3eb11461261e",
    ),
]

if __name__ == "__main__":
    main()
