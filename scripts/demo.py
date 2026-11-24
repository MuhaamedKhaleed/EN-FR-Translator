"""Explicit synthetic wiring demo. No real-data accuracy or quality claim."""

import json
import os
import sys
from pathlib import Path


def main() -> None:
    for key, value in {
        "CUDA_VISIBLE_DEVICES": "-1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "TF_CPP_MIN_LOG_LEVEL": "3",
        "TF_NUM_INTRAOP_THREADS": "1",
        "TF_NUM_INTEROP_THREADS": "1",
        "OMP_NUM_THREADS": "1",
    }.items():
        os.environ.setdefault(key, value)

    out = Path(sys.argv[1] if len(sys.argv) == 2 else "synthetic-run")
    if len(sys.argv) > 2:
        raise SystemExit("Usage: python scripts/demo.py [NEW_OUTPUT_DIRECTORY]")
    out.mkdir(parents=True, exist_ok=False)
    import sentencepiece as spm
    import torch
    import transformers

    from enfr_translator.model import train, translate

    torch.set_num_threads(1)
    torch.manual_seed(42)
    corpus = out / "corpus.txt"
    corpus.write_text(
        "hello world\nbonjour monde\ngood day\n"
        "bonne journee\nthanks friend\nmerci ami\n"
    )
    prefix = str(out / "spm")
    spm.SentencePieceTrainer.train(
        input=str(corpus),
        model_prefix=prefix,
        vocab_size=32,
        hard_vocab_limit=False,
        minloglevel=2,
    )
    pieces = spm.SentencePieceProcessor(model_file=prefix + ".model")
    vocab = {pieces.id_to_piece(i): i for i in range(pieces.get_piece_size())}
    vocab["<pad>"] = len(vocab)
    vocabfile = out / "vocab.json"
    vocabfile.write_text(json.dumps(vocab))
    tokenizer = transformers.MarianTokenizer(
        source_spm=prefix + ".model",
        target_spm=prefix + ".model",
        vocab=str(vocabfile),
        model_max_length=32,
    )
    model = transformers.MarianMTModel(
        transformers.MarianConfig(
            vocab_size=len(tokenizer),
            d_model=8,
            encoder_layers=1,
            decoder_layers=1,
            encoder_attention_heads=1,
            decoder_attention_heads=1,
            encoder_ffn_dim=16,
            decoder_ffn_dim=16,
            max_position_embeddings=64,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            decoder_start_token_id=tokenizer.pad_token_id,
        )
    )
    base = out / "base"
    base.mkdir()
    tokenizer.save_pretrained(base)
    model.save_pretrained(base, safe_serialization=True)
    pairs = (
        [("hello world", "bonjour monde"), ("good day", "bonne journee")]
        if "enfr_translator" == "enfr_translator"
        else [("hello world", "hallo welt"), ("good day", "guten tag")]
    )
    validation = [
        (
            "thanks friend",
            "merci ami" if "enfr_translator" == "enfr_translator" else "danke freund",
        )
    ]
    scores = train(base, pairs, validation, out / "model", 1, 2, 24)
    predictions = translate(out / "model", ["hello world"], 8, 1)
    assert len(predictions) == 1 and isinstance(predictions[0], str)
    result = {"training": scores, "prediction": predictions}
    scope = "invented_bilingual_strings_random_Marian_not_translation_quality"
    report = {
        "synthetic_only": True,
        "scope": scope,
        "quality_claim": False,
        "result": result,
    }
    (out / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {"synthetic_only": True, "report": str(out / "report.json")},
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
