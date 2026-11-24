import json
from pathlib import Path

import pytest

from enfr_translator.model import encode, load, train, translate

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
spm = pytest.importorskip("sentencepiece")


@pytest.fixture()
def artifact(tmp_path: Path) -> Path:
    # Invented tiny bilingual strings, random model, no pretrained downloads.
    corpus = tmp_path / "tiny.txt"
    corpus.write_text(
        "hello world\nbonjour monde\ngood day\n"
        "bonne journee\nthanks friend\nmerci ami\n"
    )
    prefix = tmp_path / "spm"
    spm.SentencePieceTrainer.train(
        input=str(corpus),
        model_prefix=str(prefix),
        vocab_size=32,
        hard_vocab_limit=False,
        minloglevel=2,
    )
    pieces = spm.SentencePieceProcessor(model_file=str(prefix) + ".model")
    vocab = {pieces.id_to_piece(i): i for i in range(pieces.get_piece_size())}
    vocab["<pad>"] = len(vocab)
    vocabfile = tmp_path / "vocab.json"
    vocabfile.write_text(json.dumps(vocab))
    tokenizer = transformers.MarianTokenizer(
        source_spm=str(prefix) + ".model",
        target_spm=str(prefix) + ".model",
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
    directory = tmp_path / "artifact"
    directory.mkdir()
    tokenizer.save_pretrained(directory)
    model.save_pretrained(directory, safe_serialization=True)
    return directory


def test_actual_target_padding_loss_and_generation(artifact: Path) -> None:
    _, model, tokenizer = load(artifact)
    batch = encode([("hello world", "bonjour monde"), ("thanks", "tag")], tokenizer, 32)
    assert bool((batch["labels"] == -100).any())
    expected = tokenizer(
        text_target=["bonjour monde", "tag"], padding=True, return_tensors="pt"
    )["input_ids"]
    expected[expected == tokenizer.pad_token_id] = -100
    assert torch.equal(batch["labels"], expected)
    loss = model(**batch).loss
    assert bool(torch.isfinite(loss))
    loss.backward()
    result = translate(artifact, ["hello world", "thanks"], 8, 1)
    assert len(result) == 2 and all(isinstance(t, str) for t in result)


def test_synthetic_local_training_roundtrip(artifact: Path, tmp_path: Path) -> None:
    output = tmp_path / "trained"
    scores = train(
        artifact,
        [("hello world", "bonjour monde"), ("good day", "bonne journee")],
        [("thanks friend", "merci ami")],
        output,
        1,
        2,
        24,
    )
    assert scores["validation_pairs"] == 1
    assert scores["validation_target_tokens"] > 0
    assert scores["validation_loss"] > 0
    assert json.loads((output / "validation.json").read_text()) == scores
    assert len(translate(output, ["hello"], 8, 1)) == 1
    with pytest.raises(ValueError):
        train(artifact, [("a", "b")], [("c", "d")], output)


def test_invalid_backend_inputs(tmp_path: Path) -> None:
    for directory in [tmp_path, tmp_path / "missing"]:
        with pytest.raises(ValueError):
            load(directory)
    for texts, length, beams in [
        ([], 8, 1),
        ([""], 8, 1),
        (["x"], 1, 1),
        (["x"], 8, 0),
        (["x"] * 33, 8, 1),
    ]:
        with pytest.raises(ValueError):
            translate(tmp_path, texts, length, beams)
    with pytest.raises(ValueError):
        encode([], None)
    with pytest.raises(ValueError):
        train(tmp_path, [("a", "b")], [(" A ", "c")], tmp_path / "new")
    with pytest.raises(ValueError):
        train(tmp_path, [("a", "b")], [("c", "d")], tmp_path / "new", epochs=0)


def test_explicit_cli_synthetic_training(artifact: Path, tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from enfr_translator.cli import app

    data = tmp_path / "pairs.csv"
    data.write_text(
        "source,target\nhello world,bonjour monde\ngood day,bonne journee\n"
        "thanks friend,merci ami\n"
    )
    runner = CliRunner()
    out = tmp_path / "cli-model"
    result = runner.invoke(
        app,
        [
            "train",
            str(artifact),
            str(data),
            str(out),
            "--epochs",
            "1",
            "--batch-size",
            "2",
            "--max-length",
            "24",
        ],
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["validation_pairs"] == 1
    result = runner.invoke(
        app, ["translate", str(out), "hello", "--max-length", "8", "--beams", "1"]
    )
    assert result.exit_code == 0, result.output
    assert len(json.loads(result.output)["translations"]) == 1


def test_encode_calls_target_mode() -> None:
    class Tokenizer:
        pad_token_id = 0

        def __call__(self, texts, **kwargs):
            assert texts == ["source sentence"]
            assert kwargs["text_target"] == ["target sentence"]
            return {
                "input_ids": torch.tensor([[1, 2]]),
                "labels": torch.tensor([[3, 0]]),
            }

    result = encode([("source sentence", "target sentence")], Tokenizer())
    assert result["labels"].tolist() == [[3, -100]]
