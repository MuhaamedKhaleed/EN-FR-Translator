from pathlib import Path

import pytest

from enfr_translator.core import read_pairs, split_pairs


def test_utf8_pairs_extra_columns_and_split(tmp_path: Path) -> None:
    path = tmp_path / "pairs.csv"
    path.write_text(
        "source,target,note\nHello,Bonjour,a\nGoodbye,Au revoir,b\nThanks,Merci,c\n"
    )
    pairs = read_pairs(path)
    assert pairs == [
        ("Hello", "Bonjour"),
        ("Goodbye", "Au revoir"),
        ("Thanks", "Merci"),
    ]
    train, val = split_pairs(pairs, 0.3, 42)
    assert len(train) == 2 and len(val) == 1
    assert set(train).isdisjoint(val)
    assert (train, val) == split_pairs(pairs, 0.3, 42)


@pytest.mark.parametrize(
    "text",
    [
        "",
        "source,other\na,b\n",
        "source,target,source\na,b,c\n",
        "source,target\n,b\n",
        "source,target\na,\n",
        "source,target\na\n",
        "source,target\na,b,c\n",
        "source,target\nHello,b\n hello ,c\n",
    ],
)
def test_bad_csv(tmp_path: Path, text: str) -> None:
    path = tmp_path / "data.csv"
    path.write_text(text)
    with pytest.raises(ValueError):
        read_pairs(path)


def test_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        read_pairs(tmp_path / "missing")


@pytest.mark.parametrize("fraction", [0, 1, -1, float("nan"), float("inf")])
def test_invalid_fraction(fraction: float) -> None:
    with pytest.raises(ValueError):
        split_pairs([("a", "b"), ("c", "d")], fraction)


def test_invalid_splits() -> None:
    for pairs in [[], [("a", "b")], [("Hello", "a"), (" hello ", "b")]]:
        with pytest.raises(ValueError):
            split_pairs(pairs)
