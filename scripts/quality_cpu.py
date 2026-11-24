"Explicit measured CPU quality experiment. No import-time work or downloads."


def main():
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--model", type=Path, help="Translation only: trusted local Marian"
    )
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output exists; use a new directory.")
    import os

    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    import bz2
    import csv
    import json
    import resource
    import time
    from pathlib import Path

    import torch
    from transformers import MarianMTModel, MarianTokenizer

    from enfr_translator.model import encode

    lang = os.environ.get("QUALITY_LANG", "fra")
    short = "de" if lang == "deu" else "fr"
    source = args.data
    root = args.output
    root.mkdir(exist_ok=False)
    links = []
    seen = set()
    with bz2.open(source / (lang + "-eng_links.tsv.bz2"), "rt") as f:
        for line in f:
            target, english = line.strip().split("\t")
            if target not in seen and english not in seen:
                links.append((target, english))
                seen.update((target, english))
            if len(links) == 1300:
                break
    wanted = {k for pair in links[1000:1300] for k in pair}
    records = {}
    for language in [lang, "eng"]:
        with bz2.open(source / (language + "_sentences_detailed.tsv.bz2"), "rt") as f:
            for row in csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
                if row[0] in wanted:
                    records[row[0]] = row
    pairs = []
    normalized = set()
    for target, en in links[1000:1300]:
        a, b = records[en], records[target]
        key = " ".join(a[2].casefold().split())
        if a[3] == r"\N" or b[3] == r"\N" or key in normalized:
            continue
        normalized.add(key)
        pairs.append(
            (
                a[2],
                b[2],
                {
                    "source_id": en,
                    "target_id": target,
                    "source_author": a[3],
                    "target_author": b[3],
                },
            )
        )
    training = [(a, b) for a, b, _ in pairs[:16]]
    dev = [(a, b) for a, b, _ in pairs[64:80]]
    test = [(a, b) for a, b, _ in pairs[96:112]]
    torch.set_num_threads(1)
    torch.manual_seed(42)
    base = args.model
    model = MarianMTModel.from_pretrained(
        base, local_files_only=True, use_safetensors=True
    )
    tokenizer = MarianTokenizer.from_pretrained(base, local_files_only=True)

    def translate(data):
        model.eval()
        out = []
        with torch.inference_mode():
            for a, _b in data:
                x = tokenizer([a], return_tensors="pt", truncation=True, max_length=64)
                y = model.generate(**x, max_new_tokens=64, num_beams=2)
                out += tokenizer.batch_decode(y, skip_special_tokens=True)
        return out

    def loss(data):
        model.eval()
        total = tokens = 0
        with torch.inference_mode():
            for pair in data:
                batch = encode([pair], tokenizer, 64)
                n = int((batch["labels"] != -100).sum())
                total += float(model(**batch).loss) * n
                tokens += n
        return total / tokens

    before = translate(test)
    start = time.time()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    history = []
    best = loss(dev)
    # Save selected checkpoints to disk, reload after freeing optimizer/model.
    model.save_pretrained(root / "best", safe_serialization=True)
    for epoch in range(1):
        model.train()
        for pair in training:
            optimizer.zero_grad()
            batch = encode([pair], tokenizer, 64)
            value = model(**batch).loss
            value.backward()
            optimizer.step()
        score = loss(dev)
        history.append({"epoch": epoch + 1, "development_token_loss": score})
        if score < best:
            best = score
            model.save_pretrained(root / "best", safe_serialization=True)
        print(history[-1], flush=True)
    del optimizer, model
    import gc

    gc.collect()
    model = MarianMTModel.from_pretrained(
        root / "best", local_files_only=True, use_safetensors=True
    )
    after = translate(test)
    from sacrebleu.metrics import BLEU, CHRF

    refs = [[b for a, b in test]]
    r = {
        "source": "https://downloads.tatoeba.org/exports/per_language/" + lang + "/",
        "model_source": "https://huggingface.co/Helsinki-NLP/opus-mt-en-" + short,
        "train_pairs": len(training),
        "development_pairs": len(dev),
        "test_pairs": len(test),
        "history": history,
        "test_before_chrf": CHRF().corpus_score(before, refs).score,
        "test_after_chrf": CHRF().corpus_score(after, refs).score,
        "test_before_bleu": BLEU().corpus_score(before, refs).score,
        "test_after_bleu": BLEU().corpus_score(after, refs).score,
        "elapsed_seconds": time.time() - start,
        "peak_rss_kb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "limits": (
            "Same named test used for before/after fixed devselected "
            "experiment, no testdriven tuning. Tatoebapretraining overlap"
            " unknown, not externalbenchmark. Model may alreadyknowthese "
            "sentences, no generalizationclaim."
        ),
    }
    (root / "report.json").write_text(json.dumps(r, indent=2))
    (root / "attribution.json").write_text(json.dumps([c for a, b, c in pairs]))
    print(json.dumps(r))


if __name__ == "__main__":
    main()
