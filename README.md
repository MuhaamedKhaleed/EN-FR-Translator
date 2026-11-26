# English/French Local Translator

A local CLI for validating parallel CSV data, translating with a local Marian model, and running explicit CPU fine-tuning. Built from an English/French translation project whose advertised API/evaluation features were never in the tracked source; this version replaces those claims with tested behavior.

## Results (honest)

- Fresh clone translates real sentences with a pinned public model (Helsinki-NLP `opus-mt-en-fr`, ~300 MB). Recorded 25 Nov 2026: "The weather is nice today." -> "Le temps est beau aujourd'hui.", "Where is the train station?" -> "Où est la gare ?".
- The bundled fine-tune runs on 12 sentence pairs. It proves the train/save/reload wiring; it is not a quality improvement.
- Validation metric is token-weighted teacher-forced loss. No BLEU, chrF, benchmark or human evaluation has been run.
- 24 tests, 96% statement coverage (tiny random Marian forward/backward, synthetic SentencePiece, safetensors save/load). Coverage proves wiring, not translation quality.

## Quickstart (fresh clone, about 3 minutes)

Needs Python 3.11 or 3.12 (locked wheels have no 3.13+ build), Linux x86_64, internet for the first install/downloads, ~2 GB free RAM.

```sh
uv sync --python 3.11 --locked --extra dev --extra ml
uv run --no-sync python scripts/fetch_real_data.py --output ./real-source      # pinned opus-mt-en-fr + Tatoeba exports
uv run --no-sync enfr translate ./real-source "The weather is nice today." "Where is the train station?"
uv run --no-sync python scripts/prepare_french.py --data ./real-source --output ./french-pairs
uv run --no-sync python scripts/train_french.py --model ./real-source --data ./french-pairs/pairs.json --output ./french-model
```

To fine-tune on your own data, build a `source,target` CSV and run `enfr train ./real-source YOUR.csv ./my-model --epochs N`.

Data checks run without the ML stack: `uv sync --locked --extra dev` then `enfr check-data --data examples/parallel.csv` returns `{"pairs": 3, "train_pairs": 2, "validation_pairs": 1}`.

## Limits

- You must supply a trusted local Marian model (loaded with `local_files_only=True`); no model is bundled or auto-downloaded, and its language direction and license are your responsibility.
- Source-disjoint splitting is not document/domain independence and not an untouched test set.
- CPU-only, deterministic beam search, bounded input lengths; truncation can drop long-text information.
- The old README's `serve`/`evaluate` commands and automatic downloads were advertised but never implemented; they are deliberately not reproduced.
- Verified on Linux x86_64 only; the container runs data checks, not translation. Hosted CI (free public-repo runners) checks tests, lint, CLI and the Docker build.

## License

MIT for the new project-owned code (see `LICENSE` and `LICENSE-SCOPE.md`). Historical notebooks, third-party models, datasets and corpora keep their own terms; no rights clearance is claimed for them. Details and evidence: `docs/verification.md`, `docs/real-data.md`, `CHANGELOG.md`.
