# English/French Local Translator

A local-only CLI for checking parallel CSV data, running an existing Marian translation model, and deliberately starting CPU fine-tuning. The repository previously advertised an implementation absent from its tracked source. This pass supplies a working local package and replaces unsupported API/evaluation claims with tested behavior. No pretrained model or corpus is bundled.

## Quick start

Python 3.11 or 3.12, Linux x86_64. Install uv first using its official instructions, then:

```sh
uv sync --locked --extra dev
uv run enfr --help
uv run enfr check-data --data examples/parallel.csv
uv run enfr check-data --config .env.example
```

Actual example output:

```json
{"pairs": 3, "train_pairs": 2, "validation_pairs": 1}
```

`source,target` are required UTF-8 CSV headers. Other uniquely named columns are allowed. Blank pairs, malformed rows, and repeated case/whitespace-normalized source sentences fail rather than being silently dropped. `--config` is an opt-in dotenv file with `PARALLEL_CSV`; explicit `--data` wins. No credentials or service calls are needed.

The lock is scoped to Linux x86_64. Other operating systems are not verified. The ordinary install and Docker image provide **data checks only**, with no Torch import. Optional ML dependencies are much larger:

```sh
uv sync --locked --extra dev --extra ml
```

## Local model workflow

`translate MODEL TEXT...` and `train MODEL DATA OUTPUT` require the optional `ml` extra and a trusted local Marian directory containing tokenizer/config files and safetensors weights. Both loading calls use `local_files_only=True`; no model is selected or downloaded automatically. Pickle model weights and remote-code loading are not supported. Do not point at untrusted artifacts.

Use `enfr-translator translate --help` and `enfr-translator train --help` for arguments. The tested end-to-end example is the synthetic offline test below, not a downloaded model demonstration. A user-supplied pretrained model's language direction and license remain the user's responsibility.

- CPU-only inference, deterministic beam search, bounded input count and lengths.
- Target tokenizer mode, dynamic batch padding, target pad labels set to `-100`.
- Seeded source-disjoint split before fitting. Normalized duplicates are rejected.
- Explicit training with finite-loss checks; no auto-training on import or overwrite of existing output directories.
- Token-count-weighted teacher-forced validation loss, not BLEU or a translation-quality claim.

The model/tokenizer architecture comes from the supplied local artifact. No pretrained artifact is bundled or selected automatically. The supplied model determines the actual language direction, so inspect it before using it for English/French. Truncation can discard long-text information. Source-disjoint splitting is not document/source/domain independence and is not an untouched evaluation set.

## Checks

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1 uv run pytest --cov=enfr_translator --cov-report=term-missing
uv run ruff check src tests
uv run ruff format --check src tests
uv run mypy src
uv run pre-commit run --all-files
uv run python -m enfr_translator --help
```

With the ML extra: **24 tests, 96% coverage**, including actual tiny random Marian forward/backward, local synthetic SentencePiece, target padding, one-epoch training, CLI inference, and safetensors save/load. This proves pipeline wiring, not translation quality. With only the light dependencies, backend tests are explicitly skipped. No real corpus fine-tuning, pretrained translation benchmark, BLEU, chrF or human evaluation was run.

## Docker

```sh
docker build --tag enfr-translator:ci .
docker run --rm enfr-translator:ci --help
docker run --rm -v "$PWD/examples/parallel.csv:/data.csv:ro" enfr-translator:ci check-data --data /data.csv
```

Multi-stage slim image, non-root runtime, no dev or ML dependencies. The container checks data, it does not translate or train. Hosted CI runs these build/run checks; see [verification](docs/verification.md) for the evidence boundary. No paid APIs, hosted training, self-hosted runners or deployment are configured. The workflow is restricted to public repositories and ordinary GitHub-hosted Linux runners.

## Layout and data flow

```text
CSV -> core.read_pairs -> source-disjoint split -> explicit CPU train
local safetensors + tokenizer -> model.encode / translate -> JSON
Typer CLI -> input validation -> core/model -> separate errors on stderr
```

`src/enfr_translator/core.py` owns CSV and split rules; `model.py` owns lazy ML loading, encoding, training and inference; `cli.py` is the interface. Tests build invented data and a random miniature model locally, with hub access disabled. CI uses pinned direct requirements via pip, while local reproduction uses `uv.lock`; pip is not claimed to enforce the full transitive uv lock.

## Existing interface changes and limits

The old README advertised `enfr train | evaluate | translate | serve`, automatic Helsinki-NLP model download, API endpoints and BLEU/chrF evaluation. No corresponding source, tests or sample CSV were tracked. Those claims were not working behavior to preserve. This pass implements `enfr check-data`, explicit `enfr train`, and `enfr translate`, **not** `serve` or `evaluate`. It does not make an untrained model bilingual by naming the repository English/French. You must supply a suitable trusted local Marian model.

`docker-compose.yml` now runs a mounted data check, not a nonexistent API. The existing MIT license is preserved. `make install`, `make lint`, `make test`, and `make check` map to the documented commands. `requirements.txt` points to the base package rather than duplicating a second dependency list.

Next work: evaluate a licensed, appropriate pretrained local model on a genuinely untouched bilingual test set, inspect generated translations and failure modes, add length/domain slices, and measure BLEU/chrF with exact versions and signatures. No English/French model-quality, broad platform compatibility or production-readiness is claimed.

See [verification and phase notes](docs/verification.md), [changelog](CHANGELOG.md), and [copy-only GitHub presentation](docs/presentation.md). This review branch does not change the repository license, create a release/tag, merge main, or publish model/data artifacts.

## Checked synthetic wiring demo

See [commands, saved output and scope](docs/synthetic-demo.md). This is invented-input execution evidence, not model accuracy or a real-data gallery.

## Real-data CPU example

See [actual results, source terms and explicit download/preparation commands](docs/real-data.md). These are small development runs, not accuracy or production guarantees. No model/data is committed.
