# Verification record

## What was missing

The initial tracked tree contained README, dependency/config files, Docker/Compose, Makefile and MIT license, but no `src`, `tests`, `examples` or Actions workflow. The advertised `enfr` console script pointed to a missing module. Docker copied an absent `src`. Claims about API endpoints, automatic model download, BLEU/chrF and a passing test suite were not backed by tracked implementation.

The existing MIT license is unchanged. This pass replaces unsupported claims with an actual local-only package. `enfr check-data`, `enfr train MODEL DATA OUTPUT` and `enfr translate MODEL TEXT...` are new explicit commands. There is no `serve` or `evaluate` command. No API, automatic downloads, deployed service or real pretrained evaluation is claimed. The supplied trusted local Marian artifact determines language direction; the repository name cannot verify it.

## Checks

| Command | Result |
| --- | --- |
| `uv sync --locked --extra dev --extra ml` | Installed pinned scoped lock |
| `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1 uv run pytest --cov=enfr_translator --cov-report=term-missing` | 24 passed, 96% statement coverage |
| `uv run ruff check src tests` | Passed |
| `uv run ruff format --check src tests` | Passed |
| `uv run mypy src` | Passed, strict package |
| `uv run pre-commit run --all-files` | Passed |
| `uv run enfr check-data --data examples/parallel.csv` | 3 pairs, 2 training, 1 validation |
| `uv run enfr check-data --config .env.example` | Same result |
| `uv run python -m enfr_translator --help` | Passed |

Tests use invented bilingual strings, a freshly trained tiny SentencePiece tokenizer and random miniature Marian weights. Actual forward/backward, target tokenizer mode, `-100` padding, one-epoch CPU training, CLI generation and safetensors save/load run offline. These establish wiring only, not linguistic quality. The light-only install runs19 tests with the ML module explicitly skipped, no Torch import.

Container build/run checks are hosted in CI, not a local Docker daemon: multi-stage build, non-root runtime, help and mounted CSV checks. Docker/Compose are data checks only, no ML packages or API. CI uses pip/direct pins, not a claim of enforcing every uv transitive lock. Python3.11/3.12 Linuxx86_64 lock scope, no cross-platform promise. Public-repository-only workflow avoids assuming paid private Actions.

## Phase accounting

1-2. Installable typed src package, separate data/model/CLI responsibilities, no unused helpers. 3. Input guards and stderr errors. 4. Optional explicit dotenv path, example and ignored secrets/model outputs. 5. Runtime/dev/ML groups and uv lock. 6. Actual offline core/CLI/ML tests. 7. Ruff/mypy/hooks. 8. Public-only CI with dependency caching and pinned actions. 9. Multi-stage non-root data image. 10. Explicit CLI. 11. Tested usage and honest limitations. 12. Review branch/changelog, no main merge/tag/release. 13. Copy-only metadata/profile notes, not account edits.

Source is adapted from the separately verified local Marian pipeline in the user's neural-machine-translation review branch, with this package's own entry point, synthetic English/French fixtures, license retention and independent tests. Shared Marian mechanics are intentional, not a claim of two independently developed translation models. Remote web-editor file commits remain inspectable; no fake dates or invented original history.

## Remaining limits

No real English/French pretrained model evaluation, BLEU/chrF, human review, untouched domain/document-level test corpus, full fine-tuning, GPU/macOS/Windows/ARM verification, API, deployment, production throughput/resource caps or atomic artifact writes. Source-disjoint normalized sentences are not proof of document/domain independence. Validation loss is teacher-forced, not translation quality. CPU fine-tuning can take substantial time. All ML loads are local-only/safetensors; use trusted artifacts.

Mechanics reference: https://huggingface.co/docs/transformers/v4.49.0/en/tasks/translation . This supports target-text tokenization and label-padding mechanics, not this repository's quality.
