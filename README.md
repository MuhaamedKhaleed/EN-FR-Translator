# EN-FR-Translator

[![CI](https://github.com/MuhaamedKhaleed/EN-FR-Translator/actions/workflows/ci.yml/badge.svg)](https://github.com/MuhaamedKhaleed/EN-FR-Translator/actions)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

English → French machine translation. Fine-tune a pre-trained MarianMT model
([`Helsinki-NLP/opus-mt-en-fr`](https://huggingface.co/Helsinki-NLP/opus-mt-en-fr)) on your own
parallel corpus, evaluate it with BLEU/chrF, and serve it through a REST API or CLI.

## Features

- **Fine-tuning pipeline**: Hugging Face `Seq2SeqTrainer`, dynamic padding, mixed precision on GPU,
  best checkpoint selected by validation BLEU.
- **Data cleaning**: NaN/empty/duplicate removal, automatic UTF-8 → latin-1 fallback, deterministic train/validation split.
- **Evaluation**: BLEU and chrF (SacreBLEU) on a held-out split.
- **REST API**: FastAPI with `/translate` and `/health`, batching, and input limits.
- **CLI**: `enfr train | evaluate | translate | serve`.
- **Deployable**: multi-stage Dockerfile (non-root, healthcheck) and `docker-compose.yml`.
- **Quality gates**: pytest suite, ruff lint/format, GitHub Actions CI.

## Quick start

### Docker (API)

```bash
docker compose up --build
curl -X POST localhost:8000/translate \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Hello, how are you?"]}'
```

The first start downloads the base model into the `models` volume, so it takes a minute.

### Local

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[train,api,dev]"

enfr translate "I love learning languages."
enfr serve                      # http://127.0.0.1:8000/docs
```

## Fine-tuning on your data

Your CSV needs two columns: `English words/sentences` and `French words/sentences`
(override with `--src-col` / `--tgt-col`). A 15-row example is in `examples/sample.csv`.

```bash
enfr train --data path/to/eng-french.csv --epochs 3 --batch-size 32
enfr evaluate --data path/to/eng-french.csv
```

The fine-tuned model and tokenizer are saved to `models/enfr-finetuned`. The API and CLI use it
automatically when it exists, otherwise they use the pre-trained base model.

## API

| Method | Path         | Description                                  |
| ------ | ------------ | -------------------------------------------- |
| GET    | `/health`    | Liveness and whether the model is loaded     |
| POST   | `/translate` | `{"texts": ["..."]}` → `{"translations": [...], "model": "..."}` |

Error codes: `422` for an empty/blank/oversized batch, `413` for a text over the character limit.
Interactive docs are at `/docs`.

## Configuration

Environment variables (see `.env.example`):

| Variable               | Default                      | Purpose                              |
| ---------------------- | ---------------------------- | ------------------------------------ |
| `ENFR_BASE_MODEL`      | `Helsinki-NLP/opus-mt-en-fr` | Pre-trained model                    |
| `ENFR_MODEL_DIR`       | `models/enfr-finetuned`      | Fine-tuned model location            |
| `ENFR_NUM_BEAMS`       | `4`                          | Beam search width                    |
| `ENFR_MAX_LENGTH`      | `128`                        | Max tokens in/out                    |
| `ENFR_BATCH_SIZE`      | `16`                         | Inference batch size                 |
| `ENFR_MAX_BATCH_ITEMS` | `32`                         | Max texts per API request            |
| `ENFR_MAX_CHARS`       | `1000`                       | Max characters per text              |
| `ENFR_LOG_LEVEL`       | `INFO`                       | Logging level                        |

## Project layout

```
src/enfr_translator/   config, data, train, evaluate, translate, api, cli
tests/                 pytest suite
examples/sample.csv    tiny dataset for smoke tests
Dockerfile, docker-compose.yml, Makefile, .github/workflows/ci.yml
```

## Development

```bash
make install   # editable install with all extras
make lint      # ruff check + format check
make test      # pytest
```

## License

MIT, see [LICENSE](LICENSE).
