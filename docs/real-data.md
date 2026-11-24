# Real English-French CPU path

A real pretrained Helsinki-NLP/opus-mt-en-fr checkpoint ran CPU inference, followed by an 8pair training/4pair development fine-tune with batch1. Actual results are in real-data-output.json. This is an English-French example even in the generic translation repo; it does not establish German-model quality. The first larger process retained two model copies and was killed for memory, while the isolated small run succeeded. No RAM minimum is asserted from this one environment. No weights or corpus are committed.

Model source: https://huggingface.co/Helsinki-NLP/opus-mt-en-fr , revision dd7f6540a7a48a7f4db59e5c0b9c42c8eea67f18, model card Apache2.0. Official main supplied pytorch_model.bin, not safetensors. The executor converted that exact official tensor state using torch.load(weights_only=True), tied missing lm_head to the shared embedding, and saved safetensors. No unrestricted pickle load or remote model code. The supported package itself still refuses pickle weights. The explicit helper downloads exactly pinned official bytes, verifies SHA-256 before tensor-only conversion, permits only the known missing tied lm_head key and deletes the pickle file afterward. This conversion is separate from training; never use it on an arbitrary checkpoint.

Dataset source: https://www.manythings.org/anki/ , fra-eng.zip, Tatoeba CC BY2.0 France with sentence-level attribution. Keep the third column and original readme/license. The script keeps attribution alongside results and does not republish sentences. Twelve first normalized-unique sources are an illustrative development slice; pretrained overlap is unknown. Not an untouched BLEU/chrF benchmark.

```sh
uv sync --locked --extra dev --extra ml
uv run --no-sync python scripts/real_data.py --model LOCAL_SAFETENSORS_MODEL --data LOCAL_FRA_ENG_ZIP --output NEW_OUTPUT
```

The published recipe now frees unused corpus lines before allocating model and optimizer state. Training and inference run in separate processes. The actual parameterized recipe completed in both packages after that memory-lifecycle fix.

Locked fresh environments passed all package tests. The explicit fetch helper now handles pinned download and conversion. It is never invoked on import/install. No RAM minimum is established; larger batches exhausted this environment. A generic package's real German model/corpus run remains separate.

## Explicit source preparation

```sh
uv sync --locked --extra dev --extra ml
uv run --no-sync python scripts/fetch_real_data.py --output ./real-source
uv run --no-sync python scripts/real_data.py --model ./real-source --data ./real-source/fra-eng.zip --output ./real-finetune
```

The download is about300MB model bytes plus the8MB corpus; dependencies and output checkpoints need additional disk/RAM. Output directory must be new. Hash changes fail closed. `--cached /path/to/source-files` avoids network but still checks every file; the cached folder must have the official pytorch_model.bin plus tokenizer/config and fra-eng.zip. Data/model rights are separate from this repo's MIT code. The source corpus retains per-sentence attribution and its readme in the original archive. No benchmark or unseen-data claim.
