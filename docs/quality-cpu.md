# Measured no-cost CPU quality experiment

16train/16dev/16evaluationTatoebapairs outside initialslice,2beam generatedtextevaluation. Fixed1epoch lowlr/devcheckpointselection. FrenchchrF74.99to75.47/BLEU59.48to59.71. Tiny same-domain/pretrainingoverlapunknown, not externalbenchmark. PeakRSS1.69GiB in1.9GiB environment. Requires pinned sacrebleu2.5.1 via uv run --with.

Run only explicitly, with previously prepared local permitted data and a new output. No automatic downloads or import-time execution. Exact measured result is quality-output.json. This is a reproducible development experiment, not a productionquality guarantee.

```sh
uv sync --locked --extra dev --extra ml
uv run --with sacrebleu==2.5.1 python scripts/quality_cpu.py --data LOCAL_PREPARED_SOURCE --output NEW_OUTPUT --model TRUSTED_LOCAL_MARIAN
```

Test sets described as finalized have now been observed. Future choices must not be justified with their scores as if they remained unseen. Retain source terms/attribution from real-data.md. No model or raw text/photos committed.
