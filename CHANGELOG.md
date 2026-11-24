# Changelog

## [Unreleased]

### Fixed
- Replaced unsupported missing-module, API and evaluation claims with working local CLI behavior.
- Corrected Docker/Compose and Makefile to use the actual package/data checks.

### Added
- Typed parallel CSV validation, source-disjoint split and explicit local Marian CPU training/inference.
- Local safetensors-only loading, target-padding guards and no output overwrite.
- Independent offline synthetic backend, data and CLI tests, scoped uv lock, Ruff/mypy/hooks and public-only CI.

### Unchanged
- Existing MIT license. No main merge, tag, release, real-model/data upload or quality claims.
