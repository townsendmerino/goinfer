# Changelog

All notable changes to goinfer are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

The forward-pass and quantization numerics are parity-gated against HuggingFace
and are the stable contract. The loader and architecture-descriptor surface is
pre-1.0 and may change as new model families and quant formats land.

## [Unreleased]

### Added
- Initial repository scaffold (README, LICENSE, NOTICE, go.mod).

### Notes
- goinfer is extracted from [`aikit`](https://github.com/townsendmerino/aikit)'s
  `decoder` / `tokenizer` / `constrain` packages so the LLM runtime can release
  on its own cadence, independent of aikit's stable retrieval core. The package
  move is tracked in `aikit/docs/aikit-module-split-plan.md`. Pre-extraction
  history for these packages lives in the aikit repository.
