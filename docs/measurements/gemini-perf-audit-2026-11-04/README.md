# An external memory and performance audit, triaged (2026-11-04)

**Question.** Which findings of an external audit of hot-path allocations are real on `main` at a628747e, and is any worth a
change? Ordinary engineering, not a registered gate; the tokenizer's timed pairs follow the bit-identical rule in `CLAUDE.md`.

**Box.** MacBook Pro, Apple M1 Pro, 16 GB, aikit v1.76.0; the probes on Go 1.27.3, the timed pairs on Go 1.27.2. Probes ran as test files
added through `go test -overlay`, so nothing they measured is in the tree.

## Verdicts

| Finding | Verdict | Evidence |
|---|---|---|
| Decoding a WAV before its length check | fixed | an 8 MiB clip at 4 kHz allocated 80 MiB before its refusal (`probes.log`); the handler now reads the length from the header (`multimodal/audio.go:ParseWAV`) |
| Tokenizer allocations, `container/heap` boxing | fixed | below |
| CPU per-layer allocation | parked, for the memory planner (`docs/tasks/task-forward-architecture-2026-10.md`) | about 6.8 MB and 0.4 ms a token on Gemma 4 E2B at depth 1,024: arithmetic from its dimensions and measured `make` times, no real-checkpoint read. The per-key MLA allocations it cites are in the parity-reference path, not the default |
| Goroutines in host sampling | no change | one fork/join a draw, 13 allocations and about 2 KB, not 600 KB; it costs about 10 µs and makes the draw 5.5× faster than serial (1.2 ms against 6.6 ms at 152k vocabulary). `workBuf` was already pooled |
| Activation fan-out at one token | no change | off on amd64, and measured a win on arm64 (`docs/measurements/s05-centering-fold-2026-09-22.md`); a persistent pool lost in `docs/completed/perf-campaign.md` |
| `unionGrammar.Clone` clones twice | no change | 4.8 µs at 32 tools against 618 µs of masking a token. Sharing the branch list, as proposed, breaks the clone's isolation: its `Reset` would reset the live grammar |
| Grammar probe allocations | no change | one allocation of at most 12 bytes per masking call of 450–620 µs |
| SSE chunks as maps | no change | a struct saves 1.3 µs a token and changes the wire's key order |
| Closures and struct copies in the walker | no change | a few small allocations a layer. Passing `arch.qwen35` and `arch.kda` by pointer, as proposed, would write to the shared architecture |
| Unbounded request bodies | wrong | every upload route is under `maxBytes` (`internal/app/serveapp/main.go:server.routes`) |
| `HashInts`, `parseBatchLines`, `SampleDraft` | do not exist | no hit in the tree or in the history of any Go file |
| Gzip without a size limit | no change | development tools reading this repo's own files |

Checking those routes found two bugs the audit did not: a large upload's temp file was never removed where a handler sees a copy
of the request, and an over-limit chunked body answered 400. Both are fixed, with `internal/app/serveapp/uploads_test.go`.

## The tokenizer change

Ten pairs, A B B A in one session under the exclusive timing lock, one second a benchmark: `main` at a628747e against the change. No
cell started above load 3.89 or below 3,422 MiB reclaimable (`tokenizer-ab-10-pairs.log`). Every pair was faster.

| Benchmark | Median | Pairs faster | ns/op | allocs/op | bytes/op |
|---|---:|---:|---|---|---|
| `BenchmarkEncode` (mixed text, 1 KB) | 1.98× | 10/10 | 190 µs to 96 µs | 5,320 to 215 | 130 KB to 40 KB |
| Go source, 27 KB, byte-level | 1.96× | 10/10 | 6.24 ms to 3.16 ms | 165,463 to 5,650 | 4.01 MB to 0.76 MB |
| Go source, 27 KB, Gemma 3's SentencePiece scores | 1.79× | 10/10 | 8.42 ms to 4.63 ms | 111,238 to 2,522 | 3.57 MB to 0.91 MB |

**Identity.** Old and new binaries, run on the same inputs, printed byte-identical hashes: 30 tokenizers (`tokenizer-differential-hashes.txt`:
byte-level, SentencePiece and Tekken), each over 6,101 inputs including 3,000 strings of raw bytes, through `Encode` with and without BOS,
without special-token parsing and through `EncodeSegments`, plus one 2.2 MB gap; and the five splitters by name over 40,102 inputs
(`tokenizer-splitter-hashes.txt`). `tokenizer-probes_test.go.txt` is the probe file. On nobara (amd64) the package reads 61 passed, 0 failed,
including the golden parity tests the Mac skips for want of fixtures (Qwen 3, Llama 3, Mellum 2, Gemma 3, Tekken). Two tests did not run on either
box: `TestLoadGGUF_byteLevelMatchesJSON` (no Llama 3.2 GGUF) and `TestLoadVocabMerges_realQwen3ASR` (heavy tier).
