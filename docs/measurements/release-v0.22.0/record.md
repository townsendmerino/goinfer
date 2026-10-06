# v0.22.0 release record (2026-10-06)

v0.22.0 adds EmbeddingGemma 2 at `/v1/embeddings` (`docs/tasks/task-embeddinggemma2.md`), hours after v0.21.0.
Two of RELEASING.md's pre-flight items are taken from v0.21.0 rather than re-run, by the owner's decision of
2026-10-06, and this file is where that is written down.

## §C1 (parity) and §C1-M (the Metal device gate): carried over from v0.21.0

**Decision (owner, 2026-10-06): "Carry over, written down."** v0.21.0's §C1 sweep is green at `7b50947a`
(`docs/measurements/release-v0.21.0/parity-sweep-2026-10-06.md`) and its Metal device gate passed at `85c50766`
(`docs/measurements/release-v0.21.0/metal-gate-2026-10-05.md`). They stand for v0.22.0 because the code they
exercise is unchanged:

- `git diff --name-only v0.21.0 <this release's commit> -- decoder metal cuda gpu` lists only
  `cuda/go.mod cuda/go.sum gpu/go.mod gpu/go.sum metal/go.mod metal/go.sum `: the submodules' requires moving to goinfer v0.21.0, v0.21.0's own release step 2. No `.go` file in
  `decoder`, `metal`, `cuda` or `gpu` changed.
- `TestParityManifest_fresh` passes: no hashed core file moved.
- What this release adds runs outside them: a new package, `embeddinggemma2/`, which `decoder.Load` never
  reaches; `internal/serveapp`'s embeddings handler and loader; and `pull.EncoderLoads`. The new package has
  its own gates, passed today: Gate 1 (tiny fixture, 12 planted defects red) and Gate 2 (the real checkpoint against
  sentence-transformers, 48 texts, ids exact, worst cosine 0.999999987 on the release's attention).

## Pre-flight 5 (a cold-user run on the previous tag): waived this once

**Decision (owner, 2026-10-06): "Skip it this once, written down."** The previous tag, v0.21.0, was published
hours before this one, and its own cold-user run (on v0.20.0) was today
(`docs/measurements/cold-user-2026-10-05-nobara-pc.md`). The next release's cold-user run is owed on v0.22.0 and
is not waived.

## Checked at the release commit (MacBook Pro M1 Pro, darwin/arm64)

- `go build ./...`; `go vet` with and without `realckpt` on the new package; staticcheck on linux/amd64; gofmt.
- `go test` on `embeddinggemma2`, `internal/serveapp`, `pull`; `TestParityManifest_fresh`,
  `TestAikitPinsAgree`, `TestEnvVars_docAndCodeAgree`.
- `GOINFER_RELEASE_TAG=v0.22.0 go test ./cmd/gate/ -run '^TestParity_noPendingGateOutlivesARelease$'`: "nothing has
  been pending since before v0.21.0".
- The hardware matrix's never-executed list is unchanged by this release: EmbeddingGemma 2 runs on the CPU only, and
  the docs say so.
