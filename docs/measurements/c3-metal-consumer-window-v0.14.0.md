# C3 — Metal consumer window, evaluated against `metal/v0.14.0` (2026-08-19)

Out-of-tree consumer evaluation of goinfer's Metal backend, run against the actual published
`metal/v0.14.0` tag (root `goinfer v0.14.0` @ `4d91858`) — the auto-pickup trigger (an aikit bump:
root moved v1.17.1 → v1.21.0 since v0.13.0). Run on `macbook-arm64` (Apple M1 Pro), Go 1.26.6, from
a scratch module outside the repo tree (`GOWORK=off`, no local replace, no checkout). Supersedes
`c3-metal-consumer-window.md` (v0.13.0) rather than editing it — that file's v0.13.0-specific
findings (the broken `go install` path) are historical record, not stale.

## Resolved dependency set

```
require (
	github.com/townsendmerino/goinfer         v0.14.0
	github.com/townsendmerino/goinfer/metal   v0.14.0
	github.com/townsendmerino/aikit           v1.21.0
	github.com/townsendmerino/aikit/gpu       v0.28.0
)
```

Resolved cleanly on the first `go get` — no version conflicts, no manual pinning.

## 1. The headline fix, verified for real: `go install …/metal/cmd/serve@v0.14.0` — **WORKS**

```
GOBIN=<scratch>/bin CGO_ENABLED=0 go install github.com/townsendmerino/goinfer/metal/cmd/serve@v0.14.0
```

Succeeds from a clean GOPATH, no checkout: **13.7 MB binary**. This is v0.13.0's real, unfixed C3
finding closing for real — the committed `replace github.com/townsendmerino/goinfer => ../` that
made `go install pkg@version` reject the module (`go install` treats the target as main and refuses
replace directives) is gone from the tagged tree, as RELEASING.md's Invariants section requires.

`otool -L` on the `serve` binary shows `libSystem.B.dylib`, `libresolv.9.dylib`, plus
`CoreFoundation.framework` and `Security.framework` — more than the pure library-path binary below,
but still no cgo/Xcode-linked dependency (those two frameworks are net/http's TLS/cert-verification
path via Go's runtime dlopen, not C code this module links against; `serve` is an HTTP server,
`main.go` below is not).

## 2. Resident-metal decode via the public API — **WORKS**, cgo-free, no regression

A minimal consumer (blank-import + `decoder.Load(Backend:"metal")` + `Generate`) built with
`CGO_ENABLED=0`:

```go
import (
	"github.com/townsendmerino/goinfer/decoder"
	_ "github.com/townsendmerino/goinfer/metal"
)
m, _ := decoder.Load(path, decoder.Options{Backend: "metal", Quant: "int4"})
ch, gen := m.Generate(ctx, promptIDs, maxTokens, decoder.SamplingParams{Temperature: 0})
```

`otool -L` on this binary: **only** `/usr/lib/libSystem.B.dylib` and `/usr/lib/libresolv.9.dylib` —
the exact library-path result v0.13.0's C3 reported. **The cgo-free / no-Xcode consumer build claim
holds unchanged.**

Run against `qwen2.5-coder-1.5b-instruct-q4_k_m.gguf` (real local checkpoint):

```
ResidentActive: true
ResidentDecline:
DecodePath: metal-resident (int4)
decoded 32 tokens: " Paris. Paris is the largest city in France and the second largest city in Europe. Paris is the most visited city in the world. Paris is the most visited"
decode-only: 65.4 tok/s
```

Resident metal was live, no fallback (`ResidentActive() == true`, `ResidentDecline() == ""`), output
is coherent, and **65.4 tok/s decode-only is consistent with v0.13.0's out-of-tree measurement (65.2
tok/s) on the same box and checkpoint** — no regression across the aikit v1.17.1 → v1.21.0 bump, the
gpt-oss/G10/G11 Metal residency work, or anything else that landed between the two tags.

## 3. What this does NOT re-establish

Same scope as v0.13.0's C3: bit-identity (`TestMetalSnapshotGolden`, in-tree only) and the
tautological-gate census (Metal has no CUDA-graph analog; see `metal/gemma_parity_test.go:84`'s
admission assertion) are in-tree properties, not consumer-observable. Not re-run here; nothing
suggests either regressed.

## Verdict

- **`go install …/metal/cmd/serve@v0.14.0`: FIXED, verified for real.** This is the actual consumer
  outcome the v0.13.0 replace-removal work (`6640dd5` and RELEASING.md's Invariants) was for — the
  first goinfer tag installable from outside the repo.
- **cgo-free / no-Xcode build: holds**, library and binary paths both, otool-confirmed.
- **Resident-metal decode via the public API: WORKS**, out-of-tree, at **65.4 tok/s** — matches the
  prior measurement, no regression.
- Both RELEASING.md Step 2 gates (`GOWORK=off go build -tags metal ./...` standalone build, and
  `scripts/gpu_gate.sh` device gate) passed clean on a committed tree at `4d91858` before this tag
  was cut — see the gate's own archived log for that run.
