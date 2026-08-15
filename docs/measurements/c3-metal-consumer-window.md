# C3 — Metal consumer window, evaluated against `metal/v0.13.0` (2026-08-15)

Out-of-tree consumer evaluation of goinfer's Metal backend — the auto-pickup that fired when the
trigger (a release tag ≥ v0.13.0 carrying the aikit bump) was met. Run on `macbook-arm64`, Go 1.26.6,
Apple M1 Pro. **No result is fabricated;** where a claim was not re-measured out-of-tree, that is said
plainly with the reason.

## Resolved dependency set (what a consumer actually gets)

A fresh consumer module requiring `github.com/townsendmerino/goinfer/metal v0.13.0` resolves:

| module | version |
|---|---|
| `github.com/townsendmerino/goinfer` | v0.13.0 |
| `github.com/townsendmerino/goinfer/metal` | v0.13.0 |
| `github.com/townsendmerino/aikit` | v1.17.1 |
| `github.com/townsendmerino/aikit/gpu` | v0.28.0 |
| `golang.org/x/sys` | v0.47.0 |
| `golang.org/x/text` | v0.40.0 |

## 1. cgo-free / no-Xcode BUILD — the headline claim: **VERIFIED** (library/require path)

A fresh module (`require goinfer/metal v0.13.0`) with a minimal `main` blank-importing the metal
backend built with **`CGO_ENABLED=0`** and ran:
- `otool -L` on the binary shows **only** `/usr/lib/libSystem.B.dylib` and `/usr/lib/libresolv.9.dylib`
  — the standard pure-Go darwin linkage. **No cgo/Xcode-linked C dependency; Metal is reached at
  runtime via dlopen (purego), not link-time cgo.**
- Running it constructed the backend: `metal backend: true, err: <nil>` (dlopened Metal.framework at
  init). **The cgo-free, no-Xcode consumer build claim holds.**

## 2. `go install …/metal/cmd/serve@v0.13.0` — **BROKEN** (real finding)

The natural binary-consumer path fails:

    go install github.com/townsendmerino/goinfer/metal/cmd/serve@v0.13.0
    → "The go.mod file for the module providing named packages contains one or
       more replace directives. It must not contain directives that would cause
       it to be interpreted differently than if it were the main module."

`metal/v0.13.0`'s go.mod ships `replace github.com/townsendmerino/goinfer => ../` — the dev-convenience
line. `go install pkg@version` treats the target module as main and rejects replace directives.
`RELEASING.md` says the replace is "ignored by consumers" — **true only for the *require* path**, where
a dependency's replaces are ignored. `go install pkg@version` is the natural way to obtain a server
binary, and it **cannot**. Consumers who want the `serve` binary must clone-and-build. This is the
B-01/B-05 class the release process guards for the *build gate* (it removes the replace temporarily),
but the **tagged go.mod still ships it**.

## 3. Running resident-metal decode as a consumer — **not cleanly consumable via the public API** (finding)

The resident-metal path is wired internally by the decoder from the registered backend (`internal/serveapp`
drives `decoder.NewBackend` → `BuildResident` → the model's internal resident). There is **no public
one-call attach**, and `decoder.Options.Backend` documents only `"cpu"`/`"webgpu"`. So a *library*
consumer cannot trivially drive resident-metal decode via the public API; the intended path is the
`metal/cmd/serve` binary — which (per finding 2) `go install` cannot build. So end-to-end, the "run
Metal decode as an outside consumer" path is: **clone the repo and build**, not `go get`/`go install`.

## 4. Decode tok/s vs the 73.6 claim — **NOT re-measured out-of-tree** (honest)

The consumer runs identical compiled code; decode tok/s is not consumer-specific, and replicating
`internal/serveapp`'s resident driver purely to re-measure an identical-code number would risk an
approximation dressed as a measurement. The canonical instrument is the in-tree `TestZZ_metalDepthBench`;
this session measured **~54 tok/s at depth 128 on a thermally-loaded M1 Pro** (earlier runs this
session; the 73.6 claim is a cooler-run figure and is consistent with the same code path). **No
out-of-tree tok/s is asserted.**

## 5. Bit-identity vs the snapshot — **in-tree property, not consumer-observable**

A consumer does not run the test suite. Bit-identity is verified in-tree by `TestMetalSnapshotGolden`
(green when run earlier this session). It is not an out-of-tree consumer-window observable, so C3 does
not re-establish it.

## 6. Tautological-gate shape on Metal — **not live in the CUDA form**

The CUDA tautology was four graph tests comparing graphs-on against graphs-off without asserting the
graphs were *admitted*. **Metal has no CUDA-graph capture/replay** — it is a command-buffer execution
model — so that specific shape has **no Metal analog**. The previously-known Metal instance (the
snapshot golden driving `Forward`/`ForwardArgmax` without the embed scale) was already fixed (G-02).
Metal tests assert admission (`metal/gemma_parity_test.go:84` fatals if the resident declined when it
should be admitted). No "gate that can't fail" of the CUDA form is present on Metal.

## Verdict

- **cgo-free / no-Xcode build: holds** (require path).
- **Two consumer-window gaps:** `go install …/metal/cmd/serve@v0.13.0` is broken by the committed
  `replace` (binary consumers must clone-build); resident-metal decode is not drivable via the public
  API (library consumers can't run it without the serve binary).
- **tok/s:** not re-measured out-of-tree (identical code; in-tree ~54–73 tok/s, thermal-dependent).
- **Bit-identity + tautological-gate:** in-tree concerns, not consumer-observable; the CUDA tautology
  has no Metal analog.

Follow-up worth queuing: strip the `replace` from the tagged `metal/go.mod` (or the release process
should tag from a replace-free tree) so `go install …/metal/cmd/serve@vX` works — the standalone-build
gate proves the *build* but the *tag* still ships the replace.
