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

## 3. Running resident-metal decode as a consumer — **WORKS via the public API** (correction)

**An earlier draft of this note claimed the resident path "is not drivable via the public API." That
was WRONG — it was read off the *stale doc comment* on `decoder.Options.Backend` ("cpu/webgpu"), not
the code.** `Options.Backend` actually accepts `cpu | webgpu | cuda | metal` (`decoder/model.go:301-309`),
and `Load` calls `withResidency()` (`decoder/residency.go:551`) which runs `NewBackend` → `BuildResident`
→ attaches the resident **automatically**. So the full consumer path is:

    import _ "github.com/townsendmerino/goinfer/metal"          // registers the backend (darwin)
    m, _ := decoder.Load(path, decoder.Options{Backend: "metal", Quant: "int4"})
    ch, _ := m.Generate(ctx, prompt, maxTokens, decoder.SamplingParams{Temperature: 0})

**Verified out-of-tree** (fresh module, `CGO_ENABLED=0`): `m.ResidentActive() == true`, `ResidentDecline()
== ""` — resident metal was live, no fallback. The only defect is the **stale doc comment**
(one-line fix; `model.go` is frozen core, so it rides a goldens-gated refresh). No public-API gap.

## 4. Decode tok/s vs the 73.6 claim — **MEASURED out-of-tree: 65.2 tok/s** (decode-only)

With the public path above, the out-of-tree consumer decoded qwen2.5-coder-1.5b (int4, resident metal):
**160 tokens, 65.2 tok/s decode-only** (timed from the first streamed token, prefill excluded), on a
thermally-loaded M1 Pro this session. Consistent with the **73.6** claim, which is a cooler-run figure
(the in-tree `TestZZ_metalDepthBench` spanned ~54–61 tok/s at various depths this session, same thermal
state). Same compiled code in-tree and out — the consumer is not slower.

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

## Verdict (revised 2026-08-15)

- **cgo-free / no-Xcode build: holds** (library/require path; otool-confirmed).
- **Resident-metal decode via the public API: WORKS** — `Load(Backend:"metal")` + blank-import auto-
  wires the resident; verified out-of-tree at **65.2 tok/s** decode-only. My earlier "not drivable"
  claim was a mis-read of a stale doc comment — corrected above. **Only defect: the stale
  `Options.Backend` comment** (one-line fix, frozen-core, goldens-gated).
- **ONE real gap, now FIXED:** `go install …/metal/cmd/serve@v0.13.0` was broken by the committed
  `replace`. **Fixed 2026-08-15** — the dev-convenience replace was removed from all four submodule
  go.mods (`gpu`/`cuda`/`metal`/`demo-agent`); metal/gpu/demo-agent verified replace-free `GOWORK=off`
  on the Mac, cuda pending the box. RELEASING.md updated to stop re-adding it; `go.work` is now the
  mandatory dev mechanism. **Consumers get it when `v0.13.1`/`v0.14.0` is cut from the replace-free tree.**
- **Bit-identity + tautological-gate:** in-tree concerns, not consumer-observable; the CUDA tautology
  has no Metal analog.
