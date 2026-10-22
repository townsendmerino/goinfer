# MEM1: decoder.Load takes the sidecar by default, 2026-10-21

The gates are registered in `docs/tasks/task-memory-campaign-2026-10.md`, §4 MEM1 (MEM1-G), committed before the code.

| | |
|---|---|
| machine | MacBook Pro, M1 Pro, 16 GB, macOS 26.6; the owner's desktop open; load average 3.11 at G5's start |
| build | branch `mem1-sidecar-2026-10`, Go 1.27.2, darwin/arm64 |
| checkpoint | local `~/models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf` (G5); a one-layer llama `.gguf` with a tokenizer, written by the test (G0 to G4) |

## Tier 0 (`decoder/sidecar_library_test.go`)

| gate | result | planted defect that turns it red |
|---|---|---|
| G0 production default (`go run ./testdata/mem1-libload`) | PASS | the default turned off |
| G1 library default opens the sidecar, builds it once, logits equal the app path's | PASS | the default turned off |
| G2 `DirectLoad` writes nothing, loads to the heap | PASS | `DirectLoad` ignored |
| G3 a test binary's zero `Options` stays direct | PASS | the default on in test binaries (found the helper reset the flag to `false` instead of restoring it, which hid this plant; fixed before the read) |
| G4 a read-only directory is refused, naming `Options.DirectLoad`, nothing written | PASS | the refusal's name removed; the refusal turned into a fallback |

## Tier 1, G5 (`g5-libload-mac.txt`)

`go run docs/measurements/mem0-2026-10/testdata/libload.go ~/models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf int4`, one run.

| | MEM0 §4.2 (before, `fc34efd8`) | MEM1 (after) |
|---|---|---|
| Go heap in use after `runtime.GC` + `FreeOSMemory` | 647 MiB | 1 MiB |

**PASS** (bar: ≤ 100 MiB). The one-time transcode took 1.0 s and wrote a 459 MB `.giw` beside the model
(`…int4.cpu-arm64.v17.giw`). The Go runtime's `Sys` read 727 MiB after the load: address space the runtime reserved
during the in-process transcode. It is not resident memory, and the gate reads heap in use.
