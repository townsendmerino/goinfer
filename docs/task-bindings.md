# task: multi-language and mobile bindings — sidecar (desktop) + c-archive FFI (mobile)

> Status: **NOT STARTED.** Drafted 2026-07-26. Supersedes an earlier WASM-based plan,
> withdrawn: `GOARCH=wasm` has no SIMD (it falls onto `linalg/dot_other.go` → `dotGeneric`),
> cannot reach the CUDA or Metal backends, and wasm32 caps usable memory near 2–3 GiB.
> Those costs bought nothing the mechanisms below don't provide better.
>
> Prior art: [`iBz-04/quaynor`](https://github.com/iBz-04/quaynor) covers the same four
> targets. Study its **binding ergonomics and packaging**, not its code — its core wraps
> `llama-cpp-2`, and its LICENSE file is GPL-3.0 while every `Cargo.toml` says MIT.

---

## 0. The split

Two deployment shapes, two mechanisms. Do not unify them; the unification attempt is what
produced the WASM plan.

- **Desktop / server → sidecar.** Python and Node ship the goinfer static binary as a
  package asset, spawn it, and speak the OpenAI/Anthropic JSON that `cmd/serve` already
  serves.
- **Mobile → c-archive FFI.** Swift, Flutter, and React Native bind one C-ABI archive
  in-process.

Both run **native ARM64/AMD64 with full SIMD**, both reach the GPU backends, neither has a
model-size ceiling.

### On the cgo-free promise

`-buildmode=c-archive` requires `CGO_ENABLED=1` **at build time, in your CI**. It does not
touch the library or the default build: `go get github.com/townsendmerino/goinfer` stays
pure Go, and an app developer consuming the `.xcframework` never sees a C toolchain. This
is the same quarantine pattern the `gpu` submodule already uses. Say so explicitly in the
bindings README so it never reads as a walk-back.

---

## A — Sidecar bindings (Python, Node)

The cheap half. `cmd/serve` already speaks OpenAI and Anthropic JSON, is already tested,
and already carries constrained decoding through `response_format` / `text.format`. The
binding is a client, not a runtime.

### A0 — Server prerequisites *(small, do first)*

1. **Unix-domain-socket / named-pipe listener.** Add `--unix <path>` to `cmd/serve`
   alongside `--port`. Avoids port collisions, avoids exposing a loopback port, and is the
   right default for an embedded sidecar. Go supports UDS on Windows 10+; fall back to a
   named pipe if that proves flaky.
2. **Readiness line.** Print a single machine-readable line to stdout on listen
   (`{"ready":true,"addr":"..."}`), so a client waits on a signal rather than polling.
3. **Parent-death handling.** The sidecar must exit if its parent does — orphaned model
   processes holding gigabytes are the failure mode users will remember. Poll parent pid,
   or take a health-ping deadline.

### A1 — Python

- Pure-Python client. `httpx` or stdlib over UDS; no compiled extension, no cffi.
- Process manager: spawn on first use, reuse, explicit `close()`, `atexit` cleanup,
  context-manager support.
- Sync API first; streaming as a generator over SSE. Async as a second surface.
- **Packaging:** per-platform wheels with the binary as package data. Go cross-compiles
  every target from one CI job — `darwin/{arm64,amd64}`, `linux/{arm64,amd64}`,
  `windows/amd64`. No manylinux toolchain, no build step for the user. The wheel matrix is
  a `GOOS`/`GOARCH` loop.
- `py.typed`, full stubs, dataclasses for request/response.
- **Headline example is constrained decoding**: a dataclass in, guaranteed-valid JSON out.
  No wrapper over llama.cpp can offer this as a first-class binding feature.

### A2 — Node

- TypeScript client, same shape. Async iterator for streaming.
- **Packaging: the esbuild pattern.** Publish per-platform packages
  (`@goinfer/darwin-arm64`, `@goinfer/linux-x64`, …) and list them as `optionalDependencies`
  of the main package with matching `os`/`cpu` fields. npm installs exactly one. This is
  the proven design for shipping a Go/Rust binary through npm — no `node-gyp`, no
  `postinstall` download, no prebuild matrix. Say that loudly; it is the pitch against
  `node-llama-cpp`.
- Types generated from the same schema as A1, not hand-written.

### A3 — Desktop Flutter / RN

Both can use the sidecar on desktop targets. Ship it as the desktop path of the mobile
packages below rather than as separate products.

---

## B — c-archive FFI (Swift / iOS, Flutter, React Native)

### B0 — Spikes, before any binding work

Three unknowns, each cheap to resolve and each capable of reshaping the task.

1. **Does `metal` work under c-archive on iOS?** This is the highest-value question in the
   whole task. The backend is purego + Obj-C with `CGO_ENABLED=0`; c-archive forces
   `CGO_ENABLED=1`. Those should coexist — cgo being *available* doesn't stop purego from
   working — but iOS also restricts `dlopen` to system frameworks and frameworks inside the
   app bundle. `Metal.framework` is a system framework, so it should resolve. **Verify on a
   real device, not the simulator.** If it works, iOS becomes your *fastest* binding and
   the strongest story in the set.
2. **Go runtime inside a host app.** Go installs signal handlers; embedded contexts on iOS
   and Android have known friction here. Confirm clean startup, that host crash reporters
   still function, and that backgrounding/suspension doesn't wedge the runtime.
3. **Binary size.** A Go archive plus weights against App Store limits and Android APK
   expectations. Measure with a real 0.5B and report it — it will drive whether models ship
   bundled or download on first run.

### B1 — The C ABI *(design once, all three targets consume it)*

**Do not design a new API surface.** Export the JSON that `cmd/serve` already speaks:

```c
void*  gi_alloc(size_t n);
void   gi_free(void* p);
int64_t gi_load(const char* cfg_json, size_t len);              // -> handle
void   gi_unload(int64_t handle);
char*  gi_request(int64_t h, const char* req, size_t len);      // -> JSON, caller frees
int64_t gi_stream(int64_t h, const char* req, size_t len,
                  void (*on_token)(int64_t stream, const char* chunk, size_t len));
void   gi_cancel(int64_t stream);
```

Seven functions. Every binding becomes a thin marshaller over a schema that already exists
and is already documented — and **every binding inherits constrained decoding on day one**,
which is the differentiator a llama.cpp wrapper structurally cannot match.

Watch two things: cgo's pointer-passing rules (no Go pointers stored on the C side), and
calling `on_token` from a Go goroutine back into a host runtime — marshal to the host's
expected thread where required.

### B2 — Build matrix

```bash
# iOS device + simulator -> lipo -> xcframework
CGO_ENABLED=1 GOOS=ios   GOARCH=arm64 go build -buildmode=c-archive -o libgoinfer-ios.a ./cmd/ffi
# Android per-ABI shared objects
CGO_ENABLED=1 GOOS=android GOARCH=arm64 go build -buildmode=c-shared -o libgoinfer.so ./cmd/ffi
```

iOS takes `c-archive` (static, standard for Go on iOS); Android takes `c-shared` per ABI
(`arm64-v8a`, `x86_64`), packaged into an `.aar`. One CI job emits the whole matrix.

### B3 — Swift / iOS

- SwiftPM package with a binary target wrapping the `.xcframework`.
- `async`/`await` surface, `Sendable` conformance, `AsyncSequence` for streaming.
- If B0.1 confirms Metal: **lead the README with it.** "GPU-accelerated on-device
  inference, no Python, no llama.cpp" is the headline.

### B4 — Flutter / Dart

- `dart:ffi` with `ffigen` generating bindings from the C header. Well-trodden with Go
  specifically.
- Run FFI calls on a background `Isolate`; `gi_request` blocks and will jank the UI thread.
- Desktop targets use the sidecar (A3); mobile uses FFI. One public Dart API over both.
- Publish to pub.dev. Match quaynor's ergonomics (`Chat.fromPath(...)`,
  `.ask(...).completed()`) — the shape is good and it's what the ecosystem expects.

### B5 — React Native

- New Architecture: a TurboModule with JSI C++ calling into the archive directly. Reuse the
  TypeScript layer from A2; swap the transport.
- Old Architecture is out of scope — say so rather than half-supporting it.

---

## Cross-cutting

1. **One conformance suite, all five bindings.** Same prompts, same seeds, same expected
   output, in CI. Bindings drift otherwise; nothing else prevents it.
2. **The JSON schema is generated from the same source as `cmd/serve`**, never
   hand-maintained. A binding that can drift from the server is a bug factory.
3. **CI builds each artifact once**; every binding consumes it. Never build per-binding.
4. `docs/capability-matrix.md` gains a bindings column, generated and freshness-gated.
5. Every README states the cgo position honestly (§0) and, where measured, real tok/s with
   provenance — matching the discipline in `docs/benchmarks.md`.

## Non-goals

- Browser / WASM. Genuinely separate — `docs/roadmap.md`'s `GOOS=js` + `syscall/js` →
  `navigator.gpu` demo is the right home for it, and it is a demo, not a binding strategy.
- Training, fine-tuning, LoRA.
- Vision or audio in bindings until text is stable across all five.
- React Native Old Architecture.

## Order of work

`A0 → A1 (Python) → A2 (Node) → B0 spikes → B1 ABI → B3 (Swift) → B4 (Flutter) → B5 (RN)`

A is weeks and nearly risk-free; ship it first and get real binding users while B0 resolves.
**B0.1 — Metal under c-archive on a real iPhone — is the single measurement that determines
how good the mobile story gets.** Run it early, even out of order.
