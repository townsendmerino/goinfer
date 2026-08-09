# Changelog

All notable changes to goinfer are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

The forward-pass and quantization numerics are parity-gated against HuggingFace
and are the stable contract. The loader and architecture-descriptor surface is
pre-1.0 and may change as new model families and quant formats land.

## [Unreleased]

### Fixed
- **`POST /admin/models/unload` now frees the model's native memory instead of leaking it.** Unload
  deleted the entry from the registry but never called `Close()`, and with no ARC or finalizers under
  purego, GC reclaimed only the Go wrappers — so each unload/reload cycle leaked a full model
  (measured ~450 MB on Metal; on an 8 GB CUDA card the reload then OOMs). Adding a bare `Close()` was
  a use-after-free: a request between resolving the model and taking its mutex touches the model
  unlocked, so `Close()` could free weights mid-request (a driver SIGSEGV on CUDA). The fix is a
  drain: a per-`*decoder.Model` liveness `RWMutex` is held by every request from resolution through
  completion (via a single `withModel` wrapper; the old `pick` is gone so no handler can skip it),
  and unload unpublishes the model, then on a detached goroutine waits that lock out before closing —
  freeing the shared model only when it is the last owner (a base and its adapters share one). Design
  and rationale: `docs/task-admin-unload-drain.md`.

### Changed
- **`POST /admin/models/unload` no longer returns `409 busy` for a model that is generating; it
  drains.** The model is unpublished immediately (unroutable at once), then the response is a bounded
  wait: `200 {"freed":true}` when the native memory is released within `-unload-drain-wait` (default
  5s), else `202 {"freed":false}` with the release continuing in the background. `GET /health` gains a
  `draining` array listing models whose memory is not yet freed; `?wait=false` returns `202`
  immediately. The old `409` was only ever safe because unload never actually freed anything.
- **New flag `-unload-drain-wait` (default `5s`)** — how long an unload waits for in-flight requests
  to drain before answering `202` (the release always completes regardless).

## [v0.10.3] — 2026-08-08

### Security
- **Request bodies are bounded before they are buffered or tokenized (G1/G2/G3).** An oversized
  body was read and tokenized in full before the context-window check ran, so a body under the
  32 MiB chat cap (e.g. 20 MB) drove RSS up by gigabytes and spent ~27 s tokenizing only to be
  rejected 400, and a ≥40 MB body tripped the reader mid-upload and closed the connection with no
  HTTP response. This ran before the request queue, so `-max-queue` did not bound it. Now, in
  increasing cost order: a **Content-Length pre-check** rejects an over-cap declared body with a 413
  before a byte is read (O(1) in body size — measured 6–145 µs, ~5 KB heap at 1/20/40 MB); the
  existing `MaxBytesReader` backstop's 413 now **names the limit and received size**; a
  **pre-tokenization guard** rejects an input whose text cannot fit the context window before the
  O(n) tokenize (the ~27 s → µs fix), as a conservative upper bound that never rejects a servable
  prompt; and the body cap is **derived per route** (`-max-body-bytes` to override every route) and
  reported on the startup line. Gated in `internal/serveapp/bodylimit_test.go`.
  - **The cap is derived per route, not globally** (pre-tag review). The text cap comes from the
    largest served *decoder's* context window; the vision routes (`/v1/chat/completions`,
    `/v1/messages`) add 32 MiB for base64 image data; and **`/v1/embeddings` gets its own 64 MiB
    cap, independent of any decoder**. That last one was a real false rejection: `/v1/embeddings` is
    served by the encoder, which is not in the decoder registry, so on an embed-only server the cap
    collapsed to the 4 MiB text floor and rejected a batch the route's own bounds
    (`maxEmbedInputs=2048`, `maxEmbedInputBytes=1 MiB`) accept — a legal 2048×4 KiB batch is ~8 MiB.
    Multimodal was checked and is safe: the pre-tokenization guard measures only `type:"text"`
    content parts, so base64 image bytes are never charged against a context window they do not
    consume. Both pinned in `internal/serveapp/bodycaps_routes_test.go`.

### Fixed
- **`temperature` is validated consistently with `top_p` (G4).** A negative `temperature` returned
  200 while `top_p=-1` returned 400. It was not silently-wrong output — the sampler treats
  `temperature <= 0` as greedy argmax before any scaling — but the inconsistency is fixed: negative
  `temperature` is now a 400; `0` stays valid as greedy/deterministic.
- **A chat request with no `messages` is now a 400 (G5)** naming the field, instead of a 200
  generated from a BOS-only prompt.
- **An unknown `model` name is rejected on a single-model server (G6).** `pick` fell back to the
  only loaded model for any name, so a wrong id produced confident output from a different model; it
  now serves an omitted name but rejects a non-empty unknown one, naming what is served (the
  multi-model behavior was already correct).
- **`POST /v1/embeddings` is registered even when no embedding model is loaded (G7).** It returned
  Go's text/plain `404 page not found` (SDKs surface it as a wrong-URL `NotFoundError`); it now
  returns a JSON 501 naming the `-embed-model` flag that enables it.
- **Sampling with `top_p`/`top_k` no longer full-sorts the whole vocabulary on the host every
  token.** The filtered-sampling path softmaxed all V logits and then ran a `sort.Slice` over all V
  — an O(V·log V) sort whose per-comparison reflection cost made it *present* as linear in V, single-
  threaded on the decode critical path. It is replaced with bounded selection in logit space
  (`decoder/sampler.go`): a k-bounded min-heap for `top_k` (O(V·log k)), an adaptive partial
  selection for the `top_p` nucleus (grown until the retained mass provably reaches `p`), and a
  `min_p` logit-space threshold — with the softmax applied **only to the retained set**, not all V.
  Measured on synthetic logits at `temperature=0.8, top_p=0.95`: selection cost **122.3 ms→1.80
  ms/tok at 152 064-vocab (68×)** and **233.0 ms→4.37 ms/tok at 262 144-vocab (53×)**; the full
  sampling path at `temperature+top_p` now runs within **1.1–1.5×** of temperature-only sampling
  (was ~7×). `top_k` no longer funnels through the general `top_p` routine. A test-only reference
  implementation gates the bounded path bit-for-bit across a wide seed sweep and both vocab sizes
  (`decoder/sampler_selection_test.go`), and a throughput gate asserts the ratio to the
  temperature-only baseline. Speculative decoding's distribution vector uses the same selection, so
  it stays lossless. **This does not address the separate `temperature==0`→nonzero cliff** (loss of
  on-device argmax → full logit readback per token); that needs device-side kernels and is scoped in
  `docs/ollama-chase.md` §8 D6.

### Changed
- **Tied-probability selection order is now specified.** The old `sort.Slice` filter was not stable,
  so tokens with equal probability were ordered arbitrarily — and since that order feeds the
  cumulative-CDF draw, it was an unspecified part of the sampling result. Ties now resolve by
  **ascending token id**, and the renormalization/`top_p` sums run in a fixed **descending-
  probability** order (a load-bearing contract: a different summation order can move the denominator
  by ULPs and flip a draw near a `top_p` boundary). Greedy (`temperature==0`) argmax is unchanged.
  Sampled output for a given seed may therefore differ from prior releases at tie/boundary points;
  the distribution is unchanged. (Same defect class as the CUDA argmax-reduce index tie-break.
  **Correction (2026-08-09):** this entry said that tie-break was "open ... not fixed here". It was
  already fixed — audit C-14, `c6600fc`, 2026-08-05 — which predates this release, so the statement
  was wrong at press time, not merely overtaken. Gate: `cuda.TestArgmaxTieBreak`.)
- **`docs/benchmarks.md` now requires sampling configuration as a per-number metadata field**, on
  the same footing as machine/driver/peer-version/date, for goinfer and the peer. Rows whose
  sampling config was not recorded are marked `sampling: unrecorded` and must not be assumed greedy
  (the one such row is the §B2 v0.32.5 re-measure box).

## [v0.10.2] — 2026-08-08

### Changed
- **The `.giw` bundle format (v5) now records the resolved quant label in its header** instead of
  leaving the reader to re-infer it (T1-6 follow-up, structural). A freshly baked bundle stores
  `int4` / `int4mix` / `int8int8` / `int8` / `native`, and the loader **prefers** that field; a
  pre-v5 bundle (or a streamed bundle, which writes the header before its layers load) has the field
  absent and falls back to the corrected tensor-kind inference — so **existing bundles keep working
  unchanged**. The reader still accepts v3/v4. Also single-sourced the "which tensors define the
  quant" fact: `quantLabel` and the recorded field now both classify over one `bodyMatmulWeights`
  (matmuls minus the int8-pinned logit tables), so the T1-6 class of bug — two lists of one fact
  drifting apart — cannot reopen within the decoder. (The cuda batched-prefill gate inspects the
  resident per-layer projections, a different type in a different module; it agrees on excluding the
  logit tables but cannot share the host-side list — documented at both sites.)

### Fixed
- **`--quant` on a prequantized `.giw` bundle is now a startup error when it conflicts, instead of
  being silently ignored** (T1-7). A `.giw` is serialized at a fixed precision, so `--quant` cannot
  re-quantize it — but it was accepted and dropped, so a user who passed `--quant int8int8` at a
  `.giw` baked at int4 got int4 with no signal. `serve` and `chat` now fail before binding a port:
  `--quant "int8int8" cannot apply to the prequantized .giw bundle <file> — it is baked at "int4",
  and a .giw carries its own quant; pass --quant int4 or omit --quant` (same shape as the
  safetensors int4mix decline). The comparison uses the corrected `Quant()` label, not the raw
  header field. A **matching** quant proceeds silently, and a bare default never conflicts — only an
  *explicit* `--quant` (global flag actually passed, or a per-model `quant=` override) is checked, so
  scripting one command across GGUF and `.giw` targets still works.
- **The submodule build command is now correct in every active doc** (field-report F2). The README
  build section already carried the verified per-backend entrypoints (v0.10.1); this sweeps the
  rest: `docs/cuda-backend.md` used the pre-split `go build -tags cuda ./cmd/serve` (the root, now a
  no-op) — replaced with the verified `CGO_ENABLED=0 go build -tags cuda github.com/townsendmerino/goinfer/cuda/cmd/serve`.
  The historical `docs/releases/v0.9.0.md` keeps its original command (correct at v0.9.0) but gains
  a forward-pointer note; `docs/completed/audit-2026-08-05.md` is left verbatim (it documents the
  finding). Every command written was run in-tree first. Note: the *published* v0.10.0 GitHub
  release notes still show the pre-split command and a literal `<exact command — fill from the repo
  before publishing>` placeholder — flagged for a maintainer decision (not editable from the repo).
- **`go build -tags cuda|gpu|metal ./cmd/serve` on the ROOT module now fails the build instead of
  silently producing a CPU binary** (audit D-B / field-report F1). Since the M-19 submodule split
  the root `cmd/serve` imports no backend, so those tags were accepted, exited 0, and yielded a
  CPU binary 48 bytes larger than the untagged one — the trap anyone upgrading from a cached
  README, blog post, or the v0.9.0 release page falls into. The v0.10.1 guard already failed the
  build, but with one generic message; it is now **one guard file per tag**, and the compiler
  error names the *exact* replacement command (`… go build -tags cuda github.com/townsendmerino/goinfer/cuda/cmd/serve`,
  and the gpu/metal equivalents). A new `TestBackendTagGuardFailsBuild` runs in the default
  `go test ./...` (no `-short`, no build tag): it shells out `go build -tags <backend> ./cmd/serve`
  for all three backends and asserts both a non-zero exit and the exact replacement command in
  stderr. The submodule entrypoints (`./cuda/cmd/serve`, `./gpu/cmd/serve`, `./metal/cmd/serve`)
  are a different module and build unaffected — verified after the change.
- **A prequantized int4 `.giw` bundle no longer mislabels itself as `int4mix`** on `/health`,
  `/v1/models`, and the startup line. `-quant int4` pins the embedding / LM head to int8 by
  default (logit-critical; `--embed-int4` opts out), and the label inference counted those tables
  — so an all-int4-projection bundle read as `int4mix` while the batched-prefill gate (which
  inspects only the seven projections) correctly batched it. The label now scans just the body
  matmuls the quant actually selects, matching the gate; the int8-pinned logit tables are
  excluded. Generation was always correct (the bundle is genuinely int4 — bit-identical logprobs
  to the GGUF int4 load); only the reported label and the KV-snapshot fingerprint were wrong.
  Direct (non-`.giw`) loads were unaffected — they report the requested quant string verbatim.
  (Reported as T1-6.)

## [v0.10.1] — 2026-08-07

### Added
- **Batched CUDA prefill for int8 weights (`int8`/`int8int8`), closing the "int4-only" TTFT gap.**
  A new `gemv_w8a8_batched.cu` (the per-row W8A8 GEMV plus a tiled multi-column loop) lets the
  dense resident prefill path batch int8 and int4mix bundles, not just int4. Because int8 is
  per-row symmetric — the dot accumulates in exact int32 and the scales apply once at the end —
  the batched result is **bit-identical** to the sequential per-token GEMV *by construction* (no
  reduction-order sensitivity, unlike the int4 lane's group-scaled float sum). Gated at three
  levels on both int4 and int8int8: kernel bit-identity at M ∈ {1,8,45,100}, prefill KV +
  logits bit-identical across all layers × all rows past a sliding window, and a 64-token greedy
  decode byte-identical. Measured TTFT (qwen2.5-coder-1.5b, RTX 2070 SUPER), sequential →
  batched: **128 tok 642→312 ms (2.06×), 512 tok 2.73→1.47 s (1.86×), 2048 tok 12.3→6.69 s
  (1.84×)**. The win is smaller than int4's (~4.4–5.7×) because int8 doubles the weight bytes per
  row on a bandwidth-bound kernel — but the ~9× sequential fallback the old int8int8 default hit
  is gone, and **every quantized mode now batches** (only native f32 stays sequential).

### Changed
- **BREAKING (behaviour): the default `--quant` moved from `int8int8` to `int4`.** int4 is
  smaller in RAM and, being the lightest quant on a bandwidth-bound kernel, the fastest to
  prefill and decode. **Output will differ from prior versions for anyone who relied on the
  default** — int4 is lossier than int8 (a decode/quality change, not just speed). Pass
  `--quant int8int8` (or `int8`/`int4mix`/`""`) to keep the old behaviour. `demo/chat` aligned
  to the same default; `demo/gemma` stays native-f32 (a faithful inspection CLI) and
  `cmd/prequant` stays `int8int8` (it bakes a `.giw` bundle that carries its own quant). The
  flag help now enumerates all five values with their accuracy/speed/RAM tradeoffs — including
  that **`--backend metal` requires `int8int8`** (int4 declines to CPU on the dense Metal
  resident path), and that all quantized modes get batched prefill while native f32 falls back to
  the sequential path. (At v0.10.0 batched prefill was int4-only, so this default flip also moved
  the out-of-box path off the ~9× slower sequential prefill; the int8 batched-prefill work above
  now removes that restriction entirely.)

### Fixed
- **The pre-v0.10.0 GPU build command no longer silently produces a CPU binary.** After M-19
  moved the backend imports to the submodule entrypoints, `go build -tags cuda|gpu|metal
  …/cmd/serve` (the *root* command in every ≤v0.9.x doc/note) kept exiting 0 while building an
  inert CPU binary — you only found out from a warning line at startup. The root `cmd/serve`,
  `demo/chat`, and `demo/gemma` now carry a `//go:build cuda || gpu || metal` guard that **fails
  the build** with a message naming the submodule entrypoint to use instead. (Reported by an
  external v0.10.0 trial, F1.)
- **The runtime "backend not registered" note pointed at the command that no-ops.** It said
  "build `-tags cuda`" — exactly what the user did — instead of naming the submodule entrypoint.
  Now: "build the submodule entrypoint (… `…/cuda/cmd/serve`) — not `-tags cuda` on the root".
- **README documents the GPU `serve` entrypoints, including the out-of-tree module-path form**
  (`go build -tags cuda github.com/townsendmerino/goinfer/cuda/cmd/serve`) and an explicit
  upgrade note. Previously the only GPU example was the in-tree `cd cuda && … ./cmd/chat` (the
  REPL, and `cd cuda` presumes a checkout an out-of-tree consumer doesn't have). (F2.)

## [v0.10.0] — 2026-08-07

### Added
- **Resolved compute paths are reported, not inferred.** Both GPU fast paths — the resident
  decode runner and the batched prefill — decline *per call* and fall back correctly but
  silently. The sharp case: `--backend cuda --quant int8int8` on a dense model builds a full
  resident decode path (looks healthy, decodes at ~0.7× int4) and then takes the sequential
  per-token prefill on every prompt, because the batched GEMV is int4-only — **TTFT 1.73 s vs
  0.19 s on a 300-token prompt (9×), 4.56 vs 0.22 CPU-seconds (20×)**, with nothing logged.
  Now:
  - serve prints the **resolved** `decode path` / `prefill path` per model at load;
  - **`GET /health`** (new) carries `decode_path`, `prefill_batched`, `prefill_path`; the same
    three fields ride on each `GET /v1/models` entry as a **vendor extension** (unknown keys
    are ignored by the Go/Python/JS OpenAI clients, but `/health` has no schema contract to
    break for strictly-typed decoders);
  - **`--require-backend`** (new, opt-in) exits non-zero at startup on either decline, so a
    batch client fails at second zero instead of discovering a 9× under load;
  - `Model.ResidentDecline` names *which* residency decline applies — module not built in, no
    usable device, or ineligible arch — since the three need different fixes and are otherwise
    indistinguishable from outside.

  The prefill report shares `prefillStaticDecline` with `prefillCore` rather than restating
  its conditions, so the startup line cannot drift from the decline it describes.

### Changed
- **BREAKING (build): the opt-in GPU/CUDA/Metal builds moved to submodule entrypoints** (audit
  M-19). The pure-Go root module no longer imports any backend, so `go install
  …/goinfer/cmd/serve@latest` — and any SBOM/vuln scan of the root — no longer resolves
  `cogentcore/webgpu`, `ebitengine/purego`, `eitamring/gocudrv`, or `aikit/gpu`. The accelerated
  binaries now live in the submodules:
  - was `go run -tags cuda ./demo/chat` → now `cd cuda && go run -tags cuda ./cmd/chat`
    (likewise `./cmd/serve`); `gpu/cmd/*` under `-tags gpu`.
  - was `go run -tags metal ./demo/gemma` → now `cd metal && go run ./cmd/gemma` (the metal
    module is `darwin`-gated, so no `-tags metal`); `metal/cmd/{serve,chat}` too.
  The root `cmd/serve`, `demo/chat`, `demo/gemma` are unchanged as **pure-Go CPU** binaries. Serve
  logic moved to importable `internal/serveapp` / `internal/chatapp` / `internal/gemmaapp`.
- **BREAKING (API): `decoder.StreamTranscodeGGUF` gained a leading `ctx context.Context`** (audit
  M-21), so a minutes-long, multi-GB `.gguf`→`.giw` transcode is cancellable — a cancelled context
  aborts at the next layer boundary. `cmd/prequant` now aborts on Ctrl-C/SIGTERM; serve's admin
  model-load cancels on client disconnect.
- **BREAKING (API): `cuda.SpecStats` renamed to `cuda.GPUSpecStats`** (audit M-22) so the GPU
  batched-verify counters no longer share a name with `decoder.SpecStats` (CPU-spec counters); the
  two have distinct, non-1:1 field sets and now carry reciprocal doc comments.

### Fixed
- **The entire 2026-08-05 code audit is closed** ([`docs/completed/audit-2026-08-05.md`](docs/completed/audit-2026-08-05.md)):
  14/14 blockers, 31/31 criticals, 6/6 gates, 23/23 majors (+ the later latent M-25), 24/24 minors.
  Behaviour-affecting highlights:
  - **No silently-wrong output:** gemma4 session KV snapshots refuse a per-layer-KV geometry the
    format can't restore instead of mis-slicing on reuse (C-05); aborted Metal command buffers surface
    as an error instead of the host reading stale logits (C-09); the resident build declines odd
    (non-%8 / non-%32) GEMV/pack widths → CPU fallback rather than corrupt logits (C-10/C-11); the cuda
    f16 scale conversion is byte-canonical (C-15); group-limited MoE masks trailing experts to −inf
    (N-02); the vision projector errors when `mm_tokens_per_image` exceeds the patch grid, instead of
    emitting silent NaNs (N-19).
  - **API / serving:** `/v1/responses` reports `status:"incomplete"` (+ `incomplete_details`) when cut
    off by `max_output_tokens` (N-15); a chat request carrying both `tools` and an image now 400s
    instead of silently dropping the tools (N-16); the backend-fallback note goes to stderr, never
    contaminating a piped token stream (N-03).
  - **Hostile-input robustness:** the resident build declines a threadgroup-memory budget over the
    device limit (M-11); the GGUF metadata reader bounds a hostile `n_dims` / nested-array header into
    a typed error instead of a ~4e9-iteration spin or stack overflow (N-17/N-18); a corrupt tokenizer
    merges array is a typed error, not a misleading "decode-only vocab" (N-20).
  - **Concurrency:** resident `GenerateSpeculative` claims (C-03) and recurrent + multimodal
    cross-sequence state resets (C-01 / M-25) are regression-gated; the adapter registry and the webgpu
    fallback counter are race-clean (C-29 / N-06).
  - **Cancellation & perf:** multi-GB `.gguf`→`.giw` transcodes are cancellable (M-21, above); the
    ~152k-entry constraint token→bytes table is built once per model, not per request (N-14); the
    fused-decode rmsnorm dispatches one workgroup, not 64 (N-07).
  - Plus test-gate hardening (env/fixture independence — G-03/G-05, `GOINFER_MODELS_DIR` — G-06) and
    dependency hygiene (pure-Go root module graph — M-19, above; `go 1.26.5` in demo/agent — N-23).
- **Two independent post-audit reviews closed** ([`docs/completed/goinfer-post-audit-review.md`](docs/completed/goinfer-post-audit-review.md),
  [`docs/completed/goinfer-fresh-review-2026-08-07b.md`](docs/completed/goinfer-fresh-review-2026-08-07b.md)):
  30 findings (R-01…R-30) + 7 follow-ups (F-01…F-07) — residuals and regressions of the audit fixes,
  all fixed and device-verified (the CUDA leg on the RTX box). Behaviour-affecting highlights:
  - an adapter (LoRA) request against a resident base returned **base-model output at HTTP 200** — now
    routed down the adapter-applying session path (R-01);
  - a transient `.giw` read error on the Metal paged 26B decode could **crash the whole server mid-token**
    — now a failed request via the command-buffer-abort path (R-02);
  - a CUDA gemma-4 MoE accumulator zero-fill raced the previous layer's kernels (R-03);
  - the webgpu **MLA (DeepSeek/Kimi) residency path was silently disabled** by an over-broad head-dim
    guard introduced with the M-12 fix (R-05);
  - the resident context cap over-rejected vision and adapter prompts the CPU path serves (F-01);
  - plus crafted-`.giw` shape validation (R-07/F-02), JSON-error Go-type-name leaks (R-11/F-03), a
    defective FMA lint that couldn't see its own kernel (R-04), and several bounded GPU resource leaks
    (R-06/R-17).
- **Docs: batched-prefill coverage was stated per family without the int4-only caveat**, in
  both `CHANGELOG` and `docs/releases/v0.9.0.md`. A family inside the covered seven, loaded at
  `--quant int8int8`, also falls back to sequential prefill. Corrected in place.
- **`TestQwen2Moe_forwardParity` `Fatalf`'d on a partial fixture** — it stat'd
  `model.safetensors` but not `config.json`, so a half-present checkpoint (interrupted HF
  download) failed as if it were a numeric parity regression, which made
  `scripts/refresh_parity_hashes.sh` refuse a provably non-numeric refresh. It now skips
  unless every file `Load` needs is present.

### Known
- The int8 prefill fallback itself is **not yet fixed** — only made visible. The fix is scoped
  in `docs/ollama-chase.md` §C6: int8 is per-row symmetric, so an int32-accumulated batched
  W8A8 GEMM is bit-identical *by construction* (unlike the int4 lane, where group scales force
  a non-associative float sum). Speed is unmeasured and deliberately not predicted.
## [v0.9.2] — 2026-08-05

Docs only; no code, `go.mod`, or numerics change from v0.9.1.

### Fixed
- **README benchmark figures corrected.** The 1.5B 2048-context row was measured before
  split-KV attention landed and understated the result (~133 tok/s / 0.71× → **160.1 tok/s
  / 0.86×**); the table now carries the full four-depth curve (128 / 512 / 2048 / 3900) from
  `46829cc` and states the decode crossover at roughly 1000 tokens of context.
- **Two overstated claims corrected** — the feature taxonomy declines an architecture a
  backend can't fully run rather than making wrong output "structurally impossible", and the
  CUDA speed claim is stated against the measured curve. Docs only; no behaviour change.

## [v0.9.1] — 2026-08-05

Release plumbing only, no code or numerics change from v0.9.0: the final step of the
tri-module tag — the root module now requires the tagged `gpu` / `cuda` / `metal` v0.9.0
(previously Aug-2 pseudo-versions), closing the module requirement cycle so the opt-in GPU
backends are `go get`-able at a real version.

## [v0.9.0] — 2026-08-04

Theme: **two cgo-free GPU decode backends land as opt-in — CUDA (Linux/NVIDIA) and Metal
(macOS/Apple Silicon) — joined by speculative decoding, batched prefill, and long-context
decode, all under a bit-identity contract that makes every GPU path byte-reproducible
against the CPU reference.** Both backends are `CGO_ENABLED=0` (pure Go via purego/dlopen),
behind build tags + `--backend`, with graceful CPU fallback — so the default pure-Go build
is unchanged. **Pre-1.0:** the full per-family parity backfill remains the 1.0 gate;
families without both a T1 committed golden and a current T3 manifest row ship
*experimental* (`docs/parity-coverage-policy.md`). See `docs/completed/task-cuda-cgofree-spike.md`,
`docs/completed/task-metal-cgofree-spike.md`, `docs/completed/metal-verdict.md`, `docs/ollama-chase.md`, and
`docs/spec/` for the full arcs, scorecards, and dead ends.

> **Peer numbers re-anchored.** Every goinfer-vs-Ollama comparison in `docs/benchmarks.md`
> was re-measured against **current Ollama v0.32.5** (the prior figures used 0.5.7 / 0.32.0,
> 12–18 months stale). Lead with absolute tok/s + the cgo-free property, never a peer
> multiple — the ratio is size-, depth-, and platform-dependent (below).

### Added
- **cgo-free CUDA backend** (`cuda/`, `-tags cuda`, `--backend cuda`) — resident decode +
  prefill via **gocudrv** (dlopen `libcuda`) + **NVRTC** (PTX JIT), no cgo. Measured on an
  RTX 2070 SUPER (real q4_K_M): **1.5B 218.6 tok/s, 0.5B 507.5 tok/s** (goinfer absolute).
  Optimization arc: coalesced W4A8 GEMV + 2× ILP unroll (a dp4a-latency frontier, ~46% of
  peak), launch diet (18→13 dispatches/layer), f16 group scales, on-device greedy argmax
  (`ResidentGreedy`), pinned logits readback, and super-kernels **K1/K3a** (fold rmsnorm
  into the QKV / gate-up GEMV, +21.3% on 0.5B; K2 measured null and reverted). Qwen3 runs
  resident (QK-norm kernel). CI builds/vets/tests `-tags cuda`; the executor tax is 15.3
  µs/token (0.34%); the forward is guarded by a mutation-proven parity gate
  (`TestRealForwardParity`, `t.Fatalf` on any position past a 3% logit-range flip).
- **cgo-free Metal backend** (`metal/`, `-tags metal`, darwin-only, `--backend metal`) —
  resident decode + prefill via **purego-objc** (dlopen Metal.framework, `objc_msgSend`,
  MSL 3.1 compiled at runtime). Defused the `LC_BUILD_VERSION`/MSL-2.4 landmine
  (golang/go#77917; explicit MSL 3.1 + read-back assertion). Decode arc **~20 → 73.6 tok/s**
  (W4A8, best-of-40 warm); the practical batch-1 ceiling on M1 without DP4A. Server-to-server
  wall-clock vs Ollama-Metal v0.32.5 (`benchmarks.md` §B3): 0.5B **0.96×**, 1.5B **0.74×** —
  Metal's story is cgo-free / no-Xcode, not raw speed.
- **Speculative decoding** — opt-in, lossless. **CPU n-gram** (`--spec ngram`, prompt-lookup
  drafter over batched verify+rollback) wins on copy-heavy traffic (rag-copy 1.55×, agent-json
  1.59× on 0.5B; code-edit ~parity) — greedy bit-exact vs sequential (`TestNgramSpeculativeGreedyParity`,
  `TestResidentSpecServe`). **Grammar-fused** drafting for constrained/tool requests (forced
  DFA bytes drafted free, lossless, modest). **Resident CUDA (D1)** — batched M=k verify,
  **1.23×@128 / 1.86×@512 / 1.18×@2048**, byte-identical to plain greedy
  (`TestSpecDecodeCurve`). Recurrent families (Mamba-2 SSM, Gated DeltaNet) and the staged
  sliding-window ring are **guarded out** → plain decode (`TestSpecRollbackSafetyGuard`).
- **Batched prefill (CUDA), default-on, bit-identical** — weight-stationary batched W4A8
  GEMV ingests the whole prompt in one M=len pass. TTFT vs sequential: 128 **5.78×**, 2048
  **6.17×** (13.1 s → 2.1 s). Families joined bit-identically: llama, mistral, phi3, qwen2,
  qwen2_5_vl, qwen3 (batched qk-norm), gemma3 (batched sandwich norms) — 7 of 23
  (`TestPrefillCoverageAudit`). **W4A8 means int4 only**: a covered family loaded at
  `--quant int8int8` still takes the sequential prefill (9× TTFT) while building a full
  resident decode path. Reported at load — see [Unreleased]. The Metal MMA prefill path (3.7× TTFT) exists but **declines
  by default** (not bit-identical — 54% stream divergence; opt-in `GOINFER_METAL_BATCHED_PREFILL=1`).
- **Split-KV long-context decode (CUDA), default-on ≥256 ctx, bit-identical** — splits the
  independent key/value axes without touching the reduction (FA's online rescale is *not*
  bit-exact, so it is deliberately not used). 2048 decode **133→160 tok/s (1.20×)**; with the
  preceding attention coalescing, **99.5→160 = 1.61× over glue** (Campaign A, closed at the
  bit-identity ceiling). Gates `TestSplitKV_bitIdentical{,_gemma3}`; `GOINFER_SPLITKV_ATTN=0`
  disables.
- **Attention coalescing** — half4/float4-coalesced decode+prefill attention K-read.
  **Metal**: bit-identical, isolated kernel 1.79× @2048, end-to-end decode **1.37×@2048 /
  1.40×@4000** (the first Metal decode speed win, long-context). **CUDA prefill**: float4 QK,
  3.1× isolated, bit-identical.
- **Model-family coverage expanded.** Resident on **both** backends: Qwen2/Llama, **Qwen3**
  (QK-norm), **Gemma3** (sandwich norm / gated-GELU / embed-scale / per-layer RoPE), **MoE**
  (router + stacked int4 experts + shared expert; Mixtral / Qwen2-MoE / Qwen3-MoE / GLM-4.5/4.6).
  Metal-only: **Mistral** (sliding-window), **Phi-3** (partial rotary). **Cohere / Command-R
  v1 + v2** land CPU-correct (cosine 1.0 vs HF; new NormParallel / interleaved-RoPE / bias-free
  LayerNorm / per-layer-NoPE / reciprocal-LogitScale primitives; committed tiny fixtures —
  below). Every GPU backend auto-declines what it cannot run.
- **Shared feature taxonomy** (`decoder/features.go`) — admission is one subset check over
  arch-flag-derived requirements vs each backend's declared implemented set; a registry-driven
  gate (`TestResidentAdmission_registryCovered`) fails CI on any unclassified arch. Eliminates
  the silent-wrong-output class (a family the backend can't fully run now declines to CPU
  instead of running with a feature dropped). Metal + CUDA + WebGPU all use it.
- **Gemma-4 26B-A4B fully GPU-resident via host↔VRAM expert paging (CUDA)** — decodes a
  ~15 GB MoE on an 8 GB card at **16.98 tok/s** by streaming routed experts host→VRAM into
  an LRU slot cache (81.6% hit at 30% residency; auto-caps slots to free VRAM, never OOMs).
  The bottleneck is **capacity, not kernels** — a model that fits would decode *faster* than a
  dense 7B. Metal MoE expert paging primitives (slot-pool LRU, pread staging default-on,
  MTLResidencySet-pinned pool) land for the same path on Apple Silicon.
- **Bit-identity gate suite** — the **Metal snapshot golden** (`TestMetalSnapshotGolden`):
  a machine+OS-pinned sha256 of Metal's own logits at fixed inputs, decoded past both
  reduction widths through two committed tiny models — the absolute stored reference that
  catches changes a self-consistent gate (paged≡non-paged, GPU-vs-CPU tolerance) is blind to
  (width sweeps, fused rewrites, accumulation-order moves). Reduction widths pinned as
  bit-identity constants (`tgReduce*`). CUDA carries `TestKernelFMALint` (build fails on any
  bare float MAC).
- **Committed tiny parity fixtures** — `testdata/cohere-tiny/` + `cohere2-tiny/` (656 KB each,
  deterministic random-weight, `scripts/pin_cohere*.py`) are committed via targeted `.gitignore`
  exceptions so the Cohere/Command-R forward-parity gates run in CI on every push, not only
  where someone regenerated the asset (same rationale as `mixtral-tiny`). The committed set is
  now enumerated as chosen policy in `docs/parity-coverage-policy.md`.
- **Test census tool** (`scripts/skip_census.py`) — the release ritual: runs `go test -json`,
  reports PASS/SKIP/FAIL with every skip bucketed by *why* (missing-fixture / missing-golden /
  no-gpu-device / heavy-model / integration-env), so a run that skipped 200 asset-gated tests
  can't be mistaken for one that exercised them. `GOINFER_REQUIRE_FIXTURES=1` turns any
  committed-fixture skip into a hard failure. It also flags native GPU-contention crashes
  (`fault 0x10`) as spurious rather than real.
- **MXFP4 / gpt-oss (CPU)** — bit-exact MXFP4 unpacker vs GGUF on real gpt-oss:20b, CPU
  forward + loader + parity; CUDA/Metal decline gpt-oss at load → CPU (never mis-run).

### Changed
- **Ollama peer re-anchored to v0.32.5** across `benchmarks.md` §B2/§B3/§B4. §B2 (CUDA): 0.5B
  still ~1.7× ahead, 1.5B short-ctx parity (~1.19×), goinfer behind at 2048 ctx and on prefill.
  §B3 (Metal): 0.96× / 0.74× (0.5B / 1.5B). "Peer version is part of the measurement" is now a
  standing method rule.
- **§B4 retracted** — the "peers fail to load the 26B" claim was false and is withdrawn: Ollama
  v0.32.5 loads and runs Gemma-4 26B-A4B via a 42%-GPU/58%-CPU split at ~24.5 tok/s (faster
  than goinfer here). The honest, narrower claim: goinfer runs it *fully GPU-resident* — an
  architecture difference, not a capability peers lack.
- **Batched prefill default flipped to ON (CUDA)** after the contraction fix made it
  bit-identical; **declined by default on Metal** (not bit-identical). Split-KV decode
  attention default-on at ctx ≥ 256.
- **Fast-math stays the Metal default**; `GOINFER_PRECISE_MATH=1` is an opt-in (measured 4–7%
  slower, no CPU-parity gain — the int8→int4 requant is the parity gap, not fast-math). Robustness
  against compiler/OS drift is carried by the snapshot golden, not by strict math.
- **aikit `gpu` module bumped to v0.25.2** — `CompileLibraryPrecise` now self-verifies
  (`setMathMode:MTLMathModeSafe` → `setFastMathEnabled:NO` fallback, reads state back and errors
  if neither selector responds), ViT reduction width pinned.

### Fixed
- **Batched-prefill / spec-decode were not bit-identical to sequential decode (CUDA) — 84%
  token-stream divergence on real weights** (invisible on uniform fixtures). Root cause:
  compiler **FMA contraction** (`facc += p*s` = mul+add / 2 roundings in decode vs fma / 1
  rounding in the batched kernel), ~1 ULP, data-dependent. Fix: explicit **`__fmaf_rn`** in
  both paths → `TestPrefillDivergenceRate` 0/50, decode speed unchanged; this is what unblocked
  lossless spec-decode. Durable guard: `TestKernelFMALint`.
- **As-cap overflow (Qwen3/Mistral):** `head_dim` independent of `hidden` (`nH·hd ≠ H`)
  overflowed a hardcoded activation-staging buffer, which would have silently broken every real
  Qwen3/Mistral size. Fixed on Metal with dynamic threadgroup memory (bit-exact to K=4096);
  CUDA audited immune.
- **Phi-3 dropped `sliding_window` on every backend (CPU included)** — `phi3Architecture` never
  set it, so Phi-3 ran full attention and diverged from HF past 2047 tokens. Now honored;
  Metal windowing works as a result.
- **`lmHeadN` dropped `LogitScale` on the batched-forward path** — surfaced by Cohere (the first
  forwardN-eligible logit-scale family); also closed 3 gemma4_text merge gaps.
- **Metal Gemma correctness:** GELU-tanh argument clamp (tanh overflow → NaN on the sink's large
  gate); prefill LM head pinned to the int8 path (was NaN on the f16-MMA path).
- **Security hardening (all shipped):** resident-KV context-cap DoS / OOB KV writes (C3, CUDA +
  Metal), speculative-rollback sliding-window leak (C1), PrefillLast unified-memory leak (C5),
  discarded CUDA launch errors + mixed-quant fuseQKV guard (M23), prompt-injection via
  special-token surface forms (M25), GGUF/VL untrusted-input hardening.

### Findings (no API change)
- **Metal decode is dispatch-/issue-bound, and the long-context gap is structural.** The
  short-context GEMV is already at the peer's effective-bandwidth band (no DP4A ⇒ integer decode
  is issue-bound). The long-context deficit (~4.2×@4000 vs Ollama's flash attention) is priced
  out by the bit-identity contract — four independent bit-identical dedup attempts (split-KV,
  two grouped, staged) were built and lost. **Decision (M1 = M-A):** stay bit-identical, accept
  the depth floor; an opt-in FA-style throughput mode (M-B) is scoped and deferred, not rejected
  (`docs/completed/metal-verdict.md`).
- **CUDA graphs are a measured null on real models** (1.01×; the ~1.4–1.7× was a
  tiny-model/dispatch-dominated artifact). Only a safe-gate shipped
  (`GOINFER_CUDA_GRAPHS=1`, promotes to live under EXCLUSIVE_PROCESS/MPS + a startup
  bit-exactness self-test); default declines.
- **Rotation + per-row scales + IMMA (tensor cores) deferred** — the cheap path (per-row scale
  search) is measured dead (per-row perplexity 108 vs per-group 28.5; the 1.24× weight-space
  error compounds ~4× at the output), and the expensive path buys only a prefill-only ~3× on an
  already-past-threshold TTFT. Format stays group-scaled int4 (`docs/completed/task-rotation-perrow-imma.md`,
  `ollama-chase.md` §7).
- **EAGLE-3 and Stage-B GPU verify are built-and-parked** — EAGLE-3 is lossless but a CPU
  wall-clock loss (needs GPU); Stage-B M=k GEMM verify is a NO-GO on small models, conditional-GO
  only for ~70B-class + short linear drafts. Both wait on GPU throughput they don't yet have
  (`docs/spec/`).

## [v0.8.0] — 2026-06-20

Theme: **the GPU resident-decode path expands from "dense Qwen2/Llama only" to most
families served, and gains a Mamba-2 SSM engine for hybrids.** (See
`docs/completed/decode-residency-campaign.md` for the full arc, scorecard, and dead ends.)

### Added
- **Resident decode for most mainstream families** (the C-lever ladder — bounded
  eligibility widenings, existing kernels reused). MoE: Mixtral / qwen2_moe / **GLM-4.5/4.6**
  (partial-RoPE) / **DeepSeek-V2/V3 + Kimi-K2** via a new **MLA latent-attention** residency
  bridge (rank-space attention + latent KV store + per-head absorb/lift kernels). **Mistral**
  (sliding-window residency) and **Mellum** (per-layer-RoPE residency). Most served models now
  decode pure-GPU instead of staged (~3× the per-token speed where it applies).
- **Resident Mamba-2 SSM decode engine** for hybrid families — the reframe that *decode is a
  bounded per-token recurrence, not the prefill scan*, so Mamba state slots onto the
  `DecodeRunner` like a KV cache (`mambaConv`/`mambaSSM`/`mambaGatedNorm` kernels, build-once
  persistent {conv-ring, ssm} state, drift-gated to 2k tokens).
- **Nemotron-H resident, DEFAULT-on at int4** — the dense squared-ReLU hybrid (Mamba-2 /
  NoPE-GQA / non-gated relu² MLP, single-op-per-block + a `relu2Quant` kernel). Near-lossless
  (perplexity 1.677 vs f32 1.695, KL 0.058; the ~7.5% greedy disagreements are 100% benign —
  99.6% top-2 agreement, every divergence at a near-tied position), ~13× the f32 CPU path.
  Guarded: default-on only at int4; int8 opt-in behind `GOINFER_SSM_RESIDENT`.
- **Granite-4.0-H resident** (Mamba-2 + attention + MoE-every-layer + the 4 Granite multipliers),
  **opt-in** — a 10× greedy speedup, but int8-quality-limited (below).
- **Decode kernel fusion** — fused q-rope + k-rope-store + v-store into one dispatch (+1.5%);
  q/k/v bias folded into the GEMV epilogue for bias models (+2.3% real Qwen2.5).
- **Mellum2 fast-load (and any safetensors MoE): prequant → int4 `.giw` + direct upload.**
  `cmd/prequant` now accepts a safetensors **directory** (not just a GGUF), producing a
  resident-loadable int4 bundle (`decoder.Load` + `SerializeWeightsTo`; tokenizer carried as the
  dir's `tokenizer.json`, loaded via `tokenizer.LoadJSONBytes`). And the resident int4 upload
  skips the per-element unpack + `packNibbles` repack: the decoder's int4 storage is
  byte-identical to the GPU packed layout for K%32==0 (`TestInt4LayoutMatch`), so it
  `CreateBufferInit`s the decoder bytes straight. Mellum2's 12B int4 resident load **~66 s →
  ~13 s warm** (the bundle skips the requant; the direct upload skips the repack); benefits every
  int4 resident load, `.giw` and direct alike. Gate: `gpu.TestGIWInt4_resident`.

### Changed
- The mmap/madvise/span-residency weight substrate moved to `aikit/mmap` (shared primitive).
- **Parity manifest: `serialize.go` split into its own `serialize` deps set** — serialize-only
  edits (a new `.giw` quant layout, the safetensors dir-input path) no longer re-stale every
  forward-parity row; the one place serialize affects numerics (the int4 deserialize→resident
  seam) keeps its own gate (`TestGIWInt4_resident`). Policy: a bare `deps_hash` re-hash of a
  validated family that changed a forward/core file is now forbidden (`parity-coverage-policy.md`).
- **`Model.Quant()` reports `int4mix` for a mixed bundle** (int4 experts + an int8-kept
  embedding/LM head) instead of mislabeling it `int8` — it scans all weight kinds. The `.giw`
  format tag (`quantMode`) is untouched, so existing bundles still load. (Heads-up: this changes
  the `.giw-kv` warm-snapshot fingerprint for mixed bundles → one-time cold prefill on upgrade.)
- **wgpu-native's benign warnings silenced** — `gpu.New()` sets `LogLevelError`, so "No
  windowing system present / No config found" no longer spams stderr on headless GPU runs.

### Fixed
- **GGUF→`.giw` dropped `rope_parameters` for MLA/YaRN families.** A nil `json.RawMessage` config
  field marshals to the literal `null` (4 bytes), which on the `.giw` round-trip
  (`json.Marshal(Cfg)` → reload) re-fired an "is it present?" check on an absent field — so a
  GGUF DeepSeek bundle failed self-check with "rope_parameters: rope_theta must be >0". `,omitempty`
  on the optional Config RawMessage fields keeps nil truly absent (`TestConfig_giwRoundTrip_nilRawMessage`).

### Findings (no API change)
- **qwen3_5_moe numbers shifted slightly under the forward refactors — re-validated, still
  gate-backed.** The v0.8.0 §1 re-run at HEAD (vs the `e3eb033` baseline) moved its int8-vs-bf16
  Gate-2 from argmax 74/80, cosine_min 0.99466 → **66/80, 0.99333**. Benign: the mean barely
  moved (0.99837), and *every* argmax divergence is a proven near-tie (worst gap 0.0045 of
  range), so the forward is sound — the flips are coin-flips on tied tokens, amplified by the
  `forward_qwen35.go` refactors (MoE demand-paging / WeightMat migration / scratch-reuse). The
  bf16 Gate-2's enforced bar is **cosine_min ≥ 0.98 + all-near-tie**, both cleared; 0.99466 was
  the *achieved* number that run, not a floor (the 66/80 + 0.9943 figure is the *separate* Q8_0
  GGUF gate's relaxed bar). deepseek_v2/v3 re-ran clean to argmax 100% / cosine 0.999, confirming
  the re-validation discriminates.
- **Granite int8 resident is quality-limited and stays opt-in/greedy-only** — characterized as a
  *fundamental* cliff (not a bug): its 64-expert top-6 MoE router turns chaotic f32-reduction-order
  perturbations into discrete expert-selection flips. Proven precision-invariant (int8 ≈ f16 ≈
  W8A16) and NOT a GPU-kernel bug (the SSM kernels are bit-correct). Nemotron-H, having no router,
  does not hit this — which is why it's default-on. Full write-up: `docs/ssm-int8-quality.md`.
- **No "wgpu-native v29 decode penalty"** — measured ≈ cogentcore/v22 (gemv + per-dispatch record);
  the real binding blocker is the go-webgpu *goffi* (zero-CGO) Go-1.26 crash, not v29. Staying on
  `cogentcore/webgpu`. (`docs/completed/gpu-gowebgpu-migration-assessment.md`.)

## [v0.7.0] — 2026-06-15

### Added
- **Qwen2.5-VL — a second vision-language family (image→text, pure Go).** goinfer's
  vision path now generalizes beyond Gemma 3 / SigLIP. The whole pipeline is gated
  against HuggingFace: HF-exact preprocessing (smart-resize + a PIL-bicubic port +
  the spatial-merge patchify), the aikit Qwen2.5-VL ViT + patch merger (dynamic
  resolution), **m-RoPE** (3D temporal/height/width positions — `applyMRoPE` + a port
  of `get_rope_index`, prefill *and* decode), and the decoder image path (merged-
  feature injection, causal attention). `serve` accepts image requests for a
  Qwen2.5-VL `--model` (OpenAI + Anthropic), validated end-to-end on the real
  Qwen2.5-VL-3B. Text decode for every other model stays byte-identical (m-RoPE
  reduces to scalar RoPE when the three position components are equal). New API:
  `decoder.GenerateQwenVL`; `cmd/serve` family auto-detection + routing.
- **Compute-time multi-adapter LoRA (`serve --adapter <name>=<base>=<dir>`) — N
  fine-tunes on one resident base.** Instead of merging a LoRA into the weights (a
  full base copy per adapter), the low-rank `A`/`B` are applied in the forward
  (`Y = W·x + s·B(A·x)`), so N adapters of one base cost ≈ base + N small deltas — the
  multi-tenant footprint win. Each adapter is a served model that shares the base's
  resident weights but keeps its own KV sessions; requests route via the OpenAI
  `model` field. Merged `--lora` stays the faster single-fine-tune default. Safetensors
  base, dense gated-MLP archs; an active adapter takes the sequential prefill path.
- **`--quant int4mix`: per-tensor mixed precision (idea #5).** A calibration spike
  found the int4→int8 quality loss is concentrated in **attention** (promoting
  `attn_output` alone recovered >half the gap), while the **FFN bulk is int4-tolerant**
  — and attention is the *cheaper* tensors. So int4mix keeps attention (q/k/v/o) +
  embed/head at int8 and the FFN (gate/up/down/experts) at int4: **near-int8 quality
  below int8 RAM** (≈0.5–0.8× int8, model-dependent on the FFN ratio). It's a
  load-time policy keyed on llama.cpp tensor names (`matmulQuant`); the resident
  weights and `.giw` carry the resolved per-tensor int8/int4 kinds (the format already
  stores per-`weightMat` kind, so a mixed model round-trips). No new kernels, no format
  change, zero decode cost. GGUF load path only. `Model.Quant()` now reports the
  requested quant for direct loads so int4/int4mix/int8 don't collide in the KV
  fingerprint. Gated by `TestInt4MixMode`.
- **Per-model `serve` overrides — a heterogeneous model zoo.** `--model` now takes
  comma-separated per-model overrides of the server-global defaults:
  `--model big=moe.giw,stream,weight-cache=16,quant=int4 --model fast=small.giw`
  streams the big MoE while the small one stays resident — the case the old flat
  global config (the `// per-model overrides are a follow-on` TODO) couldn't serve.
  Keys: `quant,lora,kv,kv-quant,stream,weight-cache,embed-int4`; overrides are
  pointer-typed so "inherit the default" stays distinct from a real `""` (f32).
  Backward compatible — no comma suffix inherits the globals. Paths may not contain
  commas.
- **`decoder.Options.Validate()`** checks the stringly-typed knobs (`Quant`,
  `Backend`, `KVPrecision`, `KVQuant`) against their allowed values, so an invalid
  enum (e.g. `-kv-quant=int8` instead of `i8`) is a clear load-time error instead of
  a silent fall-through to the default. serve calls it once per resolved model.
- **`--embed-int4`: relax the int8 embed/head pin to int4 (idea #3, opt-in lossy).**
  In int4 mode the token-embedding / LM-head table is pinned to int8 because it's
  logit-critical; for a big-vocab small model that pinned table is the single largest
  resident tensor. `--embed-int4` (decoder `Options.EmbedInt4`) stores it at int4 too,
  halving it, for ~2.3 pts top-1 (a 1.5B Q4_K_M spike — ≈0 on frequent tokens, ~3 on
  rare). Default off keeps the bit-exact int8 pin. GGUF direct-load path only (not the
  `--stream-weights` `.giw` cache — prequant with the knob to bake it). The doc's
  stronger "row-blocking" variant (b) was spiked and shelved: the int4 damage is
  entirely on tail rows, which tiering keeps at int4, so it's dominated by full-int4.
- **`serve --stream-weights` now works on a plain `.gguf` (transparent `.giw` cache,
  idea #1 "D").** Streaming needs the read-only mmap that only `.giw` provides; rather
  than make users run `prequant` by hand, serve now transcodes a `.gguf` to a sidecar
  `<model>.<quant>.giw` once (streamed, ~model-size peak RAM — no OOM) and loads that,
  reusing it on later runs and rebuilding it if the `.gguf` changes. The one-time
  transcode is logged. No per-token cost (resident bytes stay the dequant-once
  int8/int4); it's the convenience floor of #1. The transcode core moved to
  `internal/prequant` (shared by `cmd/prequant` and serve).
- **Dense weight streaming (`serve --stream-weights` on a dense `.giw`) — run a
  model bigger than RAM.** The companion to MoE expert paging for dense models
  (Llama / Qwen2 / Qwen3 / Mistral): because the transformer layer loop is
  sequential and known in advance, a sliding-window pager prefetches the next
  layer's weights (`MADV_WILLNEED`) while the current layer computes — overlapping
  the fault — and releases the layer that slides out the back (`MADV_DONTNEED`).
  Resident weight RAM is bounded to **floor + window** (sized by `--weight-cache`)
  instead of the whole model, so a model too big for RAM still runs (floored by
  NVMe bandwidth). The floor is non-zero: only the per-layer projections stream;
  embed / final-norm / LM-head stay resident (multi-GB for big-vocab models — the
  complementary lever is sub-int8 embed/head). Bit-exact by the same read-only-re-fault property as expert
  paging — validated byte-identical over a decode with a 3-layer window evicting
  and re-faulting most layers every token. Same `--stream-weights` flag, which now
  picks expert paging for MoE and layer streaming for dense; no-op when the model
  fits the budget.
- **`.giw` now round-trips the `qwen3_5_moe` DeltaNet-hybrid family (format v2).**
  The prequant `.giw` serializer dropped the hybrid's per-layer `delta` (Gated
  DeltaNet) and `qattn` (gated-softmax) tensor sets, so a `.giw`-loaded 35B-A3B
  segfaulted on the first forward (nil delta). v2 appends a one-byte-per-layer
  hybrid tail carrying those f32 tensors; v1 blobs are rejected by the version
  guard and rebuilt from the GGUF (a `.giw` is a regenerable cache). This is what
  lets MoE expert demand-paging (below) actually run its headline 35B-A3B — now
  validated end-to-end: a 512 MB expert cache against 16 GB of experts evicted and
  re-faulted 5k+ in-use experts over a decode with byte-identical output.
- **Streaming `.giw` serialization (`prequant` no longer OOMs on big models).**
  `decoder.SerializeWeightsTo(io.Writer)` + `giw.WriteStream` write the bundle
  straight to disk with a running CRC and a seek-back length patch, so peak RAM is
  ~the resident weight size instead of 2×+ (resident + full blob + bundle copy). A
  35B int4 now prequantizes at ~20 GB peak instead of thrashing into swap; the
  streamed bytes are byte-identical to the in-memory path.
- **MoE expert demand-paging (`serve --stream-weights` + `--weight-cache <GB>`) —
  run a big MoE on less RAM.** A sparse MoE (e.g. Qwen3-A3B class) resides at tens
  of GB of experts but activates only K·L per token. With `--stream-weights`, a
  `.giw` MoE model keeps its experts in the (now mmap'd, idea-Inc-1) read-only
  mapping and pages them on demand: the router's top-k selection drives an LRU
  bounded by `--weight-cache` GB (0 = auto, ~½ available RAM), releasing the tail
  with `MADV_DONTNEED` and faulting misses with `MADV_WILLNEED`. **Bit-exact** —
  the mapping is read-only and file-backed, so an evicted-then-reused expert simply
  re-faults from disk (identical bytes); the cost is the cold-miss fault, ~+24 ms/
  token at a 16 GB budget on a measured 35B-A3B (≈2× RAM reduction; see the spike
  in `decoder/moepaging_spike_test.go`). Opt-in, CPU `.giw` MoE only; a no-op for
  non-MoE / non-`.giw` / sub-page-expert models. Page-granular eviction, so it
  caps RAM on Linux firmly and best-effort on other unixes.
- **`.giw` weights are now mmap'd (pageable residency).** The prequant `.giw` fast
  path already aliased its int8/int4 weights with no per-tensor copy, but read the
  bundle with `os.ReadFile` (heap, not pageable). It now maps the file read-only
  (`MAP_PRIVATE`), so the aliased weights are views into the OS page cache —
  faulted in lazily, evictable, and shared across processes mapping the same file.
  Bit-exact and lower load time; released by `Model.Close`. The substrate the
  expert paging above (and future weight streaming) builds on. Windows falls back
  to the prior heap read.
- **Tiered KV cache (`serve --kv-idle-demote`) — demote idle warm sessions RAM →
  NVMe.** `--kv-sessions` pins RAM for every warm conversation; tiered KV adds a
  policy over the existing `--session-dir` `.giw-kv` persistence so a small-RAM box
  can hold many intermittent chats. A session untouched for `--kv-idle-demote`
  (e.g. `10m`) is snapshotted to disk and its RAM freed by a background sweep;
  capacity evictions tier the coldest session to disk instead of discarding it; the
  next request whose prompt extends a demoted session faults it back transparently.
  The fault-back continuation is **byte-identical** to a cold prefill (exact
  `.giw-kv` restore). The on-disk tier is bounded by `--kv-demoted-max` (default 64)
  and is in-process scratch (the resident tier is what survives a restart). Off by
  default; needs `--session-dir` and `--kv-sessions > 0`. Pure serve-layer policy —
  no decoder changes.

### Changed
- **aikit `v1.7.3` → `v1.8.1`.** Adds the pure-Go Qwen2.5-VL vision tower
  (`vision.LoadQwenVisionEncoder` / `Forward` / `ForwardViT`) — RMSNorm ViT, windowed
  + full attention, 2D rotary, the spatial-merge patch merger — parity-gated against
  HuggingFace. Bumped in both the root and `gpu` modules.

### Fixed
- **darwin (Apple-silicon) build break.** The MoE-paging madvise helper called
  `syscall.Madvise` under `//go:build unix`, but darwin has no `syscall.Madvise`
  (it lives in `golang.org/x/sys/unix`), so the root module — and everything
  downstream — failed to compile on macOS. Split per-platform: Linux/BSD keep
  `syscall` + `MADV_DONTNEED` (firm RAM cap, unchanged); darwin uses `x/sys/unix`
  for the `MADV_WILLNEED` prefetch and no-ops eviction (no macOS syscall reclaims a
  read-only file-backed mapping, so the `--weight-cache` cap is best-effort on
  darwin — the OS reclaims the clean, re-faultable pages under memory pressure).
  Adds `golang.org/x/sys` as a direct root dependency. (CI now runs darwin/arm64
  jobs to catch this class of platform gap.)

## [v0.6.0] — 2026-06-13

### Added
- **GPU int8 KV cache (`--kv i8`, opt-in `-tags gpu`) — 4× vs f32, ~64k context on
  8 GB.** The full-residency decode path (dense Qwen2/Llama) gains an int8 KV cache
  alongside f32 (16k ctx) and f16 (32k): per-(position, KV-head) symmetric int8,
  written and read by on-device kernels (RoPE-store / V-store quantize per head;
  attention unpacks ×scale), so decode and prefill stay on the GPU. ~6.9 GiB peak for
  a 7B int4 model + 64k KV. Lossy but argmax-faithful (≥0.99 cosine vs the f32 cache).
  The f32 and f16 KV paths are unchanged. Selected by `serve --kv i8`.

### Changed
- **aikit dependency consolidated and bumped `v1.3.0` → `v1.7.3`.** Three surfaces
  goinfer had open-coded moved onto aikit's shared types — pure de-duplication, no
  behaviour change: (1) the SigLIP vision encoder now lives in `aikit/vision` (goinfer
  keeps only the Gemma projector + soft-token glue, as package `multimodal`); (2) the
  shape-checked safetensors weight reads collapse onto
  `embed.SafetensorsFile.TensorF32` / `TensorI32`; (3) the decoder's three-precision
  quantized-weight matrix (f32 / per-row int8 / group-wise int4) is now
  `linalg.WeightMat`, with goinfer retaining only its matmul backend-routing policy —
  the `.giw` zero-copy load is preserved via aikit's `WrapInt8` / `WrapInt4`. 1.7.3
  also makes `linalg.MatmulBT` M-invariant (a row computed alone is bit-identical to
  the same row inside a batch) and fixes an amd64-only AVX2 reduction bug on
  odd-length-`K` shapes.

### Fixed
- **`--backend webgpu` falls back to CPU when the GPU device fails to initialize,
  instead of panicking.** The webgpu backend factory returned a typed-nil
  `*webgpuBackend` on adapter/device failure, which auto-converts to a *non-nil*
  `Backend` interface — so `Model.withResidency` saw a "real" backend, type-asserted
  it, and dereferenced a nil receiver (`BuildResident` → nil mutex). The factory now
  returns a literal `nil` interface on error, so a headless box (or GPU exhaustion)
  cleanly uses the CPU path with a note, as intended. `-tags gpu` only.
- **Same-model speculative decoding is now bit-exact with plain greedy.** Dense decode
  computed attention with a scalar kernel that was not bit-identical to the batched
  kernel the speculative *verify* pass uses (cosine ≥0.99, not exact), so the target
  rejected ~11% of its own draft tokens (acceptance ~0.89) and the streamed output
  could drift from greedy. Decode now runs the same batched attention as
  prefill/verify, with f64 accumulation, so decode == prefill == verify exactly —
  speculative output is token-identical to greedy and acceptance is ~1.0. This also
  removes the decode↔prefill numerics seam for all dense models; combined with aikit
  1.7.3's M-invariant `MatmulBT` it holds across f32/int8/int4. (int4 greedy decode
  now tracks the higher-precision reference one token further as a side effect.)

### Added
- **Multimodal vision input (Gemma 3 VL) — image→text end to end, pure Go.** An
  image now flows through `vision` (preprocess → SigLIP encoder → projector) into
  the text decoder's embed-by-vector seam (`decoder.GenerateVL`), and the serve
  surface accepts it on **both** the OpenAI `/v1/chat/completions` (`image_url`
  content parts) and Anthropic `/v1/messages` (`image` blocks) APIs —
  **base64/data-URI only** (a remote URL is never fetched; SSRF guard). Start with
  `--vision <dir>` (auto-discovered when the `--model` dir is a Gemma 3 VL
  checkpoint). Image tokens (256/image) count in `usage`. `demo/agent` (the web UI)
  also gains image input: drop, paste, or pick an image and the agent answers it
  via the same path. Numerics are HF-parity-gated (encoder/projector/end-to-end
  goldens). Loading a real `google/gemma-3-4b-it` now works directly (sharded +
  prefixed VL safetensors, nested `vision_config`, and the Gemma3-text class
  defaults the minimal `text_config` omits). **Caveat (CPU):** the SigLIP prefill
  is heavy on CPU (~171 s/image at 896², 4096 patches) — correct but slow. The GPU
  encoder below is the fix.
- **Resident GPU SigLIP encoder (`-tags gpu`, opt-in) — ~9× faster image
  prefill.** `--backend webgpu` runs the whole vision tower on the GPU: the tower
  uploads once, the `[4096, hidden]` activation stays on-device through all 27
  layers (patch-embed → LayerNorm → int8 qkv → bidirectional attention → gelu-tanh
  MLP → residuals → final LN), and there is one readback — paying WebGPU's
  submit/sync cost ~27× instead of the ~162× a per-op offload would (a measured
  dead end). On an RTX 2070 SUPER: **gemma-3-4b-it image prefill 171 s → 18.8 s**,
  parity **cosine 1.000000** vs the CPU W8A8 encoder (0.999959 vs the HF golden).
  New WGSL kernels (batched LayerNorm, gelu-tanh, row softmax, bias/residual
  add, per-head gather/scatter) join the existing tiled W8A8 GEMM. The pure-Go
  default build is untouched (`vision.Encoder` delegates to the device only when a
  resident backend is attached). `demo/agent` (the web UI) gets it too via
  `--vision-backend webgpu` — dropped/pasted image captions go ~9× faster. The
  attention matmuls still use a naive f32 kernel — a tiled GEMM there is the next
  lever toward the ~8–12 s estimate (`docs/completed/task-gpu-vision-tower.md`).
- **SigLIP attention vectorized** — QKᵀ/scores·V moved onto the SIMD A·Bᵀ kernels
  (QKᵀ f64-accumulating for parity), >2× faster vision prefill (>400 s → ~190 s),
  bit-faithful (encoder golden cosine 1.0).

## [v0.5.0] — 2026-06-11

_A **minor bump** per SemVer: `constrain` schema validation carries a
user-visible behavior change (under Changed) — shipped alongside the Anthropic
Messages API (Claude Code can point at a pure-Go runtime), the f16 GPU KV cache,
the ~2.4× sparse-MoE / ~3.4× dense prefill speedups, and the now soak-verified
untrusted-input fuzzing hardening._

### Changed
- **`constrain` rejects unenforceable / unsatisfiable JSON Schemas at compile**
  instead of silently accepting them. Schemas that previously compiled but could
  never be enforced now return an error: a `required` entry naming a property
  absent from `properties`, an array with `maxItems` < `minItems`, and
  negative / non-integer array bounds. **Behavior change** — a caller that relied
  on the old silent-accept (and the non-conforming output it produced) now gets a
  compile error instead. (`3cb27e6`)
- **aikit `v1.1.1` → `v1.3.0`** — a hardened `embed.parseGGUF` (overflow-safe
  length checks, map/array pre-sizing capped to the remaining input, so a hostile
  GGUF header errors instead of panicking / OOM-ing one import down) and the
  scores·V attention vectorization (`v1.2.1`), plus `MatmulBTAcc64` — an
  f64-accumulating A·Bᵀ matmul (bit-identical to a sequential-order f64 reduction)
  that the MoE attention path needs (`v1.3.0`). Bumped in both the root and `gpu`
  modules. (`b77922c`)
- **MoE attention now runs through the f64-accumulating matmul** (decode AND
  prefill). It is more accurate than — and may differ at routing near-ties from —
  the prior f32/scalar path, but stays parity-gated against the HF bf16 oracle
  (Mellum2 logit/window parity unchanged). The discrete top-k router amplifies any
  attention reassociation, so the precision is load-bearing, not cosmetic.

### Added
- **Anthropic Messages API** (`POST /v1/messages`, `POST /v1/messages/count_tokens`)
  alongside the OpenAI surface — point **Claude Code** (or any tool that speaks
  `ANTHROPIC_BASE_URL` + `ANTHROPIC_AUTH_TOKEN` + `ANTHROPIC_MODEL`) at a pure-Go
  single-binary runtime. The second de-facto-standard chat surface, served by the
  same edge-translation trick llama.cpp/Ollama/LM Studio use: the request is
  mapped into the existing internal path (system + turns + sampling + tools) and
  the result mapped back out — `drive`/`prepare` unchanged. Honors `system`
  (string or text-block array), content blocks (`text`, plus `tool_use`/
  `tool_result` conversation replay), `tools` (`input_schema`) and `tool_choice`
  (`auto`/`any`/`tool` — `any`/`tool` reuse the constrained-decoding path, so a
  malformed tool call is physically impossible), `stop_sequences`, and the named-
  event streaming SSE protocol (`message_start` → `content_block_*` →
  `message_delta` → `message_stop`, no `[DONE]`). Compatible-not-full-spec (the
  llama.cpp bar): image blocks 400; `thinking` / `cache_control` / `metadata`
  accepted and ignored (Claude Code sends `cache_control` on every request).
  Pure stdlib `net/http`, no new deps. (`d0b0f66`)
- **`demo/agent`: a fully-local stdlib RAG coding agent** over goinfer + the
  `ken` MCP retrieval server — a CLI and a single-binary web UI, kept in its own
  module so its MCP dependency stays out of the goinfer root graph. (`36eae14`)
- **f16 GPU KV cache** (`--kv f16`, opt-in; default `f32` stays bit-exact) — halves
  per-token KV bytes on the full-residency path, unlocking **32k context for a
  7B-int4 on an 8 GB card** (32k f16 fits in the same VRAM as 16k f32 — measured
  6912 vs 6926 MiB on an RTX 2070 SUPER). Manual WGSL f16 via the core
  `pack2x16float`/`unpack2x16float` builtins (no `shader-f16` device feature, so the
  CI software adapter still compiles). Decode parity vs an f32 cache: argmax
  preserved, full-logit cosine ≥0.998 over an 8000-key context. (`138d5e0`,
  `b40d8dd`)
- **Chunked-parallel Gated DeltaNet scan kernel** (`deltanet_chunked.go`) — the
  reformulation the `qwen3_5_moe` perf rewrite needs, unrolling the gated
  delta-rule recurrence over a chunk into the matmul-friendly form. Proven
  algebraically equivalent to the sequential recurrence over random
  inputs/chunk sizes (`TestGatedDeltaNet_chunkedMatchesSequential`; self-contained,
  no checkpoint/torch). Not yet on the hot path — the forward is still
  single-token streaming — so it's a zero-regression-risk reference. (`750b4ac`)

### Security
- **Hostile model files / request bodies now error, never panic.** A fuzzing pass
  over the untrusted-input surface (Go native fuzzing, CI-enforced via committed
  seed corpora) found and fixed five panic/OOM vectors, each turned into a typed
  error with a regression test:
  - GGUF: a metadata `block_count` ≥ 2³¹ overflowed `int` to a negative
    `NumLayers` → `make([]LayerWeights, …)` panic. (`9cdf98d`)
  - `.giw` bundle: a near-`maxint64` v2 length overflowed the bound check in
    `cur.take` → slice-bounds panic. (`92f0d83`)
  - `tokenizer.json`: a hostile token id overflowed / OOM'd / negative-indexed the
    `idToPiece` allocation. (`8e0b8bd`)
  - GPTQ/AWQ: dims not a multiple of 8 slipped past the integer-division shape
    check → dequant indexed out of bounds. (`aebbd45`)
  - Serialized `.giw` weights: a corrupt blob could drive a multi-GB allocation;
    the layer/expert count gates were tightened to sane bounds. (`c6b5ff2`)
  Fuzz targets now cover `constrain`, the GGUF / serialized-weights / GPTQ-AWQ
  loaders, the `.giw` frame, `tokenizer.json`, and the serve request shapers, and a
  `-race` serve soak/chaos test exercises multi-model + admin + sessions under
  concurrency (zero goroutine leaks, byte-identical warm-KV restore). The GGUF
  interpretation fuzzer was re-enabled after the aikit hardening landed (`e3bc98d`).

### Performance
- **~3.4× prompt prefill** — vectorized the prefill attention (QKᵀ + scores·V)
  onto the SIMD path. (`7fa82c2`)
- **~1.7× Gemma 4 prefill** — the same treatment for gemma4 attention. (`88b7aaa`)
- `qwen3_5_moe`: softmax attention reuses the cache scratch for scores
  (`4fcc069`), and its M=1 projections route through the SIMD A·Bᵀ kernel
  (`443ab7a`).
- **~2.4× sparse-MoE prompt prefill** (Mellum2 12B-A2.5B: 3.36 → 8.11 tok/s at a
  1024-token context). `canBatchN` now admits standard sparse-MoE (Mellum / Mixtral),
  so their prefill batches the attention onto the SIMD A·Bᵀ kernel like the dense
  families — a CPU profile put the scalar per-token attention at ~83% of MoE prefill,
  the expert matmuls only ~17% (those stay per-token: the router picks different
  experts per row). The batched attention uses the f64-accumulating `MatmulBTAcc64`
  and decode is routed through the same kernel, so the result is **bit-identical to
  the sequential forward** (the discrete router won't tolerate an f32 matmul's ~1e-6
  reassociation). The qwen3_5_moe DeltaNet hybrid and Gemma 4 keep their own forwards.

## [v0.4.0] — 2026-06-09

### Added
- **GPU full-residency decode (W4A8 int4) — run bigger models pure-GPU.** Real
  int4 `.giw` models now decode entirely on the GPU through `decoder.Generate` on
  the `webgpu` backend (dense Qwen2/Llama): the full-token forward is the resident
  DecodeRunner, not the per-matmul staged path. The win is **footprint, not a
  speed record** — int4 halves resident weights, so a **7B int4 fits and decodes
  at ~51 tok/s on an 8 GB card** (the model class that does NOT fit at int8), at
  **~71% of llama.cpp-CUDA (q4) at equal 4-bit quant** (51.7 vs 72.8 tok/s; greedy
  output matches the CPU decode bit-for-bit on the first tokens). int8 residency
  peaks ~89.7 tok/s on the 1.5B (3.5× the staged hybrid; **61% of Ollama-q8** at
  equal int8 quant, 89.7 vs 147). *(Peer figures as measured for v0.5.0, 2026-06,
  on the WebGPU backend vs then-current llama.cpp / Ollama; these are not current —
  the cgo-free CUDA backend + the Ollama v0.32.5 re-anchor in `docs/benchmarks.md`
  §B2 supersede them.)* v1 limits: **stateless `Generate` only**
  (`Session`/prefix-reuse/`GenerateSpeculative` fall back to the staged path),
  **16k context cap** (f32 KV), **eligible archs only** (dense Qwen2/Llama;
  MoE/Gemma/hybrid → staged). See `docs/completed/gpu-assessment.md` §0.0 + the §1 decision
  matrix. (The `.giw` bundle's weights length is now u64 — v2 — so int4 models
  past 4 GiB, i.e. the 7B+ class, serialize without truncation; v1 bundles still
  load.)
- **cmd/serve: multi-model + admin + Responses API.** `--model` is now repeatable
  as `name=path` to serve a model zoo from one process; requests route on the
  OpenAI `model` field (exact match, or the sole model for single-model compat),
  unknown → an OpenAI-shaped 404, and `/v1/models` lists all. Each model has its
  own mutex (distinct models run in parallel) and warm-KV dir
  (`--session-dir/<fp>/`). **Admin API** (gated behind `--allow-admin`, default
  off — RCE-adjacent): `POST /admin/models/{load,unload}`, with unload refusing a
  busy model (409) and snapshotting its warm KV. **`/v1/responses`** (OpenAI
  Responses API): `input`/`instructions`/`text.format`(→constrain)/`tools`,
  streaming event shapes, and `store`/`previous_response_id` (an in-memory ring
  that rides the per-model sessionLRU for warm KV). **Backpressure**: a bounded
  per-model queue (`--max-queue`, default 8) returns 429 + Retry-After when full
  (single decode worker per model; not continuous batching). Internally the
  generative half is a `loadedModel` registry. Pure stdlib `net/http`, no deps.
- **Qwen3.5/3.6-MoE (`qwen3_5_moe`)** — the hybrid linear/softmax-attention MoE.
  Most layers are **Gated DeltaNet** (linear attention with a recurrent matrix
  state — short causal conv + gated delta rule + gated RMSNorm, its own forward
  primitive), the rest gated softmax attention (double-width q_proj → query‖gate,
  QK-norm, partial RoPE, output gate), over a 256-expert + shared MoE on every
  layer. A **hybrid cache** holds KV for the softmax layers + a fixed-size
  `deltaState` per linear layer; prefix-reuse / speculative fall back for the
  hybrid (recurrent state isn't position-truncatable). Loads from safetensors,
  bit-exact vs the HF oracle: the DeltaNet primitive op-for-op (cosine 1.0,
  `deltanet_test.go`) and the full model argmax + cosine 1.0
  (`qwen35_forward_test.go`). **Loads from both safetensors and GGUF** — the
  `qwen3_5_moe` loader reverses llama.cpp's fused/stacked transform back to the
  per-expert layout — with **real-checkpoint int8 parity on the 35B-A3B** (Gate 2:
  argmax **74/80**, sample cosine **0.99466** vs the banked HF bf16 golden), and the
  GGUF path proven by weight-diff against the safetensors load. **Honest scope:**
  this is the **text decoder of the Qwen3-VL 35B-A3B** model (the language tower),
  and the hybrid arch runs the **staged path — not GPU residency**. Parity-first f32
  forward. See `docs/qwen3_5_moe.md`.
- **Mellum2 chat template** — `chat.Mellum2()` (a named ChatML alias) + a `Detect`
  fingerprint (its distinctive `normalize_content` macro) so JetBrains Mellum2 is
  identified as `mellum2` by `cmd/serve` / `demo/chat` rather than falling through
  to generic ChatML. Its template is ChatML byte-for-byte (`<|im_start|>`/
  `<|im_end|>` turns, stop `<|im_end|>` = EOS id 28, Hermes `<tool_call>` tools),
  verified against HF `apply_chat_template` (`testdata/chat_goldens/mellum2.json`:
  system+user, multi-turn, no-system) — the same byte-exact gate as the other five
  families. Tools ride the existing ChatML Hermes `RenderTools`/`ParseToolCalls`.

### Changed
- **Mellum2 parity-gated** — Mellum2 is no longer the README family list's parity
  exception. `TestMellum2_logitParity` pins the forward (MoE 64/top-8, 3:1
  sliding/full interleave, YaRN-on-full RoPE, QK-norm) against the HF bf16 oracle
  on a chat-templated prompt: argmax exact (`Paris`), sample-256 cosine **0.99955**
  (int8int8 vs bf16). The sliding-window EVICTION path — untested when validation
  stayed under the 1024 window — now has a model-free unit proof
  (`TestMellum2_slidingWindowEviction`: a sliding layer's output is invariant to an
  out-of-window key, a full layer's isn't) plus a real-checkpoint past-window point
  (`TestMellum2_windowParity`: 1441-token prompt, cosine **0.99636**).
- **CPU int4 decode is now usable** — the `decoder.Generate` int4 path moved onto
  aikit v1.1.1's `MatmulBTW4A8` (int4×int8 integer decode) from the f32-activation
  `MatmulBTQ4`, which was dequant-bound at M=1. CPU int4 goes from **~20–45× slower
  than int8 to ~1.4–1.9× int8** (arm64 + amd64), so int4 is now a real CPU
  footprint option, not only a GPU one.
- **aikit → v1.1.1** — adds the `MatmulBTW4A8` CPU int4 kernel above (the
  `embed`/`linalg` deps; both modules track it).

## [v0.3.0] — 2026-06-07

### Added
- **Embeddings endpoint** — `cmd/serve` serves OpenAI `/v1/embeddings` from an
  `--embed-model` CodeRankEmbed encoder (`--embed-quant f32|q8`), alongside or
  instead of the generative `--model`; endpoints register for whatever loaded.
  Honors `input` (string|array), `encoding_format` (`float`|`base64`), and
  `dimensions` (truncate + renormalize); vectors are L2-normalized. An optional
  `input_type` (`query`|`document`, default document — the Cohere/Voyage
  convention) selects the encoder's asymmetric query instruction prefix.
  `usage.prompt_tokens` is counted via the encoder's tokenizer; the encoder is
  goroutine-safe, so embeddings serve without the decoder's lock. Rides aikit
  v1.0.0's frozen `encoder` API (`Encode`/`EncodeBatch`/`HiddenDim`).
- **Prompt-prefix KV caching / cross-call session reuse** — `decoder.Session`
  pairs a KV cache with the tokens materialized in it and reuses the KV of the
  longest token prefix a new prompt shares, prefilling only the divergent suffix
  (exact — bit-identical to a cold prefill). `cmd/serve` layers a session LRU
  (`--kv-sessions`, default 4; 0 disables) so a continuing chat — or an agent
  loop with a fixed system prompt + tool specs — skips re-encoding the whole
  history; `--session-dir` persists warm sessions to CRC+identity-guarded
  `.giw-kv` snapshots and restores them across restarts. Fixes
  `KVCache.TruncateTo` to derive each layer's stride from what it holds, so it is
  correct on Gemma 4's per-layer KV widths and KV-shared tail layers. Reuse
  parity is gated id-for-id vs a cold prefill (Qwen 2.5 + Gemma 4 E2B).
- **LoRA adapter loading** — `decoder.Options.LoRA` merges a PEFT adapter
  (`adapter_config.json` + `adapter_model.safetensors`) into the base weights at
  load: W′ = W + (α/r)·B·A (α/√r under rslora), applied to the f32 weight *before*
  quantization, so a merged model costs nothing extra at decode. Targets the
  per-layer attention/MLP projections; an unsupported target (e.g. `embed_tokens`)
  or a GGUF base is a loud error. `demo/chat` and `cmd/serve` gain `--lora`.
- **Tool calling** — composes chat templates + JSON-Schema constraint + a
  per-family parser (parse against the model's template, not a naive JSON scan).
  `chat.Template.RenderTools` declares tools in each family's syntax and
  `ParseToolCalls` recovers structured calls; supported: ChatML/Qwen (Hermes
  `<tool_call>`), Mistral (`[TOOL_CALLS]`), Llama-3 (bare `{name,parameters}`),
  and Gemma 4's bespoke `<|tool_call>` micro-language (incl. its declaration
  format, byte-exact). `constrain.ToolCallGrammar` constrains a call to a tool's
  schema (the family wrapper around `{"name":const,…:<args schema>}`). `cmd/serve`
  honors OpenAI `tools` / `tool_choice`: renders declarations, constrains the call
  when unambiguous (one tool or a forced function — Gemma 4 is parse-only, no
  logit constraint), and returns `tool_calls` with `finish_reason:"tool_calls"`.
  Tested: per-family parse + round-trip, the tool-grammar property (every
  constrained generation is a valid, schema-conforming call), and a server
  integration call (Qwen → `get_weather(...)`).
- **OpenAI-compatible HTTP server** (`cmd/serve`) — pure stdlib `net/http`, no
  deps. `/v1/chat/completions`, `/v1/completions`, `/v1/models`; streaming via
  SSE; the sampling knobs (temperature/top_p/top_k/seed/frequency_penalty/
  presence_penalty/stop/logprobs) map onto goinfer's sampler; and
  **`response_format`** (`json_object` / `json_schema`) rides the constrain
  grammar for output the model cannot violate. Chat templates auto-detected per
  model (the `chat` package). Point Open WebUI / LangChain / the OpenAI SDKs at
  `http://host:8080/v1`. (Its own release; depends on the chat-template,
  sampling, and JSON-Schema work above.)
- **K-quant coverage confirmed: Q6_K and Q4_K_S** now have dedicated logit-parity
  tests (TinyLlama, same harness as the other GGUF quant tests), alongside the
  existing Q2_K/Q3_K_M/Q4_K_M/Q5_K_M. The dequant paths (via `aikit/embed`) were
  already present — most HF repos ship Q5_K_M/Q6_K next to Q4_K_M and goinfer
  loads them all; these tests pin per-quant parity (argmax exact; cosine-floored).
- **JSON Schema constrained decoding** (the v0.2 flagship) — `constrain.JSONSchema`
  compiles a JSON Schema into the existing incremental byte-level `Grammar`, so the
  streaming logit masker drives it unchanged: the model **physically cannot** emit
  non-conforming JSON (invalid tokens are −∞, i.e. unreachable). Supported subset:
  objects (required + optional, `additionalProperties:false`), arrays
  (`items`/`minItems`/`maxItems`), `string`/`number`/`integer`/`boolean`/`null`,
  `enum`/`const`, arbitrary nesting; unsupported keywords are a loud compile error.
  **`constrain.GrammarFromStruct(v)`** derives the schema from a Go struct's json
  tags — "a struct the model cannot violate": constrain → `json.Unmarshal` always
  succeeds. `demo/chat --schema <file.json>` constrains the demo. A property-based
  test asserts every constrained generation validates against its schema; the
  unconstrained path is untouched (purely additive).
- **`chat` package** — chat templating as a library feature (no Jinja engine).
  `chat.Detect(meta)` fingerprints the GGUF/HF `tokenizer.chat_template` string
  (falling back to a vocab-marker heuristic for bare checkpoints) and returns a
  `Template`; `Template.Render(system, turns)` produces the exact prompt string
  and `Template.Stops()` the turn-stop markers. Native Go renderers for **Gemma 3,
  Gemma 4, ChatML (Qwen), Llama-3, and Mistral**, each **byte-exact against HF
  `apply_chat_template`** (golden fixtures in `testdata/chat_goldens`). An
  unrecognized template is an explicit `ErrUnknownTemplate` (caller falls back to
  raw completion). New tokenizer accessors `ChatTemplate()`, `Has()`, `TokenID()`.
- **Sampling completeness** in `SamplingParams` / `Sampler` — the standard
  controls a llama.cpp/Ollama/vLLM user expects:
  - **min-p** (`MinP`) — keep tokens with prob ≥ min-p·max-prob (a relative
    floor), composable with top-k/top-p.
  - **repetition controls** — llama.cpp-style `RepeatPenalty` (scales repeated
    tokens' logits) plus OpenAI-style `PresencePenalty` and `FrequencyPenalty`,
    over a `RepeatLastN` window (the prompt is seeded so it counts). `Generate`
    wires the history automatically.
  - **`LogitBias`** — per-token logit offsets (force or ban specific tokens).
  - **logprobs out** — `Sampler.SampleWithInfo` and `SamplingParams.Logprobs` /
    `TopLogprobs` report the chosen token's log-probability and the top
    alternatives; `Generation.Logprobs` collects them per emitted token.
  - `Seed` (already present) gives reproducible draws.
  `Sample(logits)` and the greedy parity path are unchanged. `demo/chat` gains
  `--min-p`, `--repeat-penalty`, `--presence-penalty`, `--frequency-penalty`,
  `--repeat-last-n`.

### Changed
- Depends on **aikit v1.0.0** (was v0.5.2): its now-frozen Hard tier covers the
  embedding encoder behind `/v1/embeddings`. The `goinfer/gpu` submodule is
  bumped to match, so both modules ship against the same published aikit.
- `cmd/serve` `--model` is now optional when `--embed-model` is given (and vice
  versa); at least one is required. The single-model flag set moved to a config
  struct as the server grew a generative and an embedding half.
- `demo/chat` and `demo/gemma-web` now render prompts via the `chat` package
  (was duplicated per-demo). The Gemma 4 demo render matches HF exactly,
  including the `<|channel>thought` scaffold (so the model may emit reasoning).

### Removed
- `tokenizer.ChatStyle` (enum + method, shipped in v0.2.0) — superseded by
  `chat.Detect`, whose fallback subsumes the old vocab-marker heuristic.

## [v0.2.0] — 2026-06-06

### Added
- **Gemma 4 support** (HF `model_type` `gemma4_unified_text`; GGUF arch
  `gemma4`) — the **E2B** and **E4B** "E-models" plus the **12B dense**, all
  parity-gated against the HF bf16 oracle. Gemma 4 is a meaningfully new
  architecture on top of the Gemma 3 stack, driven entirely through the
  `Architecture` descriptor:
  - per-layer **head_dim** (256 local / 512 global) and per-layer **KV-head
    count**, scale-less **v-norm**, **proportional (partial-rotary) RoPE** on the
    global layers, dual-base RoPE, and the final-logit softcap (30);
  - the E-model additions — **Per-Layer Embeddings (PLE)** branch, **cross-layer
    KV sharing**, variable per-layer **FFN width**, and a per-layer output
    scalar (no AltUp/Laurel — those are Gemma 3n);
  - the 12B's **`attention_k_eq_v`** (V reuses K's projection on the global
    layers).

  Greedy generation is coherent ("Paris"). Gates: `TestGemma4_logitParity`
  (E2B, sample-256 cosine **0.99938** vs HF bf16) and `TestGemma4_12B_logitParity`
  (12B, argmax exact + cosine **0.990**). Out of scope: the 26B-A4B MoE and 31B
  multimodal vision towers (text-only runtime).
- **Gemma 4 chat template** in the tokenizer + `demo/chat` — Gemma 4 replaced
  Gemma 3's `<start_of_turn>`/`<end_of_turn>` with new `<|turn>`/`<turn|>`
  markers. `Tokenizer.ChatStyle` now detects them (`ChatStyleGemma4`), the GGUF
  loader resolves the turn-stop token, and the demo renders the right template
  so replies are clean and stop correctly.

## [v0.1.3] — 2026-06-05

### Added
- **`decoder.LoadGGUFBytes`** / **`tokenizer.LoadGGUFBytes`** — load a GGUF model
  and its tokenizer from an in-memory `[]byte`, touching no filesystem. The
  shared GGUF build core is reused by the path-based `Load` / `LoadGGUF` (both
  unchanged); EOS ids resolve from the GGUF's own metadata.
- **`decoder.SerializeWeights` / `LoadSerializedWeights`** + `Model.Weights()` /
  `Model.Quant()` / `NewModel` — a versioned binary format (`.giw`: magic +
  version + config + quant guard + CRC, lazy fallback like ken's index) for an
  already-quantized weight bundle. On load the big int8/int4 arrays are **aliased
  zero-copy** out of the blob (float scales are copied for alignment), so a
  prequant build skips dequant+requant *and* the resident-weight heap copy.
- **`cmd/prequant`** — build-time generator: turns a GGUF into a `.giw` bundle
  (serialized int8 weights + a metadata-only GGUF carrying the tokenizer).
- **`decoder.GenerateSpeculative`** + **`demo/chat --draft <gguf>`** — greedy
  speculative decoding: a small draft model proposes K tokens, the target
  verifies them in one batched pass (`forwardN`), keeping the matching prefix.
  Output is **token-identical to plain target greedy** (gated by
  `TestSpeculativeGreedyParity`); `cache.TruncateTo` does the KV rollback. On the
  pure-Go CPU int8 kernel it's ~break-even (decode is ~half compute, which
  doesn't amortize at M=K), so it's a forward-looking / bandwidth-bound-backend
  feature — correct and exact, not a CPU speedup.
- **`decoder.SetDecodeParallelThreshold`** / `DefaultDecodeParallelThreshold` —
  goinfer-owned tuning of aikit's matmul parallelism crossover for batch=1
  decode (the demo sets it; aikit's library default stays conservative).

### Changed
- **Faster decode (~48 → ~70 tok/s on the 0.5B, +42%; ~36 on the 1.5B; M-series
  CPU)** via aikit v0.5.x: a per-stream `Workspace` makes steady-state decode
  **zero-alloc** (4660 → 19 allocs/op), q/k/v and gate/up run as **batched**
  matmuls, and a decode-tuned parallelism threshold parallelizes the per-token
  weight matmuls. Numerics bit-identical (`TestDecodeParity`).
- **Batched prompt prefill** — the prompt now runs through one `M=len(prompt)`
  pass instead of sequential single-token forwards, **~1.9–2.9× faster
  time-to-first-token** on the 1.5B (e.g. ~2.7 s → ~1.3 s for an 80-token
  prompt). Seed token bit-identical.
- Requires **aikit ≥ v0.5.2** (the `Workspace` / batched / `Into` W8A8 matmul API
  and the column-blocked W8A8 kernel that reuses weights across M>1 rows).
- `demo/chat` (embed build) loads the baked-in model **from memory by default** —
  no temp file, so the single-file binary runs on a read-only / `FROM scratch`
  filesystem. `--model-tmp` / `GOINFER_MODEL_TMP=1` opts back into a temp-file +
  mmap load (lower peak RAM for large models; tmpfs caveat documented). Load
  progress prints to stderr.
- `demo/chat` gains a **`-tags prequant`** build (now `build-embed.sh`'s default)
  that embeds a `.giw` bundle and maps the int8 weights from the binary image.
  Measured on the 0.5B (Qwen2.5-Coder, M-series CPU): cold start **2.3 s → 0.48 s**
  and resident heap (`phys_footprint`) **772 MB → 78 MB**. The win scales with
  model size — a 4B int8 model no longer needs a ~5 GB weight heap per launch.
  Quant is fixed at bundle-build time; the runtime `--quant` flag and the GGUF
  fallback apply only to the `--model <gguf>` path.
- The embed build now bakes the GGUF **uncompressed** (dropping zstd — q4 weights
  are high-entropy, so it shaved only ~3% while costing inflate time + a
  full-size heap buffer). Removes the `klauspost/compress` dependency; the
  default module graph is back to `aikit` + `x/text`.
- `demo/chat` now ships in **two size tiers** built from the same program:
  Qwen2.5-Coder **0.5B** (~617 MB, ~57 tok/s — the headline) and **1.5B**
  (~1.7 GB, ~26 tok/s — bigger, smarter, still one file). `build-embed.sh --name`
  parameterizes the output basename so the tiers build side by side without
  clobbering. Prequant keeps the 1.5B's resident heap at ~87 MB (≈ the 0.5B), and
  the 1.5B binary stays under GitHub's 2 GiB asset cap.

## [v0.1.2] — 2026-06-04

### Added
- **`demo/chat`** — "an entire LLM in one file": an interactive REPL with a
  zstd-`//go:embed`-ed Qwen2.5-Coder-0.5B-Instruct model baked into a static,
  no-cgo, cross-compiled binary (macOS / Linux / Windows × amd64/arm64).
  Defaults to the fast int8×int8 (W8A8) kernel; ships live prompt-tuning
  slash-commands and canned `/demo` prompts.

### Changed
- Requires `aikit ≥ v0.4.1`, which platform-guards the embed loader's mmap so the
  decoder (and the demo binary) cross-compiles to `windows/amd64`.

## [v0.1.1] — 2026-06-04

### Added
- **`decoder`** — generic decoder-only forward pass with HuggingFace logit
  parity: Gemma 3, Qwen 2.5/3, Llama 2/3, Mistral, Mixtral, GPT-2, Mellum.
  Loads safetensors (single or sharded), GGUF, GPTQ, and AWQ checkpoints;
  f32/bf16/f16 plus int8 (weight-only and W8A8) and group-wise int4
  quantization; KV-cache; the standard samplers; and a pluggable matmul
  `Backend` (default pure-Go SIMD).
- **`tokenizer`** — byte-level BPE and SentencePiece byte-fallback tokenizers
  for the decoder LLMs, from `tokenizer.json` or a bare `.gguf`, id-exact
  against HuggingFace.
- **`constrain`** — constrained / structured decoding via a logit mask that
  forces output to satisfy a grammar; ships a streaming JSON grammar.
- **`gpu`** (opt-in submodule, `-tags gpu`) — WebGPU compute backend
  (Metal / Vulkan / DX12) that registers a resident-weight matmul into both
  the goinfer `decoder` and the aikit `encoder`. The cgo
  `github.com/cogentcore/webgpu` dependency is confined to this submodule;
  the default goinfer build is pure Go, no cgo.
- `demo/gemma` and `demo/gemma-web` — CLI and HTTP examples wiring tokenizer +
  decoder with optional sampling and JSON-constrained output.

### Notes
- goinfer is extracted from [`aikit`](https://github.com/townsendmerino/aikit)'s
  `decoder` / `tokenizer` / `constrain` packages so the LLM runtime can release
  on its own cadence, independent of aikit's stable retrieval core. It depends
  inward on `aikit/embed` + `aikit/linalg` (≥ v0.4.0, which promoted `linalg`
  to public and gave `encoder` a pluggable GPU `Backend`). The split is tracked
  in `aikit/docs/aikit-module-split-plan.md`. Pre-extraction history for these
  packages lives in the aikit repository.
