# goinfer — Code Review: Fixes Needed

**Date:** 2026-07-18
**Reviewed at:** working tree on `main` (754ab66 + uncommitted changes to `README.md`, `decoder/features.go`, new `decoder/hardware_matrix_test.go`)
**Scope:** all four modules — root (`decoder`, `tokenizer`, `chat`, `constrain`, `multimodal`, `internal`, `cmd`), `gpu/`, `metal/`, `cuda/` — non-test sources, plus module/workspace layout.
**Method:** `go build` + `go vet` on all four modules (all clean; `metal` vetted under `GOOS=darwin`, `cuda` under `-tags cuda`), `gofmt -l` (clean except the in-flight `decoder/features.go` edit), followed by a multi-pass manual review with every reported finding re-verified against the source before inclusion. Line numbers refer to the current working tree.

Severity scale: **Critical** = silent wrong output, memory corruption, or crash on a supported path. **Major** = correctness/robustness defect reachable in real use, or a contract the code documents but does not keep. **Minor** = robustness gaps, latent traps, API-compat divergences. **Nit** = polish. Items tagged *(verify)* are high-likelihood but depend on a checkpoint config or spec detail worth confirming before fixing.

---

## 1. Overall assessment

This is a strong codebase. The parity-gating discipline (HF oracles, argmax-exact gates), the feature-taxonomy admission design in `decoder/features.go` (an architecture a backend can't fully run is declined, never mis-run), the f64-accumulation strategy that makes decode ≡ batched prefill ≡ speculative verify bit-exact, the GGUF transform-reversal work, and the streaming-quant memory design are all carefully engineered and unusually well documented at the site of the decision. Concurrency in the serving layer is simple and mostly sound (one decode worker per model, honest 429 backpressure). `go vet` is clean across all four modules and the fuzz harnesses encode exactly the right invariants.

The defects cluster in four places, and almost all of them share one shape — **a guarantee enforced in one layer that a sibling layer silently fails to keep**:

1. **The `.giw` serialization format trails the loaders.** New `LayerWeights` fields (MLA, Mamba, Gemma-4 PLE, `TiedLMHead` finalization) were added to the loaders but not the writer/reader, and nothing guards the gap — so prequant/stream-transcode produces structurally valid bundles that load cleanly and then generate garbage or panic (C2).
2. **Rewind/rollback invariants are documented but unenforced.** `ring.truncate` computes the exactness flag the design doc calls for and the one call site throws it away (C1); session reconciliation and error-path truncation have similar holes (M10, M11).
3. **The GPU backends don't enforce their own capacity/shape limits.** Fixed context caps (4096 / 16384), a 256-expert kernel clamp, and head-dim ≤ 128 kernels are all real limits with no host-side check, so exceeding them corrupts memory or silently degrades output instead of erroring (C3, C4, M20).
4. **The serving layer trusts too much.** Generation errors are never surfaced (M1), stop sequences leak across token boundaries (M2), request bodies are unbounded (M3), and two registry/session races are reachable with documented flag combinations (M4, M5).

None of this contradicts the project's stated correctness bar — it's the same silent-wrong-output failure class the architecture was explicitly built to prevent, showing up in the seams *between* the well-built subsystems. The fixes below are ordered so the highest-leverage guards (a `canSerialize` predicate, cap checks in `Forward`, consuming the exactness flag) close whole classes at once.

---

## 2. Critical — fix first

### C1. Sliding-window KV: ring rewind exactness flag is computed and then discarded
`decoder/kvcache.go:413` (flag produced at `ring.truncate`, `kvcache.go:220`)

```go
if r := c.rings[l]; r != nil {
    r.truncate(pos) // exactness flag consumed by the rewind rule (Inc 2)
    continue
}
```

`ring.truncate` returns whether a rewind is exact; no code in the repo consumes it (this is the only call site). `docs/completed/task-kv-ring-eviction.md` shows Increment 2 — "TruncateTo rewind rule … a deeper-than-ring rewind triggers cold prefill, never wrong output" — was never shipped. Once a ring has wrapped (`count > W`), a rewind deeper than one position leaves window slots physically holding the *dropped* positions' K/V, and subsequent attention reads them as history.

Two live triggers, both routine:
- **Session prefix rewind** — `Session.Generate` → `s.cache.TruncateTo(matched)` (`session.go:82`) on any divergent chat history once context exceeds the window (Gemma-3 local layers W=512/1024; Mistral/Phi-3 similar). This breaks the documented "bit-identical to a full prefill" guarantee, silently.
- **Speculative rollback** — every rejection of ≥2 draft tokens (`spec_ngram.go` / `speculative.go` → `TruncateTo(C+acc)`) on a wrapped ring. `specRollbackSafe` screens recurrent families but not rings, so the "lossless" speculative guarantee is violated for exactly the sliding-window models rings serve.

**Fix:** propagate the flag out of `TruncateTo` (e.g. `TruncateTo(pos) (exact bool)`), and on an inexact rewind force a cold prefill (sessions: clamp `matched` to what's exactly rewindable, or Reset; speculative: gate spec on `count ≤ W` for ring caches or fall back to re-prefill). Add a regression test: wrapped ring → rewind ≥2 → compare against cold prefill.

### C2. `.giw` serialization silently cannot represent MLA / Mamba / Gemma-4 / tied-LM-head models — and nothing refuses them
`decoder/serialize.go:396-461, 185`; `decoder/gguf.go:937, 1174, 1566, 1738`; `decoder/kvsnapshot.go:66`

Four related gaps, one root cause — no guard connects "the loader produced X" to "the serializer can express X":

1. **`giwWriter` drops `l.mla` and `l.mamba` silently** (handles only `delta`/`qattn`; `default: w.raw([]byte{0})`), and never writes the Gemma-4 PLE fields (`PLEGate/PLEProj/PostPLENorm/LayerScalar/KVShared/VFromK`, per-layer token embeds). A DeepSeek GGUF stream-transcoded to `.giw` produces a CRC-valid bundle that loads cleanly and **panics at the first forward** (`w := lw.mla` nil-deref, `forward_deepseek.go:81`); Granite/Nemotron via `prequant` on a safetensors dir lose all SSM state the same way.
2. **`LoadSerializedWeights` never finalizes `TiedLMHead`** (`serialize.go` contains no reference to it; every other loader sets it from `lm_head`/`output.weight` presence — `gguf.go:1163`, `weights.go:496`). A tied checkpoint (Qwen3-0.6B/1.7B, Qwen2.5-0.5B, Llama-3.2 GGUFs with no `output.weight`) round-tripped through `.giw` comes back `TiedLMHead=false` with an empty LMHead → **all-zero logits: greedy emits token 0 forever, sampling is uniform noise. No error.** `prequant`'s `selfCheck` passes because the bundle parses. Fix: `arch.TiedLMHead = w.LMHead.Rows() == 0` after reading the head (or persist the flag).
3. **`StreamTranscodeGGUF`'s sink guard covers only qwen35/gemma4** (`gguf.go:937`: `if arch.qwen35 != nil || arch.gemma4 != nil`). The granite (`:1566`), nemotron (`:1738`), and llama4 branches return after `parallelLayers` without ever calling `sink.layer(...)` — with a sink set they do a **full resident load** (defeating the "peak RAM ≈ one layer" contract for exactly the big models streaming targets) and then emit a header that declares N layers followed by no layers.
4. **`Snapshot` refuses only `c.delta`** (`kvsnapshot.go:66`) but `NewCache` also creates `c.mamba` (Granite/Nemotron) and `c.mlaLatent` (DeepSeek). Snapshotted sessions for those families restore with zeroed recurrent state / empty latents and continue from silently wrong state.

**Fix (one mechanism):** add a `canSerialize(arch) / canSnapshot(cache)` predicate; make `SerializeWeightsTo`, `StreamTranscodeGGUF`, and `Snapshot` refuse unsupported families with a typed error; make `giwWriter.layer` refuse (not skip) non-nil fields it doesn't write; finalize `TiedLMHead` on load. A table-driven round-trip test per registered architecture would have caught every one of these.

### C3. Metal and CUDA resident backends: nothing enforces the 4096-position cap — out-of-bounds KV writes on long generations
`metal/model.go:18` (`metalCtxCap = 4096`), `metal/backend.go` `Forward`; `cuda/resident.go:24` (`cudaCtxCap = 4096`), `cuda/resident.go:205` `Forward`; decoder side `decoder/model.go` decode loop (`gpuPos++`, unbounded)

The only cap check in either backend is `PrefillLast`'s prompt-length guard (`metal/backend.go:115`). Decode has none: the decoder's generate loop increments `gpuPos` without bound and calls `resident.Forward(emb, gpuPos)`. Once `pos ≥ 4096`:
- **Metal:** `kv_store` writes `kc[pos*kvDim+i]` past the MTLBuffer — on unified memory this silently corrupts adjacent allocations (other models' weights included), and the attention kernel's `threadgroup float sc[4096]` overflows once `nKeys > 4096`.
- **CUDA:** `rope_kv` stores past the `cudaCtxCap*kvDim` allocation (device OOB, UB), and the attention launch's shared-memory request eventually exceeds the 48KB block limit — an error the code discards (see M23), so attention silently stops running.

A prompt of ≤4096 plus `max_tokens` large enough to cross the cap is all it takes (`cmd/serve` lets clients set `max_tokens`; nothing clamps prompt+output to 4096). Worse, on Metal the prompt guard *funnels into* the bug: a >4096-token prompt makes `PrefillLast` decline, and the fallback prefills sequentially via `Forward(emb, i)` — guaranteed OOB for exactly the prompts the guard rejected.

**Fix:** `Forward`/`ForwardN`/`ForwardArgmax` must return an error when `pos+K > ctxCap` (both backends), and the decoder loop must stop with that error (or fall back to the staged path). Make the cap a queryable property of the resident interface so `generateInto`/serve can clamp `max_tokens` up front. The same check belongs in `gpu/` (16384/32768/65536 caps in `gpu/residency.go:131` — same gap, see M20).

### C4. WebGPU partial-RoPE: the un-rotated K tail is never written to the KV cache — silently wrong logits for GLM-family resident models
`gpu/attention.go:176` (`qkvFinalize`), plus `ropeStore` (~:106), `ropeStoreF16` (~:288), `ropeStoreI8` (~:393)

All four K-store kernels write only the rotated span. In `qkvFinalize`:

```wgsl
if (i < p.nKV * p.half) { ...
    kCache[p.base + off + d]          = x1 * c - x2 * s;
    kCache[p.base + off + p.half + d] = x2 * c + x1 * s;
}
```

That covers `[0, 2*half)` per head. For partial rotary (`half = rotaryDim/2 < hd/2` — GLM, some Phi; admitted by the residency layer as "Lever C5"), the pass-through tail `[2*half, hd)` is **never stored** — attention reads zeros for those key dims at every cached position. The CPU reference stores the full `k` (`decoder/rope.go:22`). The f16/i8 variants additionally index `invFreq` out of bounds for the tail. This is not hypothetical: **the CUDA backend contains the explicit fix for this exact bug** — `cuda/resident.go:577-583` dispatches `nKV*(hd-2*rhalf)` extra tail threads, with a comment and a dedicated `cuda/rope_partial_test.go` describing the failure ("produces plausible-looking logits"). The WebGPU kernels have the pre-fix structure, and the GLM e2e test skips without a local checkpoint and gates only the first greedy token.

**Fix:** mirror the CUDA tail-store dispatch in `qkvFinalize` and all three `ropeStore` variants; clamp `invFreq` indexing to `dd < 2*half`. Add a partial-rotary CPU-oracle parity test (`hd=64, rotaryDim=32`) so the gate doesn't depend on a local GLM checkpoint.

### C5. Metal `PrefillLast` allocates ~20 buffers per request onto the device ledger and never frees them — serve leaks unified memory until an unrecovered OOM panic
`metal/prefill.go:309-339`; ledger mechanics `metal/metal.go:194-201` (`mustBuf` → `d.allocs`), freed only by `ReleaseAll` (Close)

Every `PrefillLast` call allocates fresh f16 scratch (`xF`, `normF`, `qkvF`, `ctxF`, `guF`, `dqF`, `posB`, ~17 uniform buffers — 22 `NewBuffer*`/`mustBuf` sites in the file, zero `Release` calls). All go through `mustBuf`, which appends to the device ledger; nothing releases them until `Resident.Close()`. `cmd/serve` calls `PrefillLast` once per request (prompt ≥ 8), so a 7B model leaks on the order of 100–150 MB of unified memory per request (`guF` alone is `Mpad*2I*2` bytes) and ratchets until `mustBuf` panics — and that panic is only `recover`ed on the `BuildResident` path, so **on the prefill path it kills the serve process**. `close_leak_test.go` pins load/Forward/Close cycles, not per-request growth, which is why the gate is green.

**Fix:** preallocate max-size prefill scratch once on the `Resident` (the decode path already does exactly this), or give `Device` a per-buffer release API and free at the end of the call. Extend `close_leak_test` with an N-request RSS gate. Related: `BuildResident`'s error/decline paths also return without `ReleaseAll` (M24).

### C6. `FeatRopeMscale` is derived from layer 0 only — Mellum's YaRN is missed, over-admitting it as resident; and the hand profile vs. the derivation are never cross-checked *(verify)*
`decoder/features.go:60`; hand profile `decoder/features_test.go:30`; generated matrix `docs/hardware-matrix.md` / `decoder/hardware_matrix_test.go`

`residentFeatures()` derives `FeatRopeMscale` from `a.ropeMscale(0) != 1` — **layer 0 only**. Mellum interleaves sliding/full attention with YaRN on the `full_attention` layers only; its layer 0 is a `sliding_attention`/`default` layer (mscale 1), so the check samples the wrong layer and Mellum's required set omits `yarn-mscale`. Verified: `residentFeatures(representativeConfig("mellum"))` = `[moe per-layer-rope qk-norm sliding-window]`, and `ResidentEligible(mellum, cuda)` / `(mellum, metal)` both return `true`. (`ropeUniform` at `:100` loops all layers to detect that mscale *varies* → `FeatPerLayerRoPE`; the yarn check right beside it does not.)

Two consequences:
- **Generated matrix over-reports.** `docs/hardware-matrix.md` shows Mellum2 resident on CUDA and Metal — neither declares `FeatRopeMscale`. The feature's own comment says *"(Mellum, long-ctx)"* and `archFeatureProfile["mellum"]` lists `yarn-mscale`, so author intent + hand profile both contradict the derivation. This is why the README's compact table cannot be safely synced to the generated slice yet — one of its cells is wrong.
- **Potential silent-wrong (the reason for the tag).** If the CUDA/Metal resident path does not apply Mellum's full-layer YaRN `attention_factor` (CUDA's docs note "no yarn-mscale"), a real Mellum2 is admitted resident and runs with the scaling dropped on the full layers — the exact class this taxonomy prevents. Confirm against a real Mellum2 CUDA-vs-CPU run whether it's live or masked by the per-layer-RoPE (`ropeScale`) handling.

**Root gap:** `archFeatureProfile` (hand table, drives `TestResidentAdmission_matrix`) and `residentFeatures()` (derivation, drives the generator + `TestHardwareMatrix_fresh`) are two sources of truth that are **never cross-checked**, so they silently disagree on Mellum and nothing goes red.

**Fix:** derive `FeatRopeMscale` from *any* layer's mscale (`for i … if a.ropeMscale(i) != 1`), not just layer 0; add a test asserting `archFeatureProfile[name] == residentFeatures(representativeConfig(name))` for every registered arch (ties the two sources together — would have caught this, and confirms the qwen2_moe `FeatMoEGatedShared` addition); regenerate `hardware-matrix.md`. The real-Mellum2 check decides *which* artifact was wrong, but the derivation fix + cross-check test is the same either way.

---

## 3. Major

### Serving layer (`cmd/serve`)

**M1. Generation errors are never surfaced — failures return HTTP 200 with truncated/empty content.**
`cmd/serve/openai.go` (`drive`, ~:581); grep confirms zero `.Err()` calls in the package. `decoder.Generation` documents "check Err after the range loop"; `generateInto` sets it on prefill failure, forward/backend failure, and sampler error. `drive` returns only `(finish, n, logprobs, stopHit)`, `driveVL` discards the Generation entirely, so every handler reports `finish_reason:"stop"` and 200 on a failed generation. Compounding: `encode()` drops tokenizer errors (`ids, _ := lm.tk.Encode(...)`, `openai.go:496`). **Fix:** after `streamTokens` returns, check `gen.Err()` (ignore `context.Canceled` from stop/client-cancel): non-stream → 500 with an error body; stream → an error SSE event before closing. Surface `encode` errors as 400s.

**M2. Stop sequences that span token boundaries leak their already-emitted prefix.**
`cmd/serve/openai.go:609-645` (`streamTokens`). The loop flushes up to `completeUTF8(text)` with no holdback for a partial stop-string match; a stop of `"END"` arriving as `"E"`+`"ND"` emits `"E"` to the client (SSE already sent; non-stream keeps it in `sb`). OpenAI/Anthropic semantics require the stop sequence removed entirely; multi-token stops (`"\n\nUser:"`) make this routine. **Fix:** in addition to the UTF-8 holdback, hold back the longest suffix of `text` that is a proper prefix of any stop string (standard partial-stop logic).

**M3. No request body size limit on any endpoint.**
All eight POST handlers `json.NewDecoder(r.Body).Decode(&req)` the raw body; no `http.MaxBytesReader` anywhere in the package. A multi-GB JSON body (or giant base64 `image_url`, which is then decoded into a second copy) is buffered before validation — memory-exhaustion DoS, squarely in the threat model `images.go`'s own header describes. **Fix:** wrap every body with `http.MaxBytesReader` (a few MB default; larger for vision endpoints), return 413. While there: the `http.Server` at `main.go:324` sets no `ReadHeaderTimeout`/`IdleTimeout` (slowloris; `WriteTimeout` must stay 0 for SSE) — add both.

**M4. `demoteLoop` and the shutdown goroutine iterate `srv.models` without `regMu` — concurrent-map panic with `--allow-admin`.**
`cmd/serve/main.go:715` and `:349` iterate the map with only `lm.mu`; `handleAdminLoad`/`Unload` write it under `regMu` (`admin.go:82-85`). `--allow-admin` + `-kv-idle-demote` is a legal combination; an admin load during a demote sweep is a concurrent map iteration+write — a runtime-fatal panic, not just a race. **Fix:** snapshot the model list under `regMu.RLock()` in both places.

**M5. Admin load publishes the model before restoring its sessions, racing request handlers.**
`cmd/serve/admin.go:82-86`: `s.models[lm.name] = lm; s.regMu.Unlock()` then `lm.sessions.load(...)` runs with no lock. `sessionLRU` is documented not-goroutine-safe (guarded by `lm.mu`, which load doesn't hold); a request can acquire the model and touch the LRU while `load` is still appending. KV snapshots can be hundreds of MB — the window is seconds. **Fix:** run `sessions.load` before publishing (or hold `lm.mu` around it).

**M6. `/v1/responses` tool path generates with `context.Background()` — client disconnect never cancels.**
`cmd/serve/responses.go:205`: `lm.drive(context.Background(), gr, ...)` (the helper doesn't take `*http.Request`). A disconnected client leaves the model decoding to `max_output_tokens` while holding `lm.mu` and a queue slot; retries amplify into a DoS. Every sibling path passes `r.Context()`. **Fix:** thread `r` through, as `handleChatTools` does.

**M7. Adapter models silently generate with base weights on a GPU-resident backend — and can run concurrently against one resident KV.** *(verify)*
`cmd/serve/openai.go:539`: the resident path (`lm.model.ResidentActive()`) uses a fresh cache with no adapter binding (adapters live on per-session caches; the resident path never sees them), so requests to an adapter's served name are answered by the base model with no error. Separately, base and adapter `loadedModel`s have distinct `mu`s but share one resident runner/KV — concurrent generations interleave positions on-device (see also M12). `loadAdapters` rejects `--stream-weights` but not resident backends. **Fix:** make adapter-backed models take the non-resident path (or reject `--adapter` + resident backend at startup).

### Decoder core

**M8. `Model.Generate`'s channel send ignores ctx — an abandoned consumer leaks the goroutine (and wedges its Session) permanently.**
`decoder/model.go` emit loop (~:768): `out <- next` is a bare send; the ctx check is a separate `select { case <-ctx.Done(): … default: }` at the top of the loop. A consumer that stops ranging — even one that then cancels ctx, the documented stop mechanism — leaves the goroutine blocked on send forever, holding the KV cache; for `Session.Generate` the reconciliation after `generateInto` never runs, so the session is permanently wedged. All speculative paths already do this right (`select { case <-ctx.Done(): return; case out <- tok: }`). **Fix:** use the same select in `generateInto`, `GenerateVL`, `GenerateQwenVL`.

**M9. Concurrent `Generate` on a residency-enabled model corrupts the shared resident KV.**
`decoder/model.go` doc promises "distinct sequences can run concurrently"; with `m.resident` set, every generation drives the single per-model positional resident cache (`m.resident.Forward(emb, gpuPos)`) with no lock or in-flight guard — two concurrent calls interleave writes at overlapping positions. `cmd/serve` is safe only because `lm.mu` serializes; the library contract is not. **Fix:** document "one in-flight generation when resident", or guard with an atomic in-use flag and fall back to the staged path.

**M10. Session reconciliation overclaims the cache when prefill fails, poisoning later prefix reuse.**
`decoder/session.go:88-99`: `seq` starts as the full prompt; the goroutine unconditionally runs `s.tokens = seq; s.cache.TruncateTo(len(seq))`. On a partial prefill failure the cache holds fewer positions, `TruncateTo(pos > c.pos)` no-ops (`kvcache.go:404`), and `s.tokens` claims the whole prompt — the next Generate "reuses" KV that was never written and prefills at the wrong position. Silently wrong output. **Fix:** on `g.err != nil` (or whenever `s.cache.Pos() < len(seq)`) clamp `seq = seq[:s.cache.Pos()]` or Reset. Same pattern in `genSpec` and the grammar path. Related: an empty-prompt call runs `TruncateTo(0)` *before* the validation error, wiping a warm session's KV (minor but same family).

**M11. `TruncateTo`'s len/pos stride derivation corrupts layers left ragged by a mid-sweep forward error.**
`decoder/kvcache.go:432`: `perPos := len(c.keys[l]) / c.pos` assumes every layer holds exactly `c.pos` rows. If a forward errors between layers, earlier layers hold `c.pos+1` rows; the derived stride is then wrong whenever `c.pos < width`, and the slice cuts mid-row — permanently misaligning that layer. This undercuts the "clean rollback if a forward errors mid-stream" claim (`session.go:84`). **Fix:** record each layer's stride at first append instead of deriving it; truncate ragged layers to `pos` rows explicitly.

**M12. Batched verify (`forwardN`) ignores an active compute-time LoRA — speculative decoding silently verifies with the base model.**
`decoder/forwardn.go:458` guards only `!m.canBatchN(K)`; `prefillLogits` (`:500`) guards `cache.lora != nil || !m.canBatchN(...)`. Path: `Session.UseAdapter` → ngram/grammar speculative → prompt KV prefilled *with* the adapter, every verify block projected *without* it, K/V committed to the same cache. The KVCache field doc even notes lora isn't wired into `forwardN`; nothing enforces it. **Fix:** add `cache.lora != nil` to `forwardN`'s sequential-fallback condition (the sequential path applies LoRA correctly), or reject adapters in the spec validators.

**M13. Greedy speculative paths skip repetition penalties and LogitBias, breaking "token-identical to plain greedy".**
Plain greedy sampling applies LogitBias + penalties before argmax (`sampler.go` `SampleWithInfo`); every greedy spec loop verifies with raw `argmax(logitsN[i])` (`spec_ngram.go:289/378`, `speculative.go:196/252`, `spec_grammar.go:245/293`, both EAGLE paths) and no validator rejects penalties/bias at Temperature 0. `RepeatPenalty` + greedy therefore silently diverges from `Generate` — the exact property the parity tests claim. The *sampled* path handles history correctly via `distVectorHist`. **Fix:** reject penalties/bias in greedy spec mode, or thread penalized history as the sampled path does. (Also update the stale doc claiming the sampled path rejects them — it threads them now.)

**M14. `Load` proceeds with a nil backend when `Options.Backend == "metal"` on an untagged build — panic at first matmul.**
`decoder/backend.go` `NewBackend`: "webgpu"/"cuda" return `(&cpuBackend{}, note)`; "metal" (accepted by `Options.Validate`, `model.go:191`) falls to `return nil, fmt.Errorf("unknown backend %q (have: cpu, webgpu, cuda)")` — and `Load` deliberately treats `beErr` as a note and continues (`model.go:88`), now with `be == nil`. **Fix:** give "metal" the same CPU-fallback+note treatment (and add it to the error's have-list); abort `Load` if `be == nil`.

**M15. EAGLE tree drafting builds a full B-ary tree, not the documented B×D chains — exponential verify cost and wrong stats.**
`decoder/eagle.go:263` (`DraftTree`): every frontier node expands B children (`topKDraftIdx(parentLogits[fi], b)` inside the frontier loop), so B=2,D=4 verifies 30 nodes, not 8; `eagle_accept_test.go:83` confirms ("full binary tree to depth 5 = 62 nodes") while the API docs promise "top-B first tokens, each a depth-D chain". `stats.Drafted += td.B * td.D` undercounts, and the cache `capHint` under-sizes. **Fix:** branch only at depth 1 (chains after), or fix docs + `Drafted += len(td.Tokens)` + capHint. Related robustness: the EAGLE entry points validate neither `capLayers` (needs exactly 3) nor batchability — unsupported targets panic (nil `cache.captured` / out-of-range fuse) instead of erroring.

**M16. GGUF config builders divide by / allocate from unvalidated metadata before `validateGGUFDims` runs.**
`decoder/gguf.go:730` (`headDim := hidden / heads` — phi3; missing key → `u()` returns 0 → integer-divide panic), same class at `:537` (granite `MambaDHead`), `:620` (nemotron); and `make([]string, nLayers)` / `make([]int, nLayers)` from raw `block_count` at `:501`, `:591`, `:800` (+ the qwen35 LayerTypes append loop) — makeslice panic on wrapped-negative, unbounded allocation on huge. All inside `ggufConfig`, i.e. before the validator whose doc says it exists to stop exactly this; `FuzzGGUFConfig`'s contract is "must never panic" but its seeds don't cover these seven architectures. **Fix:** validate `block_count`/core dims generically before dispatching to family builders (or hoist the guards into each), and add fuzz seeds for every GGUF arch. Same family: `permMat` divide-by-zero when `HeadDim == 1` (`:1108`); granite/llama4 registry adapters divide by unvalidated `cfg.NumHeads` on the safetensors/.giw paths (`registry.go:841/1192`); deepseek/llama4 accept rope base 0 → RoPE silently never applied (`registry.go:979/1183`) — add the `>0` guard phi3 has.

**M17. `LoadSerializedWeights` and `LoadSession` panic or over-allocate on malformed CRC-valid input, violating their documented never-panic contracts.**
(a) `serialize.go:581` `giwReader.weightMat` passes blob-controlled `rows/cols/group` straight to `linalg.WrapInt4/WrapInt8`, which **panic** on `group <= 0` or length mismatch (verified in aikit v1.9.0) — a one-byte kind/group flip with recomputed CRC crashes the loader; kind-1 f32 defers the panic to first use. (b) `kvsnapshot.go:194` `LoadSession` has no sanity ceilings (unlike serialize.go's `maxSerializedLayers`): `pos=0xFFFFFFFF` → multi-TB makeslice; blob-controlled ring `stride`; unchecked `copy(rr.kq[so:so+st], kb[do:do+st])` on inconsistent lengths → slice panics. CRC is integrity, not authenticity. **Fix:** validate group/lengths before wrapping (typed `*SerializeError`); bound `pos`, validate strides against model widths, check array-length consistency in `LoadSession`.

**M18. LoRA merge silently no-ops for VL-prefixed checkpoints and qwen35 — defeating `validateTargets`' stated purpose.**
`decoder/lora.go:142`: validation builds its known-set from unprefixed `tensorName(...)`, but the loader merges via prefixed `tn(...)` ("language_model.model.layers…" Gemma-3 VL, "model.language_model.layers…" qwen35-VL) — so on those checkpoints validation passes and the adapter is ignored. qwen35 safetensors + LoRA passes validation but `loadQwen35Attn` never consults the adapter. The function's doc says it exists to "fail loudly rather than silently no-op". **Fix:** validate against the same prefixed names the loader requests; reject qwen35 like the other special-forward families. Related: `LoadAdapter` closes a replaced adapter's mmap while live sessions may still reference it (`lora.go:277`) — document or refcount.

**M19. Gemma-3-27B attention scale is wrong in two independent places.** *(verify against a 27B config)*
(a) `decoder/gguf.go:221`: `cfg.QueryPreAttnScalar = float64(cfg.HeadDim)` — true for 1B/4B/12B (256) but 27B uses `hidden/heads = 5376/32 = 168` while `head_dim` is 128 (llama.cpp special-cases exactly this). (b) `decoder/residency.go` `Model.AttnScale()` returns `1/√HeadDim`, ignoring the registry-resolved `arch.AttnScale` (`Pow(QueryPreAttnScalar, -0.5)`) — consumed by the Metal/CUDA resident paths. Both are invisible on the small models the parity fixtures use (scalar == head_dim there) and mis-scale every attention logit by ~√(168/128) on 27B. **Fix:** mirror llama.cpp's 27B geometry handling in the GGUF builder (or reject loudly), and make `AttnScale()` return `float32(m.w.arch.AttnScale)`.

### GPU backends (`gpu/`, `metal/`, `cuda/`)

**M20. WebGPU: resident context caps (16384/32768/65536) unenforced — same class as C3.**
`gpu/residency.go:131` sizes every KV buffer from `ctxCap`; neither `DecodeRunner.Run` nor the decode loop checks `pos < ctxCap`. Models on this path advertise 32k–131k contexts; past the cap, WGSL robust-access clamps the writes and attention reads garbage — silent corruption, no error. **Fix:** plumb `ctxCap` into the runner and error when exceeded (see C3's queryable-cap suggestion).

**M21. WebGPU: `newDecodeRunner` swallows buffer-allocation errors and panics on bind failure — OOM crashes instead of falling back.**
`gpu/decoderunner.go:219-237`: `b, _ := c.device.CreateBuffer(...)` in `storF`/`uni`/`storFZ`/logits staging; nil buffers flow into `bind()` whose failure handler is `r.release(); panic(e)` — in a constructor that already returns `error`, on exactly the path (VRAM exhaustion) callers need an error to fall back to CPU. `DecodeTokenFused`/`Batched` repeat the pattern. **Fix:** accumulate and return errors (the `vision_encoder.go` `up`/`upW` pattern); never panic in library code.

**M22. WebGPU MoE router silently clamps to 256 experts / 32 groups — Kimi K2 (384 experts) would route wrong.**
`gpu/moe.go:25`: `const MAXE: u32 = 256u; let nE = min(p.nE, MAXE);` with fixed `array<f32, 256>`/`array<f32, 32>`; no host-side check in `BuildResident`/`moeResidentEligible`, and `kimi_k2` is a registered arch with 384 routed experts (`registry.go:39`). Experts 256+ would simply never be considered — plausible-looking wrong output. **Fix:** reject `nE > 256 || nGroups > 32` at build with an explicit decline (staged fallback). Same pattern to add while there: attention kernels hard-cap `hd ≤ 128` with no eligibility guard (`gpu/attention.go:59`) — latent today, add the check.

**M23. CUDA: launch errors discarded across the dense hot chain; mixed-quant checkpoints hit a nil-buffer kernel via the `fuseQKV` guard.**
(a) `cuda/resident.go:557-641`: `_ = r.launch(...)` for k/v proj, rope_kv, attention, quant, sandwich block, unfused FFN — `cuLaunchKernel` config errors (bad shared-mem size — see C3's overflow — bad args) are returned immediately and are *not* sticky, so the token "succeeds" with stale buffers. The MoE path checks every launch; the dense path's discards are drift, not policy. **Fix:** check every launch. (b) `cuda/backend.go:467`: `r.fuseQKV` is derived from Q/K/V kinds only but also gates `fused_rms_gu`, which reads gate/up as int4+f16-scales; an int4-QKV + int8-gate/up checkpoint passes a nil `ws16` — crash or garbage on the executor goroutine (which has no `recover`, so the process dies). **Fix:** require g/u (and d) int4 in the guard, or gate fGU per layer.

**M24. Metal: `BuildResident` error/decline paths leak everything allocated so far; `Close` leaks all non-buffer objc objects.**
(a) Mid-construction errors (sandwich shape check `model.go:286`, `buildMoE`, LM-head int8) return without `ReleaseAll`, and the `mustBuf` OOM panic is recovered in `backend.go` as a clean decline — leaving gigabytes on the ledger while serve continues on CPU. **Fix:** `defer` cleanup unless construction completed; release in the recover. (b) `Close` → `ReleaseAll` frees buffers only: the MTLDevice, command queue, ~40 pipelines, 2 libraries, and every intermediate `MTLFunction` (never released even on success, `metal.go:176`) leak per load/unload cycle. **Fix:** track and release them. Related FFI hygiene: `BuildResident`/`ensurePrefill` call `newLibraryWithSource:`/pipeline creation with no autorelease pool on an unpinned thread (`metal.go:110`) — wrap in `LockOSThread` + pool like `Forward` does.

### Text stack (`tokenizer/`, `chat/`, `constrain/`)

**M25. Prompt injection: special-token surface forms in user/tool content become real control tokens.**
`chat/templates.go:66` (and every renderer): untrusted content is concatenated verbatim (`"<|im_start|>" + t.Role + "\n" + t.Content + …`), and `Tokenizer.Encode` unconditionally matches added-token surface forms anywhere in the text (`tokenizer/added.go` trie), emitting their control ids. So a user message or tool result containing `<|im_end|>\n<|im_start|>system\n…` forges turn boundaries end-to-end (serve and demos both do `Render` → `Encode`; there is no sanitization and no plain-text encode mode — verified by search). **Fix:** add an Encode mode/segment API that skips added-token matching for content spans (Render returning marked segments is the clean shape), or neutralize marker strings in content before rendering. This is the standard llama.cpp `parse_special=false` distinction; the stack currently has no equivalent.

**M26. Constrained decoding: token ids with empty surface bytes are never masked — padded-vocab models can livelock or emit undecodable ids.**
`constrain/constrain.go:213`: `TryBytes(m.tokenBytes(id))` — `tokenBytes` returns nil for out-of-table ids (verified), and every grammar's `TryBytes(nil)` is vacuously true (verified: the byte loop simply doesn't run). Callers build the table from the *model's* vocab size (`demo/chat/main.go:472`), which exceeds the tokenizer table on Qwen/Gemma (padded vocab) — those ids stay legal at every step; sampling one never advances the grammar (livelock to maxTokens) and `Decode` then fails. **Fix:** mask any non-EOS id whose byte surface is empty. Inconsistency in the same file: `Commit`/`MaskAt` index `m.tokens[id]` unchecked (`:139`) — same configs panic; use `tokenBytes` in both.

**M27. `constrain` schema compilation: silently ignored keywords and compilable-but-unsatisfiable property names — both against documented contracts.**
(a) `schema.go:79`: the package doc promises "unsupported keywords are a compile error rather than a silent no-op", but `pattern`, `minimum`, `oneOf`, `$ref`, `uniqueItems`, etc. compile cleanly with the constraint dropped — callers believe a constraint is in force that isn't. **Fix:** error on unknown non-annotation keys. (b) `schema.go:187`/`schema_grammar.go:393`: property names containing `"`, `\`, or control bytes compile but can never be matched byte-wise (`keyStep` treats `"` as terminator, rejects `\`) — mid-generation every logit becomes −∞, the exact "unsatisfiable" state `fuzz_test.go:106` declares a bug (schemas are per-request attacker-supplied via serve's `response_format`). **Fix:** reject or JSON-escape such names at compile; seed the fuzzer with one. (c) `reflect.go:99`: `SchemaFromStruct` recurses without a visited set — a self-referential struct (`type Node struct { Children []Node }`) is a fatal stack overflow, not an error. **Fix:** track visited types, return "recursive type unsupported". Also `{"type":"object"}` with no `properties` silently compiles to "`{}` only" (`schema.go:142`) — over-constraining the standard freeform-arguments shape; error instead.

**M28. Tokenizer: GGUF special-token ids unvalidated; Gemma path appends BOS without the `>= 0` guard; O(n²) merge on hostile input.**
(a) `tokenizer/gguf.go:316`: `ggufTokenID` returns `int(v)` unchecked — the common `0xFFFFFFFF` "none" sentinel (and any hostile value) flows into Encode as a garbage id and then out-of-range indexes the embedding downstream. Clamp to `[0, len(tokens))` else −1. (b) `sentencepiece.go:385`: `if addBOS { out = append(out, int32(t.special.BOS)) }` lacks byte-level's `BOS >= 0` guard — GGUF "llama"-family models without a BOS key emit id −1. (c) `sentencepiece.go:509`: `mergeSymbols` rescans all pairs per merge (O(n²)); modeGemma has no pretokenizer so the whole inter-added-token gap is one unit — a few-hundred-KB request body is minutes of CPU (serve feeds client text straight in; byte-level is equally exposed via one long pretoken). Use the standard heap/linked-list merge or cap unit length.

---
## 4. Minor

Grouped by area. Each entry: location — issue → fix.

### `cmd/serve` — API compatibility & robustness

- **`openai.go` (sampling struct)** — `max_completion_tokens` is not read (only legacy `max_tokens`; verified absent from the package). Current OpenAI SDKs send only the new field → silent truncation at the 512 default with `finish_reason:"length"`. → Add the field, prefer it when set.
- **`helpers.go:65` `writeErr`** — hardcodes `"type":"invalid_request_error"` for every status including 429 (queue full) and 500; typed client retry logic keys off this (the Anthropic side gets kinds right). → Derive type from status (`rate_limit_error`, `api_error`); consider `code:"model_not_found"` on 404.
- **`openai.go:346` `/v1/completions`** — `firstString(req.Prompt)` silently serves only the first element of an array prompt, and the legal token-id-array form decodes to `""` → BOS-only prompt → unrelated 200 output. → Return 400 for unsupported prompt shapes.
- **`responses.go:164`** — Responses SSE emits a `[DONE]` terminator (a Chat Completions convention, not part of the Responses stream format), omits `response.output_item.added`/`content_part.added`/`output_text.done`/`output_item.done`, and discards `finish` so a length-truncated response still reports `"status":"completed"` instead of `"incomplete"`. → Drop `[DONE]`, add lifecycle events, map finish→status.
- **`responses.go:215`** — model-parsed `function_call` items emit empty `call_id` (the ID fallback used by `toAPICalls`/`toolUseBlock` is missing here), breaking `function_call_output` correlation. → Apply the same `"call_" + reqID()` fallback.
- **`openai.go:311`** — `stream:true` + `logprobs:true` silently drops logprobs (stream path discards them; `chatChunk` has no field). → Attach to the final chunk or reject the combination.
- **`responses.go:47`** — `responseEntry.model` is stored, never read: `previous_response_id` created under model A continues under model B unchecked (OpenAI 400s this). → Validate or drop the field.
- **`openai.go:620` `streamTokens`** — re-decodes the entire sequence every token: O(n²) in completion length (~8M redundant byte-copies at 4k tokens, material at 16–32k). → Incremental decode with a bounded tail for stop-matching. (Demos share the pattern, bounded by their max tokens.)
- **`main.go` shutdown** — after `Shutdown`'s 30s timeout, the checkpoint loop blocks on `lm.mu` behind a long generation; a second-SIGINT force-exit path would help. Also `writeErr`-style: no auth-token option exists at all, so `--allow-admin` on a non-loopback `-addr` exposes arbitrary-path model loading network-wide — worth a startup warning at minimum.
- **`cmd/prequant/main.go:34` + `internal/prequant/prequant.go:59`** — `Transcode` writes the output `.giw` in place (`os.Create(out)`); a crash mid-write leaves a corrupt cache with a fresh mtime, which `cacheFresh` then treats as valid → every later start fails until the user deletes it by hand; two concurrent transcodes interleave. → Write `out+".tmp"`, `os.Rename` after `selfCheck`.

### `decoder`

- **`model.go:254` `Close`** — never closes `w.st` (the safetensors mmap + per-shard fds from `openCheckpointMmap`); serve's load/unload cycles retain mappings until GC. → Close and nil it (the GGUF path already defer-Closes).
- **`kvsnapshot.go:237`** — restored int8-global layers alias the caller's snapshot blob (`r.i8()` wraps `r.data` via `unsafe.Slice`; scales/f32/rings copy). Because `unsafe.Slice` caps at `len`, the first append reallocates — so no cross-array corruption — but a post-restore `TruncateTo` shrink makes later appends write through into the caller's `[]byte`. Callers today pass freshly-read private buffers, so this is an API-contract landmine rather than a live bug. → Copy at restore, or reslice with a three-index expression; document that `LoadSession` takes ownership either way.
- **`kvsnapshot.go:60`** — snapshots drop `mropePos`/`mropeDelta`/`imgBlocks`: restored Qwen2.5-VL sessions decode at un-shifted RoPE positions (delta lost). → Persist the fields or refuse to snapshot when set (as done for `c.delta`). Also `Snapshot` signals "unsupported" by returning nil rather than a typed error — easy for callers to miss.
- **`spec_eagle.go:106` / `speculative.go:262`** — two-model `GenerateSpeculative` never updates `SpecStats.Evaluated` (always-0 `EvalAcceptanceRate`); ngram/grammar/EAGLE maintain it. → `evaluated := accepted; if !allAccept { evaluated++ }`.
- **Speculative paths + `SamplingParams.Logprobs`** — silently never populated and never rejected (`model.go:826` documents the field as per-token). → Reject in the spec validators or document per entry point.
- **`eagle.go:115` `LoadEagleHead`** — every post-open error path leaks the safetensors mmap (`st.Close()` only on success). → `defer` a conditional close.
- **`model.go:636`** — `Generate` allocates the full CPU KV cache (hundreds of MB at long maxTokens) even on the GPU-resident path that never touches it. → Allocate behind the `!useGPU` decision. Related: `RouterDrafter.RecordOutcome` is wired only in the grammar loop, not plain ngram — the router never learns there.
- **`forwardn.go:463`** — the sequential fallback silently ignores `treeRowPos`/`treeMask` (tree nodes attended as a linear chain); the tree entry point checks `specRollbackSafe` but not `canBatchN`. → Error when `cache.treeMask != nil && !canBatchN`.
- **`scratch.go:85`** — exact-fit growth (`if cap < n { make(n) }`) reallocates + zeroes multi-MB attention scratch every token in steady-state decode. → Grow with headroom (`max(2*cap, n)`).
- **`forward_deepseek.go:340` `mlaRope`** — allocates a loop-invariant de-interleave temp per cached key: O(layers × context) small allocs per decoded token on the V3-default interleave path. → Hoist to the caller or store latents pre-de-interleaved.
- **`forward_granite.go:63`** — `os.Getenv` (and `strconv.Atoi`) inside the per-layer/per-token loop; the package convention elsewhere hoists these to package vars. → Hoist.
- **`gguf.go:85` llama builder** — ignores `llama.attention.key_length` (explicit-head-dim models die with an opaque dims error; qwen3/glm/nemotron read it) and never consumes/synthesizes `rope_freqs.weight` — Llama-3.1/3.2 GGUFs silently lose llama3 long-context RoPE scaling (the safetensors path applies it; the llama4 builder documents the equivalent trade-off, this one doesn't). *(verify)* → Read `key_length`; detect `rope_freqs.weight` and at minimum warn.
- **`gptq.go:50` `parseQuantConfig`** — ignores `checkpoint_format`/`version`: GPTQ-v2 (true zero point, no +1) and AWQ-GEMV packing decode silently wrong (the implemented v1/GEMM math itself is correct). *(verify)* → Parse and reject unsupported variants loudly.
- **`weights.go:498` (+4 sibling sites)** — LM-head probe pattern `if head, herr := loadMat(...); herr == nil { … }` conflates "absent → tied" with "present but malformed → silently tied". → Distinguish not-found from real errors.
- **`registry.go:164` gemma4** — the only adapter with no validate* at all (and a "⚠️ verify against the real GGUF" note on `RMSAddOne`); zero dims/eps sail through to opaque later failures. → Add `validateGemma4`.
- **`features.go`** — `add(a.EmbedScale > 1, …)` vs the forward's `!= 0 && != 1` guard: an EmbedScale in (0,1) would dodge admission. No current arch hits it; the predicate should mirror the forward exactly (this file's stated purpose).
- **`sampler.go:347`** — top-p accumulates probabilities normalized over the full vocab after top-k/min-p cuts (llama.cpp-style), while HF renormalizes between filters — combined-filter results differ near the boundary. A convention choice, but undocumented. → Doc note (or renormalize). Also: default `Seed=0` means identical sampled streams unless callers vary it; and a LogitProcessor that masks the entire vocab yields NaN softmax → arbitrary token rather than an error.
- **`mamba2_chunked.go:119` (+ `deltanet_chunked.go:162`)** — chunked scans divide cumulative f32 decay products (`Pc[i]/Pc[m]`); with realistic per-step decay a chunk of ~64 underflows to 0 → 0/0 NaN into state. Unwired today (reference kernels for the prefill rewrite) — fix before wiring: segsum in log space, or renormalize per sub-block.
- **`kvcache.go:287`** — `Append`'s doc ("returns the position index just written") is off by one on the last layer (auto-advance runs first). No caller uses the return today. → Fix doc or capture before advancing.

### `gpu` (WebGPU)

- **`backend.go:149`** — `b.fallbacks++` outside the mutex in `MatmulBT` (every sibling increments under it); racy under the concurrent callers the mutex exists for. → Move under the lock.
- **`decoderunner.go:983` `runBatch`** — on a failed `MapAsync` status, later rows' successfully-mapped persistent staging buffers are never unmapped — the next `ForwardN` fails on a still-mapped buffer; one transient failure poisons batched verify until rebuild. → Unmap remaining rows on the error path.
- **`decodelayer.go:163` `attnBlockInto`** — returns `nil, err` after accumulating `frees`, leaking the scratch created so far (`mlpInto` returns `frees, err` correctly); several `pbuf, _ :=` CreateBufferInit errors ignored in the same family. → Match `mlpInto`'s convention.
- **`residency.go:467` nemotron branch** — attention-layer KV allocated unconditionally f32 while model-wide `kvI8`/`kvF16` flags are set later: kvI8 binds nil scale buffers → build panic; kvF16 silently oversizes. → Route through the same precision switch as the standard branch, or reject the combination.
- **`decodetoken_batched.go:114`** — dispatches the thin-M kernel without the `M ≤ 16` guard the standalone wrapper enforces (WGSL `array<i32,16>` accumulator). Test-only today. → Add the same rejection.
- **`gemv_w4a8.go:258` area** — W4A8 activation buffers are sized `padK(K)` (16) while the kernel reads `padK32(K)/16` vec4s and pad nibbles decode to −8, not 0: for `K % 32 == 16` the last read is OOB/garbage-contributing. All current dims are multiples of 32 — latent. → Size activations by the consuming weight's `kPad()`, or assert `K%32==0` at upload.
- **`gemv_w4a8.go:118`** — two near-duplicate f32→f16 converters with different rounding (`f32to16` truncation-ish/round-half-up + NaN→inf vs `mamba_f16.go`'s correct RNE+subnormals). → Consolidate on the RNE one.
- **`gpu.go:52` `Context`** — ~50 pipeline/shader triples with hand-maintained lifecycle; `Close` releases roughly a third (misses qkvFin*, gemvBias*, int8-KV trio, qkNorm*, mamba, W8A16, relu2, MoE, MLA, vision — whose shader modules aren't even retained), and no `GetBindGroupLayout` reference is ever released. Embedders that cycle Contexts (vision_register does) leak per cycle. → Central registry of compiled pipelines + release list; make Close a loop.

### `metal` / `cuda`

- **`metal/go.mod`** — missing `require` for `github.com/townsendmerino/goinfer` and `aikit` (the `replace` is inert without them; verified — only purego is required). Builds only inside this go.work; fails standalone/published. `cuda/go.mod` does it right. → Add requires + go.sum.
- **`cuda/glue.cu:202` argmax reduce** — tie-break keeps the lower *thread*, not the lower *index*; CPU scan (and the Metal twin, which merges `cv==v && ci<idx`) return first-max. `ForwardArgmax` is documented argmax-exact and feeds the greedy fast path. → Carry (v, idx), break ties on index in both stages.
- **`cuda/kernels.go:68` `f32tof16`** — comment says round-to-nearest-even; code truncates and flushes subnormal group scales to zero (tiny int4 groups dequantize to exactly 0). Metal's converter is correct — backends diverge per checkpoint. → Share one correct helper.
- **`cuda/resident.go:344` `Close`** — frees buffers meticulously but never `Unload`s the four JIT'd modules; per load/unload cycle leak in the multi-model scenario Close's own comment targets. → Retain handles, unload in Close.
- **`cuda/driver.go`** — the `driver` interface, `stubDriver`, `dim3`, `errCUDANotWired` are dead scaffolding; the resident path calls gocudrv directly. → Delete or actually route through it.
- **`metal/metal.go:260` `Buffer.n`** — means bytes in some constructors, elements in others; `Floats()/U16s()` build `unsafe.Slice(ptr, b.n)` assuming elements, so f16-KV `U16s()` views claim 2× the real capacity (4× for `Floats()` on byte-sized buffers) — any full-slice write corrupts adjacent memory. Current callers touch valid prefixes only. → Store byte length + element kind, or per-type view methods.
- **`metal/kernels.go:131` SA-family row indexing** — derives the row from the *current group's* `threads_per_threadgroup`, which Metal shrinks for the trailing partial group: any row count not a multiple of 8 double-computes some rows (`_wacc` variants double-accumulate) and drops the top ones; `nTiles := V/8` truncates similarly. All shipped shapes are multiples of 8 — latent, unasserted. → Use `dispatch_threads_per_threadgroup`/a uniform, add a `row >= N` guard, assert shapes at build.
- **`metal/prefill.go:366`** — prefill binds the model-wide sliding window to every layer while decode binds per-layer windows; an admitted arch with mixed local/global layers (none today — Gemma declines on other features) would prefill global layers windowed. → Decline mixed-window archs in `prefillFeatures` or pass per-layer windows.
- **`metal/model.go:469` / cuda equivalent** — use-after-Close sends on a nil channel: permanent goroutine hang instead of an error. → Closed-flag check returning an error.

### `tokenizer` / `chat` / `constrain` / `multimodal`

- **`tokenizer/gguf.go:154`** — chat-turn-marker resolution iterates a `map[string]*int` literal, so a vocab containing two marker styles resolves nondeterministically per process. → Ordered slice, first-match-wins.
- **`tokenizer/gguf.go:261`** — the "GPT-2 takes one digit" knob is wrong: real GPT-2 groups unbounded digit runs with case-sensitive contractions; a genuine GPT-2 checkpoint mis-tokenizes numbers against the package's exact-id bar, and *unknown* `tokenizer.ggml.pre` values silently get the same defaults. *(verify)* → Add a gpt-2 knob set; refuse unknown `pre` values loudly.
- **`tokenizer/sentencepiece.go:567` `DecodePiece`** — routes through `Decode`, which strips the leading space per call: per-token streaming on prependSpace (Llama-2/Mistral GGUF) models produces "helloworld". Doc advertises it "for token streaming". → Bypass the whole-sequence strip; document the UTF-8 caveat instead.
- **`constrain/tool_grammar.go:34`** — `paramSchema` spliced into a JSON template via `fmt.Sprintf` and `mustMap` discards the Unmarshal error: invalid input surfaces as a misleading unrelated error; a crafted valid fragment can inject members into the wrapper object (duplicate keys — last wins — can even override the `"name":{"const":…}` pin). → `json.Valid` check; build via `map[string]any` + Marshal.
- **`chat/gemma4_tools.go:216`** — call-body parsing splits on commas without brace/bracket depth, mangling the nested values the *renderer itself emits* (`cfg:{"a":1,"b":2}` → key `cfg` = `{"a":1`). Round-tripping tool history corrupts nested args. → Track depth in `splitGemmaPairs`.
- **`chat/tools.go:139`** — ChatML tool-history rendering diverges from the Qwen/Hermes template: per-turn `<tool_response>` user blocks instead of merged consecutive results, and an extra newline on empty-content assistant call turns. Goldens deliberately don't byte-match tool history — which is exactly where the drift is. *(verify)* → Match the reference template; byte-gate one tool-history golden.
- **`multimodal/projector.go:60`** — divide-by-zero on absent `patch_size`/`mm_tokens_per_image`; transpose indexes `projW` without a length check: hostile/truncated VL configs panic instead of erroring (the tokenizer loaders validate; this one should too). → Validate dims and tensor lengths.
- **`multimodal/qwen_preprocess.go:152`** — extreme aspect ratios floor a grid dimension to 0 → empty pixel_values with no error (HF raises at ratio > 200); and decoded image dims aren't capped before the h*w*3 float alloc (a small crafted PNG header can demand GBs). → Replicate the ratio guard; cap dims before allocating.
- **`internal/prequant/ggufmeta.go:73`** — `need` uses the overflow-prone `c.off+n > len(c.b)` form that `internal/giw/bundle.go` explicitly documents avoiding; a crafted u64 string length panics `metadataPrefixLen` (reachable from `Transcode`/`EnsureCachedGIW`). Also `skipValue` treats negative array counts as 0 and recurses unboundedly on nested arrays. → Use the `n > len(c.b)-c.off` form; bound depth.
- **`internal/giw/bundle.go:105` `ReadTokFile`** — allocates the tok buffer from an untrusted u32 before reading (4 GiB on a corrupt length). → Clamp against file size first.
- **`decoder/serialize.go:333` `giwWriter`** — every length cast to u32 unchecked; a ≥2^32-element array wraps silently and the CRC covers the wrong bytes (confusing "truncated body" far from the cause). Same in kvsnapshot's `ints`. → Error from `writeBundle` past MaxUint32.

---

## 5. Nits & polish

- **Doc rot (user-facing):** `decoder.Load`/`Model` docs still say "Gemma 3 snapshot … CPU backend … the only one wired" (`model.go:21,84-86` — verified); `config.go`'s opening doc says the struct "captures the Gemma 3 architecture constants"; `gguf.go:25` file header lists 5 architectures, `doc.go` similar; the unsupported-arch error at `gguf.go:61` omits `llama4` which the switch dispatches. For a library, exported-API docs lagging this far hurts adopters first. Consider deriving the arch list from the registry so it can't drift.
- **Library prints to stdout:** `gguf.go:1019` `fmt.Println(beErr)` (every sibling uses stderr) corrupts piped output; `model.go:805` decode timing prints to stdout (env-gated, still surprising in a server). `LoadGGUFBytes` also silently ignores `opts.LoRA`/`StreamWeights` unlike `Load`.
- **Error joining:** `gguf.go:1315` (+4 sites) `"%v / %v / %v"` renders `<nil>` noise and breaks `errors.Is/As`; prefer `errors.Join` wrapped with layer context.
- **Dead code:** `Config.AttnScaleMul` (`config.go:84`) — declared, documented, never referenced (the fold-into-AttnScale is still a registry TODO). Delete or wire.
- **`decoder/features.go`** is currently not gofmt-clean (the in-flight `FeatMoEGatedShared` edit) — `gofmt -w` before commit.
- **Version-directive drift:** root `go.mod` says `go 1.26.3`, `go.work` 1.26.5, `metal` 1.26.5, `gpu`/`cuda` 1.26.3 — harmless but worth aligning.
- **`decoder/attention.go:169`** — `attendQuery`'s doc claims it's shared with `causalAttention`/`forwardN`; callers are now only the four own-path families, its ring/int8 branches are unreachable, and it already lacks the image-block `attendHi` bound the SIMD twin has. Update the doc; delete the dead branches or pin `attendQuery ≡ attendBatchedHeads` with a test.
- **`decoder/forwardn.go:330`** — `attendBatchedHeads` shadows its load-bearing `base` (absolute-position) parameter with three byte-offset locals also named `base`; correct but hostile to the reader in exactly the function where the distinction matters. Rename the offsets.
- **`forward_qwen35.go:81` (+granite/nemotron)** — dereference `cache.scr` without the nil guard deepseek/llama4 have; a cache built via `NewKVCache` directly panics. Add the two-line guard.
- **Package-level test seams** (`moeSelTrace`, `mlaForceNaive`, `ssmForceF32`, `g4trace*`, env knobs in `gpu` constructors) make the packages non-reentrant while set — fine under the serial-test convention, worth a doc note in each file.
- **`chat.Render`** silently drops `Turn.ToolCalls`/tool fields for templates without tool support — better an error (or a documented loud fallback) than dropped data.
- **`constrain`'s JSON grammar** deliberately has no nesting-depth limit (fine under a token budget) — document it for consumers.
- **`spec_ngram.go:154` + `spec_sample.go:16`** — stale comments describing the pre-threading design of sampled-mode penalties (code now threads them).

---

## 6. Architecture notes

**What's working well.** The module topology is right: heavy/optional GPU deps isolated in `gpu/`, `metal/`, `cuda/` submodules so the root stays cgo-free and dependency-light, stitched by `go.work` (with `cuda` deliberately opt-in). The edge-translation design in `cmd/serve` (three API dialects → one internal request vocabulary → shared `prepare`/`drive`/`streamTokens`) keeps per-dialect duplication low. `decoder`'s two-tier forward (descriptor-driven shared core + six own-path families) splits at a principled boundary, and the features.go admission matrix is the standout idea — it just needs to *cover more axes* (context caps, expert counts, head dims, per-layer windows — every C3/M20/M22-class bug is a missing taxonomy dimension). Numerics discipline (f64 accumulation for cross-path bit-exactness, documented at the site) is exemplary.

**Recommendations, highest leverage first:**

1. **Make representability explicit.** One `canSerialize(arch)` / `canSnapshot(cache)` predicate consulted by `SerializeWeightsTo`, `StreamTranscodeGGUF`, `Snapshot`, and prequant turns the whole C2 class into load-time errors, and a per-registered-arch round-trip test keeps it honest forever.
2. **Backends should own and advertise their limits.** Add `ctxCap()`/shape constraints to the resident interface; have `generateInto` clamp or fall back, and have `BuildResident`-equivalents *decline* (the design's own verb) anything outside kernel limits (nE > 256, hd > 128, mixed windows) rather than clamp.
3. **Consolidate the own-path dispatch lists.** The `a.gemma4 == nil && a.qwen35 == nil && …` chains repeat across ~6 sites (`canBatchN`, `NewCache` ×2, `ForwardCapture`, `decodeRunnerEligible`, dispatch); one `ownForward` marker (plus per-arch capability bits) collapses them and removes the "new family forgot site #5" failure mode. Same story for the ~15-field boilerplate repeated across 21 registry adapters — a base-descriptor builder would halve `registry.go` and single-source conventions like TiedLMHead finalization.
4. **Split the two megafiles along their existing seams.** `gguf.go` (76KB) is three files (13 per-family config builders / `buildWeightsFromGGUF`'s six inline family branches / permutation math) — `gguf_qwen35.go` already proves the per-family shape, and the sink-guard omissions would have been structurally visible. `weights.go` (71KB) similarly splits by family; the `load→quantizeWM` idiom repeated ~30× wants a helper.
5. **Give the serve drive layer a fake-model seam.** The findings that matter most there (M1, M2, M6) all live in `drive`/`streamTokens`, which today can only be exercised against real GGUFs (env-gated). A scripted-token-stream decoder interface would make error propagation, stop holdback, and cancellation hermetically testable — and the chaos test should add `-session-dir` + `-kv-idle-demote` + admin churn to hit M4/M5's seams.
6. **Unify GPU kernel semantics with shared golden tests.** Three hand-mirrored kernel suites (WGSL/MSL/CUDA) invite drift — the argmax tie-break and f16-conversion divergences are the observed cases. A tiny backend-agnostic semantics suite (router selection, argmax ties, quant rounding, partial-RoPE tail) run against all three would catch the class. In `gpu/`, a pipeline registry (the `buildCompute` helper generalized) makes `Close` a loop and ends the ensure*/Close drift.
7. **Introduce a content/control boundary in the text stack.** M25 needs `Render` to return segments (or `Encode` to take them); this is also the natural place to fix per-segment BPE cost caps (M28c).
8. **Seed the fuzzers you already built.** `FuzzGGUFConfig` (7 unseeded archs), constrain's satisfiability fuzzer (quoted property names, unknown keywords), `FuzzLoadSerializedWeights` (group=0 kind flips), and a kvsnapshot fuzzer would have caught M16, M17, M27 mechanically. The harness contracts are already exactly right.

---

## 7. Suggested order of attack

| Order | Items | Rationale |
|---|---|---|
| 1 | C3 + M20 (cap enforcement), M4, M14, M21 | Small diffs, crash/corruption class, no design work |
| 1b | C6 (yarn-mscale all-layer derive + profile↔derivation cross-check test) | Small taxonomy fix; closes a source-of-truth gap + a silent-wrong-admission risk. Independent of the rest; (verify) the Mellum decline direction against a real checkpoint |
| 2 | C2 (canSerialize + TiedLMHead), M17 | Closes every silent-garbage `.giw` path; mostly guards |
| 3 | C1, M10, M11 | One coherent "rewind correctness" change in kvcache/session |
| 4 | C4, C5, M23, M24 | GPU backend correctness + lifecycle; CUDA tail port is a template for C4 |
| 5 | M1, M2, M3, M5, M6 | Serving correctness/robustness sweep; add the fake-model seam while there |
| 6 | M8, M9, M12, M13, M15 | Generation API contracts; mostly guard-and-reject |
| 7 | M16, M25–M28 | Untrusted-input hardening (GGUF metadata, schemas, tokenizer, injection) |
| 8 | Minors by area, then §5 | Batch by file to amortize context |

*Review produced by an automated multi-pass code review (8 parallel per-subsystem deep reviews + independent verification of every Critical/Major finding against the source). Items tagged (verify) depend on external checkpoint/spec details — confirm before fixing. Everything else was re-verified line-by-line in this tree.*
