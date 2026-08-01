# goinfer `decoder/` — Engineering Audit

**Date:** 2026-07-25 · **Scope:** `decoder/` (~35k LOC) and how it is consumed
**Categories:** bugs & correctness, performance, architecture & maintainability

**Build/tooling status.** `go.mod`/`go.work` declare Go 1.26.5; only 1.24.7/1.25.1 were installed and the module proxy was blocked. A working build was obtained by disabling `go.work`, pinning the go directive to 1.25, adding local `replace` directives for `aikit`/`x/text`/`x/sys` (stubbing `x/text/unicode/norm`, used only by `aikit/embed`), patching `constrain/reflect.go`'s Go-1.26 `reflect.Type.Fields()` to `NumField()`, and building `GOARCH=riscv64` (the `aikit/linalg` `.s` files weren't available). Result: `go build ./decoder/...` clean, `go vet ./decoder/...` **clean**, `gofmt -l decoder/` **clean**. `golangci-lint` with `staticcheck,govet,ineffassign,unused,gosec,nilerr,makezero` on non-test sources finds no correctness defects — only `gosec` conversion noise, one `QF1011`, and 8 genuinely-unused functions. Everything below comes from reading the source.

**Re `CODE_REVIEW.md` (2026-07-18).** Verified as **shipped since**: C1's ring-exactness flag (`kvcache.go:420` now returns `exact`, `session.go:73` consumes it), C2 (`canSerialize` at `serialize.go:79`, `TiedLMHead` at `serialize.go:224`, `Snapshot` refusals at `kvsnapshot.go:66`), C6 (all-layer `yarnMscale` loop, `features.go:67`), M10 (`Session.reconcile`), M11 (`c.stride[]`), M14 (metal CPU fallback), M17. Verified **still open**: M8, M9, M12, M13, M15, M19a/b, `scratch.go` exact-fit growth, `model.go` cache-on-GPU-path. Those are not repeated as-is; where still open, the reachability analysis that decides severity is added.

---

## Critical & High

### 1. `specRollbackSafe`'s ring exemption is keyed on the wrong predicate — C1 is still reachable on three of the four speculative loops

**Critical** — `decoder/forwardn.go:46`

```go
if m.resident == nil && a.SlidingWindow > 0 {
    return false
}
```

The exemption assumes that when `m.resident != nil` the speculative loop runs on the resident (positional, eviction-free) cache. That is false for three of the four loops:

- `genGrammarInto` has **no resident branch at all** — `tc := cache; if tc == nil { tc = target.NewCache(...) }` (`spec_grammar.go:231`), then `tc.TruncateTo(tpos)` (`spec_grammar.go:331`).
- Both EAGLE loops likewise always use `tc := m.NewCache(...)` (`spec_eagle.go:45`, `:202`) with `tc.TruncateTo(C)` / `TruncateTo(C+acc)` (`:135`, `:289`).
- `genNgramInto`'s resident test is `resident := cache == nil && target.resident != nil && target.DecodeRunnerEligible()` (`spec_ngram.go:238`) — so any **Session**-backed call (`cache != nil`) takes the staged path while `specRollbackSafe` already waved it through.

And `m.NewCache` *does* enable rings for these families: `if a.gemma4 == nil && a.qwen35 == nil && a.granite == nil && a.nemotron == nil { c.enableRings(a.SlidingWindow, a.isGlobalLayer) }` (`model.go:289`). Every one of those five `TruncateTo` call sites discards the `exact` bool that C1's fix added.

**Failure scenario.** Mellum2 (`SlidingWindow > 0`, `layerIsGlobal` from `layer_types` 3:1, and resident-eligible — `decodeRunnerEligible` explicitly admits sliding-window via "Lever C6" and Mellum's local/global inv-freq tables are the same length so `ropeResidentCompatible()` passes) loaded with `--backend cuda|metal|webgpu`. Call `GenerateGrammarSpeculative`; generate past the window so the sliding layers' rings wrap (`r.count > r.w`); a round rejects ≥2 draft tokens → `TruncateTo(base+1+accepted)` → `ring.truncate` returns `false`, nobody reads it, and the window slots still physically hold the dropped positions. Attention then reads them as history. Silently wrong tokens, no error, and the "grammar-spec output is bit-exact with constrained `Generate`" guarantee is broken. Same for Gemma-3 and Phi-3-mini-4k.

**Fix.** Make the guard reflect the path actually taken: replace the `m.resident == nil` condition with an explicit parameter (e.g. `specRollbackSafe(staged bool)`) computed the same way each loop computes `resident`, and additionally consume the flag — `if !tc.TruncateTo(keep) { /* cold re-prefill or abort the round */ }` at all five sites. The grammar/EAGLE loops should simply reject `a.SlidingWindow > 0` unconditionally, since they have no resident path.

---

### 2. `Session.Reset()` and prefix rewind never reset Mamba-2 / Gated-DeltaNet recurrent state — cross-conversation state leak in `serve`

**Critical** — `decoder/kvcache.go:420` (via `decoder/session.go:64`)

`TruncateTo` handles MLA latents, rings, and global f32/int8 stores. It never touches `c.delta` or `c.mamba`. Those slices are written in exactly one place — `NewCache` (`model.go:299`, `:310`, `:322`) — and never reset anywhere:

```go
func (s *Session) Reset() {
	s.cache.TruncateTo(0)
	s.tokens = s.tokens[:0]
}
```

`deltaState`'s own doc says it is "NOT position-truncatable (why qwen3_5_moe falls back from prefix reuse / speculative)" — but only the *speculative* fallback exists (`specRollbackSafe`). Nothing gates `Session`.

**Failure scenario (two, both live).**
(a) `cmd/serve` with the default `-kv-sessions 4` (`main.go:264`). `sessionLRU.fresh()` evicts the coldest session and hands it to a **different conversation**: `s := l.order[len(l.order)-1]; s.Reset()` (`cmd/serve/sessions.go:186-187`). On Granite-4.0-H / Nemotron-H / Qwen3.5-MoE, conversation B's prefill then runs `mamba2Step` / `gatedDeltaNetStep` on top of conversation A's `st.ssm` and `st.convWin`. User B gets output conditioned on user A's tokens. (The tiered path can't save this either: `Snapshot` returns `nil` for recurrent caches, so `tierOut` fails and it falls through to the same `Reset`.)
(b) Within one conversation, `rewindForReuse` (`session.go:75`) rewinds KV to `matched` but leaves the recurrent state at the *previous* full position, then re-prefills `prompt[matched:]` on top of it.

**Fix.** Give `KVCache` a `resetRecurrent()` that re-news every non-nil `delta[l]`/`mamba[l]`, call it from `TruncateTo` when `pos == 0`; and make `rewindForReuse` return `0` (forcing a cold prefill) whenever `c.delta != nil || len(c.mamba) > 0`, mirroring the inexact-ring path it already has. A `canRewind(cache)` predicate next to `canSerialize`/`canSnapshot` is the natural home.

---

### 3. Batched forward skips the hidden-state capture seam on MoE layers → EAGLE panics on every MoE target

**High** — `decoder/forwardn.go:240`

The MoE branch of `runLayersFromEmbedN` ends with a bare `continue` (`forwardn.go:240`) that jumps over the capture block at `forwardn.go:273-279`:

```go
			}
			continue          // ← skips the captureLayers block below
		}
		...
		if cache.captureLayers != nil {
			for ci, cl := range cache.captureLayers { if cl == l { cache.captured[ci] = append(...) } }
		}
```

For Mixtral / Mellum / Qwen2-MoE every layer has `lw.Experts != nil`, so `cache.captured[ci]` is never written and stays `nil`.

**Failure scenario.** `GenerateEagleSpeculative` on any MoE target (`canBatchN` explicitly permits MoE; `specRollbackSafe` permits it too). `captureN(prompt)` runs the batched path, `captured[ci]` stays `nil`, then `fuseAt` does `row := tc.captured[ci][i*hidden : (i+1)*hidden]` (`spec_eagle.go:49`, `:208`) → `panic: slice bounds out of range [:4096] with capacity 0`, inside the generation goroutine with no `recover` → process death. Note the sequential `runLayersFromEmbed` captures correctly (`model.go:481`), so `ForwardCapture` works and only the batched path is broken — which is why no test caught it.

**Fix.** Move the capture block above the MoE `continue`, or restructure the branch so the residual-add/capture tail is shared. Add a `TestForwardCaptureN_MoE` asserting `len(captured[ci]) == K*hidden`.

---

### 4. EAGLE entry points validate neither `capLayers` nor batchability

**High** — `decoder/spec_eagle.go:49` and `:208`

`ForwardCapture` bounds-checks layer indices (`model.go:548-552`) and rejects own-forward families (`model.go:545`). `GenerateEagleSpeculative`/`Tree` validate only `head != nil`, greedy, `specRollbackSafe`, `head.Hidden()`, and `len(prompt) >= 2` — then call `forwardN` directly.

**Failure scenario (two).**
(a) `capLayers = []int{2, 40, 60}` on a 28-layer model: `captured[1]`/`captured[2]` stay `nil` → same out-of-range panic as #3.
(b) A non-batchable target — DeepSeek (`a.mla != nil`) passes `specRollbackSafe` but fails `canBatchN` (`forwardn.go:22`). `forwardN` then falls back to the per-token loop at `forwardn.go:474-483`, and `m.forward` writes `cache.captured[ci] = append(cache.captured[ci][:0], h...)` — one `hidden`-wide row, **overwritten each iteration**. After a 500-token prompt, `captured[ci]` holds only the last position, so `fuseAt(0)` returns the *wrong* feature and `fuseAt(1)` panics on `[hidden:2*hidden]` of a `hidden`-length slice.

**Fix.** In both entry points: bounds-check `capLayers` against `[0, NumLayers)` (and require `len(capLayers) == 3`, which `Fuse`'s `3*hidden` matmul assumes), and reject targets where `!m.canBatchN(2)`. Better: route capture through a `forwardCaptureN` that errors instead of silently degrading when it takes the sequential fallback.

---

### 5. `forwardN` ignores an active compute-time LoRA — speculative verify silently uses base weights

**High** — `decoder/forwardn.go:474`

`prefillLogits` correctly forces the sequential path (`if cache.lora != nil || !m.canBatchN(len(prompt))`, `forwardn.go:511`). `forwardN` guards only `if !m.canBatchN(K)`, and `runLayersFromEmbedN` contains no reference to `lora` at all. `validateNgramSpec`/`validateGrammarSpec` don't reject adapters either.

**Failure scenario.** `s := m.NewSession(n); s.UseAdapter("sql")` (sets `s.cache.lora`), then `s.GenerateNgramSpeculative(...)`. The prompt KV is prefilled **with** the adapter (sequential path), every verify block's q/k/v/o/gate/up/down is projected **without** it, and the resulting K/V is committed to the same cache. Output is a silent hybrid of adapter and base — while the API documents the path as lossless.

**Fix.** Add `cache.lora != nil` to `forwardN`'s fallback condition (the sequential path applies LoRA correctly), or reject `cache.lora != nil` in the three spec validators.

---

### 6. Greedy speculative paths drop `RepeatPenalty` / `PresencePenalty` / `FrequencyPenalty` / `LogitBias`

**High** — `decoder/spec_ngram.go:373` (also `:402`, `spec_grammar.go:293`/`:314`, `spec_eagle.go:129`/`:286`, `speculative.go`)

`sampled := sp.Temperature > 0`; the `Sampler` is only constructed inside `if sampled` (`spec_ngram.go:217`). Greedy verification is raw `ti := argmax(logitsN[i])`. Meanwhile plain `Generate` applies bias+penalties before argmax (`sampler.go:119-127`). No validator rejects the combination: `validateNgramSpec` (`spec_ngram.go:164`) checks only nil drafter, `LogitProcessor`, and rollback safety.

The *sampled* path handles this correctly via `distVectorHist` (`spec_sample.go:51`) — so the machinery exists; greedy just doesn't use it.

**Failure scenario.** `sp = SamplingParams{Temperature: 0, RepeatPenalty: 1.15}`. `Generate` divides repeated tokens' positive logits by 1.15; `GenerateNgramSpeculative` doesn't. The streams diverge at the first repeat, violating the documented "token-identical to plain greedy". Silently — the API returns no error.

**Fix.** Either reject `sampler.needsHistory()` in greedy mode in all three validators (one line, matches the existing `LogitProcessor` rejection), or thread `ph` through the greedy branch the way the sampled branch already does and argmax over `distVectorHist`.

---

### 7. `Model.AttnScale()` ignores the registry-resolved scale — resident backends mis-scale every attention logit

**High** — `decoder/residency.go:292`

```go
func (m *Model) AttnScale() float32 {
	return float32(1.0 / math.Sqrt(float64(m.w.arch.HeadDim)))
}
```

`arch.AttnScale` is the authoritative value (`Pow(QueryPreAttnScalar, -0.5)` for Gemma-3, `1.0` for Gemma-4, `Pow(hd,-0.5)` elsewhere) and is what the CPU path uses (`forwardn.go:324`, `attention.go:200`). The three resident backends all consume the derived method instead: `metal/model.go:371`, `cuda/backend.go:296`, `gpu/residency.go:735`.

**Failure scenario.** Gemma-3-27B from safetensors: `query_pre_attn_scalar = 168`, `head_dim = 128`, so `arch.AttnScale = 168^-0.5 ≈ 0.0772` but `AttnScale() = 128^-0.5 ≈ 0.0884`. Every pre-softmax logit is scaled ~14.5% too large on the resident path — a sharper distribution and a different argmax at near-ties, with no CPU-vs-GPU parity failure on the 1B/4B/12B fixtures where `QueryPreAttnScalar == HeadDim`. (Related and separate: the GGUF builder hardcodes `cfg.QueryPreAttnScalar = float64(cfg.HeadDim)` at `decoder/gguf.go:221`, so a 27B GGUF is wrong on *both* paths; llama.cpp special-cases exactly this geometry.)

**Fix.** `return float32(m.w.arch.AttnScale)`. Add a test asserting `m.AttnScale() == float32(arch.AttnScale)` for every registered arch — that also pins Gemma-4's `1.0`.

---

### 8. `Model.pager` / `Model.layerPager` are model-scoped, unsynchronised, and mutated per layer — the documented concurrency contract is unsound

**High** — `decoder/layerpaging.go:96` and `decoder/moepaging.go:26`

`Model`'s doc promises "per-sequence state (the KV cache) is owned by each Generate call, so distinct sequences can run concurrently" (`model.go:22-24`). But `m.layerPager` and `m.pager` live on `Model`, and `expertPager`'s own comment says "**Not goroutine-safe** — one decode stream touches it at a time, like the KV cache" — which is exactly wrong about its scope. `layerPager.enterLayer` mutates `p.state[t]` (a `[]bool`) and `p.prefetches`/`p.evictions` on every layer of every token, and `finishLayers` is `defer`red from both `runLayersFromEmbed` (`model.go:434`) and `runLayersFromEmbedN` (`forwardn.go:131`).

**Failure scenario.** Two concurrent `Generate` calls on a model loaded with `Options.StreamWeights` (the paths that build these pagers). Genuine unsynchronised concurrent writes to `p.state[t]` and `p.prefetches` — the race detector fires, and functionally stream A's `finishLayers()` issues `madvise(DONTNEED)` across every span stream B just prefetched, so B re-faults its whole layer window from disk each token. On the measured 35B-A3B that's the ~24 ms/token cold-miss cost paid on nearly every layer. For `expertPager` the shared `mmap.SpanCache` LRU (a map + list) is mutated from both streams via `pager.touch` in `moeMLP` (`mlp.go:109`) — a concurrent map write is runtime-fatal, not just racy.

**Fix.** Either move both pagers behind a mutex (they're I/O-hint-bound, so the lock is free relative to `madvise`), or make residency per-stream by hanging the pager state off `KVCache` alongside `scr`. Whichever you choose, correct the `Model` doc — the same applies to `m.resident` (M9, still open at `model.go:678`: the resident decode path drives one shared positional device KV with no in-flight guard).

---

### 9. Steady-state decode reallocates and zeroes context-sized scratch **every token** (exact-fit growth)

**High (performance)** — `decoder/scratch.go:90` (also `:76`, `:100`)

```go
	if cap(s.akh) < nKeys*hd {
		s.akh = make([]float32, nKeys*hd)
		s.avt = make([]float32, nKeys*hd)
	}
```

`nKeys` grows by exactly 1 per decoded token, and `make` returns `cap == len`, so the test is true on **every** token forever. `attnBatchBufs` is now on the main decode path for all families — `causalAttention` routes single-token decode through `attendBatchedHeads` (`attention.go:135`, `:147`, `:153`).

**Measurable cost.** Llama-3-8B (`hd=128`) at 4096 context: `2 × 4097 × 128 × 4 B ≈ 4.2 MB` malloc'd **and zeroed** per token, plus the GC pressure of a 2 MB-class object pair per token. On the int8-KV path it is far worse: `causalAttention`'s `kvI8` branch calls `scr.localBufs(cache.storedRows(layer, kvDim) * kvDim)` (`attention.go:145`), i.e. `2 × 4097 × 1024 × 4 B ≈ 33.6 MB` per token at `kvDim=1024`. Windowed models stop growing at `W`; full-attention models (dense Llama/Qwen/Mistral-v0.2+) pay forever.

**Fix.** Grow with headroom in all three helpers: `if cap(s.akh) < n { n2 := max(2*cap(s.akh), n); s.akh = make([]float32, n2)[:n] }` (and return `s.akh[:n]`). That turns O(tokens) reallocations into O(log tokens).

---

### 10. `Generate` allocates the entire CPU KV cache on the GPU-resident path, where it is never touched

**High (performance)** — `decoder/model.go:641`

```go
cache := m.NewCache(len(prompt) + maxTokens)
go func() { ... m.generateInto(ctx, out, g, cache, prompt, 0, maxTokens, sp, nil) }()
```

`useGPU` is decided *inside* `generateInto` (`model.go:678`), long after the cache exists. `NewKVCache` pre-sizes `make([]float32, 0, capHint*kvDim)` for keys **and** values on **every** layer (`kvcache.go:163-164`).

**Measurable cost.** Llama-3-8B, `len(prompt)+maxTokens = 4096`, 32 layers, `kvDim = 8*128 = 1024`: `32 × 4096 × 1024 × 4 B × 2 ≈ 1.07 GB` allocated and zero-filled per `Generate` call, entirely unused when `m.resident != nil`. That is a multi-hundred-millisecond zeroing stall on TTFT plus 1 GB of RSS per in-flight request.

**Fix.** Hoist the `useGPU` decision (it depends only on `m.resident`, `prefillFrom`, `commit`) above the allocation and pass `nil`; or lazily construct the cache in `generateInto`'s `else` branch.

---

## Medium

### 11. `out <- next` is an unguarded send — abandons the goroutine and permanently wedges the Session

**Medium** — `decoder/model.go:786`

The ctx check is a separate non-blocking `select` at the top of the loop (`model.go:741-746`); the send itself is bare. Every speculative loop does this correctly (`spec_ngram.go:298-303`, `spec_eagle.go:80-85`, `spec_grammar.go:251-256`), so this is drift, not policy.

**Failure scenario.** A consumer that `break`s out of `for tok := range ch` — even one that then cancels ctx, the documented stop mechanism — leaves the goroutine blocked on `out <- next` forever, pinning the KV cache (up to the 1 GB of #10). For `Session.Generate` the goroutine never reaches `s.reconcile(seq)` (`session.go:124`), so `s.tokens` stays stale and `s.cache` stays locked at a partial position: the session is wedged for the process's lifetime and every later `rewindForReuse` on it prefixes against a lie.

**Fix.** `select { case <-ctx.Done(): g.err = ctx.Err(); return; case out <- next: }` in `generateInto`, `GenerateVL`, and `GenerateQwenVL`.

### 12. MoE FFN allocates per-expert scratch on every expert evaluation

**Medium (performance)** — `decoder/mlp.go:245-246`

```go
func swiGLUExpert(ex *expertWeights, h, dst []float32, inter int, be Backend) {
	gate := make([]float32, inter)
	up := make([]float32, inter)
```

The dense path was converted to `scr.gate`/`scr.up` reuse (`mlp.go:331`); the MoE path was not. Per `moeMLP` call there is also `logits` (`mlp.go:90`), `out` and `expOut` (`:117-118`), plus `routeExperts`' `scores`/`sel`/`wts`, `groupLimit`'s three slices, and `topK`'s `idx`/`val`/`used` — and `mlp()` then does a redundant `copy(out, g)` (`mlp.go:57`).

**Measurable cost.** DeepSeek-V3 class (`inter = 2048`, `TopK = 8`, 61 layers): `8 experts × 2 slices × 2048 × 4 B = 131 KB` per layer per token → **~8 MB per decoded token** of pure churn, plus a second shared-expert pair. In the batched prefill this is multiplied by `K` — `runLayersFromEmbedN` calls `moeMLP` per row (`forwardn.go:228`), so a 2048-token prompt allocates ~16 GB cumulative through this one function.

**Fix.** Add `moeGate`/`moeUp`/`moeOut`/`moeExpOut`/`moeLogits`/`moeScores` to `decodeScratch`, pass `scr` into `swiGLUExpert`/`moeMLP`/`routeExperts`, and have `moeMLP` write into the caller's `out` (dropping the `copy`). For the batched path allocate them once outside the layer loop.

### 13. `forwardN` allocates ~16 buffers per call, including two `maxKeys*hd` arrays — once per speculative round

**Medium (performance)** — `decoder/forwardn.go:97-121`

`runLayersFromEmbedN` allocates `norm,q,k,v,ctx,att,gate,up,mlpOut,aqh,akh,avt,ascores,ach` (+ `alk,alv` for ring/int8 caches) on entry, then `lmHeadN` allocates `M*VocabSize` (`forwardn.go:451`). Fine for a one-shot prefill; but `forwardN` is the per-round verify of every speculative loop.

**Measurable cost.** Qwen2.5-7B (`hd=128`, vocab 152 064) at 4096 context, K=5: `akh + avt = 2 × 4101 × 128 × 4 ≈ 4.2 MB`, `ascores = 5 × 4101 × 4 ≈ 82 KB`, `logits = 5 × 152064 × 4 ≈ 3.0 MB` — **~7.5 MB allocated and zeroed per speculative round**, i.e. per ~2–3 emitted tokens. That is a large fraction of the amortization speculation is supposed to buy.

**Fix.** Hang a `batchScratch` off `KVCache` next to `scr`, keyed by `(K, maxKeys)` with the same headroom growth as #9. `lmHeadN` should write into a reused vocab-sized buffer.

### 14. Sampled decode sorts the full vocabulary and allocates ~5 MB per token

**Medium (performance)** — `decoder/sampler.go:325-329`

```go
	ips := make([]indexedProb, len(probs))
	for i, p := range probs { ips[i] = indexedProb{id: i, p: p} }
	sort.Slice(ips, func(a, b int) bool { return ips[a].p > ips[b].p })
	if topK > 0 && topK < len(ips) { ips = ips[:topK] }
```

Plus `softmaxStable`'s `make([]float64, len(logits))` (`sampler.go:303`). `applyPenaltiesOver` additionally builds a fresh `map[int]int` over the whole window each call (`sampler.go:202`) — the default `RepeatLastN == 0` means the *entire* history.

**Measurable cost.** Vocab 152 064: `ips` is `152064 × 24 B ≈ 3.6 MB`, `probs` is `1.2 MB`, and the sort is ~152 k·log₂(152 k) ≈ 2.6 M comparisons through a closure — **per sampled token**, to select `TopK = 40`. In sampled speculative mode `distVectorFrom` runs this for *every* draft position (`spec_sample.go:36`), so K=8 means ~43 MB and 21 M comparisons per round.

**Fix.** Replace the full sort with a bounded selection: when `TopK > 0`, a size-`TopK` min-heap is O(V log K); when `TopK == 0`, quickselect on the min-p/top-p threshold. Reuse `probs`/`ips` from a sampler-owned scratch. Keep the penalty counts in a reused `map` (or a small ring of counts) rather than rebuilding it.

### 15. `applyRoPE` recomputes `sin`/`cos` per layer and separately for Q and K

**Medium (performance)** — `decoder/rope.go:33-34`

```go
	for d := range half {
		theta := posF * invFreq[d]
		c := math.Cos(theta) * scale
		s := math.Sin(theta) * scale
```

For one position the `(cos, sin)` table depends only on `(pos, invFreq, scale)` — and `causalAttention` calls `ropeAt` back-to-back for Q and K with identical arguments (`attention.go:107-108`), while every layer sharing a rope spec (all global layers share one `ropeInvFreqGlobal` table, all local layers one `ropeInvFreqLocal`) recomputes the same values.

**Measurable cost.** Llama-3-8B (32 layers, `half = 64`, one table): `32 × 2 × 64 = 4096` `Cos` + 4096 `Sin` per token, of which 64 pairs are distinct — a 64× redundancy, ~165 µs/token of pure transcendental work. The prefill case is the material one: `runLayersFromEmbedN` calls `ropeAt` per row per layer for Q and K (`forwardn.go:174-175`), so a 2048-token prompt does `2048 × 32 × 2 × 64 ≈ 8.4 M` `(cos, sin)` pairs ≈ **0.34 s of trig on TTFT alone**.

**Fix.** Compute a `[half]` cos/sin pair array once per `(position, rope-spec)` and pass it into `applyRoPE`. In the batched path build a `[K][half]` table per rope spec before the layer loop (two specs max) and index it — the arithmetic is unchanged, so parity is bit-exact.

### 16. `NgramDrafter.Draft` clones the full history every round and scans it O(n·maxMatch²)

**Medium (performance)** — `decoder/spec_ngram.go:313` and `:124-135`

```go
lookupCtx := append(slices.Clone(hist), cur)      // :313 — full-history copy per round
...
for L := hi; L >= minM; L-- {                      // :124  up to 15 lengths
    pat := ctx[n-L:]
    for s := n - L - 1; s >= 0; s-- {              // :127  up to n starts
        if !slices.Equal(ctx[s:s+L], pat) { continue }   // :128  O(L) each
```

**Measurable cost.** At 4096 context: the clone is a 32 KB allocation per round. The scan, on a **miss** (the common case on novel prose, and the case the drafter is documented to make "free"), is `Σ_{L=2..16} n·L ≈ 4096 × 135 ≈ 550 k` element comparisons per round — comparable to a whole decode step's non-matmul work, and paid on every round including the ones that draft nothing.

**Fix.** Keep the confirmed context in one append-only slice the loop already owns and pass `hist` + `cur` as two arguments instead of cloning. For the scan, index the context with a rolling hash of the last `MinMatch` tokens (`map[uint64][]int` of candidate starts) so a miss is O(1) and a hit verifies only the handful of candidates — this is the "suffix automaton" the doc already flags as the intended follow-up.

### 17. GPTQ/AWQ reconstruction writes with stride `in` — a cache-hostile inline transpose at load

**Medium (performance)** — `decoder/gptq.go:116`, `decoder/awq.go:78`

```go
	for i := range in {
		...
		for j := range out {
			...
			res[j*in+i] = float32(code-zero) * sc[scRow+j]
```

The inner loop advances `j`, so consecutive stores are `in` floats apart. For `in = out = 4096` that is a 16 KB stride: every store touches a fresh cache line and a fresh TLB entry, and one full pass dirties 64 MB with essentially zero locality.

**Measurable cost.** 16.7 M scattered stores per linear × ~7 linears × 32 layers ≈ **3.7 G cache-missing stores** for a 7B GPTQ checkpoint — this is the dominant term in GPTQ/AWQ load time, and it is entirely avoidable.

**Fix.** Write `[in, out]` row-major (sequential stores, matching the packing order), then transpose with a blocked transpose (32×32 or 64×64 tiles) — or better, since `streamQuantized` already wants one row at a time, restructure to emit `[out, in]` rows directly by iterating `j` outer and gathering `i` inner from `qw`.

### 18. `softmaxStable` produces all-NaN on a fully-masked vocabulary; the sampler then silently picks the last token id

**Medium** — `decoder/sampler.go:293-313`

With every logit `-Inf`, `maxv = -Inf` and `math.Exp((-Inf) - (-Inf))` = `Exp(NaN)` = `NaN`, so `out[i] = NaN` and `sum = NaN`. `drawFull`'s `if r < cum` is false for all `i` (NaN compares false), so it falls through to `return len(probs) - 1` (`sampler.go:279`) — the highest vocab id. The greedy path is equally silent: `argmax` over all `-Inf` returns `0` (`sampler.go:282-290`, strict `>`).

**Failure scenario.** A `LogitProcessor` from `constrain` reaches an unsatisfiable grammar state (the `constrain` fuzzer declares this a bug, and schemas are per-request attacker-supplied via serve's `response_format`). Instead of an error, generation emits token `VocabSize-1` (or `0`), commits it to the cache and the grammar, and continues — producing plausible-looking garbage rather than a 400.

**Fix.** After the accumulation, `if sum == 0 || math.IsNaN(sum) { return error }` and surface it from `SampleWithInfo` (which already returns `error`). Same guard for the greedy branch: if `argmax`'s best is `-Inf` or NaN, error.

### 19. `Session.genSpec` and `GenerateGrammarSpeculative` wipe a warm session on a rejected call

**Medium** — `decoder/session.go:188` (and `:168`)

`Session.Generate` deliberately guards the cache mutation (`if len(prompt) > 0 { matched = s.rewindForReuse(prompt) }`, `session.go:107`) precisely so a rejected empty-prompt call leaves a warm session intact. `genSpec` calls `s.rewindForReuse(prompt)` unconditionally, before `genNgramInto` reports the empty-prompt error.

**Failure scenario.** `len(prompt) == 0`: `commonPrefixLen(s.tokens, nil) = 0`, `matched = max(min(0, -1), 0) = 0`, so `rewindForReuse` runs `TruncateTo(0)` — discarding a 4000-token warm prefix — and only then does the goroutine set `g.err`. The next real request re-prefills from scratch. `validateNgramSpec` doesn't check the prompt either, so nothing stops it earlier.

**Fix.** Add the `len(prompt) == 0` check to `validateNgramSpec`/`validateGrammarSpec` (synchronous, before any cache mutation), matching what the `Model`-level entry points already do at `spec_ngram.go:184` / `spec_grammar.go:201`.

### 20. `DraftTree` builds a full B-ary tree, and `stats.Drafted` undercounts it ~4×

**Medium** — `decoder/eagle.go:263` (accounting at `decoder/spec_eagle.go:106`)

`cands := topKDraftIdx(parentLogits[fi], b)` sits **inside** the `for fi, node := range frontier` loop, so every frontier node expands `b` children at every depth: `b + b² + … + b^d` nodes, not the `B×D` chains the API documents. Meanwhile `stats.Drafted += td.B * td.D`.

**Failure scenario.** `B=2, D=4`: 30 tree nodes verified, `Drafted` reports 8. The verify pass is 3.75× the documented cost (so a "speedup" measurement using `AcceptanceRate = Accepted/Drafted` reads ~3.75× too optimistic), and `m.NewCache(len(prompt)+maxTokens+B*D+8)` (`spec_eagle.go:45`) under-sizes the cache by the same factor. At `B=3, D=5` it is 363 nodes vs. a claimed 15 — the batched verify's `ascores = K*maxKeys` and `treeMask` (`n×n` bools) become the dominant cost.

**Fix.** Pick one: branch only at depth 1 (`if depth == 1 { cands = topKDraftIdx(...) } else { cands = topKDraftIdx(..., 1) }`) to match the docs, or keep the tree and fix `Drafted += len(td.Tokens)`, the `capHint`, and the doc comment.

### 21. `ArgmaxEquivalent` omits `LogitProcessor` from its own contract

**Medium** — `decoder/sampler.go:108-110`

```go
func (s *Sampler) ArgmaxEquivalent() bool {
	return s.p.Temperature <= 0 && len(s.p.LogitBias) == 0 && !s.penaltiesActive() && !s.p.Logprobs
}
```

The doc directly above it says "When true (**and `SamplingParams.LogitProcessor` is nil**), a backend … may skip the full-logits readback" — i.e. the predicate is documented as incomplete, and its own comment insists that "adding a new one that mutates or reads logits must be reflected here, so a fast path can never silently diverge". The one in-tree caller does remember (`model.go:720`: `sp.LogitProcessor == nil && sampler.ArgmaxEquivalent()`), which makes the trap invisible today.

**Failure scenario.** Any new caller (or a resident backend adding its own on-device argmax path) that trusts the method name skips the logits readback while a `constrain` mask is active → constrained decoding is silently unconstrained.

**Fix.** Fold the condition in: `&& s.p.LogitProcessor == nil`. Also note `penaltiesActive()` requires non-empty history, so a fresh sampler with `RepeatPenalty` set returns `true` on the *first* token and `false` after — use `penaltiesConfigured()` here instead.

### 22. Own-forward family lists are duplicated across seven sites with divergent membership

**Medium (maintainability)** — `decoder/model.go:280`, `:289`; `decoder/forwardn.go:22`, `:38`; `decoder/model.go:545`, `:571`; `decoder/residency.go:130`

The `a.gemma4 == nil && a.qwen35 == nil && …` chain appears seven times with **different** member sets. Two concrete divergences:

- `canBatchN` excludes `a.mla` and `a.llama4` (`forwardn.go:22`), but the kvI8 guard does not: `if m.kvI8 && a.gemma4 == nil && a.qwen35 == nil && a.granite == nil && a.nemotron == nil && a.MoE == nil` (`model.go:280`).
- `enableRings` excludes gemma4/qwen35/granite/nemotron but not `mla`/`llama4` (`model.go:289`).

Both are latent only by accident: llama4 and DeepSeek both set `MoE != nil` (so the kvI8 guard incidentally catches them) and neither sets `SlidingWindow`/`layerIsGlobal` (so `enableRings` returns early). The failure mode if either accident changes is a hard panic, not a decline: `forward_llama4.go:88` and `forward_qwen35.go:80` size the scores buffer from the **f32** store —

```go
nKeys := len(cache.Keys(layer)) / (nKV * hd)
attendQuery(q, ctx, cache.scr.scoresBuf(nKeys), cache, layer, pos, true, arch)
```

— while `attendQuery` re-derives `nKeys` from `r.count` (ring) or `len(cache.keysQ[layer])/kvDim` (int8). Under either mode `cache.Keys(layer)` is empty, so `scoresBuf(0)` returns a zero-length slice and the callee's first `scores[s] = …` panics.

**Fix.** One `ownForward` marker plus per-arch capability bits (`canRing`, `canKVQuant`, `canBatch`, `canCapture`, `canRewind`) set once in the registry adapter, consulted everywhere. That is the same "make representability explicit" mechanism `canSerialize` already established — and it removes the entire "new family forgot site #5" class. While there, make `attendQuery`'s callers derive `nKeys` from `cache.storedRows(layer, kvDim)` (which already handles all three storage modes) instead of `len(cache.Keys(layer))`.

---

## Low

### 23. `Snapshot` still silently drops m-RoPE and image-block state

**Low** — `decoder/kvsnapshot.go:66-95` — The refusal list now covers `c.delta`, `c.mamba`, `c.mlaLatent`, but the writer never emits `mropePos`, `mropeDelta`, or `imgBlocks`. A snapshotted Qwen2.5-VL session restores with `mropeDelta == 0`, so every decode position past the prefill rotates at `seqPos` instead of `seqPos + delta` (typically a large negative offset because the image grid compresses positions) — wrong RoPE for the whole continuation. **Fix:** persist the three fields, or add them to the same refusal predicate that already guards the recurrent families.

### 24. `Session` can never use the GPU-resident path

**Low (architecture)** — `decoder/model.go:678` — `useGPU := m.resident != nil && prefillFrom == 0 && commit == nil`. `Session.Generate` always passes a non-nil `commit` (`session.go:119`), so residency is structurally unreachable for sessions — and `cmd/serve` runs everything through `sessionLRU`. The net effect is that on a resident build, serve pays for resident weights (device VRAM) and then decodes on the CPU staged path for every request that goes through a session. It's a documented consequence of prefix reuse not being wired to the device KV, but nothing in the API surfaces it and the resident-vs-staged choice is invisible to operators. **Fix:** at minimum log/expose which path a request took; longer term, `ResidentForward` needs an `UploadKV`-based prefix-reuse seam (Metal already has `UploadKV`).

### 25. `logitsFromHidden` normalizes the caller's buffer in place

**Low** — `decoder/model.go:608` — `normalize(arch, h, m.w.FinalNorm, …)` mutates `h`. On the `forward` path `h` is `scr.h` (fine), but `forwardFromEmbed(h, cache)` (`model.go:594`) is the multimodal embed-by-vector seam and takes a caller-owned slice — after the call it holds the final-normed hidden, not the projected vision embedding the caller passed. **Fix:** normalize into `scr.norm` and matmul from there; document the buffer contract on `forwardFromEmbed`.

### 26. Dead code the linter can prove

**Low** — `unused` reports, all confirmed by grep: `Architecture.ropeBase` (`arch.go:352`), `gatedDeltaNet` (`deltanet.go:174`), `gatedDeltaNetChunked` + `scanChunk` (`deltanet_chunked.go:28`, `:122`), `mamba2Seq` (`mamba2.go:254`), `mamba2Chunked` (`mamba2_chunked.go:29`), `layerPager.stats` (`layerpaging.go:133`). Separately, `attendQuery`'s ring and int8 branches (`attention.go:189-192`, and all of `attendQueryI8`) are unreachable in production: the only four non-test callers (`forward_granite.go:100`, `forward_nemotron.go:61`, `forward_qwen35.go:81`, `forward_llama4.go:89`) all pass `global=true` on families that `NewCache` excludes from both rings and kvI8. The file's own doc still claims it is "shared by causalAttention (M=1) and the batched forwardN", which is no longer true. **Fix:** delete the unreachable branches (or pin `attendQuery ≡ attendBatchedHeads` with a test and keep them as the reference), and correct the doc. The chunked SSM kernels are deliberate future work — worth a `//lint:ignore` with the reason rather than leaving them ambiguous.

### 27. `parseQuantConfig` reads `sym` and `descAct` but ignores `checkpoint_format`

**Low** — `decoder/gptq.go:36-64` — `quantConfig.sym` is parsed and never read anywhere; `descAct` is unused too (correctly, since `g_idx` is always honoured). But `checkpoint_format`/`version` is not parsed at all, so a `gptq_v2` checkpoint — which stores the true zero point with **no** `+1` — silently decodes through the v1 formula `zero := (… & 0xF) + 1` (`gptq.go:115`), shifting every weight by one quantization step. **Fix:** parse `checkpoint_format`/`version` and reject anything but v1/GEMM loudly; delete `sym`/`descAct` or wire them.

### 28. `EmbedScale` admission predicate doesn't mirror the forward

**Low** — `decoder/features.go:76` — `add(a.EmbedScale > 1, FeatEmbedScale)` vs the forward's `if arch.EmbedScale != 0 && arch.EmbedScale != 1` (`model.go:397`, `forwardn.go:72`, `residency.go`'s `embedResident`). An `EmbedScale` in `(0, 1)` would be applied by the forward but not declared to the admission matrix, so a backend that doesn't implement it would be admitted and run without it. No current arch hits it; the point of this file is that the predicate cannot drift from the forward. **Fix:** `add(a.EmbedScale != 0 && a.EmbedScale != 1, FeatEmbedScale)`.

---

## What's solid

Checked rather than assumed:

- **The f64-accumulation strategy.** Routing K=1 decode through `attendBatchedHeads` with `MatmulBTAcc64` (`attention.go:114-125`, `forwardn.go:328-334`) so decode ≡ batched prefill ≡ speculative verify bit-exactly is the right call, and the comment explains *why* (f32's reduction is M-dependent, which flipped ~11% of argmaxes and broke spec parity). The deferred ring write in `batchReadLocal`/`commitBatch` (`kvcache.go:470`, `:528`) — reading the window before overwriting it so a `K > W` batch can't evict in-batch history — is a subtle ordering constraint, correctly identified and correctly implemented.
- **RoPE math.** `applyYarnScaling` (`ropescale.go:242`), `applyLlama3Scaling` (`:266`), `newYarnScaling`'s `get_mscale` default (`:176`), and `applyMRoPE`'s section→component mapping (`rope.go:90`) were checked line-by-line against HF's `_compute_yarn_parameters` / `_compute_llama3_parameters` / `apply_multimodal_rotary_pos_emb`. All correct, including the `low == high` singularity guard and the fact that `d` and `half+d` fall in the same m-RoPE section. `mropePositions`' `st = base + max(t, hm, wm)` (`rope.go:154`) correctly reproduces `max(llm_pos_ids) + 1`.
- **`ring.truncate`'s exactness predicate** (`kvcache.go:230`): `r.count <= r.w || p >= r.count-1`, evaluated on the *pre*-truncation count. Correct and conservative in the right direction across the wrapped, unwrapped, and drop-exactly-one cases. The bug is entirely in the callers (#1), not here.
- **Sampled speculative decoding** (`spec_sample.go`). The point-mass reduction of the Leviathan/Chen rule, the residual draw `(p − δ_x)+` with its `denom <= 0` and float-rounding guards, and `distVectorHist` threading per-position penalty/bias history — all correct, and honest about being distribution-lossless rather than token-identical.
- **Determinism hygiene.** `argmax` (`sampler.go:284`) and `topK` (`mlp.go:288`) both use strict `>`, so ties break to the lowest index — matching the documented first-max convention and the CUDA/Metal argmax kernels' intent.
- **Quantized unpacking.** `gptqReconstruct` and `awqReconstruct` validate shapes before indexing (multiple-of-8 checks with the reasoning spelled out, `g_idx` range check, `qzeros` length vs `nGroups*out/8`), and AWQ's `AWQ_REVERSE_ORDER` nibble de-interleave is right. The GPTQ v1 `(q − (z+1)) · s` formula is correct for what it claims to support.
- **`config.go`'s per-family validators** are unusually thorough — e.g. `NRoutedExperts % NGroup == 0` and `topk_group ∈ [1, n_group]` (`config.go:413-416`) are exactly the guards `groupLimit`'s `gsz := len(sel)/nGroup` needs, and they're there.
- **The `features.go` admission taxonomy** remains the standout design idea, and the fixes shipped since the last review (`canSerialize`, all-layer `yarnMscale`, `TiedLMHead` finalization, the `Snapshot` refusal list, `ContextCap`) show the mechanism working as intended: representability turned into load-time errors.

---

## Suggested priority order

| # | Items | Rationale |
|---|---|---|
| 1 | **#1** (spec rollback predicate), **#2** (recurrent state on Reset/rewind) | Both are silent-wrong-output on supported configurations, both reachable from `cmd/serve` defaults, and #2 leaks state *across users*. Each is a small diff: a parameter on `specRollbackSafe` + consuming the `exact` bool; a `resetRecurrent()` + a `canRewind` check. |
| 2 | **#3, #4** (EAGLE panics), **#5** (LoRA in verify), **#6** (greedy spec penalties) | Crash-or-silent-divergence in the speculation family; all four are guard-and-reject or a moved `continue`. Do them as one sweep with a table-driven "every spec entry point × every registered arch" rejection test. |
| 3 | **#7** (`AttnScale`), **#28**, and the `gguf.go:221` 27B geometry | One-line fix plus a cross-check test (`m.AttnScale() == arch.AttnScale` per arch) that closes the whole class of derived-vs-resolved drift. |
| 4 | **#9, #10** | The two highest-leverage performance fixes in the package: headroom growth in `decodeScratch` and not allocating a 1 GB cache the resident path never reads. Both are a handful of lines with no design work. |
| 5 | **#8** (pager races), **#11** (unguarded send) | Correct the `Model` concurrency contract *and* the code. #11 is three `select` statements copied from the spec loops. |
| 6 | **#12, #13, #14, #15** | The remaining allocation/CPU hot spots, in descending payoff. #12 and #13 want the same `batchScratch`-on-`KVCache` mechanism, so do them together; #14 and #15 are self-contained. |
| 7 | **#22** (dispatch-list consolidation) | The structural fix. Best done *after* 1–3, because those sweeps will have added two or three more per-family predicates and the right factoring will be obvious by then. Retire the latent `scoresBuf(0)` panic as part of it. |
| 8 | **#16–#21, #23–#27** | Load-time performance, robustness, and hygiene; batch by file to amortize context. **#18** deserves a nudge up if untrusted `response_format` schemas reach the sampler in your deployment. |
