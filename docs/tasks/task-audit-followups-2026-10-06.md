# Task — follow-ups from the 2026-10-06 external quality audit

**Status:** FILED 2026-10-06, not started (owner: "file stuff that's real and needs fixing as a task, we can start
that after 22").
**Source:** an external audit ("Comprehensive Quality Audit: goinfer Code, Architecture, and Performance",
Antigravity, 2026-10-06, taken at `0937d069`, before the v0.21.0 tag), pasted into a Mac session. Its 33 findings
were checked against the tree at v0.21.0 / v0.22.0's release commit; this file keeps the ones that are real and need
fixing, and says why the others were set aside. Nothing here blocked v0.22.0: every finding is in code v0.22.0 does
not change, and was in v0.21.0 too.

## To fix

### A1 · Two exclusive resident claims bypass MC3's holder count (audit 1.1, filed as Critical; real for library callers)

**Status 2026-10-06: FIXED.** `GenerateSpeculative`'s target and draft claims and `BlockSpec.generate`'s claim now
go through `tryClaimResident()`. `decoder/resident_claim_mc3_test.go` tests both callers with one MC3 place held:
block-spec generation returns `ErrBlockSpecResidentBusy` before any device call, and speculative decoding runs on
the CPU fallback with zero resident forwards. Each test was red on the old code: the block-spec one ran past its
claim (into a stub panic), and the speculative one, with only its bare CAS restored, reported "the target's resident
ran 4 forwards while an MC3 generation held a place". The existing MC3, speculative and block-spec tests pass; the
parity manifest is unaffected (neither file is hashed).

- **Where:** `decoder/speculative.go:137` (`GenerateSpeculative`, the target model's claim, and `:150` for the draft
  model) and `decoder/blockspec.go:229` (`BlockSpec`) take the resident with a bare
  `atomic.CompareAndSwapInt32(&m.resBusy, 0, 1)`. `m.tryClaimResident()` (`decoder/model.go:1737`) routes through
  `m.batcher.claimExclusive`, which also requires `holders == 0`; MC3's `claim` (`decoder/mc3_batch.go:158`) reads
  `resBusy` but never sets it. So with resident concurrency on, these two paths can take the resident while batched
  holders are mid-step on the same KV.
- **Reachability, checked:** not through `goinfer-serve`. serve never calls `GenerateSpeculative`; a block drafter
  forces one generation (`internal/serveapp/openai.go`, `setConcurrency`), so no batcher exists; `--spec-adaptive`
  combines MC3 only with `--spec ngram`, whose claim is already `claimExclusive` (`decoder/spec_ngram.go`). It is
  reachable from a Go program that calls `EnableResidentConcurrency` and runs `GenerateSpeculative` or a block
  drafter concurrently with other generations: silent KV corruption, the worst failure shape.
- **Fix:** use `tryClaimResident()` (the draft model's claim too) and keep the losers' existing fallbacks. A test
  through the caller (CLAUDE.md: a component's contract that depends on how it is called is tested through the
  caller): a resident with a batcher and one live holder must refuse both claims. Touches a decoder file, so it is a
  normal core edit (parity hashes refresh), not a release-day change.

### A2 · No runtime check that Metal's prefill logits are finite (audit 1.4, Medium) — CLOSED, already covered

**Status 2026-10-06: not real; it was filed in error.** `metalResident.PrefillLast` (`metal/backend.go`) already scans
the logits with `firstNonFinite` straight after readback and declines to the sequential path on any NaN or Inf
(A-C02, `docs/audit-metal-2026-09-30.md`), with a test hook that poisons a logit. The check tests the float's bit
pattern, so the grep for `math.IsNaN` that this item was filed on missed it. The layer-major paged paths (M26, the
generic paged MoE) return without the scan; they run decode's own kernels, which carry float32 and are bit-identical
to the sequential fallback, so declining there would change nothing. The original text follows.


- **Where:** `metal/` `PrefillLast` returns the last row's logits with no finiteness check. Finite output is asserted
  by tests (`metal/prefill_nan_test.go`, `prefill_parity_test.go`) and by the Metal device gate's "prefill emits
  finite logits" group (`go run ./cmd/gate gpu`), not at run time; the f16-MMA path has shipped NaN before (Gemma's
  GELU-tanh overflow).
- **Fix:** check the row on readback (it is already on the host); on any NaN or Inf, warn once and decline to the
  sequential path for that prompt, the same decline shape the batched pass already has. Cost is one pass over
  `vocab` floats per prompt.

### A3 · To confirm first: does the host-RAM fit guard price Metal's KV at f32? (audit 1.2, filed as High)

**Status 2026-10-06: CONFIRMED by reading, and wider than the audit said; the fix needs a design choice.** Both host
guards run for every backend (`guardGIWFit` at `decoder/model.go:610`, `guardFit` at `:705`; neither checks the
backend). Both price KV from `opts.KVPrecision == "f16"`, and serve's `-kv` defaults to `f32`
(`internal/loadflags/loadflags.go:65`), so a default Metal load is priced at 4 bytes an element while Metal allocates
f16 only. And with no `-ctx`, the guard prices the model's whole `max_position_embeddings` (`guardGIWFit`'s
`effCtx`), not Metal's 4,096-position default (`decoder/fitplan.go`, `MetalCtxDefault`): on a Qwen2.5 at 32,768
positions that is about 16× the KV the resident will hold. When the guard auto-pins a smaller context it writes
`opts.ResidentContext`, which Metal reads, so the pin can move Metal's context away from its own default in either
direction. Not yet measured: a load where the overcharge refuses or pins something it should not. **Fix shape to
decide:** price KV per backend from what the backend will allocate (Metal: f16, `MetalCtxDefault` unless pinned),
through a value `decoder` already owns (`fitplan.go` holds the Metal constants) since it cannot import `metal`; and
make the guard's pin never raise a backend's context above what it would have allocated. Then a load-time test that
shows the pin's effect on a Metal plan.


- **The claim:** `decoder/fitguard.go` prices KV from `opts.KVPrecision == "f16"` (`:393`, `:820`), so a default
  Metal load (no `-kv`) would be priced at 4 bytes an element while Metal always allocates f16 (2 bytes).
- **What was checked:** Metal's own resident guard prices f16, the only KV it ships (`metal/backend.go:311`). Not
  yet checked: whether the decoder's host guard also runs for a Metal-resident load, and with which options, so the
  double-pricing may or may not happen. If it does, it refuses or caps large-context Metal loads that would fit; it
  never lets an oversize load through.
- **Do first:** a load-time test with `Backend: "metal"` and no `KVPrecision` on a model whose context makes the
  difference visible, asserting the KV the host guard priced. Fix only if it double-prices.

## Set aside, with the reason

- **1.3 (auto-pinned context above Metal's 32k ceiling falls back to the CPU):** already handled.
  `decoder/fitplan.go` clamps the planned context at `MetalCtxCeiling` and names the refusal it prevents (C-C01);
  the function the audit names does not exist.
- **3.2 (decode flash attention lacks head dim 64 and query groups 2, 4, 8) and 3.3 (paged-MoE and DeltaNet prefill
  run token by token):** stale. B-P01 and B-P02 (block decode attention at head dim 64 and groups 2-8), the
  layer-major paged-MoE prefill with D-P01, and D-B01 (DeltaNet batched prefill) all shipped in v0.21.0.
- **2.2 (`attention_prefill_steel` untested in default CI):** F-G01 tests it by default against a float64 reference
  on the Mac; CI's macOS runner cannot run Metal at all (RELEASING.md §C1-M), which is why the Metal device gate is a
  manual release step.
- **1.5 (stop-string tail tokens in restored sessions):** stated as a possibility with no failing sequence; reopen
  only with a reproduction.
- **2.1 (backend constants in `decoder/`), 2.3 (a busy resident sends the second request to the CPU), 3.1 (W4F16
  GEMM tuning ideas):** design and performance notes, not defects. 2.3 in particular is a deliberate choice already
  in the docs; a bounded wait for the GPU would be a design change with its own measurement.
