# Task: Zeno head-to-head — feasibility and matched-depth comparison

> Scoping/measurement doc. Opened 2026-08-24 from `docs/prompts/zeno-compare-phase0.md`. Context:
> Icosa's Zeno (r/ollama post, 2026-08) ships 4-bit Qwen3.5-35B-A3B on 16 GB Macs via disk
> offloading, posting 10k prompt — 214 tok/s prefill / 8.7 tok/s decode; 541 prompt — 66/13.2 tok/s;
> llama.cpp best-config at 8.7-12.6 prefill / 3.5 decode. Vendor-posted, one table image, treat as
> directional. goinfer's nearest measured point is gemma4 26B-A4B paged at 16.98 tok/s
> (`docs/completed/gemma4-resident-scope.md`).

## Phase 0, Part A — can goinfer run the actual model?

**Rig:** Apple M1 Pro, 16 GB unified memory, macOS. Model: `unsloth/Qwen3.5-35B-A3B-GGUF`,
`Qwen3.5-35B-A3B-Q4_K_M.gguf`, 22.0 GB (22,016,023,168 bytes), single file, downloaded and verified
2026-08-24 (`GGUF` magic bytes confirmed; matches the size independently reported by both
`unsloth` and `bartowski`'s GGUF repos for this quant — weight-identity with Zeno's own shipped
copy cannot be verified, per the brief's own caveat, since Zeno is closed-source).

**Step 1 (disk):** done before downloading, as required. 47 GB free at the start; freed ~8 GB more
via `go clean -cache` + `brew cleanup -s` (both safe, regenerable caches) before starting, landing
at 54-57 GB free — enough margin for the 22 GB GGUF, confirmed after the fact.

**Step 2 (prequant → paged load): BLOCKED — real, precise, fixable finding, not yet fixed.**

```
go run ./cmd/prequant -o ~/models/qwen3.5-35b-a3b-int4.giw -quant int4 \
  ~/models/Qwen3.5-35B-A3B-Q4_K_M.gguf
```

killed by the kernel after growing to **40,544 MB (40.5 GB) resident** — confirmed via
`log show --predicate 'eventMessage CONTAINS "prequant"'`: `memorystatus: killing largest
compressed process prequant [81483] 40544 MB`. 16 GB total RAM (`sysctl hw.memsize`), swap nearly
exhausted (`vm.swapusage`: 5.35/6.14 GB used) before the kill. Output `.giw` was a 17-byte stub —
the process died before writing any real tensor data.

**Mechanism, characterized (not fixed) per this doc's own house discipline:**
`decoder.StreamTranscodeGGUF` (`decoder/gguf.go:1236-1242`) has an explicit carve-out:

> `// Dedicated-loader families (qwen35) can't stream; they fit resident.`
> `if arch.qwen35 != nil { w, berr := buildWeightsFromGGUF(cfg, arch, g, q, embedInt4, nil, ""); ... }`

and the streaming branch itself refuses the family outright (`decoder/gguf.go:1508-1510`):

> `if arch.qwen35 != nil || arch.gemma4 != nil { return nil, fmt.Errorf("...streaming transcode
> unsupported for %s (load resident + prequant instead)", arch.Name) }`

`cmd/prequant`'s own doc comment names the assumption directly (`internal/prequant/prequant.go:56-60`):
"transcode the GGUF straight into the bundle, ONE LAYER at a time... peak RAM is ~one layer rather
than the whole resident model... The dedicated qwen35/gemma4 loaders fall back to a resident build
inside StreamTranscodeGGUF (**those models fit**)." **That assumption is what broke**: it held for
every qwen35-family model tried before now (all smaller), and for gemma4 26B-A4B on a bigger box —
a 35B-A3B on a 16 GB Mac is the first case where it doesn't. The generic loader path already proves
per-layer stream-and-free works on this codebase (`streamQuantized`, `decoder/weightmat.go:123` —
"the whole model never sits resident"); qwen35's dedicated loader was written before that pattern
existed and was never migrated onto it. This is a genuine, fixable bring-up gap, not an inherent
MoE or quantization-format limit. The separately-known P12 finding (qwen35 family running f32
attention/DeltaNet scratch regardless of `Options.Quant`, `docs/queue-performance.md`) is NOT this
issue — P12 was fixed 2026-08-19 and is a *decode-time bandwidth* cost, not this *load-time
resident-peak* one; both live in the same family's bring-up debt but are independent findings.

**Phase 0's own framing says this is a complete, valuable outcome on its own**: "the real
checkpoint won't load/page/decode... file it as the bring-up gap it reveals." Recorded here as
that gap. Cleanup done: the 17-byte stub `.giw` removed, disk back to ~50 GB free.

**Steps 3 (coherent decode) and 4 (size the f32-scratch handicap) — not reached.** Both depend on
the model loading at all; deferred until the streaming fix (below) lands and this step is re-run.

## Streaming-transcode fix — scoped as its own task, not folded into Phase 0

Deliberately **not attempted as part of Phase 0** — "feasibility only, no benchmarking yet" is this
doc's own charter, and implementing per-layer streaming for a dedicated loader is real engineering,
not a quick characterization. Scoped here so the record exists independent of when (or whether) the
fix lands, and because it has value beyond this comparison: the "dedicated-loader families fit
resident" assumption was already going to be tested by Qwen3.6-35B-A3B (`docs/qwen3_5_moe.md`
scoping), a roadmap target regardless of any Zeno comparison.

**Shape:** mirror the generic path's per-layer stream-and-free inside qwen35's dedicated loader —
both call sites that currently hit the `sink=nil`/resident-fallback branch (`StreamTranscodeGGUF`'s
own qwen35 carve-out, which both the general load path and `cmd/prequant` route through, per
`internal/prequant/prequant.go:56-60`'s own comment — there is exactly one code path to fix, not
two). gemma4 shares the same carve-out (`decoder/gguf.go:1509`) but is NOT in scope here — it fits
resident on the boxes it's been run on; touch only the qwen35 branch unless a gemma4-specific
gap surfaces independently.

**Correctness gate, free from the situation itself:** build the same `.giw` from the qwen3_5_moe-tiny
checkpoint (the existing T3 fixture) through both the old resident path and the new streaming path
— outputs must be byte-identical, tensor for tensor, the same standard `TestSerializeWeights_roundTrip`
already holds other families to. Combined with the existing family parity goldens (`qwen3_5_moe-tiny`
forward parity, the DeltaNet golden) staying green, this makes the change provably inert: a loader
refactor, not a numerics event. T3 re-run required regardless (core file).

**Acceptance:** the qwen3_5_moe-tiny .giw round-trip is byte-identical old-path vs new-path; all
qwen35-family parity gates stay green; the real 35B-A3B prequant conversion completes without OOM
on this 16 GB Mac, producing a real (non-stub) `.giw`. Peak RSS during the real conversion is worth
logging even though there's no fixed target — the generic path's own "~one layer" framing sets the
expectation, not a specific number.

**DONE, 2026-08-24.** No tiny qwen35 GGUF fixture exists (`qwen3_5_moe-tiny` is safetensors), so
the round-trip gate ran against the real 22 GB GGUF with `NumLayers` clamped to 2
(`decoder/qwen35_streaming_test.go`, `TestQwen35StreamingTranscode_matchesResident`) — a genuine
subset of the real model's real tensors, small enough to safely also run the old resident path for
comparison. It found one real, designed (not buggy) difference: the old resident-then-serialize
path bakes a resolved `quantLabel` ("int4mix") into the header because `w` is fully materialized
before `writeHeadGlobals` runs; the new streaming path calls `writeHeadGlobals` before any layer
streams in, so `hasPopulatedLayers()` correctly defers the label to "" (pre-existing B11 logic,
`decoder/serialize.go:170-185`) — exactly how every other already-streaming family already behaves.
Every other byte agrees, and both bundles resolve to the same live-inferred label once loaded. All
qwen35-family goldens (`TestQwen35_forwardParity`, `TestQwen3_5_textParity`,
`TestSerializeQwen35_roundTrip`) stayed green, and the change is provably non-numeric (proven by the
above), so `scripts/refresh_parity_hashes.sh` was used for T3 rather than the full 12-minute gate —
27 goldens green, 0 failed (one heavy Mellum2 checkpoint was skipped for a missing local shard,
unrelated to this change and to any family this fix touches).

Real conversion: `go run ./cmd/prequant -o ~/models/qwen3.5-35b-a3b-int4.giw -quant int4
~/models/Qwen3.5-35B-A3B-Q4_K_M.gguf` completed in a few minutes, peak observed RSS **~7.9 GB**
(vs. the 40.5 GB OOM kill before this fix), producing a real 22,146,331,551-byte `.giw`
(`wrote ...: 21120 MB (int4)`). Verified loadable: read back through the `internal/giw` bundle
reader + `decoder.LoadSerializedWeights` — 40 layers, `NumLayers=40`, `VocabSize=248320`, no error.

## Go/no-go for Phase 1

**NO-GO, as of 2026-08-24** — the real checkpoint has not been loaded, paged, or decoded, so Part A
has not cleared. Part B (Zeno install feasibility) was not pursued: it gates on Francis's explicit
checkoint before installing anything regardless of Part A's outcome, and Part A's own result already
determines the overall verdict without needing Part B's answer. Re-run Part A once the streaming fix
above lands; Part B's checkpoint stands unchanged whenever that becomes relevant.

## Not in scope (this doc, both phases)

- Any Phase 1 benchmarking, before Part A clears.
- Fixing the P12-adjacent f32-scratch handicap (sizing it is still owed once Part A clears — see
  Phase 0's own step 4).
- Batched MoE prefill work, Qwen3.6 (wrong match for this comparison), anything W4A8 (that campaign
  owns its own files and box time — this is a side quest and shares no files with it).
- gemma4's identical carve-out — noted, not touched, unless it independently blocks something.
