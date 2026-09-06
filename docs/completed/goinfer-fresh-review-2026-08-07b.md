# goinfer fresh review — 2026-08-07 (post R-01..R-30 fixes)

> **ARCHIVED — a record, not instructions.** This file is closed work kept for its reasoning and
> its numbers. Checkboxes record the state at the moment it was archived: an unticked box means
> "not ticked when this closed", **not** "still to do", and nothing in `docs/completed/` is
> actionable. If you need a task, use the live docs; if something here reads as an instruction to
> a future reader, it was missed at archival — see the doc-closeout rule in
> `docs/parity-coverage-policy.md`, and move it to live policy or strike it.


**Scope:** the tree at HEAD `6ad3ebc` (six fix commits since the previously reviewed `53c67eb`; +650/−138 across 51 files). Every hunk of the fix diff was read line-by-line by the lead pass and cross-checked by an independent fresh-eyes reviewer; the residual candidates below were then re-verified directly (file:line, mechanism traced).

**Toolchain (go1.26.5):** `go build` clean — root workspace, `-tags gpu`, cuda standalone, metal darwin/arm64 cross. `go vet` clean on all four modules — including native-GOOS `go vet ./metal/`, which R-12's tag fixed (reproduced here). `gofmt -l` clean. **Every test suite green with zero failures** — root (now including `TestParityManifest_fresh` and `TestCapabilityMatrix`, since this sync carried the re-baked `testdata/parity_manifest.json`), gpu, cuda incl. the new R-gates (`TestBuildScoreRank_equalScoresShareRank`, `TestRunJob_recoversPanic`, lint + coverage tests), serveapp suites.

## Verdict

**All 30 findings are fixed, and fixed well.** Each fix matches or improves on the suggested shape; several came with new gate tests; the cuda ones are additionally device-verified on the RTX box per the committed tracker (R-03's ordered clear is bit-exact across the graphs suite). Two of the thirty warrant a follow-up because the fix's placement creates a new over-restriction (F-01, one root cause, two victims), and two fixes are correct but still incomplete against their finding's full surface (F-02, F-03). Everything else verified clean.

---

## New findings this round

### F-01 · The R-10 cap is applied per-model, not per-path — vision and adapter requests are now over-rejected
`internal/serveapp/openai.go:569` (prepare) · **regression introduced by the R-10 fix** · confirmed independently by both review passes

`prepare` now lowers the context bound to `lm.model.ResidentContextCap()` whenever the model has a resident runner. But the cap is a property of the **stateless resident decode path**, and two request classes never take that path:

- **Vision requests.** Both vision handlers call `lm.prepare` (vision_serve.go:138, :208), yet `GenerateVL`/`GenerateQwenVL` are stateless **CPU-only** by design (generate_vl.go:14-16) — the resident cap never binds their decode. A Qwen2.5-VL served with a resident text decoder (webgpu rc=16384, or metal/cuda rc=4096) now 400s `context_length_exceeded` for image-heavy prompts the CPU VL path served before the fix, and silently over-clamps `max_tokens` (`rc − promptLen`) below that. Image-token runs make long prompts routine here, so the 4096 caps are realistically reachable with a screenshot or two.
- **Adapter models.** The R-01 fix (correctly) routes `lm.adapter != ""` down the session path, which decodes staged/CPU — bounded by `MaxPositions`, not `rc`. But prepare still caps adapter requests at the base's `rc` (the adapter shares the base's resident model, so `ResidentContextCap() > 0`): prompts in `[rc, MaxPositions)` → 400, shorter ones get `max_tokens` clamped, though the very path R-01 chose for correctness would serve both.

Fix shape: decide the cap where the path is decided — apply `rc` in prepare only when the request will actually run stateless-resident (`lm.adapter == ""`, not a vision request), or move the bound into `drive()`. Side note (fine as-is, just intentional-check): with prompt ≥ rc now rejected at prepare, R-09's clamp-to-zero case is no longer reachable over HTTP — the R-09 fix still matters for library callers.

### F-02 · validateShapes: the R-07 extension still misses the family-specific tensors
`decoder/serialize.go:302-421` vs reader `:921-975` · **residual of R-07 (itself a residual of C-06)** · confirmed by both passes

The fix added the standard per-layer vector pass and the gemma4moe router/experts/`perExpertScale` — all correct. Still blob-controlled and unvalidated, same crafted-`.giw` threat model via exported `LoadSerializedWeights`:

- `routerScale` (reader :975) — indexed to `hidden` at forward_gemma4_moe.go:78; short → index-out-of-range panic in the decode goroutine (named in R-07's own text; the adjacent `perExpertScale` got the check, this didn't).
- `PLEGate`/`PLEProj` (reader :950-951) — matmul'd at forward_gemma4.go:203/208; wrong rows → OOB/panic or silently truncated PLE activations.
- `PostPLENorm` (:952) and the gemma4moe `postFFNNorm1`/`preFFNNorm2`/`postFFNNorm2` (:972-974) — rmsNorm'd at `hidden`; short → panic.
- `RouterBias` (:921) — `addBias` over `NumExperts` logits at forward_gptoss.go:168; short → panic.
- `SharedExpert.Gate/Up/Down` + `SharedGate` (:935-938) — rows drive writes into `SharedIntermediateDim`/`hidden` scratch (llama4/GLM shared-expert path).

Fix shape: extend the existing `vec()`/`eq()` passes to these fields — the machinery from the R-07 fix makes each a two-liner.

### F-03 · decodeJSON still leaks internal type names for composite fields
`internal/serveapp/helpers.go:100` · **residual of R-11** · empirically reproduced

The `UnmarshalTypeError` branch prints `ute.Type`, which for struct/slice-typed fields renders named internal types: `{"messages": 42}` → `field "messages" has the wrong type (expected []serveapp.chatMessage)`; likewise `serveapp.chatMessage`, `[]serveapp.apiToolCall`, `[]serveapp.toolSpec`, `[]serveapp.anthropicMessage`. Scalar fields (the M-06 bool case) are fixed. Fix shape: print `ute.Type` only for scalar kinds, else a generic "wrong type".

### Nits and notes

- **F-04** `metal/model.go:606` — the R-28 width checks use `maxNHhd`/`maxKvDim` (maxima over layer geoms); a two-geom arch whose *smaller* q/kv width is non-%8 while the larger is %8 still admits. Latent one level below the original R-28; checking each distinct geom closes it.
- **F-05** `cuda/resident.go:352` — R-25's `cacheExperts = false` prevents the nil-deref, but the comment's promised "fully-resident" fallback still doesn't exist: `upExperts` left the stacks host-mapped-only, so the expert GEMVs would bind zero-value device buffers → persistent launch errors rather than working decode. Unreachable today (needs `GOINFER_MOE_CACHE_EXPERTS=1` plus a degenerate blob); align the comment or allocate the device copies in the decline branch.
- **F-06** `metal/gemma4_moe.go` (R-02 recover) — correctness verified (slot `where[]` only updates after a successful stage; `recordExecErr`→`takeExecErr` is same-goroutine on the paged path), two benign side effects: each recovered panic abandons an un-ended encoder/command buffer (small native leak per abort), and a stale `slotExpert` tag can trigger one spurious re-stage on a later eviction. Perf-only, bounded by abort count.
- **F-07** test gap, acknowledged in your tracker: `gpu/mla_test.go:355` still passes `vHead` as the head dim, so nothing gates "real MLA geometry (hd=192) is admitted" or the new rank-1024 decline — a future guard re-tightening would regress MLA residency silently again. Worth the small fixture when the DeepSeek checkpoint pass happens.
- Cosmetic, pre-existing: `/v1/responses` SSE terminal event is named `response.completed` even when `status:"incomplete"` (both branches).

---

## Fix verification, R-01..R-30 — all confirmed

R-01 ✓ adapter models bypass the resident fast path (`&& lm.adapter == ""`); session path verified to actually apply LoRA (staged decode: `useGPU` requires `commit == nil`; `genNgramInto`'s resident verify requires `cache == nil`, so the spec path stays staged too; grammar/EAGLE never touch the resident; VL can't combine with adapters — `loadVisionTower` requires a single loaded model). R-02 ✓ recover→`recordExecErr` around `forwardLogitsPaged` (see F-06 notes). R-03 ✓ `r.stream.Sync()` before the zero-fill, correctly placed in the dynamic gap (never on a capturing stream); box-verified bit-exact. R-04 ✓ for-header stripped instead of line-skipped, both router MACs now `__fmaf_rn`; box regen byte-identical; lint + coverage gates green here. R-05 ✓ MLA exempted from the GQA guard + R-24's rank-1024 decline added (test gap noted, F-07). R-06 ✓ all allocs checked with cumulative releases in all three sites. R-07 ✓ as-scoped (see F-02 for the remainder). R-08 ✓ Anthropic tools+image 400, mirroring the OpenAI guard incl. the `tool_choice:"none"` exemption. R-09 ✓ `BudgetClamped` flag; single setter/clamp site; gate test now pins clamp-to-zero. R-10 ✓ as-scoped (see F-01 for the placement issue). R-11 ✓ as-scoped (see F-03). R-12 ✓ tag added; native metal vet reproduced clean. R-13 ✓ reconcile skipped on empty prompt. R-14 ✓ recurrent sessions reset-to-cold on rollback; verified it cannot fire on clean finishes (commit-after-forward ordering) and covers the realistic mid-sweep error shapes. R-15 ✓ count>0-on-unwritten-ring rejected; writer invariant confirmed (can't reject a legitimate snapshot). R-16 ✓ tools branch threads real finish → `respStatus` + `incomplete_details`. R-17 ✓ `bgl()` tracks every layout; `c.layout` released in Close; zero untracked `GetBindGroupLayout` sites left in production code; no double-release. R-18 ✓ `rmsQuant` on `padK32`; remaining `padK` uses all feed W8A8/W8A16 consumers — consistent. R-19 ✓ batched rms dispatches 1 workgroup. R-20 ✓ decline keyed on aikit's exact "device allocation failed" sentinel (verified against the dep source); programming panics now surface. R-21 ✓ empty ForwardN no-ops; both callers always pass ≥1 row. R-22 ✓ dead `pGemm` removed. R-23 ✓ request-time recover around batched prefill. R-25 ✓ flag cleared (see F-05 for the comment). R-26 ✓ all four comment/provenance corrections in place. R-27 ✓ `startPos < 0` guarded. R-28 ✓ q-width + kv-width %8 checks (sum property holds; see F-04). R-29 ✓ equal scores share a rank so the heap's leftIndex breaks ties — matches llama.cpp's leftmost-first; collision-free key verified; new gate + full tokenizer suite green. R-30 ✓ guard-on-last (attn, vision — guard fields verified genuinely last-built) / per-field guards (layer, fuse); retry rebuilds bounded and tracked.

## Environment caveats + cleanup

Same sandbox as the previous round: static analysis + CPU tests only, no GPU device; metal cross-compiled, not run (your tracker's on-device runs cover that side). Sandbox-only `replace` blocks in `go.work`/`cuda/go.mod`; never part of any finding.

To delete on the Mac: `~/tmcode/goinfer/_to_delete/cowork-stage-src2.tgz` (this round's staged snapshot).
