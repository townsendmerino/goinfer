# Task: modularize goinfer further — cut the long functions first (2026-11)

> **Status 2026-11-13: SCOPED, nothing started.** Opened from a read of the tree at df4d023fc, prompted by a Gemini
> list of five suggestions (§1). The finding: goinfer's monoliths are a handful of very long functions and two very wide
> state structs, and they sit in the files that change most. Splitting those files leaves every one of them whole, so the
> order here is functions first, then the backend view, then two small package moves. Two of Gemini's items are declined
> (§1). No test was run for this doc and no code changed.
>
> **Rigor:** ordinary engineering ("Rigor matched to stakes", CLAUDE.md). Every row is a no-behavior-change refactor: a
> byte-identity or census gate where one exists, `touched.py` and the lint green, and a PR. No measurement record and no
> registration section, unless a row turns out to change outputs, which stops it with DECISION NEEDED.

<!-- ledger -->
| id | state | next | evidence |
|---|---|---|---|
| MZ1 | active | `backend/cuda/resident_build.go:buildState` carries eight of `backend/cuda/backend.go:cudaBackend.buildResidentMode`'s phases: `admit` (#934), `planLayers` (#943), `newResident` (#955), `bindKernels` (#957), `bindFamilyState` (#960), `uploadLayers` (#961), `checkSublayers` with `sizeAttnScratch` (#963) and `allocKV` (the KV fit, the cache-width check and the pools, 38 lines; branch `mz1-p8-2026-11-13`). Next the remaining decode scratch, then the slot grants, then the post-setup flags. Gate: the SB2 admission and memory censuses and `TestMemGate_cuda`'s census lines identical to origin/main on nobara | §2, §3.1 |
| MZ2 | active | Metal: done. `backend/metal/model.go:buildResident` is 80 lines, a sequence of twelve phases on `backend/metal/resident_build.go:residentBuild` (`compileKernels`, `bindPipelines`, `buildMixers`, `selectLanes`, `uploadLayers` with `buildLayer`, `allocKVTables`, `uploadHead`, `checkWidths`, `selectGemvRows`, `sizeDecodeScratch`, `pinResidencySet`, `finish`); #938 and #939 merge the first ten, the last two are on branch `mz2-metal-buildresident-3-2026-11-13`. WebGPU `backend/webgpu/residency.go:webgpuBackend.BuildResident` (was 1,049 lines), nobara-mz2w's lane: `backend/webgpu/residency_build.go:webgpuBuild` carries the upload helpers (#942), the geometry (`newBuild`) and the model-level upload (`uploadModelLevel`) (#945), and `backend/webgpu/residency_layer.go` the per-layer mixers (`nemotronLayer`, `deltaNetMixer`, `gatedAttnMixer`, `graniteMambaMixer`, `mlaMixer`, `attnMixer`) (#946), then the feed-forward and tail (`g4MoEFFN`, `sparseMoEFFN`, `denseFFN`, `layerTail`) (#947), then the layer loop (`uploadLayers`) (#948), then the runner and prefill closure (`bindRunner`) (phase 6 of 6, PR open); `BuildResident` is 35 lines (WebGPU: 244 builds held, 39 declined, identical at each cut). Gate: each backend's SB2 census and memory gate unchanged (Metal: 840 census lines; 286 builds held, 87 declined, identical at each of the twelve cuts) | §2, §3.1 |
| MZ3 | active | cut `decoder/gguf.go:buildWeightsFromGGUF` (1,469 lines) and `decoder/weights.go:buildWeightsFromSafetensorsTo` (627 lines) into per-family builders. After CH4 closes (`docs/tasks/task-code-health.md`), behind its gate (`decoder/ch4_giw_gate_test.go:TestCH4_giwByteIdentity`); it touches the parity manifests, so the coordinator clears the parity lane first. This replaces "split `weights.go` by family" in that doc's §3.4. First cut: `decoder/gguf_build.go:ggufBuild` carries the shared readers and `loadQwen35`, `loadGptOss`, `loadLaguna` and `loadGranite` are the qwen35, gpt-oss, laguna and granite branches moved whole (every old line is still present, in `gguf.go` or `gguf_build.go`); next: the nemotron, gemma4 and llama4 branches, then the default per-layer loop, then the safetensors builder | §2, §3.1 |
| MZ4 | open | move CUDA and WebGPU onto `decoder/backend_view.go:BackendModelView` and delete the `Model.*Resident` forwarders in `decoder/residency.go` as their last caller goes. Then stop: §3.2's package question is Francis's | §3.2 |
| MZ5 | decision | whether the resident contract leaves `decoder` for a leaf package, and whether the weight data types follow. Decide with SUB1 (`docs/tasks/task-forward-and-memory-2026-11.md`), which already owns "which interfaces stay" | §3.2 |
| MZ6 | done | none | `backend/cuda/internal/fitdecide` holds `capSlots`, `slotsDecision`, `weightsFitDecision`, `kvFitDecision` and their pure tests; it imports `gpudev` for the quantum, neither `decoder` nor `decoder/fit`. `kvFitDecision` takes the KV price as a function (`kvChargedBytes` stays in cuda), so cuda keeps one-line adapters and `TestKVFitDecision*` stay and run through them. Ratchet cuda Reach 307 to 306 (`slotcap_test.go` left). §3.3 |
| MZ7 | done | none | `genEngine` no longer links `engine/enginetest` (rule `serveapp-fakes-test-only`; tests set `fakeEngine` to the test-file `scriptedEngine`). Francis approved the move (relayed by the coordinator, 11:30): `internal/app/serveapp/jobs` now holds `job.go`, `jobeventlog.go`, `jobjournal.go`, `turn_timings.go` (611 lines; the earlier "1,611" was a typo) and `usage.go`, plus `job_test.go` and `jobeventlog_test.go`; serveapp keeps `usage` and `prefillReuseSources` as aliases of `jobs.Usage` and `jobs.PrefillReuseSources`, which moved because `Job` holds a `Usage`. The journal's two permission constants are its own (`dirPerm`, `filePerm`, pinned to 0o700 and 0o600 by `TestJobJournal_permissions`), so `sessions.go` is untouched. `Job.Result` is exported with `json:"-"`, so the JSON is unchanged. The survey of the 32 files that touch neither `server` nor `loadedModel` (4,709 lines by an identifier scan; the row's 29 differs by counting method): remaining leaves with no outward need are `sessions.go` (534; `sessionLRU`, `newSessionLRU`, `sessionDirOK`), `carried_ids.go` (358; 12 names used outside), `feature_cache.go`, `kvhost.go`, `tower_swap.go`, `version.go`. Coupled, not first: the tower cluster (`tower_reserve.go`, `gemma3_tower.go`, `webgpu_tower_fit.go` need `config` and `visionModelType` from `main.go`), `helpers.go` (SSE and body helpers), `reasoning_items.go` and `responses_stream.go` (need `carriedMint`, `thinkOut`). §3.4 |
| MZ8 | active | `decoder/model.go`: `Options` and `Options.Validate` to their own file, then cut `decoder/model.go:Model.generateInto` (657 lines, 10 parameters) into its phases. Touches the parity manifests (a non-test `decoder` file), so it shares MZ3's lane. First half done: `Options`, `Options.Validate` and `Options.layoutBackend` are `decoder/options.go` (a pure move: every line of the old `model.go` is in `model.go` or `options.go`, plus the package and import lines); next: `generateInto`'s phases, after MZ3 | §2, §3.5 |

## 1 · Gemini's five suggestions, checked against the tree

| # | suggestion | verdict | why |
|---|---|---|---|
| 2 | move the `.giw` serializer and sidecar entry points out of `decoder` | declined | decided the same day (`docs/tasks/task-code-health.md` §3.4): the serializer is built on `decoder`'s weight types, and `internal/runtime/prequant` also calls `decoder.Load`, so its import would stay |
| 2 | split `decoder/weights.go` by family | replaced by MZ3 | a file move leaves the two builder functions whole |
| 3 | split `backend/cuda/resident.go` and `backend/metal/model.go` by responsibility | replaced by MZ1, MZ2, MZ6 | they are the right files (§2's churn) cut the wrong way, and the list misses `backend/cuda/backend.go`, which holds the longest function in the repository |
| 4 | route `serveapp`'s `decoder` calls through `engine` | declined, except MZ7 | the rule `engine-below-the-shell` (`internal/layout/imports.json`) forbids `engine` importing the shell, not the shell importing `decoder`. `serveapp` is the composition root: it loads the models it serves. Its 50 `decoder` names concentrate in `model_handle.go` and `sessions.go` |
| 5 | keep the layout docs in sync | already LAY1 | `docs/tasks/task-forward-and-memory-2026-11.md` |

## 2 · Where the weight is

Read at df4d023fc. Lengths are a function's first line to its closing brace, from two instruments that agree (ken's
outline and a brace-matching script). Of about 6,900 non-test functions, 69 are 150 lines or longer, and they hold
about 23,000 lines, a tenth of the non-test code.

| lines | function |
|---:|---|
| 1,988 | `backend/cuda/backend.go:cudaBackend.buildResidentMode` |
| 1,469 | `decoder/gguf.go:buildWeightsFromGGUF` |
| 1,197 | `backend/webgpu/internal/kernels/decoderunner.go:Context.newDecodeRunner` |
| 1,107 | `backend/metal/model.go:buildResident` |
| 1,049 | `backend/webgpu/residency.go:webgpuBackend.BuildResident` |
| 921 | `backend/cuda/prefill.go:cudaResident.prefillCore` |
| 657 | `decoder/model.go:Model.generateInto` |
| 627 | `decoder/weights.go:buildWeightsFromSafetensorsTo` |

The state structs have the same shape: `cudaResident` (`backend/cuda/resident.go`) declares about 545 lines of fields
and Metal's `resident` (`backend/metal/model.go`) 422.

**These are the files that change most.** Commits in the 30 days to 2026-11-13, counting a file's path before and
after the move under `backend/` together: `backend/cuda/resident.go` 109, `backend/metal/model.go` 101,
`decoder/model.go` 99 (the most-changed file that did not move), `decoder/residency.go` 50. Within `decoder/model.go`
the changed hunks fall on `Options` (27), `Load` (26), the `Model` struct (24) and `modelFromOptions` (20): a new option
touches all four.

**The package seam, for scale.** Non-test code outside `decoder` names these many distinct package-level `decoder`
identifiers: `backend/metal` 102, `backend/cuda` 90, `internal/app/serveapp` 43, `backend/webgpu` at least 37.
Type-resolved (`ken go uses . <package> decoder`, three build configurations, 2026-11-13); WebGPU's is a lower bound,
because its configuration type-checks with errors where cgo is off. Counting methods and fields as well, non-test code
names 303 in metal, 334 in cuda, 148 in webgpu and 126 in serveapp. Test code is the wider seam: 187 package-level
names in metal's and 189 in cuda's, led by `decoder.Options` (622 test lines in cuda) and the two test hooks
`Model.EmbedResidentForTest` and `Model.ResidentForwardForTest` (287 and 281).

*Corrected 2026-11-13.* This paragraph first read 117, 111, 50 and 41, from a regular expression that also counted
names in comments, and concluded from the October census (`docs/completed/task-subsystems-2026-10.md` §2.2: 103, 93,
52 and 35) that the backend seams had widened. The typed counts do not support that: they sit at or below October's,
whose method is not recorded, so the two are not comparable and no trend is claimed.

## 3 · The rows

### 3.1 · MZ1 to MZ3: the builders

A resident build is a sequence (fit and plan, per-layer weight upload, KV and scratch allocation, family-specific
blocks, graph or pipeline capture), written today as one function per backend. Each phase becomes a named method on a
small build-state value, moved without reordering a single allocation or upload, since the lease census and the memory
gates read their order and sizes. The loaders are the same job by family: `buildWeightsFromGGUF` already dispatches on
`decoder/registry.go:ggufRegistry`, and the per-family bodies move out behind it.

`prefillCore` and `newDecodeRunner` are not rows yet. `prefillCore` is under LW-cu's open reads, and WebGPU keeps the
RF1 adapter (OP4-W, killed), so its runner is not being reshaped. Add rows when those settle.

### 3.2 · MZ4 and MZ5: the backend view and the contract

- **The view is half adopted.** `Model.BackendView` has one backend call site, in Metal's `buildResident`. CUDA and
  WebGPU read the model through about 60 one-line `Model.*Resident` forwarders in `decoder/residency.go`, and non-test
  backend code names `decoder.Model` on 58 (metal), 28 (cuda) and 8 (webgpu) lines. MZ4 finishes the adoption and
  deletes the forwarders. It changes no behavior and needs no decision.
- **The contract is nearly free of `decoder`.** `decoder` declares 56 resident interfaces with 233 methods; only two,
  `ResidencyBackend` and `ResidencyPricer`, name `*Model` in a signature. With MZ4 done they could take the view, and
  the interfaces and their value types could move to a Tier 3 leaf, aliased from `decoder` as `decoder/fit` is.
- **What that would and would not buy (MZ5).** `BackendModelView.Weights` returns `*Weights`, so a backend keeps
  importing `decoder` until `Weights`, `LayerWeights` and the family weight structs move as data, the larger move §3.4 of
  the code-health doc names. And backend tests load through `decoder.Load` (test lines naming `decoder.Options`: 622 in
  cuda, 546 in metal, 236 in webgpu), so the layout ratchet's reach figure would not fall from a contract move alone.
  The gain is a contract a reader can hold and a fake model a backend test could be built on, not test time today.

### 3.3 · MZ6: CUDA's fit decisions

`kvFitDecision`, `slotsDecision`, `weightsFitDecision` and `capSlots` take plain numbers and small structs, and their
bodies name `decoder/fit` and nothing else from goinfer. `backend/cuda/internal/launchgeom` and
`backend/cuda/internal/wpack` are the precedent. The functions that take a model (`ctxForSlots`, `resolveCtxCapFit`
and its variants) were not read and are not in the row. Guard thresholds do not change: this is a move, and any figure
that differs after it stops the row.

### 3.4 · MZ7: the serving shell

- `loadedModel.genEngine` builds an `enginetest.Scripted` when the test seam `lm.scripted` is set, which links the
  engine's test fakes into the server. Tests can set `fakeEngine` to the same value.
- `internal/app/serveapp` has 80 non-test files and 20,324 lines; 128 methods hang on `server` and 67 on
  `loadedModel`. 29 files (3,317 lines) touch neither: the carried-id codec, the tower reserve, the job journal and
  event log, the image history, among others. They are candidates for subpackages. Their unexported cross-references
  were not read, so the row is a survey first.

### 3.5 · MZ8: `decoder/model.go`

Moving `Options` to its own file is the cheap half and takes the most-edited declaration out of the most-edited
file. `generateInto` is the hard half: it carries prefill, the resident and CPU decode loops, the speculative chains
and the commit in one body.

## 4 · What was not looked at

`cmd/gate` (16,349 non-test lines, the fourth-largest package), `backend/webgpu/internal/kernels` (17,752), and
`decoder`'s own internal structure beyond its longest functions. `decoder` is 55,140 non-test and 104,663 test lines in
one package; nothing here proposes a new Tier 3 subpackage for it beyond MZ5's question.
