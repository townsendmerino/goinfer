# Option × path and state × lifecycle grids

**Generated** from `decoder/optiongrid.go` and `decoder/cachestate.go` by `TestOptionStateGrid_fresh`; do not edit.
Regenerate: `go test ./decoder -run OptionStateGrid -update`. Design record and history (archived):
[`completed/task-option-path-admission-2026-10.md`](completed/task-option-path-admission-2026-10.md) §4.

## Policy: how a new option, path or kind of state joins a grid

- A new `decoder.Options` field fails `TestOptionGrid_everyOptionClassified` until `decoder/optiongrid.go` classifies it on every execution path: tested, declined or n/a, with a reason. It cannot be untested: `TestOptionGrid_ratchet`'s ceiling is 0.
- A cell that names a test or a decline must name one that exists in some module (`TestOptionGrid_cellsCarryEvidence`).
- The untested count can only fall: a rise fails `TestOptionGrid_ratchet`, and a fall is booked into its ceiling in the same change.
- A new execution path is not detected on its own: add it to `ogPaths`, which then fails every option until each is classified.
- A new field on `KVCache` fails `TestKVCache_everyFieldHasAState` until it maps to a kind of state (or is geometry, a counter or scratch, with a reason).
  A new kind of state needs a cell on every lifecycle path (`TestCacheStateGrid_everyCellDeclared`) and `holdsState` must know it (`TestCacheStateGrid_holdsStateKnowsEveryKind`).
- This page is generated: a hand edit fails `TestOptionStateGrid_fresh`.
- A cell moves from untested to tested with the test that drives it, one cell at a time; a path-admission refactor is not required and changes no behaviour by itself.

## Load options × execution paths

**tested** names a test that drives the option through the path; **declined** names the function where the path refuses it;
**untested** runs today with no test that drives it (0 cells; `TestOptionGrid_ratchet` lets that number only fall);
**n/a**: the option does not reach the path (reasons below the table).

| option | CPU decode | CPU batched prefill | CPU batched decode | GPU resident decode | GPU resident prefill | speculative verify | session reuse and snapshot |
|---|---|---|---|---|---|---|---|
| `ActQuantGroup` | tested: `TestActQuantGroup_perModel` | tested: `TestOptionPath_cpuBatchedPrefill` | tested: `TestOptionPath_cpuBatchedDecode` | tested: `TestActGroup_phi3ResidentMatchesCPU` | declined at `actGroupResidentDecline` (`TestOptionPathMetal_actQuantGroupDeclines`) | tested: `TestOptionPath_specVerify` | n/a |
| `Backend` | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| `CPUBatchDecode` | n/a | n/a | tested: `TestEnableCPUBatch_policy` | n/a | n/a | tested: `TestOptionPath_specVerify` | tested: `TestOptionPath_sessionSnapshot` |
| `EmbedInt4` | tested: `TestOptionPath_cpuBatchedPrefill` | tested: `TestOptionPath_cpuBatchedPrefill` | tested: `TestOptionPath_cpuBatchedDecode` | declined at `int8BufOn` (`TestOptionPathMetal_embedInt4Declines`) | declined at `int8BufOn` (`TestOptionPathMetal_embedInt4Declines`) | tested: `TestOptionPath_specVerify` | n/a |
| `ExactPrefill` | n/a | tested: `TestOptionPath_cpuBatchedPrefill` | n/a | n/a | tested: `TestOptionPathMetal_exactPrefill` | tested: `TestOptionPath_specVerify` | n/a |
| `KVHostBytes` | n/a | n/a | n/a | n/a | tested: `TestKV7_optionKVHostBytes_resident` | n/a | n/a |
| `KVPrecision` | n/a | n/a | n/a | tested: `TestOptionPathMetal_kvPrecision`; CUDA f16 tested: `TestKV16_everyAttentionPathStaysCloseToF32`; CUDA i8 declined at `ResolveKV` (`TestKVPrecisionDelivered`) | tested: `TestPrefill_int8KVOnThePass`; CUDA f16 tested: `TestKV16_everyAttentionPathStaysCloseToF32`; CUDA i8 declined at `ResolveKV` (`TestKVPrecisionDelivered`) | tested: `TestOptionPathMetal_kvPrecision`; CUDA i8 declined at `ResolveKV` (`TestKVPrecisionDelivered`) | tested: `TestOptionPathMetal_kvPrecision`; CUDA f16 tested: `TestKV7_restoreFull`; CUDA i8 declined at `ResolveKV` (`TestKVPrecisionDelivered`) |
| `KVQuant` | tested: `TestOptionPath_cpuBatchedPrefill` | tested: `TestOptionPath_cpuBatchedPrefill` | declined at `cpuBatchCacheEligible` (`TestCPUBatch_ineligibleCachesBypass`) | n/a | n/a | tested: `TestOptionPath_specVerify` | tested: `TestKVI8_snapshotRoundtrip` |
| `KVShareOff` | n/a | n/a | n/a | n/a | tested: `TestKV2_optionKVShareOff_resident` | n/a | tested: `TestSessionLRU_sharesAPreamble` |
| `Knobs` | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| `MoECacheExperts` | n/a | n/a | n/a | tested: `TestMoECacheExperts_bitExactMetal` | tested: `TestMoECacheExperts_bitExactMetal` | tested: `TestMoECacheExperts_specAndSessionMetal` | tested: `TestMoECacheExperts_specAndSessionMetal` |
| `MoECacheSlots` | n/a | n/a | n/a | tested: `TestMoECacheExperts_bitExactMetal` | tested: `TestMoECacheExperts_bitExactMetal` | tested: `TestMoECacheExperts_specAndSessionMetal` | tested: `TestMoECacheExperts_specAndSessionMetal` |
| `MoEPager` | tested: `TestOptionPath_moePaging` | tested: `TestOptionPath_moePaging` | declined at `cpuBatchModelEligible` (`TestOptionPath_moePagingDeclinesCPUBatch`) | n/a | n/a | tested: `TestOptionPath_moePaging` | n/a |
| `Quant` | tested: `TestDecodeParityInt4` | tested: `TestOptionPath_cpuBatchedPrefill` | tested: `TestOptionPath_cpuBatchedDecode` | tested: `TestOptionPathMetal_quant` | tested: `TestOptionPathMetal_quant` | declined at `SpecDecodeConflict` (`TestSpecDecodeConflict_refusesStagedWebGPUInt4`) | tested: `TestOptionPath_sessionSnapshot` |
| `ResidentContext` | n/a | n/a | n/a | tested: `TestOptionPathMetal_neutralOptions` | tested: `TestOptionPathMetal_neutralOptions` | tested: `TestOptionPathMetal_neutralOptions` | tested: `TestOptionPathMetal_neutralOptions` |
| `ResidentKVSlots` | n/a | n/a | n/a | tested: `TestOptionPathMetal_neutralOptions` | tested: `TestOptionPathMetal_neutralOptions` | tested: `TestOptionPathMetal_neutralOptions` | tested: `TestOptionPathMetal_neutralOptions` |
| `ResidentPrefillChunk` | n/a | n/a | n/a | n/a | tested: `TestOptionPathMetal_prefillChunk` | n/a | n/a |
| `ResidentWindowedKV` | n/a | n/a | n/a | tested: `TestWindowedKV_decodeIdentity` | tested: `TestWindowedKV_prefillIdentity` | tested: `TestWindowedKV_verifyIdentity` | tested: `TestWindowedKV_reuseFloor` |
| `StreamWeights` | tested: `TestOptionPath_moePaging` | tested: `TestOptionPath_moePaging` | declined at `cpuBatchModelEligible` (`TestOptionPath_moePagingDeclinesCPUBatch`) | n/a | n/a | tested: `TestOptionPath_moePaging` | n/a |
| `WeightCacheBytes` | tested: `TestOptionPath_moePaging` | tested: `TestOptionPath_moePaging` | declined at `cpuBatchModelEligible` (`TestOptionPath_moePagingDeclinesCPUBatch`) | n/a | n/a | tested: `TestOptionPath_moePaging` | n/a |

**n/a reasons.**

- `ActQuantGroup`: an activation-quantization setting; the session's cache does not depend on it
- `Backend`: selects which paths run; it is the row, not an input to one
- `CPUBatchDecode`: selects whether decode joins a batch; a single generation is unaffected
- `CPUBatchDecode`: a decode setting
- `CPUBatchDecode`: chooses the CPU's MC3c batching; a resident batches through its own MC3
- `EmbedInt4`: a weight format; the session's cache does not depend on it
- `ExactPrefill`: selects how a prompt is ingested; decode is one token
- `ExactPrefill`: selects prefill numerics; a session's reuse rules do not depend on them
- `KVHostBytes`: snapshots hold a GPU resident's KV in host memory; the CPU path keeps its own session LRU
- `KVHostBytes`: a snapshot is taken after the prefill, before the first decode step, and restored before the prefill; decode extends the slot's own rows
- `KVHostBytes`: snapshots act as a request takes its slot and as its prefill ends, before any speculative step
- `KVHostBytes`: a session's own KV is the CPU path's; this is the resident slot's
- `KVPrecision`: "Ignored off the residency path"
- `KVQuant`: the CPU KV cache's precision ("selects the CPU KV cache storage precision"); a resident uses KVPrecision
- `KVShareOff`: sharing chooses the rows a request starts from as it takes its session; the forward that follows is unchanged
- `KVShareOff`: a copy happens once, as a request takes its slot and before its prompt is prefilled; decode extends the slot's own rows
- `KVShareOff`: sharing acts as a request takes its slot, before any speculative step
- `Knobs`: routes per-model operator knobs by name to the code that reads each (knobs.go); each knob is an environment read listed in testdata/env_reads.txt and owned there
- `MoECacheExperts`: "CUDA and Metal residency; the CPU's expert paging is StreamWeights"
- `MoECacheSlots`: "Only meaningful with MoECacheExperts", a residency option
- `MoEPager`: the CPU expert pager's backing mode
- `MoEPager`: pages weights, not the session's cache
- `ResidentContext`: "Ignored off the residency path"
- `ResidentKVSlots`: asks a GPU-resident backend for independent KV caches
- `ResidentPrefillChunk`: chunks a resident's batched prefill under MC3
- `ResidentPrefillChunk`: chunks prefill; decode is one token
- `ResidentPrefillChunk`: chunking is mc3Prefill's, which only an MC3 holder's generation runs; the speculative paths claim the resident exclusively (claimExclusive) and prefill whole
- `ResidentPrefillChunk`: schedules a prefill; the session's cache is the same either way (chunk-invariant)
- `ResidentWindowedKV`: a CUDA resident's KV layout; the CPU cache keeps its own ring
- `StreamWeights`: CPU expert paging of an mmap-backed .giw ("the CPU's expert paging is StreamWeights")
- `StreamWeights`: pages weights, not the session's cache
- `WeightCacheBytes`: the budget of StreamWeights, a CPU paging option
- `WeightCacheBytes`: budgets weights, not the session's cache

**Load-only options** (read while loading, by no path afterwards):

- `AcceptSlowMoE`: an acknowledgement checked when a paged-MoE load is planned; a no-op afterwards
- `BackendAuto`: tells a backend at load that "auto" chose it, so it may decline a precision it would requantize
- `DirectLoad`: chooses the file Load opens (the source or its sidecar .giw); the paths run whatever was opened
- `DisableFit`: selects load-time fit-by-default behaviour (resolveCtxCapFit, guardFit)
- `Drafter`: tells ResolveKV at load that a block drafter will attach: auto stays f32 on CUDA, and an explicit f16 refuses
- `ExtraResidentBytes`: prices a companion allocation when the resident is planned
- `ExtraResidentKVPerPosition`: prices a companion allocation's K/V when the resident context is chosen
- `LoRA`: merged into the base weights at load ("merged into the base at load"), so every path runs the merged weights; compute-time adapters are Session state, in the cache-state grid
- `LoadAbort`: checked between layers during a GGUF weight build, a .giw load and a backend's resident build; nil afterwards in effect
- `ResidentKVSlotsDefault`: tells the backend at load that ResidentKVSlots is the caller's default, not a chosen count
- `noSelfTest`: the resident self-test's own fixture loads; read by the self-test at load
- `planResidency`: FitQuery's open of a .giw, which builds no resident and is closed before it runs anything

## Kinds of cache state × lifecycle paths

Every `KVCache` field belongs to one of these kinds, or is declared geometry, a counter or scratch (`TestKVCache_everyFieldHasAState`).
`Snapshot` and a partial `TruncateTo` read this table.

| kind | partial rewind | full reset | snapshot | session prefix reuse | speculative rollback | publish | share | evict | checkpoint save | checkpoint restore |
|---|---|---|---|---|---|---|---|---|---|---|
| positional KV | exact | cleared | persisted | exact | exact | exact | exact | cleared | kept | exact |
| sliding-window ring | exact unless wrapped | cleared | persisted | exact unless wrapped | refused | exact unless wrapped | exact unless wrapped | cleared | kept | exact unless wrapped |
| int8 KV + scales | exact | cleared | persisted | exact | exact | exact | cold on mismatch | cleared | kept | exact |
| MLA latent | exact | cleared | refused | exact | exact | exact | exact | cleared | kept | exact |
| DeltaNet state | inexact → cold | cleared | refused | inexact → cold | refused | kept | inexact → cold | cleared | exact | exact |
| short-conv window | inexact → cold | cleared | refused | inexact → cold | refused | kept | inexact → cold | cleared | exact | exact |
| Mamba-2 state | inexact → cold | cleared | refused | inexact → cold | refused | kept | inexact → cold | cleared | exact | exact |
| KDA state | inexact → cold | cleared | refused | inexact → cold | refused | kept | inexact → cold | cleared | exact | exact |
| image blocks | inexact → cold | cleared | refused | inexact → cold | transient | exact | cold on mismatch | cleared | transient | refused |
| m-RoPE positions | inexact → cold | cleared | refused | inexact → cold | transient | caller-bound | caller-bound | cleared | transient | refused |
| deepstack rows | transient | transient | transient | transient | transient | transient | transient | transient | transient | transient |
| capture rows | transient | transient | transient | transient | transient | transient | transient | transient | transient | transient |
| tree-verify mask | transient | transient | transient | transient | transient | transient | transient | transient | transient | transient |
| LoRA adapter binding | kept | kept | persisted | cold on mismatch | kept | kept | cold on mismatch | kept | kept | cold on mismatch |

**Reasons.**

- positional KV × publish: the full blocks of a completed generation become immutable and are indexed under their chain hash; the partial tail stays its sequence's own
- positional KV × share: rows match on token ids under an equal root, and the last token is always recomputed
- positional KV × checkpoint save: not copied: the rows stay in the slot by position
- positional KV × checkpoint restore: truncated to c by position: the prefill from c overwrites what lies past it, and attention reads pos+1 keys
- sliding-window ring × partial rewind: ring.truncate returns false for a wrapped ring rewound by more than one position (C1); TruncateTo passes it on
- sliding-window ring × snapshot: the snapshot writes each ring's physical rows and count (C2)
- sliding-window ring × session prefix reuse: rewindForReuse goes cold on an inexact TruncateTo
- sliding-window ring × speculative rollback: specRollbackSafe refuses every SlidingWindow > 0 model: rollback sites do not consume an inexact rewind (audit C-04)
- sliding-window ring × publish: a resident may keep only a window of these layers' rows; the block is published and the resident answers for what it still holds
- sliding-window ring × share: residentReuseFloor asks the resident (ResidentReusableKV): a prefix whose window was dropped is served as 0, and the prompt is prefilled whole
- sliding-window ring × checkpoint save: not copied: the ring stays in the slot
- sliding-window ring × checkpoint restore: exact while the window holds what c attends to; once dropped, ResidentReusableKV's floor sends the request cold before any restore
- int8 KV + scales × publish: the scales are stored with the rows they scale
- int8 KV + scales × share: the KV element type is in the root every block is keyed under: rows of another precision match nothing
- int8 KV + scales × checkpoint save: not copied: the rows stay in the slot by position
- int8 KV + scales × checkpoint restore: truncated to c by position, as the f32 rows
- MLA latent × partial rewind: TruncateTo re-slices the per-layer latent by latentDim
- MLA latent × snapshot: the snapshot format does not carry the latent store
- MLA latent × publish: the latent is the row its group stores
- MLA latent × share: as positional rows
- MLA latent × checkpoint save: not copied: the rows stay in the slot by position
- MLA latent × checkpoint restore: truncated to c by position, as the f32 rows
- DeltaNet state × partial rewind: mutated in place per token; no per-position history to rewind to
- DeltaNet state × full reset: resetRecurrent (audit C-01, C-03)
- DeltaNet state × snapshot: the snapshot format does not carry recurrent state
- DeltaNet state × session prefix reuse: follows the rewind; rewindForReuse goes cold
- DeltaNet state × speculative rollback: specRollbackSafe refuses every model with recurrent state on a loop that rolls back with TruncateTo; BlockSpec on a resident whose fold path serves every recurrent layer (ResidentRecurrentSpec, SP2: Gated DeltaNet on CUDA) verifies without writing it and folds the committed rows in
- DeltaNet state × publish: the state is not in any block: it stays live in the resident, which is why such a model has one slot (residentSlotCount)
- DeltaNet state × share: a strict extension of the whole sequence continues from where the state stopped; a request that matches it only through its checkpoint restarts from there (lcCkptRestore); on a resident whose prefix snapshots carry the state (ResidentRecurrentSnapshots: CUDA, Gated-DeltaNet), a prompt that repeats a snapshot's exact prefix restores it and continues (B35, kvsnap.go); anything else is cold
- DeltaNet state × evict: the resident re-zeroes it at position 0 of the prefill that follows; the sequence's checkpoint record goes with it
- DeltaNet state × checkpoint save: the resident copies every recurrent layer's state and conv window at c, the end of the prompt (ResidentRecurrentCheckpoint); one that cannot copy every recurrent kind the model has keeps none
- DeltaNet state × checkpoint restore: copied back whole at c, conv window included; the prefill continues from c
- short-conv window × partial rewind: mutated in place per token; no per-position history to rewind to
- short-conv window × full reset: resetRecurrent (audit C-01, C-03)
- short-conv window × snapshot: the snapshot format does not carry recurrent state
- short-conv window × session prefix reuse: follows the rewind; rewindForReuse goes cold
- short-conv window × speculative rollback: specRollbackSafe refuses every model with recurrent state on a loop that rolls back with TruncateTo; BlockSpec on a resident whose fold path serves every recurrent layer (ResidentRecurrentSpec, SP2: Gated DeltaNet on CUDA) verifies without writing it and folds the committed rows in
- short-conv window × publish: the state is not in any block: it stays live in the resident, which is why such a model has one slot (residentSlotCount)
- short-conv window × share: a strict extension of the whole sequence continues from where the state stopped; a request that matches it only through its checkpoint restarts from there (lcCkptRestore); on a resident whose prefix snapshots carry the state (ResidentRecurrentSnapshots: CUDA, Gated-DeltaNet), a prompt that repeats a snapshot's exact prefix restores it and continues (B35, kvsnap.go); anything else is cold
- short-conv window × evict: the resident re-zeroes it at position 0 of the prefill that follows; the sequence's checkpoint record goes with it
- short-conv window × checkpoint save: the resident copies every recurrent layer's state and conv window at c, the end of the prompt (ResidentRecurrentCheckpoint); one that cannot copy every recurrent kind the model has keeps none
- short-conv window × checkpoint restore: copied back whole at c, conv window included; the prefill continues from c
- Mamba-2 state × partial rewind: mutated in place per token; no per-position history to rewind to
- Mamba-2 state × full reset: resetRecurrent (audit C-01, C-03)
- Mamba-2 state × snapshot: the snapshot format does not carry recurrent state
- Mamba-2 state × session prefix reuse: follows the rewind; rewindForReuse goes cold
- Mamba-2 state × speculative rollback: specRollbackSafe refuses every model with recurrent state on a loop that rolls back with TruncateTo; BlockSpec on a resident whose fold path serves every recurrent layer (ResidentRecurrentSpec, SP2: Gated DeltaNet on CUDA) verifies without writing it and folds the committed rows in
- Mamba-2 state × publish: the state is not in any block: it stays live in the resident, which is why such a model has one slot (residentSlotCount)
- Mamba-2 state × share: a strict extension of the whole sequence continues from where the state stopped; a request that matches it only through its checkpoint restarts from there (lcCkptRestore); on a resident whose prefix snapshots carry the state (ResidentRecurrentSnapshots: CUDA, Gated-DeltaNet), a prompt that repeats a snapshot's exact prefix restores it and continues (B35, kvsnap.go); anything else is cold
- Mamba-2 state × evict: the resident re-zeroes it at position 0 of the prefill that follows; the sequence's checkpoint record goes with it
- Mamba-2 state × checkpoint save: the resident copies every recurrent layer's state and conv window at c, the end of the prompt (ResidentRecurrentCheckpoint); one that cannot copy every recurrent kind the model has keeps none
- Mamba-2 state × checkpoint restore: copied back whole at c, conv window included; the prefill continues from c
- KDA state × partial rewind: mutated in place per token; no per-position history to rewind to
- KDA state × full reset: resetRecurrent (audit C-01, C-03)
- KDA state × snapshot: the snapshot format does not carry recurrent state
- KDA state × session prefix reuse: follows the rewind; rewindForReuse goes cold
- KDA state × speculative rollback: specRollbackSafe refuses every model with recurrent state on a loop that rolls back with TruncateTo; BlockSpec on a resident whose fold path serves every recurrent layer (ResidentRecurrentSpec, SP2: Gated DeltaNet on CUDA) verifies without writing it and folds the committed rows in
- KDA state × publish: the state is not in any block: it stays live in the resident, which is why such a model has one slot (residentSlotCount)
- KDA state × share: a strict extension of the whole sequence continues from where the state stopped; a request that matches it only through its checkpoint restarts from there (lcCkptRestore); on a resident whose prefix snapshots carry the state (ResidentRecurrentSnapshots: CUDA, Gated-DeltaNet), a prompt that repeats a snapshot's exact prefix restores it and continues (B35, kvsnap.go); anything else is cold
- KDA state × evict: the resident re-zeroes it at position 0 of the prefill that follows; the sequence's checkpoint record goes with it
- KDA state × checkpoint save: the resident copies every recurrent layer's state and conv window at c, the end of the prompt (ResidentRecurrentCheckpoint); one that cannot copy every recurrent kind the model has keeps none
- KDA state × checkpoint restore: copied back whole at c, conv window included; the prefill continues from c
- image blocks × partial rewind: the blocks are position ranges with no per-position rewind; a rewind below one would let new text attend bidirectionally
- image blocks × full reset: resetMultimodal (audit M-25)
- image blocks × snapshot: the snapshot format does not carry image blocks
- image blocks × session prefix reuse: follows the rewind
- image blocks × speculative rollback: the speculative loops take text prompts; nothing they run puts an image block in their cache
- image blocks × publish: each span's content hash, offset and length are in the key of every block it touches
- image blocks × share: a span is taken whole, and only when the request claims the same image at the same place; otherwise the match stops at its start
- image blocks × evict: residentForgetIDs clears resImgBlocks with the ids, always
- image blocks × checkpoint save: the saving path (generateInto) takes text prompts: no image block is in what it saves
- image blocks × checkpoint restore: scoreRecurrent gives no checkpoint hit to a sequence or a request with an image span
- m-RoPE positions × partial rewind: positions and the decode delta are set by the image prefill and have no per-position rewind
- m-RoPE positions × full reset: resetMultimodal (audit M-25)
- m-RoPE positions × snapshot: the snapshot format does not carry m-RoPE positions
- m-RoPE positions × session prefix reuse: follows the rewind
- m-RoPE positions × speculative rollback: the speculative loops take text prompts; nothing they run sets m-RoPE positions in their cache
- m-RoPE positions × publish: the positions of published rows are in their K rows and follow from the image spans, which are in the key; the decode delta is not stored
- m-RoPE positions × share: GenerateQwenVL recomputes the positions and the delta from the prompt's placeholder run and image grid on every call
- m-RoPE positions × evict: nothing of it is kept outside the rows
- m-RoPE positions × checkpoint save: the saving path (generateInto) takes text prompts
- m-RoPE positions × checkpoint restore: the VL paths never restore one
- deepstack rows × partial rewind: GenerateQwenVL sets the rows for the image prefill and clears them right after (generate_vl.go)
- deepstack rows × full reset: cleared after the prefill that set them
- deepstack rows × snapshot: cleared after the prefill that set them
- deepstack rows × session prefix reuse: cleared after the prefill that set them
- deepstack rows × speculative rollback: cleared after the prefill that set them
- deepstack rows × publish: cleared after the prefill that set them
- deepstack rows × share: cleared after the prefill that set them
- deepstack rows × evict: cleared after the prefill that set them
- deepstack rows × checkpoint save: cleared after the prefill that set them
- deepstack rows × checkpoint restore: cleared after the prefill that set them
- capture rows × partial rewind: captured rows are overwritten by every forward before anything reads them
- capture rows × full reset: overwritten by every forward
- capture rows × snapshot: overwritten by every forward
- capture rows × session prefix reuse: overwritten by every forward
- capture rows × speculative rollback: overwritten by every forward
- capture rows × publish: overwritten by every forward; a hidden-state route never commits (it calls residentForgetIDs)
- capture rows × share: overwritten by every forward
- capture rows × evict: overwritten by every forward
- capture rows × checkpoint save: overwritten by every forward
- capture rows × checkpoint restore: overwritten by every forward
- tree-verify mask × partial rewind: nothing in production sets the tree fields since EAGLE tree drafting was removed (be9aeea8, 2026-09-24); only tests do, on caches they own
- tree-verify mask × full reset: unset in production (be9aeea8)
- tree-verify mask × snapshot: unset in production (be9aeea8)
- tree-verify mask × session prefix reuse: unset in production (be9aeea8)
- tree-verify mask × speculative rollback: unset in production (be9aeea8)
- tree-verify mask × publish: unset in production (be9aeea8)
- tree-verify mask × share: unset in production (be9aeea8)
- tree-verify mask × evict: unset in production (be9aeea8)
- tree-verify mask × checkpoint save: unset in production (be9aeea8)
- tree-verify mask × checkpoint restore: unset in production (be9aeea8)
- LoRA adapter binding × partial rewind: the adapter is the stream's binding, not sequence state: positions are rewound under the same projections
- LoRA adapter binding × full reset: a reset session keeps its adapter; serve's per-adapter session LRUs rely on it
- LoRA adapter binding × snapshot: format v3 records the adapter name; LoadSession rebinds it, or refuses when the model has not loaded it
- LoRA adapter binding × session prefix reuse: rewindForReuse goes cold when the bound adapter is not the one the cached prefix was built under
- LoRA adapter binding × speculative rollback: a rollback stays under the same projections
- LoRA adapter binding × publish: the adapter's identity is in the root every block of the sequence is keyed under
- LoRA adapter binding × share: another adapter, or none, is another root: nothing matches (rule 4 of resident_reuse.go)
- LoRA adapter binding × evict: the binding is the generation's, not the slot's
- LoRA adapter binding × checkpoint save: the adapter is in the root the sequence, and so its checkpoint, is keyed under
- LoRA adapter binding × checkpoint restore: another adapter, or none, is another root: Score matches nothing
