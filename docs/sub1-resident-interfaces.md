<!-- citations-at: 12bf57d5b -->
# SUB1 inventory: the old Resident* interfaces (48 declared in decoder; 5 are the contract + provider, 43 below)

Generated 2026-11-14 from origin/main 12bf57d5b by an AST scan (identifier uses, comments excluded) over every module, cross-checked with grep -w.
"bridge" = decoder/resident_subsystems.go (the legacy* wrappers that make a test fake satisfy the contracts) and each backend's subsystems.go. "pin" = a backend `var _ decoder.X = (*resident)(nil)`.

| interface | file | replaced by (contract: methods) | users outside the bridge | verdict |
|---|---|---|---|---|
| ResidentAdapter | decoder/residency.go | Runtime: SetAdapter | 3 backend pin(s); decoder prod: decoder/model.go:1306(field) | delete (tier B: decoder production names it, call site moves to the contract) |
| ResidentBatchStepper | decoder/residency.go | Decode: BatchStepRange, StepBatch | 1 backend pin(s) | delete (tier A: no user beyond the bridge and the backend pins) |
| ResidentBlockDrafter | decoder/blockspec.go | — | 1 backend pin(s); backend prod: backend/cuda/drafter.go:714(param/result); decoder prod: decoder/blockspec.go:61(param/result), decoder/blockspec.go:203(field), decoder/testhooks.go:763(param/result); 10 decoder test site(s) in 6 file(s); 2 backend test site(s) in 1 file(s) | STAYS: a value the Runtime contract returns (ResidentRuntimeSubsystem.AttachBlockDrafter): its DraftBlock/FuseContext methods are the drafter handle, not a capability probe |
| ResidentCapsReporter | decoder/residency.go | Runtime: ResidentCaps | 3 backend pin(s); 14 decoder test site(s) in 2 file(s) | delete (tier C: tests name it, they move to the contract) |
| ResidentCost | decoder/cost_resident.go | Runtime: CostSpan | 1 backend pin(s); decoder prod: decoder/cost_resident.go:15(param/result), decoder/cost_resident.go:36(param/result), decoder/model.go:1314(field) | delete (tier B: decoder production names it, call site moves to the contract) |
| ResidentDrafterHost | decoder/blockspec.go | Runtime: AttachBlockDrafter, PrefillLastNArgmax, SetBatchedCapture, BatchedCapture | 1 backend pin(s); decoder prod: decoder/blockdrafter.go:137(iface-embed), decoder/blockspec.go:202(field), decoder/blockspec.go:279(param/result); 4 decoder test site(s) in 4 file(s) | STAYS (revised in tier B): `BlockSpec` drives the raw resident as its host (`SubsystemsOf` is taken from it, `StreamingDrafterHost` embeds it, a backend pins it), so it is a handle like `ResidentForward`, not a view of a contract the caller already holds |
| ResidentExpertCache | decoder/residency.go | Runtime: ExpertCacheStats, ResizeExpertCache | none | delete (tier A: no user beyond the bridge and the backend pins) |
| ResidentForward | decoder/residency.go | — | 4 backend pin(s); backend prod: backend/cuda/backend.go:139(param/result), backend/cuda/backend.go:155(param/result), backend/cuda/backend.go:161(param/result), backend/cuda/backend.go:173(param/result), backend/cuda/backend.go:183(param/result), backend/metal/backend.go:78(param/result), backend/webgpu/residency.go:235(param/result), backend/webgpu/residency.go:254(param/result); decoder prod: decoder/mc3_batch.go:65(field), decoder/model.go:53(field), decoder/residency.go:487(param/result), decoder/residency.go:575(param/result), decoder/residency.go:1505(param/result), decoder/resident_reuse.go:564(param/result), decoder/selftest_gpu.go:264(param/result), decoder/step.go:161(assert), decoder/step.go:262(assert), decoder/testhooks.go:78(param/result); 57 decoder test site(s) in 36 file(s); 14 backend test site(s) in 11 file(s) | STAYS: the base of every resident (Model.resident, Forward/ForwardN/UploadKV/TruncateTo/Reset/Close are in no subsystem); 73 test uses |
| ResidentGreedy | decoder/residency.go | Decode: ForwardArgmax | 1 backend pin(s); 1 decoder test site(s) in 1 file(s) | delete (tier C: tests name it, they move to the contract) |
| ResidentGreedyChain | decoder/residency.go | Decode: GreedyChainAvailable, GreedyChainStart, GreedyChainNext, GreedyChainStop | 1 backend pin(s); decoder prod: decoder/residency.go:194(iface-embed) | delete (tier B: decoder production names it, call site moves to the contract) |
| ResidentHiddenLast | decoder/residency.go | Prefill: HiddenLast | none | delete (tier A: no user beyond the bridge and the backend pins) |
| ResidentHybridMRoPEPrefill | decoder/residency.go | Prefill: HybridMRoPEPrefill | 3 backend test site(s) in 3 file(s) | STAYS (revised in tier C): the three backends' `support_table_test.go` ask whether the resident *type* implements it, which a contract cannot answer without a built resident; `backendTypeAssertAllowed` names those three and may only shrink |
| ResidentImageBlocksPrefill | decoder/residency.go | Prefill: PrefillImageBlocksLast | 1 backend pin(s) | delete (tier A: no user beyond the bridge and the backend pins) |
| ResidentImagePrefill | decoder/residency.go | Prefill: PrefillImageLast | 1 backend pin(s); decoder prod: decoder/testhooks.go:115(param/result), decoder/testhooks.go:120(param/result) | delete (tier B: decoder production names it, call site moves to the contract) |
| ResidentKVCopier | decoder/resident_reuse.go | KV: CopyKV | 3 backend pin(s) | delete (tier A: no user beyond the bridge and the backend pins) |
| ResidentKVDemoter | decoder/resident_reuse.go | KV: SetKVHostBudget, PromoteKV, CopyDemotedKV, ReserveKVHost | 1 backend pin(s) | delete (tier A: no user beyond the bridge and the backend pins) |
| ResidentKVPoolInfo | decoder/resident_reuse.go | KV: KVPool | 1 backend pin(s) | delete (tier A: no user beyond the bridge and the backend pins) |
| ResidentKVSlotActivity | decoder/resident_reuse.go | KV: SetKVSlotBusy | 1 backend pin(s) | delete (tier A: no user beyond the bridge and the backend pins) |
| ResidentKVSlotter | decoder/residency.go | KV: KVSlots, UseKVSlot | 2 backend pin(s); 2 decoder test site(s) in 1 file(s) | delete (tier C: tests name it, they move to the contract) |
| ResidentKVSnapshot | decoder/kvsnap.go | — | backend prod: backend/cuda/kvsnap.go:189(param/result), backend/cuda/kvsnap.go:287(param/result); decoder prod: decoder/kvsnap.go:53(param/result), decoder/kvsnap.go:56(param/result), decoder/kvsnap.go:131(field); 3 decoder test site(s) in 2 file(s); 5 backend test site(s) in 3 file(s) | STAYS: a handle the KV contract returns (SnapshotKV): Rows/Bytes/Serves/Release are the handle, not a capability probe |
| ResidentKVSnapshotter | decoder/kvsnap.go | KV: SnapshotKV, RestoreKV, KVSnapshotBytes, KVSnapshotRollback, KVSnapshotRecurrent | 1 backend pin(s); decoder prod: decoder/kvsnap.go:411(param/result); 1 decoder test site(s) in 1 file(s) | delete (tier B+C: decoder production and tests name it) |
| ResidentKVUploadPass | decoder/resident_reuse.go | KV: BeginKVUpload, EndKVUpload | 1 backend pin(s) | delete (tier A: no user beyond the bridge and the backend pins) |
| ResidentLayerView | decoder/backend_view.go | — | decoder prod: decoder/backend_view.go:72(param/result), decoder/backend_view.go:308(param/result) | STAYS: not a capability: BackendModelView.Layer returns it (MZ4 owns it); 24 methods, no contract method matches |
| ResidentMRoPE | decoder/residency.go | Decode: ForwardMRoPE | 3 backend pin(s) | delete (tier A: no user beyond the bridge and the backend pins) |
| ResidentMRoPEDeepstackPrefill | decoder/residency.go | Prefill: PrefillMRoPEDeepstackLast | none | delete (tier A: no user beyond the bridge and the backend pins) |
| ResidentMRoPEPrefill | decoder/residency.go | Prefill: PrefillMRoPELast | 1 backend pin(s); decoder prod: decoder/testhooks.go:127(param/result), decoder/testhooks.go:133(param/result) | delete (tier B: decoder production names it, call site moves to the contract) |
| ResidentPrefillBlocks | decoder/residency.go | Prefill: PrefillBlockRows, PrefillScratchBytes | 1 backend pin(s); decoder prod: decoder/residency.go:504(param/result) | delete (tier B: decoder production names it, call site moves to the contract) |
| ResidentPrefillChunks | decoder/prefill_chunk_scratch.go | Prefill: PrefillChunkRows, PrefillScratchBytes | none | delete (tier A: no user beyond the bridge and the backend pins) |
| ResidentPrefillKV | decoder/residency.go | Decode: ForwardNoLogits | 1 backend pin(s) | delete (tier A: no user beyond the bridge and the backend pins) |
| ResidentPrefillMarked | decoder/kvsnap.go | Prefill: PrefillMarkAlign, PrefillLastMarked | 1 backend pin(s) | delete (tier A: no user beyond the bridge and the backend pins) |
| ResidentPrefillRows | decoder/prefill_rows.go | Prefill: PrefillLastRows | 2 backend pin(s) | delete (tier A: no user beyond the bridge and the backend pins) |
| ResidentPrefillSpanScratch | decoder/prefill_chunk_scratch.go | Prefill: PrefillSpanScratchBytes | none | delete (tier A: no user beyond the bridge and the backend pins) |
| ResidentRaggedStepper | decoder/residency.go | Decode: StepRagged | 1 backend pin(s) | delete (tier A: no user beyond the bridge and the backend pins) |
| ResidentRecurrentCheckpoint | decoder/recurrent_checkpoint.go | KV: RecurrentCheckpointBytes, SaveRecurrentCheckpoint, RestoreRecurrentCheckpoint, RecurrentCheckpointHeld | 2 backend pin(s); decoder prod: decoder/recurrent_checkpoint.go:37(param/result); 2 decoder test site(s) in 2 file(s) | delete (tier B+C: decoder production and tests name it) |
| ResidentRecurrentSpec | decoder/blockspec.go | Runtime: PrepareRecurrentSpec, RecurrentSpecRows, VerifyLastNArgmax, FoldVerified, FlushFold, DropFold | 1 backend pin(s); decoder prod: decoder/blockspec.go:306(param/result), decoder/forwardn.go:113(param/result); 1 decoder test site(s) in 1 file(s) | delete (tier B+C: decoder production and tests name it) |
| ResidentResidualAll | decoder/residency.go | Prefill: ResidualAll | 2 backend pin(s) | delete (tier A: no user beyond the bridge and the backend pins) |
| ResidentReusableKV | decoder/resident_reuse.go | KV: ReusableKV | 1 backend pin(s); 1 decoder test site(s) in 1 file(s) | delete (tier C: tests name it, they move to the contract) |
| ResidentReusableKVSlot | decoder/resident_reuse.go | KV: ReusableKVSlot | 2 backend pin(s); decoder prod: decoder/resident_reuse.go:589(param/result) | delete (tier B: decoder production names it, call site moves to the contract) |
| ResidentSample | decoder/residency.go | Decode: SampleAvailable, ForwardSample | 4 backend pin(s) | delete (tier A: no user beyond the bridge and the backend pins) |
| ResidentSampleChain | decoder/residency.go | Decode: SampleChainAvailable, SampleChainStart, SampleChainNext | 1 backend pin(s); decoder prod: decoder/model.go:1332(field) | delete (tier B: decoder production names it, call site moves to the contract) |
| ResidentSeedArgmax | decoder/blockspec.go | Runtime: PrefillSeedArgmax | none | delete (tier A: no user beyond the bridge and the backend pins) |
| ResidentSpanPrefill | decoder/residency.go | Prefill: PrefillSpansLast | 1 backend pin(s) | delete (tier A: no user beyond the bridge and the backend pins) |
| ResidentTopK | decoder/residency.go | Decode: TopKAvailable, ForwardTopK | 1 backend pin(s) | delete (tier A: no user beyond the bridge and the backend pins) |

## Tiers

- **A** (22): ResidentBatchStepper, ResidentExpertCache, ResidentHiddenLast, ResidentImageBlocksPrefill, ResidentKVCopier, ResidentKVDemoter, ResidentKVPoolInfo, ResidentKVSlotActivity, ResidentKVUploadPass, ResidentMRoPE, ResidentMRoPEDeepstackPrefill, ResidentPrefillChunks, ResidentPrefillKV, ResidentPrefillMarked, ResidentPrefillRows, ResidentPrefillSpanScratch, ResidentRaggedStepper, ResidentResidualAll, ResidentSample, ResidentSeedArgmax, ResidentSpanPrefill, ResidentTopK
- **B** (12): ResidentAdapter, ResidentCost, ResidentDrafterHost, ResidentGreedyChain, ResidentImagePrefill, ResidentKVSnapshotter, ResidentMRoPEPrefill, ResidentPrefillBlocks, ResidentRecurrentCheckpoint, ResidentRecurrentSpec, ResidentReusableKVSlot, ResidentSampleChain
- **C** (5): ResidentCapsReporter, ResidentGreedy, ResidentHybridMRoPEPrefill, ResidentKVSlotter, ResidentReusableKV
- **stays** (4): ResidentBlockDrafter, ResidentForward, ResidentKVSnapshot, ResidentLayerView

## What "delete" means here

Every method of every deletable interface is already a method of a contract (the table), so no contract changes. A deleted interface is removed from the exported surface: the legacy* wrappers in `decoder/resident_subsystems.go` keep their structural probe for test fakes as an unexported interface of the same method set (a fake that implements `ForwardArgmax` still satisfies the Decode contract through the bridge); the backends' pins are deleted, since each backend already pins the four contracts in `subsystems.go`; tests that name an old interface assert on the contract through `SubsystemsOf`. No method body changes.
Removing the legacy* wrappers waited on every test fake implementing a contract; that is done ("Bridge tail" below). Not planned here: MZ5's leaf move, `ResidentLayerView` (MZ4).

## Progress

- **PR1 (tier A, 22):** the backends' pins are gone (commit 1) and the 22 are unexported in decoder as `legacy<Name>` probes inside the bridge (commit 2). The old exported names no longer exist; the comments in the backends point at the contract that carries the method.
- **PR2 (tier B, 11 of 12):** the backends' pins of 15 interfaces are deleted (commit 1). Where decoder production already held the contract value, the narrow type is replaced by the contract (`ResidentRuntimeSubsystem` for `resAdapter`, `resCost`, `residentCostFor`, `costSpan`, `recurrentFold`; `ResidentDecodeSubsystem` for `sChainRF`; `ResidentKVSubsystem` for `snapshotter` and `recurrentCheckpointer`; `ResidentPrefillSubsystem` for `prefillBlocks`); every method call resolves to the contract's declaration of the same method on the same dynamic value. `kvOtherDonors` and `specRollbackSafeVia` keep a narrow parameter because decoder tests pass them fakes that implement only that interface. The 11 are unexported as `legacy<Name>` (commit 2). `ResidentDrafterHost` moves to **stays** (see its row): a handle on the raw resident, not a contract view. Tier B's four exported test hooks (`Model.ResidentImagePrefillForTest` and its three siblings) now take an unexported interface type; the backend tests that call them pass a concrete resident, which still compiles.
- **PR3 (tier C, 4 of 5):** the backends' pins of `ResidentCapsReporter`, `ResidentGreedy`, `ResidentKVSlotter` and `ResidentReusableKV` are deleted and the four are unexported as `legacy<Name>` (the decoder tests that name them follow the rename; their fakes still satisfy the probes). `ResidentHybridMRoPEPrefill` stays (see its row).

## Where this leaves the exported surface

Six old interfaces stay exported, each for a stated reason: `ResidentForward` (the base), `ResidentDrafterHost` (the raw host `BlockSpec` drives), `ResidentBlockDrafter` and `ResidentKVSnapshot` (values the contracts return), `ResidentLayerView` (BackendModelView's layer), `ResidentHybridMRoPEPrefill` (a type-level check the backends' support tests make). The other 37 were unexported probes inside the bridge and are gone ("Bridge tail" below); the four contracts and `ResidentSubsystemProvider` are the surface a backend implements. Not done: MZ5's leaf move.

## Bridge tail (the last step)

Goal: no resident in the tree reaches the legacy bridge in `decoder/resident_subsystems.go`, so the bridge and its probes go. State before this step: 38 unexported `legacy*` probes (below; `legacyLastNArgmax` is declared inline in the bridge file), four bridge types (`legacyDecodeSub`, `legacyPrefillSub`, `legacyKVSub`, `legacyRuntimeSub`), `SubsystemsOf` falling back to them for any value that is not a `ResidentSubsystemProvider`. No backend resident reaches the bridge: `cudaResident`, `metalResident` and `residentDecoder` (webgpu) each pin `ResidentSubsystemProvider` at compile time (`backend/*/subsystems.go`). Only test fakes did.

Method: a temporary probe (not committed) logged the dynamic type of every value that took a non-provider path in `SubsystemsOf` while `go test ./decoder/ ./internal/app/serveapp/ ./internal/runtime/program/` ran, then a reflection census over every type declared in the decoder test files under the four build-tag configs (default, `goinfer_testhooks`, `realckpt`, both) listed those that implement a probe and are not providers, so a fake only a tagged or skipped test builds is counted too.

### Probes

| probe | file | methods | test fakes that satisfied it |
|---|---|---|---|
| legacyAdapter | decoder/residency.go | SetAdapter | 1 |
| legacyBatchStepper | decoder/residency.go | BatchStepRange, StepBatch | 10 |
| legacyCapsReporter | decoder/residency.go | ResidentCaps | 44 |
| legacyCost | decoder/cost_resident.go | CostSpan | 0 |
| legacyExpertCache | decoder/residency.go | ExpertCacheStats, ResizeExpertCache | 0 |
| legacyGreedy | decoder/residency.go | ForwardArgmax | 8 |
| legacyGreedyChain | decoder/residency.go | GreedyChainAvailable, GreedyChainStart, GreedyChainNext, GreedyChainStop | 6 |
| legacyHiddenLast | decoder/residency.go | HiddenLast | 3 |
| legacyImageBlocksPrefill | decoder/residency.go | PrefillImageBlocksLast | 1 |
| legacyImagePrefill | decoder/residency.go | PrefillImageLast | 1 |
| legacyKVCopier | decoder/resident_reuse.go | CopyKV | 3 |
| legacyKVDemoter | decoder/resident_reuse.go | SetKVHostBudget, PromoteKV, CopyDemotedKV, ReserveKVHost | 0 |
| legacyKVPoolInfo | decoder/resident_reuse.go | KVPool | 0 |
| legacyKVSlotActivity | decoder/resident_reuse.go | SetKVSlotBusy | 0 |
| legacyKVSlotter | decoder/residency.go | KVSlots, UseKVSlot | 18 |
| legacyKVSnapshotter | decoder/kvsnap.go | SnapshotKV, RestoreKV, KVSnapshotBytes, KVSnapshotRollback, KVSnapshotRecurrent | 2 |
| legacyKVUploadPass | decoder/resident_reuse.go | BeginKVUpload, EndKVUpload | 1 |
| legacyLastNArgmax | decoder/resident_subsystems.go | PrefillLastNArgmax | 12 |
| legacyMRoPE | decoder/residency.go | ForwardMRoPE | 37 |
| legacyMRoPEDeepstackPrefill | decoder/residency.go | PrefillMRoPEDeepstackLast | 1 |
| legacyMRoPEPrefill | decoder/residency.go | PrefillMRoPELast | 1 |
| legacyPrefillBlocks | decoder/residency.go | PrefillBlockRows, PrefillScratchBytes | 1 |
| legacyPrefillChunks | decoder/prefill_chunk_scratch.go | PrefillChunkRows, PrefillScratchBytes | 1 |
| legacyPrefillKV | decoder/residency.go | ForwardNoLogits | 1 |
| legacyPrefillMarked | decoder/kvsnap.go | PrefillMarkAlign, PrefillLastMarked | 3 |
| legacyPrefillRows | decoder/prefill_rows.go | PrefillLastRows | 1 |
| legacyPrefillSpanScratch | decoder/prefill_chunk_scratch.go | PrefillSpanScratchBytes | 1 |
| legacyRaggedStepper | decoder/residency.go | StepRagged | 2 |
| legacyRecurrentCheckpoint | decoder/recurrent_checkpoint.go | RecurrentCheckpointBytes, SaveRecurrentCheckpoint, RestoreRecurrentCheckpoint, RecurrentCheckpointHeld | 5 |
| legacyRecurrentSpec | decoder/blockspec.go | PrepareRecurrentSpec, RecurrentSpecRows, VerifyLastNArgmax, FoldVerified, FlushFold, DropFold | 3 |
| legacyResidualAll | decoder/residency.go | ResidualAll | 3 |
| legacyReusableKV | decoder/resident_reuse.go | ReusableKV | 4 |
| legacyReusableKVSlot | decoder/resident_reuse.go | ReusableKVSlot | 2 |
| legacySample | decoder/residency.go | SampleAvailable, ForwardSample | 14 |
| legacySampleChain | decoder/residency.go | SampleChainAvailable, SampleChainStart, SampleChainNext | 1 |
| legacySeedArgmax | decoder/blockspec.go | PrefillSeedArgmax | 2 |
| legacySpanPrefill | decoder/residency.go | PrefillSpansLast | 1 |
| legacyTopK | decoder/residency.go | TopKAvailable, ForwardTopK | 13 |

Also asserted structurally by the bridge, and kept: `Prefiller`, `PrefillPathReporter`, `PrefillTailExact`, `VerifyPathReporter`, `VerifyCostReporter`, `DecodeVerifyDiverger`, `ExactAttentionScoper`, `ResidentDrafterHost`, `StreamingDrafterHost`, `ResidentHybridMRoPEPrefill` (exported; the fakes that satisfied them are in the next table).

### Test fakes that reached the bridge

| type | file | role | probes it satisfied |
|---|---|---|---|
| quantReporting | decoder/auto_backend_test.go | root fake | legacyCapsReporter |
| recurrentSnapFake | decoder/b35_snapshot_test.go | embeds `snapFake` | Prefiller, legacyBatchStepper, legacyCapsReporter, legacyKVSlotter, legacyKVSnapshotter, legacyMRoPE, legacyPrefillMarked, legacySample, legacyTopK |
| blockspecCommitStubHost | decoder/blockspec_commit_test.go | host (runtime contract only) | ResidentDrafterHost, legacyLastNArgmax |
| blockspecWriteHost | decoder/blockspec_commit_test.go | host (runtime contract only) | ResidentDrafterHost, legacyLastNArgmax |
| blockspecStubHost | decoder/blockspec_residentbusy_test.go | host (runtime contract only) | ResidentDrafterHost, legacyLastNArgmax |
| blockspecReuseStubHost | decoder/blockspec_reuse_test.go | host (runtime contract only) | ResidentDrafterHost, legacyLastNArgmax |
| widthHost | decoder/blockspec_width_test.go | host (runtime contract only) | ResidentDrafterHost, legacyLastNArgmax |
| windowResident | decoder/cachestate_kvmgr_test.go | embeds `slotResident` | legacyCapsReporter, legacyKVSlotter, legacyMRoPE, legacyReusableKV |
| slotHiddenLastResident | decoder/embed_slot_test.go | embeds `fakeResident` | legacyCapsReporter, legacyHiddenLast, legacyKVSlotter, legacyMRoPE, legacyResidualAll |
| flintWindowResidentStub | decoder/flint_parity_test.go | root fake | legacyReusableKV |
| uploadFailResident | decoder/gemma4_vl_resident_test.go | embeds `fakeResident` | legacyCapsReporter, legacyMRoPE |
| identResident | decoder/generate_identity_test.go | root fake | legacyCapsReporter |
| identArgmax | decoder/generate_identity_test.go | embeds `identResident` | legacyCapsReporter, legacyGreedy |
| identChain | decoder/generate_identity_test.go | embeds `identArgmax` | legacyCapsReporter, legacyGreedy, legacyGreedyChain |
| identTopK | decoder/generate_identity_test.go | embeds `identResident` | legacyCapsReporter, legacyTopK |
| identSample | decoder/generate_identity_test.go | embeds `identResident` | legacyCapsReporter, legacySample |
| identSampleChain | decoder/generate_identity_test.go | embeds `identChain` | legacyCapsReporter, legacyGreedy, legacyGreedyChain, legacySample, legacySampleChain |
| residentForwardOnly | decoder/generate_vl_resident_test.go | root fake | legacyCapsReporter |
| chainGateFake | decoder/hl_chain_gate_test.go | embeds `fakeResident` | legacyCapsReporter, legacyGreedy, legacyGreedyChain, legacyMRoPE |
| donorFake | decoder/kv2_donor_test.go | embeds `copyFake` | Prefiller, legacyBatchStepper, legacyCapsReporter, legacyKVCopier, legacyKVSlotter, legacyMRoPE, legacyReusableKVSlot, legacySample, legacyTopK |
| twoClassFake | decoder/kv2_fu1_test.go | embeds `copyFake` | PrefillTailExact, Prefiller, legacyBatchStepper, legacyCapsReporter, legacyKVCopier, legacyKVSlotter, legacyMRoPE, legacySample, legacyTopK |
| copyFake | decoder/kv2_share_test.go | embeds `mc3Fake` | Prefiller, legacyBatchStepper, legacyCapsReporter, legacyKVCopier, legacyKVSlotter, legacyMRoPE, legacySample, legacyTopK |
| recurrentFake | decoder/kv8a_generate_test.go | embeds `fakeResident` | Prefiller, legacyCapsReporter, legacyGreedyChain, legacyMRoPE, legacyRecurrentCheckpoint |
| slottedRecurrentFake | decoder/kv8a_generate_test.go | embeds `recurrentFake` | Prefiller, legacyCapsReporter, legacyGreedyChain, legacyKVSlotter, legacyMRoPE, legacyRecurrentCheckpoint |
| oneCopyRecurrentFake | decoder/kv8a_generate_test.go | embeds `slottedRecurrentFake` | Prefiller, legacyCapsReporter, legacyGreedyChain, legacyKVSlotter, legacyMRoPE, legacyRecurrentCheckpoint |
| snapFake | decoder/kvsnap_test.go | embeds `mc3Fake` | Prefiller, legacyBatchStepper, legacyCapsReporter, legacyKVSlotter, legacyKVSnapshotter, legacyMRoPE, legacyPrefillMarked, legacySample, legacyTopK |
| fakeGreedyResident | decoder/logit_gate_test.go | embeds `fakeResident` | legacyCapsReporter, legacyGreedy, legacyMRoPE |
| mc3Fake | decoder/mc3_batch_test.go | embeds `fakeResident` | Prefiller, legacyBatchStepper, legacyCapsReporter, legacyKVSlotter, legacyMRoPE, legacySample, legacyTopK |
| erroringResident | decoder/model_cancel_commit_test.go | root fake |  |
| blockFake | decoder/prefill_blocks_test.go | embeds `mc3Fake` | PrefillTailExact, Prefiller, legacyBatchStepper, legacyCapsReporter, legacyKVSlotter, legacyMRoPE, legacyPrefillBlocks, legacySample, legacyTopK |
| chunkFake | decoder/prefill_chunks_test.go | embeds `mc3Fake` | Prefiller, legacyBatchStepper, legacyCapsReporter, legacyKVSlotter, legacyMRoPE, legacyPrefillChunks, legacyPrefillSpanScratch, legacySample, legacyTopK |
| decliningPrefiller | decoder/prefill_decline_warn_test.go | embeds `fakeResident` | Prefiller, legacyCapsReporter, legacyMRoPE |
| prefillingResident | decoder/prefillpath_test.go | embeds `fakeResident` | PrefillPathReporter, Prefiller, legacyCapsReporter, legacyMRoPE |
| prefillerOnly | decoder/prefillpath_test.go | root fake | Prefiller |
| stubResidualAll | decoder/prompt_hidden_all_test.go | root fake | legacyResidualAll |
| ckptResident | decoder/recurrent_checkpoint_test.go | embeds `slotResident` | legacyCapsReporter, legacyKVSlotter, legacyMRoPE, legacyRecurrentCheckpoint |
| ckptWindowResident | decoder/recurrent_checkpoint_test.go | embeds `ckptResident` | legacyCapsReporter, legacyKVSlotter, legacyMRoPE, legacyRecurrentCheckpoint, legacyReusableKV |
| recurrentSpecStub | decoder/recurrent_spec_test.go | host (runtime contract only) | legacyRecurrentSpec |
| prepHost | decoder/recurrent_spec_test.go | host (runtime contract only) | ResidentDrafterHost, legacyLastNArgmax, legacyRecurrentSpec |
| foldHost | decoder/recurrent_spec_test.go | host (runtime contract only) | ResidentDrafterHost, legacyLastNArgmax, legacyRecurrentSpec |
| captureHost | decoder/recurrent_spec_test.go | host (runtime contract only) | ResidentDrafterHost, legacyLastNArgmax |
| fakeResidentAdapter | decoder/resident_adapter_seam_test.go | embeds `fakeResident` | Prefiller, legacyAdapter, legacyCapsReporter, legacyMRoPE |
| fakeResidentHiddenLast | decoder/resident_embed_seam_test.go | embeds `fakeResident` | legacyCapsReporter, legacyHiddenLast, legacyMRoPE |
| fastPrefillResident | decoder/resident_leadreuse_test.go | embeds `slotResident` | legacyCapsReporter, legacyKVSlotter, legacyMRoPE |
| noCapResident | decoder/resident_reporters_test.go | root fake |  |
| fastFloorResident | decoder/resident_reporters_test.go | embeds `fakeResident` | legacyCapsReporter, legacyMRoPE |
| contextNoteResident | decoder/resident_reporters_test.go | embeds `fakeResident` | legacyCapsReporter, legacyMRoPE |
| kernelFloorResident | decoder/resident_reporters_test.go | embeds `fakeResident` | legacyCapsReporter, legacyMRoPE |
| equivResident | decoder/resident_reuse_equiv_test.go | embeds `fakeResident` | legacyCapsReporter, legacyKVSlotter, legacyMRoPE, legacyReusableKV |
| uploadPassFails | decoder/resident_reuse_test.go | embeds `fakeResident` | legacyCapsReporter, legacyKVUploadPass, legacyMRoPE |
| fakeResident | decoder/resident_seam_test.go | root fake | legacyCapsReporter, legacyMRoPE |
| slotResident | decoder/resident_slots_test.go | embeds `fakeResident` | legacyCapsReporter, legacyKVSlotter, legacyMRoPE |
| uncomparableValueResident | decoder/resident_subsystems_test.go | root fake | VerifyCostReporter |
| rideFake | decoder/rf3_ride_test.go | embeds `mc3Fake` | Prefiller, legacyBatchStepper, legacyCapsReporter, legacyKVSlotter, legacyMRoPE, legacyRaggedStepper, legacySample, legacyTopK |
| costReportingResident | decoder/spec_adaptive_cost_test.go | root fake | VerifyCostReporter |
| fakeThetaResident | decoder/spec_adaptive_test.go | embeds `fakeResident` | PrefillPathReporter, Prefiller, legacyCapsReporter, legacyMRoPE |
| fakeVerifyResident | decoder/spec_adaptive_test.go | embeds `fakeResident` | VerifyPathReporter, legacyCapsReporter, legacyMRoPE |
| stubResident | decoder/spec_ngram_test.go | root fake |  |
| divergingResident | decoder/spec_verify_guard_test.go | embeds `fakeResident` | DecodeVerifyDiverger, legacyCapsReporter, legacyMRoPE |
| quantReportingResident | decoder/staged_device_note_test.go | embeds `fakeResident` | legacyCapsReporter, legacyMRoPE |
| callLogResident | decoder/step_callers_test.go | embeds `fakeResident` | legacyCapsReporter, legacyGreedy, legacyLastNArgmax, legacyMRoPE, legacySample, legacyTopK |
| callLogHost | decoder/step_callers_test.go | host (runtime contract only) | ResidentDrafterHost, legacyLastNArgmax |
| callLogSeedHost | decoder/step_callers_test.go | host (runtime contract only) | ResidentDrafterHost, legacyLastNArgmax, legacySeedArgmax |
| stepLogResident | decoder/step_test.go | root fake | Prefiller, ResidentHybridMRoPEPrefill, legacyBatchStepper, legacyGreedy, legacyHiddenLast, legacyImageBlocksPrefill, legacyImagePrefill, legacyLastNArgmax, legacyMRoPE, legacyMRoPEDeepstackPrefill, legacyMRoPEPrefill, legacyPrefillKV, legacyPrefillMarked, legacyPrefillRows, legacyRaggedStepper, legacyResidualAll, legacySample, legacySeedArgmax, legacySpanPrefill, legacyTopK |
| stepPlainResident | decoder/step_test.go | root fake |  |
| stepPrefillOnly | decoder/step_test.go | embeds `stepPlainResident` | Prefiller |
| stepNoAllocResident | decoder/step_test.go | embeds `stepPlainResident` | legacyGreedy, legacySample, legacyTopK |
| fakeResident | internal/app/serveapp/residentpath_test.go | root fake (runtime contract only) | ResidentCapsReporter |

### Result

- `SubsystemsOf` returns a `ResidentSubsystemProvider`'s own subsystems and the no-op subsystems for anything else (nil included). The four bridge types, the 38 probes (the doc comment of each moved onto its capability in `decoder/resident_subsystems.go`), `cachedResidentSubsystems`, `sameResident` and `Model.resSubP` are deleted.
- Test fakes: `fakeSubsystems` (the four `Noop*` types; a decoder test helper) is embedded by a fake that serves contract methods; it overrides the methods and `*Available` gates it serves and returns itself from `ResidentSubsystems`. A BlockSpec host embeds `NoopResidentRuntime` and serves the runtime contract only. A fake that models a plain resident (`stepPlainResident`, `noCapResident`, `erroringResident`, `stubResident`, `residentForwardOnly`) stays a non-provider and gets the no-op subsystems. A fake that embeds another through a pointer a test may leave nil embeds `fakeSubsystems` too, so the defaults stay reachable.
- Retyped: `specRollbackSafeVia` takes a `ResidentRuntimeSubsystem`, `kvOtherDonors` a `ResidentKVSubsystem`, and the four `Resident*PrefillForTest` hooks take `any` (a resident; they resolve the prefill contract inside).
- No contract gained or lost a method. The exported surface is unchanged: the four contracts, `ResidentSubsystemProvider`, and the six interfaces with reasons above.
- A test that checked only the bridge: `TestResidentSubsystems_DirectProviderAndZeroAlloc` asserted that `providerFakeResident` does not implement `legacyGreedy`; it now asserts the outer struct is not a `ResidentDecodeSubsystem`. `bannedMicroAssertions` drops the 38 names (a new assertion on one cannot compile) and keeps the exported ones.

## MZ5 option B: the resident contract leaf (plan, 2026-11-15)

Francis's word 2026-11-15 (the options: a private task doc, row SUB1 and its MZ5 sentence): move the contract a backend implements into a small leaf, aliased from decoder; behavior unchanged. Option C (the weight types, `Options`) stays parked.

**What moves.** A name-level closure over the package-level declarations of decoder, rooted at the four subsystem contracts, `ResidentSubsystemProvider`, `ResidentSubsystems`, `ResidentForward` and `ResidentDrafterHost`, reaches 26 types and nothing else: no function, no constant, no variable, no method of `Model`, and none of `Model`, `Options`, `Weights`, `LayerWeights`, `Config`, `Architecture` or `KVCache`. They are the four contracts, the provider and the subsystems struct, `ResidentForward` (the base every resident implements), `ResidentDrafterHost` with `ResidentBlockDrafter`, `BlockDrafterWeights`, `DrafterGeometry` and `DrafterLayerWeights`, `ResidentKVSnapshot`, and the small data the signatures need (`TopKRow`, `ResidentBatchSeq`, `ResidentBatchDraw`, `ResidentBatchOut`, `ResidentRaggedSeq`, `ResidentRaggedPrefill`, `ResidentRaggedOut`, `ResidentCaps`, `ExpertCacheStats`, `KVHostBudget`, `KVPoolDetail`, `PrefillTailExact`). `ResidentAdapterLayer` is already an alias of `internal/runtime/adapter`'s type. The leaf imports `internal/cost` and `internal/runtime/adapter` and nothing else of goinfer.

**What does not move, and why.** `ResidencyBackend` and `ResidencyPricer` name `*Model`; `BackendModelView` returns `*Weights` and `Layer` views over `Weights`, so the view cannot move either (that is option C). `SubsystemsOf` and the `legacy*` probes stay in decoder: they are the bridge for test fakes. `StreamingDrafterHost` embeds `ResidentDrafterHost` and stays beside `BlockSpec`. No contract method's signature drags in a non-movable type, so B needs no narrower cut.

**Name and place.** `decoder/resident` (package `resident`), after `decoder/fit` and `decoder/config`: under `decoder/`, a leaf, aliased from decoder in `decoder/residentalias.go` so no caller changes in PR 1. In the leaf the types drop the `Resident` prefix (`resident.DecodeSubsystem`, `resident.BatchSeq`); decoder keeps the old names as aliases. Layout rule `resident-leaf`: `decoder/resident` may import `internal/cost` and `internal/runtime/adapter` and no other goinfer package; decoder, the backends and the tests may import it.

**PRs.** (1) the leaf, the aliases, the layout rule and the docs rows. (2) `ResidencyBackend.BuildResident` and `ResidencyPricer.PriceResident` take `decoder.BackendModelView` instead of `*Model` (the view stays in decoder; it cannot move), backends updated. (3) the seam's proof: one backend's tests that load a checkpoint for geometry take a fake view instead; what it saves is stated there. Each is a small PR and carries the SUB1 row's update.

**Outcome (Francis, 2026-11-15: "2"):** B stops after PR 1. The probe behind PRs 2 and 3: switching `BuildResident` and `PriceResident` to `decoder.BackendModelView` needs the view (47 methods) to grow by about 55 forwarding accessors, because the builders read `Model` directly (distinct missing methods: WebGPU 11, CUDA 31, Metal 39; 101 non-test uses of `*decoder.Model` in the backends; about 40 backend test call sites pass a `*Model` to `BuildResident`). The signature stays `*Model`. Revisit when a backend test needs a fake model, with a narrow interface per backend as the likely shape.
