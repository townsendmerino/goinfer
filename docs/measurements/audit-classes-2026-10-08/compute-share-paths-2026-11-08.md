# The audit classes' compute share, mapped to program paths (2026-11-08)

AC (ledger row in `docs/tasks/task-forward-and-memory-2026-11.md`). Read against main `5c5fe5db2`.

The question comes from the owner's hybrid of 2026-10-21 (`docs/completed/task-forward-architecture-2026-10.md`'s status block). It
concerns the 20 family-by-path findings of the 71 counted on 2026-10-08 (`../audit-classes-2026-10-08.md` §2, `final.json`'s
`rb_kind` = `family`). The 7 on loading, serializing, tokenizer dispatch and chat-template detection are `docs/queue-correctness.md`
G47's. The other 13 are the compute, fit-guard and speculative share. For each, the table says what decides that path today, and
whether it is the program (`decoder/program.go`, `internal/runtime/program`) or still a hand list.

| finding | family × the path that missed it | what decides it now | the program? |
|---|---|---|---|
| 08-05 C-02 | recurrent targets × `GenerateSpeculative`'s rollback check | `Model.specRollbackSafeVia` reads `hasRecurrentState`, which is `Architecture.recurrent`: a walk of every layer's program ops for a recurrent kind. `decoder/cachestate.go`'s lifecycle table states the rule per state kind (`lcSpecRollback`). | yes |
| 08-05 M-07 | families without a capture seam × the EAGLE entry guard | The path is gone: EAGLE-3 was removed (`be9aeea81`, no callers). The capture consumer left, `Model.ForwardSubCapture`, refuses by `program.SubCaptureUncovered`, held by `TestSeams_subCaptureRefusedWhereGateSCannotRun`. | yes, for what is left |
| 09-02 C-01 | LFM2 × batched-prefill admission (`canBatchN`) | `Model.canBatchN` is `program.BatchEligible`. | yes |
| 09-02 M-01 | Gemma 4 MoE, Qwen3.5, Mamba-2, MLA, LFM2's conv, PLE × `ResidentWeightBytes` and Metal's fit guard | `Model.residentWeightBytesSplit` sums a hand list of `LayerWeights` matrices and the experts, plus `denseVectorBytes`. A miss is caught per registered fixture by the memory gates, which hold each plan line to the build's own ledger (`TestMemGate_metal`, `TestMemGate_webgpu`), not by a declaration. | no: a hand list, gated |
| 09-02 M-14 | MoE, K=V, non-uniform, non-int4 on CUDA × `verifyTheta` | `Model.verifyTheta` asks the resident whether its verify forward is batched (`VerifyPathReporter`, else `canStep`); there is no family list. | the resident's own report |
| 09-10 C-04 | Cohere × CUDA's tail norm dispatch, `prefillStaticDecline`, the tail logit scale | CUDA's layer steps come from the program's ops (`cudaSublayersOf`, `cudaPrefillStepsOf`), and its admission reads `program.CUDA` (OP4cu-c2, FA §4.9a). The logit scale is `FeatLogitScale`, a host-side multiply. | yes (steps and admission); a feature (the scale) |
| 09-10 M-08 | the families and backends `ResidentEligible` declines × `Model.Plan`'s admission | `Model.planIneligible` is `planAdmitDecline` first: `Admit` over the backend's program tables. WebGPU keeps two hand checks after it: the routed experts' kind, and a family list (Nemotron, gated Qwen3.5, MLA) for the KV-precision flags. | yes, but for WebGPU's two |
| 09-10 M-23 and Metal 09-12 M-06 | Gemma 3 (`FeatPerLayerRoPE`) × Metal's batched-prefill admission | Metal's f16 MMA prefill admits by a feature set (`prefillFeatures`, which now holds `FeatPerLayerRoPE`) against decoder's feature registry, plus two named declines (per-layer geometry, Gemma 4's MoE block). | no: the feature registry |
| 09-10 M-28 | hybrids, sliding window, MLA × the fit guard's KV (`kvBytesPerPosition`, `AdmitPrefillMemory`) | The host estimate resolves the architecture and sums per-layer KV (`estimateKVBytes` into `kvBytesForCtx`: `kvDimAt`, KV-shared layers). Metal's KV counts the rows each layer's program declares (`metalKVParts` over `LayerBuffers`). | Metal yes; the host by per-layer architecture |
| 09-10 M-29 | f32-pinned Mamba-2 and MLA mixers, GPTQ/AWQ × the fit guard's weight estimators | `estimateGGUFWeightBytes` and its safetensors twin price by tensor name and architecture (`fit.GGUFWeightBytes`), at rates from the loader's own quantizer rules (`fitRates`: `MambaProjInt8`, `MLAProjInt8`). | no: name-driven |
| 09-10 M-34 | gpt-oss, Llama 4, Nemotron 3 Nano × the expert pager's touch hooks | The program's MoE op touches through `moeMLP` (`decoder/program_cpu.go`). The three own-forward loops touch by hand (`decoder/forward_gptoss.go`, `forward_llama4.go`, `forward_nemotron.go`). | yes for program families; by hand for own-forward ones |
| 09-10 M-35 | Nemotron-H MoE with no shared expert × WebGPU's `BuildResident` branch | WebGPU's resident build still branches by family (`Model.NemotronResidentParams` and its siblings). OP4-W, WebGPU on the program, was killed: it stays on the RF1 adapter. | no, by decision |

## What is left

Five of the 13 are not read from the program, or only partly.
- **09-02 M-01:** the weight sum is a hand list. The memory gates catch a miss on every registered fixture, but not on a family with no fixture.
- **09-10 M-23 and 09-12 M-06:** Metal's f16 MMA prefill admits by its feature set, not by the program's ops.
- **09-10 M-29:** the weight estimators read tensor names.
- **09-10 M-34:** the own-forward MoE loops wire the pager by hand.
- **09-10 M-35:** WebGPU's families stay hand-branched, by OP4-W's kill.

Plus WebGPU's two checks beside `Admit` (09-10 M-08).

Each is a place a new family can be missed again. Whether any becomes work is the owner's call. The candidates, smallest first:
- a test that every own-forward MoE loop touches the pager;
- Metal's f16 prefill admission read from the program's ops;
- the weight sum read from the program's declared weights.
