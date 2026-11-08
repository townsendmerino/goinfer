# Forward-architecture prior art, the part OP0 left for RF0 (2026-11-07)

**Status: a reading, not a measurement.** It closes the prior-art half of RF0
(`docs/tasks/task-forward-architecture-2026-10.md` §7a and the "Not read, and left for RF0" paragraph at the end of §10a). Nothing was
run from any of these repositories and nothing here is a gate. Each source was read from a shallow clone at the revision below by
three read-only readers, one per group of sources, and then spot-checked: ten load-bearing claims were looked up again in the clones by
`grep` (the vLLM scheduler's "no decoding phase" comment, `HolisticPlanInfo` and `CTA_TILE_Q_SIZES` in FlashInfer's `scheduler.cuh`, the
POD wrapper's hard-coded `logits_soft_cap = 0.0`, `GGML_VK_DISABLE_FUSION`, llama.cpp's recurrent `seq_rm` comment, Ollama's unbuffered
`Requests` channel, TVM's `is_decode_request_`, flash-attention's `page_block_size % 256` check, Sarathi-Serve's
`chunk_size - num_batched_tokens`) and all ten were found as reported. The rest is the readers' account and is labeled so.

Labels: **READ** is something the code says; **INFERRED** is a reading of it for goinfer. A citation names the file and then the declaration in it, never a line; these are other projects' files, so they are not goinfer citations.

| source | revision (commit date) |
|---|---|
| vLLM | `fd8f3f3` (2026-11-07) |
| FlashInfer | `e14ddfd9` (2026-11-08) |
| flash-attention | `b2edfab` (2026-11-07) |
| llama.cpp | `f477f79` (2026-11-08) |
| Ollama | `d63b702` (2026-11-07) |
| Sarathi-Serve | `96f9911` (2026-01-08) |
| TVM (Relax, the MLC runtime) | `1fa4b86` (2026-11-08) |

## What RF2 and RF3 take from it (INFERRED, in one place)

- **One varlen call over paged KV is how the mainstream backends run a mixed batch.** vLLM's FlashAttention and Triton backends make one call with per-row `query_start_loc`, `seq_lens` and a block table; splitting decode rows from prefill rows is a per-backend optimization (FlashInfer's builder does it), not a requirement. RF2's "decode, verify and prefill rows in one pass" is the shape the flat-batch design already has.
- **Verification rows are multi-token query rows with several logit rows.** vLLM gives request r `num_draft_r + 1` logit rows at the end of its query span, and orders rows verify/decode, then short extend, then prefill.
- **The scheduler needs no phase.** vLLM assigns tokens so that `num_computed_tokens` catches up with `num_tokens_with_spec`; chunking, prefix hits and speculation are the same rule. RF3's token budget is that rule plus a cap.
- **Graphs need a ladder, not one shape.** vLLM keeps uniform decode graphs keyed by (request count, query length), and token-count graphs for the rest, and pads the batch to the nearest captured size. RF2's fixed-grid requirement (FlashInfer's plan) is the same constraint seen from the kernel.
- **A fused mixed kernel is rare.** Only FlashInfer has one (POD), and its softcap and sink are not wired. TVM/MLC picks a separate decode or prefill kernel per call shape. flash-attention has none.
- **Declared fusion tables are goinfer's own design, not borrowed.** llama.cpp's Vulkan fusions are hard-coded in the compute loop with per-pattern and per-device predicates; there is no list a backend declares.

## 1. vLLM (V1 runner and scheduler)

The runner is the v2 layout under `vllm/v1/worker/gpu/`; `gpu_model_runner.py` and `gpu_input_batch.py` do not exist at this revision.

READ:
- **The flat batch.** `vllm/v1/worker/gpu/input_batch.py`, `InputBuffers` preallocates `input_ids`, `positions`, `query_start_loc`, `seq_lens` and a padding mask. `model_runner.py`, `GPUModelRunner.prepare_inputs` builds `query_start_loc` on the CPU and fills the rest with Triton kernels (`prepare_prefill_inputs`, `prepare_pos_seq_lens`, `combine_sampled_and_draft_tokens`).
- **Block table and slots.** `GPUModelRunner.prepare_attn` gathers a per-request block table (`block_table.py`, `BlockTables.gather_block_tables`) and computes slot mappings (`compute_slot_mappings`: `block_table[req, pos // block_size] * block_size + pos % block_size`), with a pad slot for padding rows.
- **Which rows get logits.** `GPUModelRunner.sample` takes `hidden_states[input_batch.logits_indices]`. Without drafts, one row per request (its last query token); with drafts, `num_draft_r + 1` rows at the end of the span. A prefill-chunk row's logit is ignored unless the chunk ends the prompt.
- **Row order.** `model_runner.py`, `sort_batch_req_ids` sorts by (has draft, query_len != decode_query_len, query_len); its comment reads verification/decode, then short extend, then prefill.
- **Attention metadata.** `vllm/v1/attention/backend.py`, `CommonAttentionMetadata` carries `query_start_loc`, `seq_lens`, `max_query_len`, `block_table_tensor`, `slot_mapping`, `causal`, `is_prefilling`, `positions`. `attention/backends/utils.py`, `split_decodes_and_prefills` splits at the first row with `query_len` above `reorder_batch_threshold` (default 1; `max(1, max_decode_query_len)` where speculation counts as decode). `attention/backends/flashinfer.py`, `FlashInferMetadataBuilder.build` uses the split; `attention/backends/flash_attn.py`, `FlashAttentionMetadataBuilder` makes one `flash_attn_varlen_func` call over the whole batch, and `triton_attn.py` one `unified_attention` call.
- **Scheduler.** `vllm/v1/core/sched/scheduler.py`, `Scheduler.schedule`: its comment says there is no decoding or prefill phase; each request has `num_computed_tokens` and `num_tokens_with_spec` and tokens are assigned so the first catches up with the second. Running requests are scheduled first, then waiting ones, and waiting ones are admitted only if nothing was preempted this step, the running count is under `max_num_active_reqs` and budget remains. A request's tokens are capped by `long_prefill_token_threshold` (default 0, off; an adaptive variant floors it at `max_num_batched_tokens // num_eligible_reqs`), then by the remaining budget. New decode-like requests are padded to `1 + num_spec_tokens` placeholders to keep full-graph batches uniform.
- **KV exhaustion.** `allocate_slots` returning none triggers preemption in a loop: first policy pops the last-admitted running request, priority policy takes the worst (priority, arrival); a victim's blocks are freed, `num_computed_tokens` returns to 0 and it goes to the front of `waiting`.
- **Graphs.** `vllm/v1/attention/backend.py`, `AttentionCGSupport` (ALWAYS, UNIFORM_BATCH, UNIFORM_SINGLE_TOKEN_DECODE, NEVER); `flash_attn.py` is ALWAYS for FA3/FA4 and UNIFORM_BATCH for FA2; `worker/gpu/cudagraph_utils.py`, `CudaGraphManager` keys uniform decode graphs by (n, q) and the others by token count, and falls back to eager when no captured size covers the batch.

NOT READ: the older V1 runner layout (absent here), and the rejection sampler's body.

## 2. Sarathi-Serve (code, against the paper §10a already read)

READ:
- **Fill order.** `sarathi/core/scheduler/sarathi_scheduler.py`, `SarathiScheduler._schedule`: running sequences that finished prefill first (one token each, preempting the lowest priority when no slot can be appended), then running sequences mid-prefill, then new waiting ones if they can be allocated and the running count is under the limit.
- **Chunk.** `_get_seq_next_num_prefill_tokens` returns `min(prompt_len - processed, chunk_size - num_batched_tokens)`: one budget shared with the decodes.
- **The budget is a configured value.** `sarathi/config/config.py`, `SarathiSchedulerConfig.chunk_size` defaults to 512; the capacity-search configs sweep fixed values 256 to 4096. `profile_num_available_blocks` profiles KV memory only.
- **Rounding.** The fixed path rounds nothing; the last chunk is the remainder. The optional dynamic schedule (`_compute_chunk_size_schedule`, experimental per its README) asserts `low/high_chunk_size % 32 == 0`.
- **Executor.** `sarathi/model_executor/attention/flashinfer_attention_wrapper.py` runs two batch-prefill wrappers, one on the prefill slice and one on the decode slice; `model_runner.py`, `_prepare_inputs` lays out prefill chunks, then decodes, padded to a multiple of 8. No CUDA-graph code.

**A correction to §10a.** Its table lists, from the paper, "a budget profiled once against the time-between-tokens target" and "chunk sizes rounded to the kernel's tile". The code has neither: the budget is a configured number and the tile rounding exists only in the experimental dynamic schedule. Those two takes are the paper's design, not the repository's.

## 3. FlashInfer

READ:
- **Two POD paths.** `include/flashinfer/attention/pod.cuh`, `PODWithKVCacheTensorKernel` (single prefill plus batch decode) and `batch_pod.cuh`, `BatchPODWithKVCacheTensorKernel` (batch prefill plus batch prefill, the "decode" half using the prefill device function with a small tile). Python: `flashinfer/pod.py`, `PODWithPagedKVCacheWrapper`, `BatchPODWithPagedKVCacheWrapper`.
- **The split is a runtime SM-aware scheduler.** In `BatchPODWithKVCacheTensorKernel`, one thread reads `%smid` and increments a per-SM counter in a `num_sm + 2` int buffer that is zeroed before each launch; the counter assigns the CTA to prefill or decode work round-robin in the planned ratio, and a CTA whose side is exhausted takes the other. The caller, not the kernel, decides which requests are prefill and which are decode: `BatchPODWithPagedKVCacheWrapper.plan` takes separate `_p` and `_d` index arrays.
- **Host plan.** `include/flashinfer/attention/scheduler.cuh`, `PrefillPlanImpl` and `PrefillSplitQOKVIndptr`: choose `cta_tile_q` once (`utils.cuh`, `FA2DetermineCtaTileQ`: 16 when the average packed query length is at most 16, covering decode and speculative decode), expand (request, q-tile, kv-chunk) into work items, pick the kv chunk size by binary search (`PrefillBinarySearchKVChunkSize`) so the item count stays within `(2 * SMs - num_colocated_ctas) / num_kv_heads`, and copy the plan to the device from a pinned buffer. Under graph mode the grid is `padded_batch_size = max(max_batch_size_if_split, total_num_tiles_q)` with fixed user buffers (`qo_indptr_buf`, `paged_kv_indptr_buf`, `paged_kv_indices_buf`, `paged_kv_last_page_len_buf`) in `flashinfer/prefill.py`, `BatchPrefillWithPagedKVCacheWrapper.__init__`; the plan itself cannot run inside a graph.
- **A persistent alternative.** `scheduler.cuh`, `HolisticPlanInfo` with `persistent.cuh` and Python `flashinfer/attention/_core.py`, `BatchAttention`: two tile classes (`CTA_TILE_Q_SIZES = {128, 16}`, rows with packed query length above 16 to the first), work balanced across clusters with a min-cost heap, partial outputs merged. Its plan takes `causal`, `logits_soft_cap` and `window_left`.
- **Paged KV inputs.** `page.cuh`, `paged_kv_t`: `indptr [batch + 1]`, `indices [nnz_pages]`, `last_page_len [batch]`, HND or NHD layouts; the KV length is `(npages - 1) * page_size + last_page_len`. The query side is `q_indptr` in `default_prefill_params.cuh`, `BatchPrefillPagedParams`. The dedicated decode kernel is `decode.cuh`, `BatchDecodeWithPagedKVCacheKernel`; `decode.py` has a `use_tensor_cores` switch that routes decode through the prefill kernel.
- **Variant hooks.** `variant_helper.cuh`: `QueryTransform`, `KeyTransform`, `LogitsTransform`, `LogitsMask`, `update_m_d`, `OutputTransform`. `variants.cuh`, `DefaultAttention` implements soft cap (`LogitsTransform`, tanh with a pre-scale, and a rescaled `sm_scale_log2`) and sliding window (`LogitsMask`), plus ALiBi and custom masks. The attention sink is JIT source in `flashinfer/jit/attention/variants.py` (`attention_sink_fa2_decl`, `attention_sink_fa3_decl`): `update_m_d` folds the per-head sink into the running max and denominator on the first KV tile.
- **Split-KV for decode.** `scheduler.cuh`, `BatchDecodeWithPagedKVCacheWorkEstimationDispatched` does not split when `batch * kv_heads` already fills the grid, otherwise partitions pages (minimum `128 / page_size` per batch row); under graph capture it always uses the split kernel.
- **The POD wrappers do not support soft cap or sink:** `PODWithPagedKVCacheWrapper.plan` sets `logits_soft_cap = 0.0` with the comment that it is not supported.

INFERRED: the ragged-single-kernel analogue for RF2 is the holistic plan (two tile classes, host balancing, merge), not POD, which fuses two separately planned launches to overlap compute-bound and memory-bound work on one SM. Whether RF2 needs that overlap is a separate question for RF2's own gates.

NOT READ: how `BatchAttention` handles a sink (skimmed only).

## 4. flash-attention

READ:
- **FA2.** `flash_attn/flash_attn_interface.py`, `flash_attn_varlen_func`: `cu_seqlens_q`, `cu_seqlens_k`, `max_seqlen_q`, `max_seqlen_k`, `softcap`, `window_size`, `block_table`; `flash_attn_with_kvcache`: per-row `cache_seqlens`, an int32 `block_table [B, max_blocks]`, `num_splits`, `softcap`. `csrc/flash_attn/flash_api.cpp` requires `page_block_size % 256 == 0` for paged KV. The soft cap is a runtime value that selects a template variant (`Is_softcap` in `flash_fwd_launch_template.h`), rejected with dropout. The varlen paged path checks `num_splits <= 1`.
- **FA3.** `hopper/flash_attn_interface.py`: `seqused_q/k`, `page_table`, `scheduler_metadata`, `num_splits`, `pack_gqa`, `softcap` (compile-time, `Has_softcap_`). No page-size multiple check was found in `hopper/flash_api.cpp`; TMA paged loads need `page_size % kBlockN == 0` (`get_pagedkv_tma`), otherwise a non-TMA path runs.
- **Split-KV.** `csrc/flash_attn/flash_api.cpp`, `num_splits_heuristic`: the smallest split count reaching 85% of the best wave efficiency, capped at 128. `hopper/heuristics.h`, `num_splits_heuristic` returns 1 when the blocks already fill 0.8 of the SMs or the KV is at most 4 blocks long; for varlen `hopper/flash_prepare_scheduler.cu` sets a per-batch split count with a 10% margin.

NOT FOUND: attention sinks anywhere in the repository; a mixed prefill+decode (POD) path.

## 5. llama.cpp

READ:
- **Vulkan fusion** (`ggml/src/ggml-vulkan/ggml-vulkan.cpp`). `ggml_backend_vk_graph_compute` tries, node by node, a priority-ordered chain of `ggml_vk_can_fuse(ctx, cgraph, i, {ops...})` and related `ggml_vk_can_fuse_*` predicates, built on `ggml_can_fuse` / `ggml_can_fuse_subgraph`. Fused sequences include multi-add, MUL_MAT(+ADD(+ADD)), MUL_MAT_ID with ADD_ID and MUL, RMS_NORM with MUL, ADD and ROPE (and SET_ROWS), ROPE+SET_ROWS, unary+MUL (GELU, SIGMOID, SILU, SOFTPLUS), SSM_CONV+SILU, and five TOPK_MOE modes. `GGML_VK_DISABLE_FUSION` sets `device->disable_fusion`; a runtime toggle is exposed through `ggml_backend_vk_get_proc_address`.
- **Per-device capability checks exist inside the predicates:** `ggml_vk_can_fuse_topk_moe` needs subgroup arithmetic, shuffle and full-subgroup support and refuses the sigmoid variant on MoltenVK; `ggml_vk_fuse_multi_add` needs `device->multi_add`; `add_rms_fusion` is off on Intel. A fused destination overlapping a non-elided source cancels the fusion.
- **Recurrent memory** (`src/llama-memory-recurrent.cpp`). One cell per sequence per layer (`r_l`, `s_l`), `mem_size * (1 + n_rs_seq)` rows. `seq_rm` accepts a full-range removal; a partial range only as a tail rollback from per-token snapshots (`0 < p0 <= cell.pos < p1`, the cell unshared, no rollback pending, `1 <= rollback <= cell.rs_max`); the comment says Mamba and RWKV "can't have a state partially erased at the end". `seq_cp` shares the source's cell rather than copying. `find_slot` asserts `ubatch.equal_seqs()`.
- **Hybrid memory** (`src/llama-memory-hybrid.cpp`). `init_batch` splits the batch into ubatches once (`split_equal`, keeping the last `1 + n_rs_seq` tokens of each sequence together), then prepares the recurrent cache before the attention cache. `seq_rm` tries the recurrent side first because it may refuse, then attention. Attention KV `seq_rm` is a plain position-range removal.

INFERRED: llama.cpp has no declared per-backend list of operation kinds or fusions; the patterns live in the compute loop and the capability gating inside each predicate. Rejected-draft rollback for recurrent state is real but bounded by `n_rs_seq` snapshot planes.

## 6. Ollama's MLX runner

READ (`mlxrunner/`):
- **No batching.** `server.go` makes an unbuffered `Requests` channel; `runner.go`, `Runner.Run` handles one request at a time in a single goroutine (`runRequest`), embeddings in the same loop. `batch.Batch` is shaped (B, L) with per-row offsets and query lengths, but the pipeline builds single-row batches.
- **Prefix cache.** `prefix_cache.go`, `prefixCache` is a compressed token trie (`cache_trie.go`), one live path with others paged out from snapshots; a request resumes at the deepest match, and whole-state layers (recurrent, rotating) must hold a snapshot there. Prefill snapshots every 8,192 tokens; paged-out bytes are capped at 8 GiB.
- **Recurrent caches.** `cache/recurrent.go`, `RecurrentCache` is not position-sliceable: `Restore` succeeds only when the target equals the current offset or a snapshot's. Speculation rolls back through per-token snapshots (`speculate.go`).
- **Models** are Go per-model implementations over an `mlx.Array` API, registered by name (`model/model.go`, `model/qwen3_5/qwen3_5.go`).

## 7. TVM / Relax (MLC's runtime)

READ:
- **Separate kernels per call shape.** `python/tvm/relax/frontend/nn/llm/kv_cache.py`, `PagedKVCache` passes prefill kernels (`_prefill_kernels.py`, `_attention_prefill`, `_attention_prefill_ragged`) and decode kernels (`_decode_kernels.py`, `_attention_decode`) as separate arguments. `src/runtime/vm/paged_kv_cache.cc`, `BeginForward` sets `is_decode_request_` true and clears it if any `append_lengths[i] != 1`; the dispatch then runs the decode kernel only for an all-decode batch and the prefill kernel otherwise, merging partial states with `f_merge_inplace_`. A mixed batch goes entirely to the prefill kernel.
- **No soft cap.** Searching `python/` and `src/` found `softcap` only in `src/runtime/vm/attn_backend.h`, which passes `logits_soft_cap = 0.0` to FlashInfer's decode plan. This is consistent with the MLC comment §10a already quotes ("We ignore attn_logit_softcapping in the gemma2 implementation for now").
- **Recurrent state** is a separate `RNNState` (`src/runtime/vm/rnn_state.cc`, `PopN`); `LinearAttention` in `paged_kv_cache.cc` is an unimplemented stub.

NOT READ: `rnn_state.cc` beyond finding `PopN`; the TIR kernel bodies.

## What is still not read

The kernel bodies of FlashInfer's and flash-attention's mainloops (the reading covered interfaces, planning and hooks, which is what RF2's gates need); how FlashInfer's persistent `BatchAttention` treats a sink; the rejection sampler in vLLM; TVM's TIR kernel bodies and `rnn_state.cc`. None blocks RF1 or RF2's registration; each is named here so a later reader does not assume it was covered.
