# MEM12-scratch: MoE output, LM-head logits and sampler scratch reused, 2026-10-21

Gates: `docs/tasks/task-memory-campaign-2026-10.md`, §4 MEM12, "MEM12-scratch gates" (SCR-G), registered before the code.
Machine: MacBook Pro, M1 Pro, 16 GB, macOS 26.6.2, Go 1.27.2, darwin/arm64; the owner's desktop open. Main arm: a detached
worktree of `origin/main` at 35006c8d with this branch's test files copied in (the sampler test without its `Release`).

| gate | reading (final build, `GOMAXPROCS=1`) | result |
|---|---|---|
| SCR-G1 MoE | **As amended 2026-10-22 (per buffer, the owner's call):** each of the eight expert-major buffers is allocated exactly once in the 64-token, 16-row-chunk prefill (`g1-per-buffer.log`). As first registered (the summed bar, kept as context): main 410,384 bytes after the first chunk, the first build 94,592, a fall of 315,792 against 138,240; with the expert buffers sized by the chunk's rows, 70,016 | **PASS** (per buffer) |
| SCR-G2 logits | `llama-tiny`, mean of 50 warm iterations: verify round (K = 4) main 80,288, branch 76,192, a fall of 4,096 = 4 × 256 × 4 (bar 4,055); prefill main 222,400, branch 221,376, a fall of 1,024 = 256 × 4 (bar 1,014) | **FAIL as registered; PASS as amended** |
| SCR-G3 sampler | vocabulary 151,936: request 1 5,626,864 bytes, requests 2–10 92,488–105,352 (≤ 1.9%, bar 5%); main 5.58–5.59 MB every request | **PASS** |
| SCR-G4 identity | `TestMEM12Scratch_identity` (chunked prefill and verify rows byte-equal to the sequential decode on `mixtral-tiny` with expert-major on and on `llama-tiny`, twice on one cache); 36 speculative, prompt-many and f16-KV tests PASS, including `TestSpeculativeGreedyParity_draftTarget` on the 1.5B/0.5B pair from `~/models` (`g4.log`); `internal/sampling` ok; `scripts/refresh_parity_hashes.sh` green, goldens 33 passed, 0 failed, 31 skipped | **PASS**, with three skips below |

**G2, read twice, as it happened.** First reading, a single `TotalAlloc` difference with the scheduler free: the round fell
3,904 and the prefill 832, both short (`g1-g2-first.log`). Two findings came out of it. (1) The build embedded the MoE
scratch in `prefillPassScratch` by value, so every pass, dense ones included, allocated about 200 bytes more; it is now a
pointer started by the first MoE layer (`prefillPassScratch.moeScratch`). That was a defect of the build, and the gate found it.
(2) The instrument could not resolve the prefill's 1,024 bytes: main itself read 222,400 and 223,840 on two runs, and a
mean of 50 still moved by 340 bytes between runs. With `GOMAXPROCS=1` both arms read identically to the byte on three runs
each, which points at per-P `sync.Pool` caches whose hits depend on scheduling. **Amendment, after the reading:** G2 is the
mean of 50 warm iterations at `GOMAXPROCS=1`, on both arms; the bars are unchanged. G1 and G3 were read the same way and
clear their bars by a wide margin in every reading.

**Skips in G4, named:** `TestMoEExpertMajor_bitIdentical` needs the Mellum checkpoint (not on this Mac); its question,
expert-major against per-row, is the one `TestMEM12Scratch_identity` asks on `mixtral-tiny`, through the scratch's reuse.
`TestNgramSpecMLA_parity` needs a multi-GB DeepSeek load (`GOINFER_HEAVY_TESTS`), not run. `TestMoEExpertMajor_endToEnd`
is a timed A/B, not an identity test.

Planted defects, run once each on the committed tree and reverted. The MoE branch allocating its output again, as
registered: **G1 stayed green** (143,744 bytes, a fall of 266,640): the output is 49,152 of the 153,600 arithmetic bytes, so
a bar on their sum cannot see that term alone. The whole MoE path allocating again (output and `moeMLPBatchWith`'s buffers):
410,384 bytes, main's figure, G1 red. The LM head allocating again: 80,288 and 222,400, main's figures, G2 red. The release
not returning the scratch: every request 5.58 MB, G3 red.

The grammar speculative loop has no sampler (greedy only), so the release sits in `generateInto` and the n-gram loop.

**Found by PR #135's CI, then fixed (2026-10-21, before merge).** (1) `TestQwen3ASR_tinyTextParity` and
`TestVoxtral_tinyTextParity` failed: each keeps `prefillLogits`'s row and runs decode forwards before reading it, and the
build had put that row in the decode logits buffer, which the forwards overwrite. The SCR-G4 consumer audit had checked the
production callers only. The prefill row now goes in the cache's batched-pass buffer (`KVCache.headLogits`, the one a verify
round uses), never the decode buffer, so it outlives decode forwards and is reused only by the next batched prefill or
verify; both tests pass, SCR-G2 reads the same bytes (76,192 and 221,376 at `GOMAXPROCS=1`), and the whole `decoder`
package passes locally (944 PASS, 164 skips, 0 FAIL, 17.6 min). (2) `TestSamplerScratch_pooled` missed the pool on two of
nine requests on Linux: `sync.Pool` caches per P and the race detector drops a share of puts on purpose. The test now pins
one P and skips under `-race`; the gate's reading above, taken without the pin, is unchanged.

**SCR-G1 re-registered per buffer (2026-10-22, the owner's call, before the re-read).** The summed bar could not see one
term alone: the output-only plant left it green. Each buffer now counts its own fresh allocations (`moeScratchGrows`), and the
bar is exactly one each over the whole prefill. That needed the gather, gate, up and down buffers sized by the chunk's rows
(the bound on any expert's rows) rather than the busiest expert's rows in that chunk, which can grow from chunk to chunk;
the bytes after the first chunk fell again, 94,592 to 70,016. Plants, one per buffer, each run once and reverted: every
plant turned its own row red (16 allocations: 4 chunks x 4 layers) and left the other seven at 1, the output-only plant
included. The parity refresh is green (goldens 33/0/31).
