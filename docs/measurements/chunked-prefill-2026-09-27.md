# Chunked prefill under MC3 SHIPS (512-token chunks): Metal prefill is chunk-invariant; the decoders' longest stall during a ~3k-token newcomer's prefill 5.38 s → 1.23 s (0.229×), every reply identical, wall 1.045× (2026-09-27)

Chunked prefill of [`task-concurrency-2026-09.md`](../tasks/task-concurrency-2026-09.md), unparked from MC5 by the
owner on 2026-09-27. Under MC3 ([`concurrency-mc3-2026-09-26.md`](concurrency-mc3-2026-09-26.md)), a newcomer's
prefill runs whole in the resident's exclusive section, so every decoding conversation stops for the entire prompt.
Chunking takes a long prompt in pieces while others decode, yielding one decode step between pieces.

**Result.**
- The revised candidate — **512-token chunks, and a tail rule that never leaves a long final pass** — passes all five
  pre-registered gates. It ships on by default: `serve -prefill-chunk 512`.
- The first candidate (256-token chunks) cut the stall about as much but missed its wall-time gate. It was not
  shipped, and its record follows unchanged.

**Findings.**
1. **Metal's fast prefill is chunk-invariant**, bit for bit: 0 of 14,336,000 KV elements and 0 of 151,936 logits
   differ at every chunk size tried. So chunking changes no reply.
2. The first registered candidate (256-token chunks, `d4293d41`) cut the decoders' longest wait during a ~3k-token
   newcomer's prefill **from 5.35 s to 1.14 s (0.213×)**. It kept every reply identical and left a lone request's
   TTFT untouched (1.000×).
3. **Gate 4 failed:** the cell's wall time grew 1.052× against a ≤ 1.05× bar. By the pre-registered rule it does not
   ship. Chunking is now an explicit `decoder.Options.ResidentPrefillChunk`, 0 (off) by default.
4. **The diagnosis names two costs.**
   - A pass costs a 64-row GEMM tile's worth at minimum: ~70 ms of matmul per 64 tokens on the 1.5B.
   - The candidate's tail rule left the final pass up to 511 tokens long, and that pass was the whole of the longest
     stall.

## 1. Chunk invariance (the question the design turns on)

`TestMC5_prefillChunkInvariance` (`metal/mc5_chunk_test.go`):
- qwen2.5-coder-1.5b int4 on Metal, 2 contiguous f16 KV slots.
- A 1000-token prompt is prefilled whole (`PrefillLast(all, 0)`) into slot 0, then into slot 1 as chunks of C in
  sequence (`PrefillLast(chunk, start)`).
- Every K and V element of every layer, and the last token's logits, are compared bit for bit.

| C | KV elements differing (of 14,336,000) | logits differing (of 151,936) |
|---:|---:|---:|
| 64 | **0** | **0** |
| 128 | **0** | **0** |
| 256 | **0** | **0** |
| 384 (ragged tail) | **0** | **0** |

The control reruns the chunked prefill with one token changed, at position 700. Nothing differs before 700 and
3,746,400 elements differ from it on. The comparison sees a difference and places it.
Log: [`mc5-chunk-invariance.log`](chunked-prefill-2026-09-27/mc5-chunk-invariance.log).

**2026-10-01 addendum: the steel build** (F-G02, `docs/audit-metal-2026-09-30.md`). The table above is from the
fused-kernel build. Since R19 (2026-09-27) production's head-dim-128 prefill attention is `attention_prefill_steel`. The
same model and prompt on the steel build, with three more chunk sizes: 512 (serve's default) and 100 and 77, which
start chunks off the kernel's 32-row tiles:

| C | KV elements differing (of 14,336,000) | logits differing (of 151,936) |
|---:|---:|---:|
| 64 | **0** | **0** |
| 128 | **0** | **0** |
| 256 | **0** | **0** |
| 384 (ragged tail) | **0** | **0** |
| 512 (ragged tail) | **0** | **0** |
| 100 (off the tiles) | **0** | **0** |
| 77 (off the tiles) | **0** | **0** |

The control: nothing differs before position 700, and 3,747,365 elements differ from it on.
Log: [`mc5-chunk-invariance-steel-2026-10-01.log`](chunked-prefill-2026-09-27/mc5-chunk-invariance-steel-2026-10-01.log).

The test now lives in `metal/mc5_chunk_invariance_test.go` and runs by default on a generated fixture
(`metal/mc3_fixture_test.go`). The off-tile sizes are there because the aligned ones cannot see a bug at a tile edge:
with the kernel's causal limit moved by one key, every aligned size still matched bit for bit on the fixture, while 100
and 77 differed from the first chunk boundary on.

So a chunked reply is the unchunked reply. Chunking can therefore switch on only under load (while another generation
decodes), and a lone request keeps today's path exactly.

## 2. The first candidate and its grading

**The build** (`d4293d41`, `decoder/model.go` `mc3Prefill`, `decoder/mc3_batch.go`):
- Under MC3, a suffix of ≥ 512 tokens prefills in 256-token chunks while another generation is decoding.
- Each chunk runs in its own exclusive section, with one decode step yielded between chunks (`yieldToDecode`,
  bounded at 50 ms).
- A chunk was cut only while two chunks' worth remained, so no tail could drop below the 8 tokens the batched path
  needs.
- Fake-resident `TestMC3_longPrefillChunksWhileOthersDecode` passes, and goes red with no yield and with chunking off.

**The grading** was registered in `d8ae520a` before any timing, with `scripts/bench_prefill_stall.py`:
- 3 decoder clients each stream 400 greedy tokens; after 2 s, 3 newcomers arrive one after another, each with a
  ~2400-word prompt (~3k tokens) and 16 answer tokens.
- A solo arm runs the newcomers with no decoders.
- *old* = `serve-metal` at `731f4f4e` (S3), *new* = at `d4293d41`; serve defaults, the 1.5B from `~/models`.
- Idle-gated, 3 interleaved pairs per arm; 2026-09-27 04:43–04:48 PDT.

Raw: [`stall-c256.json`](chunked-prefill-2026-09-27/stall-c256.json), [`stall-run-c256.log`](chunked-prefill-2026-09-27/stall-run-c256.log),
[`stall-servers-c256.log`](chunked-prefill-2026-09-27/stall-servers-c256.log), [`run-stall-c256.sh`](chunked-prefill-2026-09-27/run-stall-c256.sh);
gates by [`gates.py`](chunked-prefill-2026-09-27/gates.py) → [`gates-output-c256.txt`](chunked-prefill-2026-09-27/gates-output-c256.txt).

The bench's summary print crashed for the solo arm (no decoder gaps to format), after each cell had saved its result.
It is fixed; no data was lost.

| gate (pre-registered) | result |
|---|---|
| 1. identity: every reply equal, decoders and newcomers, every cell (hard) | **pass**: 6 stall and 6 solo cells; the newcomers' replies also equal across the two arms |
| 2. the stall: decoders' max inter-token gap, new ÷ old, median ≤ 0.5× | **pass: 0.213×**, 5.35–5.41 s → 1.14–1.17 s |
| 3. newcomer TTFT, new ÷ old, median ≤ 1.5× | **pass: 1.137×**, 5.27–5.40 s → 5.96–6.06 s |
| 4. cell wall time, new ÷ old, median ≤ 1.05× | **FAIL: 1.052×**, 24.4–24.7 s → 25.7–25.8 s |
| 5. solo guard: newcomer TTFT alone, new ÷ old ≤ 1.05× (hard) | **pass: 1.000×** |

The decoders' p99 gap rose from 22–30 ms to ~660 ms. Their p50 stayed at 19.3 ms. Fewer, shorter stalls now spread
over more gaps; the whole-prefill arm had one 5.4 s gap per newcomer.

**Decision, by the pre-registered rule: not shipped.** Gate 4 misses by 0.2%. The stall reduction is large, but a
wall-time cost was a registered gate, and a gate that fails is not reread. Chunking stays in the tree as
`Options.ResidentPrefillChunk`, 0 (off). Serve does not set it.

## 3. Where chunking's cost comes from (exploratory, after the grading)

`TestMC5_chunkCost` times a 3000-token prefill alone, with no decoding between passes. It is cut as the candidate cut
it (while two chunks remain), 3 reps interleaved:

| chunk | passes | total | longest pass |
|---:|---:|---:|---:|
| whole | 1 | 5.37 s | 5.37 s |
| 128 | 23 | 6.28 s (+17%) | 0.56 s |
| **256** | 11 | 5.79 s (+7.9%) | **1.20 s** |
| 512 | 5 | 5.68 s (+5.8%) | 2.31 s |

`TestMC5_passCost` times one pass by length and depth:

| startPos | C = 16 | C = 64 | C = 128 | C = 256 |
|---:|---:|---:|---:|---:|
| 64 | 95.8 ms | 104.8 | 154.2 | 290.0 |
| 2048 | 180.4 | 265.1 | 318.4 | 599.7 |

Reading:
- **A pass's floor is the prefill GEMM's 64-row tile.** 16 and 64 tokens cost the same, and each further 64 costs
  ~70 ms at shallow depth. S0 measured the same GEMM flat across M = 1–64
  ([`concurrency-mc3-s0-2026-09-26.md`](concurrency-mc3-s0-2026-09-26.md)).
- A 3000-token prompt uses 47 tiles whole and 47 in 256-token chunks, so the tiles are not the whole +7.9%. The rest
  is per-pass: per-call scratch (about 15 MB allocated and zeroed per pass), and attention run in more, smaller
  dispatches. It was not separated further.
- **The longest stall was the tail.** "Cut while two chunks remain" leaves a final pass of up to 2C − 1 tokens, run
  at the prompt's full depth. That is the 1.20 s pass above, and the 1.14 s gap the bench measured. The tail only
  needs to be at least 8 tokens long.
- Bigger chunks cost less wall time and stall longer. With the tail fixed, the stall is set by one C-token pass at
  full depth.

Logs: [`mc5-chunk-cost.log`](chunked-prefill-2026-09-27/mc5-chunk-cost.log), [`mc5-pass-cost.log`](chunked-prefill-2026-09-27/mc5-pass-cost.log).

## 4. The revised candidate, and its grading

Registered in `b69c73ad`, the build it grades, before any timing of it. Two changes, each from §3's diagnosis:
- **512-token chunks.** Half the passes, so half of chunking's per-pass cost.
- **The tail rule.** Cut while a chunk plus 8 tokens remain, so the last pass is 8..C+7 tokens, never the up-to-2C
  tail that had been the longest stall.

*new* = `serve-metal` at `b69c73ad` with `-prefill-chunk=512`; *old* = at `731f4f4e` (S3). The bench, cells, pairs and
five gates are the first candidate's, bars unchanged; 2026-09-27 04:58–05:03 PDT.
- A first launch (04:57) was stopped in its first cell, before any new cell ran: the run script passed `--serve-args
  -prefill-chunk=512` as two words, which argparse refuses
  ([`stall-c512-run-attempt1-argparse.log`](chunked-prefill-2026-09-27/stall-c512-run-attempt1-argparse.log)). It
  was relaunched with `--serve-args=-prefill-chunk=512`.

Raw: [`stall-c512.json`](chunked-prefill-2026-09-27/stall-c512.json), [`stall-c512-run.log`](chunked-prefill-2026-09-27/stall-c512-run.log),
[`stall-c512-servers.log`](chunked-prefill-2026-09-27/stall-c512-servers.log), [`run-stall-c512.sh`](chunked-prefill-2026-09-27/run-stall-c512.sh);
→ [`gates-output-c512.txt`](chunked-prefill-2026-09-27/gates-output-c512.txt).

| gate (pre-registered) | result |
|---|---|
| 1. identity, every reply, every cell (hard) | **pass** (12 cells) |
| 2. decoders' max inter-token gap, new ÷ old ≤ 0.5× | **pass: 0.229×**, 5.36–5.42 s → 1.23–1.27 s |
| 3. newcomer TTFT, new ÷ old ≤ 1.5× | **pass: 1.107×**, 5.28–5.41 s → 5.86–5.91 s |
| 4. cell wall, new ÷ old ≤ 1.05× | **pass: 1.045×**, 24.4–24.6 s → 25.6 s |
| 5. solo guard, TTFT alone ≤ 1.05× (hard) | **pass: 0.998×** |

**Reading.**
- **The stall is now one 512-token pass at the prompt's full depth.** The decoders' p99 gap is ~1.13 s, and several
  of their gaps are that long (one per chunk after the first few), where before one gap per newcomer was 5.4 s.
- **The wall gate passes with little room** (1.045× against 1.05×). The per-pass cost of §3 remains. Lowering it
  (reused per-call scratch, the attention dispatch shape) is what would let smaller chunks, and so shorter stalls,
  pay.
- **A lone request is untouched:** nothing chunks with nobody decoding.
- **W7 is unaffected.** Its turns prefill ~150-token suffixes, below the chunking threshold (a chunk plus 8 tokens).
