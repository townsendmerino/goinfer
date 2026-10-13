# KV0 — prefill invariance on CUDA and WebGPU, and the continuation effect (2026-10-13)

`docs/tasks/task-kv-architecture-2026-10.md` §5.6 and §6 KV0, item 3. Machine: nobara-pc (amd64, RTX 2070 SUPER 8 GB, CUDA and
WebGPU/Vulkan). The tests are `cuda/kv0_invariance_test.go` (`cuda && goinfer_testhooks`) and `gpu/kv0_invariance_test.go`
(`gpu && goinfer_testhooks`). No product code changed.

## 1. Registration (written and committed before the first run)

**The rule.** An arm is *invariant* when **0 elements** of every layer's K and V rows `[0, n)` differ, compared bit for bit
(`math.Float32bits`), and **0 of the last token's logits** differ. Anything else is *not invariant* and is recorded with the
lowest position where a K/V element differs (and its layer). There is no tolerance, no "close", and no re-baselining a bar
because a number moved. A backend with a non-invariant arm keeps sharing within one conversation until it is fixed (§5.6).

**The comparison.** Each arm starts by overwriting rows `[0, n)` of every K and V cache with NaN, so a row an arm fails to write
cannot match the row an earlier arm left behind. Then it prefills, then reads every K and V row back.

**Fixtures.** One tiny fixture and one real checkpoint per backend.
- CUDA: `testdata/llama-attnfa-tiny` (hd 128, 2 layers, hidden 512, int4) and Qwen2.5-Coder-1.5B-Instruct q4_k_m at int4.
- WebGPU: `testdata/llama-attnfa-tiny` and the same 1.5B at int4. (Amended before the WebGPU run: `llama-tiny` allows 128 positions, too few for a 1,000-token prompt.)
- Prompt: n = 1000 tokens. The tiny fixtures take seeded random token ids; the 1.5B takes the first 1000 tokens of this
  repository's `decoder/*.go` under the model's tokenizer.

**Arms (all against the whole prompt prefilled in one call).**

| arm | what runs | registered? |
|---|---|---|
| chunk = 64, 128, 256, 512 | one `PrefillLast` over 1000 rows, `GOINFER_PREFILL_CHUNK` set (tails of 40, 104, 232, 488 rows) | yes (the brief) |
| split @ 256, 512, 768 | `PrefillLast(rows [0,s))`, then `PrefillLast(rows [s,1000), startPos s)`: what a prefix hit does | yes (the brief) |
| control | the same prompt with the token at position 700 changed, chunk 256 | yes (the brief) |
| split @ 511 and @ 984, 985 | probes of the class map's two boundaries (the 512-token floor, the 16-row minimum) | added here, before the run |
| chunk = 496 | a chunk width that leaves an 8-row tail | added here, before the run |
| exact-class whole | the whole prompt with `GOINFER_CUDA_FAST_PREFILL_FLOOR=1000000` | reference, CUDA only |

**The control passes** when nothing differs below position 700, the lowest differing position is exactly 700, something
differs from there on, and the last logits differ. If it does not, the comparison cannot see a difference and no arm's
verdict counts.

**CUDA class map (read from `cuda/prefill.go`, before measuring).** A pass of m rows starting at s runs the fast kernels
(`attn_fused`, `gemm_w4a8_mma`) when all of these hold: the tail is an ordinary `PrefillLast` (or a non-final chunk of one);
m ≥ 16 (`attnFusedMinRows`, `gemmMMAMinRows`); the prompt length the pass is judged on is ≥ 512 (`fastPrefillFloor`); the
head dim is 64 or 128 for attention, and K % 32 == 0 for the GEMM; and the model is not a DeltaNet, E-model or DeepStack
model. **The prompt length a pass is judged on is `max(chunkPromptLen, s + m)`**: the whole prompt's length for a pass inside
a multi-chunk prefill, and the pass's own end otherwise. So it is the pass's length, not the prompt's, for a prefill that
arrives in two calls.

Predictions that follow, registered:
1. `chunk=64/128/256/512`: every pass is judged on 1000 and has m ≥ 16. **Invariant.**
2. `split@512`, `split@768`: the first call is judged on 512 or 768, the second on 1000, and both have m ≥ 16. **Invariant.**
3. `split@256`: the first call is judged on 256 < 512, so rows [0, 256) are written by the exact class and the whole prompt
   would have written them with the fast class. **Not invariant, first differing position 0.** Rows [0, 256) must equal the
   exact-class whole prompt's rows [0, 256) bit for bit (they read only exact-class history).
4. `split@511`: the same, from position 0.
5. `split@984`: the second call has m = 16, fast. **Invariant.** `split@985`: m = 15, exact. **Not invariant, first differing
   position 985.**
6. `chunk=496`: the 8-row tail is exact. **Not invariant, first differing position 992.**
7. The fast and exact whole-prompt prefills differ (otherwise no arm can see a class mismatch). Launch counts: `attn_fused`
   and `gemm_w4a8_mma` are launched once per layer per fast pass and never in an exact pass, which the test checks per arm.

The tests assert predictions 1–7, so a class map that is wrong fails the run and is not recorded as a surprise.

**WebGPU class map (read from `gpu/residency.go:residentDecoder.prefillBlocks` and `gpu/prefillrunner.go`).** The batched pass
is built from per-row kernels (`attnBatchedShaderWGSL` takes each row's own causal limit; `ropeBatchedShaderWGSL` recovers
row r's position as start + r) and is documented bit-equivalent to M sequential decode calls. Nothing reads the prompt's
length. The one selector that reads the row count is `w4MultiFor` (the row-shaped int4 kernel up to `w4MultiMaxRows`, the
tiled one beyond), which its comment says changes speed only. A prefill is cut into passes of `prefillBlockRows` = 512 rows.
Registered prediction: **every arm is invariant on WebGPU**, and the two int4 kernel kinds give the same bits (the test pins
each kind in turn through `w4MultiPin` and compares).

**The continuation effect (CUDA).** Per trial, with the same teacher-forced tokens (np = 600 prompt tokens, d = 32 more):
A = prefill 600, then decode 32 rows one at a time; B = prefill all 632 in one pass. Both on the fast class. Reported over 50
trials (50 different windows of the text): the K/V elements of the 32 decode rows that differ between A and B, their maximum
absolute difference and the rms difference relative to the rms value, the next-token logits' cosine (min, mean) and argmax
agreement, and the logits that differ bit for bit. Two controls per trial: A' and C, the same two runs with the floor raised
above the prompt so every pass is exact class. **Registered prediction: A' equals C exactly** (the exact class is documented
bit-identical to decode: `cuda/prefill.go:prefillChunked`, `useAttnFused`), and A differs from B (decode is not the fast
class). No number is predicted for the size of A against B: it is what is being measured. The 1.5B's reading is the
reference one; the tiny fixture's random weights make its argmax meaningless and it is reported as a plumbing check.

**Cost and stopping rule.** Fixed arms, no sampling, no early stop. Estimated wall time: tiny fixtures under 1 minute each; the
1.5B arms about 2 minutes and the 50-trial continuation about 3 minutes (4 prefills of ≤ 632 rows and 64 decode steps per
trial). Run with `-v`, `-timeout 30m`, a heartbeat line at least every 30 s, one package per invocation, logs under
`docs/measurements/kv0-invariance-2026-10-13/`. By day: nothing here exceeds ~10 minutes.

## 2. Readings

(filled in below after the runs)
