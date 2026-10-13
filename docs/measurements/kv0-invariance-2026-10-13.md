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
bit-identical to decode: `cuda/prefill.go:cudaResident.prefillChunked`, `cuda/prefill.go:cudaResident.useAttnFused`), and A differs from B (decode is not the fast
class). No number is predicted for the size of A against B: it is what is being measured. The 1.5B's reading is the
reference one; the tiny fixture's random weights make its argmax meaningless and it is reported as a plumbing check.

**Cost and stopping rule.** Fixed arms, no sampling, no early stop. Estimated wall time: tiny fixtures under 1 minute each; the
1.5B arms about 2 minutes and the 50-trial continuation about 3 minutes (4 prefills of ≤ 632 rows and 64 decode steps per
trial). Run with `-v`, `-timeout 30m`, a heartbeat line at least every 30 s, one package per invocation, logs under
`docs/measurements/kv0-invariance-2026-10-13/`. By day: nothing here exceeds ~10 minutes.


### 1a. Addendum, registered before the f16 runs

After the first CUDA runs (below, §2.1), a read of `cuda/kvpolicy.go` and commit 60222412 found that **f16 KV is the CUDA
default** for llama, mistral, qwen2, qwen3 and phi3 since 2026-10-13 (`Options.KVPrecision` "" or "auto"), so `serve` runs the
models of this study with f16 KV. The registered runs loaded with the test's own options and asserted f32 K/V rows, so they
measured f32. This addendum adds the configuration `serve` ships.

- Every CUDA arm and the continuation arm are repeated with `KVPrecision: "f16"`, and the f32 arms with `KVPrecision: "f32"`
  named explicitly, from the same test version. K/V rows are read as f16 bit patterns and compared through their exact f32
  images (a widening is injective, so equal images mean equal bits).
- **Predictions, unchanged from §1 for the invariance arms:** the class map reads the same quantities (pass rows, prompt length,
  tail), the f16 kernels are the same source compiled with `-DKV_F16`, and both arms of a comparison round the same values to
  f16 on store. So chunk arms and the splits at 512, 768 and 984 are **invariant**; the splits at 256 and 511, the split at 985
  and the 496-row chunk are **not invariant**, starting at 0, 0, 985 and 992.
- The exact-class property of the continuation arm (A′ equals C) is predicted to hold for f16 as it did for f32. The size of the
  f16 continuation effect is not predicted; it is expected to differ from the f32 figure in the fraction of differing elements
  (f16 rounding can absorb a small difference), not in sign.
- WebGPU is unchanged: its KV is f32 by default and its arms ran as registered.

## 2. Readings

Logs are in `kv0-invariance-2026-10-13/` (`cuda-tiny.log`, `cuda-real1p5B.log`: the first, registered runs with the test's own
options, f32 K/V; `cuda-f32-f16.log`: every CUDA arm again with `KVPrecision` named, one run per precision and fixture;
`cuda-tiny-rerun-after-assertion-change.log`; `webgpu.log`). Every number below is read from those logs. Commit of the tests
and of the run: this branch, `kv0-2026-10`.

### 2.1 Verdicts under the registered rule

"INV" is invariant (0 K/V elements and 0 logits differ); "NOT" is not invariant, with the lowest position where a K/V element
differs. Every cell is the same for both fixtures and for f32 and f16 unless a second value is shown.

| arm | CUDA (class map predicts) | CUDA observed | WebGPU observed |
|---|---|---|---|
| chunk = 64, 128, 256, 512 (tails 40, 104, 232, 488 rows) | INV | **INV** (all four, both fixtures, f32 and f16) | **INV** |
| split @ 512, 768 | INV | **INV** | **INV** |
| split @ 256 | NOT, from 0 | **NOT, from position 0** (tiny f16: position 1) | **INV** |
| split @ 511 | NOT, from 0 | **NOT, from position 0** (tiny f16: position 1) | **INV** |
| split @ 984 (second call 16 rows) | INV | **INV** | **INV** |
| split @ 985 (second call 15 rows) | NOT, from 985 | **NOT, from position 985** | **INV** |
| chunk = 496 (an 8-row tail) | NOT, from 992 | **NOT, from position 992** | **INV** |
| control: token 700 changed, chunk 256 | nothing before 700, first difference at 700 | **0 differ below 700, first at 700** (1.5B f32: 4,147,121 elements from 700 on; f16: 4,124,507) | **same** (1.5B: 4,147,653 from 700 on) |
| exact-class rows of `split@256` and `split@511` against the exact-class whole prompt | 0 differ | **0 of 3,670,016 and of 7,325,696** (1.5B f32) | n/a |
| fast whole prompt against exact whole prompt | differ | 13,837,756 of 14,336,000 elements (1.5B f32); 13,583,400 (f16) | n/a |
| WebGPU `w4Multi` pinned to the row kernel, then to the tiled kernel | n/a | n/a | **INV** both, both fixtures |

Sizes of the non-invariant arms on the 1.5B at f32: `split@256` 13,446,456 of 14,336,000 K/V elements differ and 151,935 of
151,936 logits (max |diff| 0.972); `split@511` 13,580,961 and all 151,936 (1.19); `split@985` 207,586 and all 151,936 (0.829);
`chunk=496` 110,739 and all 151,936 (0.878). On the tiny fixture the logits differ in all 256 and the largest difference is 0.02.

**Prediction scorecard.** Every invariance verdict matched the class map as registered, in all 44 CUDA arm results (2 fixtures × 2
precisions × 11 arms) and all 13 WebGPU arms of both fixtures; the launch counts matched (`attn_fused` and `gemm_w4a8_mma`
once per layer per fast pass, never in an exact pass) in every arm; the control passed everywhere. **One detail of one
prediction was wrong:** it said the first differing position of a non-invariant arm is the first exact-class row. On the tiny f16
fixture it is position 1 for `split@256` and `split@511` (the first exact-class row is 0). Position 0 attends only to itself,
so its attention is the identity under either class, and an f16 store absorbed the rest of the difference at that row. The
test asserted equality, failed on that cell (`cuda-f32-f16.log`, "FAIL: TestKV0_prefillInvariance_tiny/f16"), and the
assertion was relaxed to what is true: no row before the first exact-class pass may differ, and the first differing row must be
within 16 of it (`cuda-tiny-rerun-after-assertion-change.log`, pass). No invariance verdict depends on that assertion.

### 2.2 The class map, as measured

CUDA: the same as registered. A pass of m rows from s runs the fast kernels iff m ≥ 16 and max(promptLen, s + m) ≥ 512, with
promptLen the whole prompt for a pass inside a multi-chunk prefill. **What decides the class is the pass's own length, not the
prompt's**, for a prefill that arrives in two calls: `split@256` at n = 1000 wrote rows 0–255 with the exact class although the
prompt is 1,000 tokens, because its first call was 256 tokens long; that is what `decoder/resident_reuse.go:Model.declineShortLeadReuse` guards. The
second condition, m ≥ 16, is not part of `decoder/resident_reuse.go:Model.declineShortLeadReuse`'s rule (it compares the prompt's length with the floor): `split@985` and `chunk=496` show it makes a ≤ 15-row pass exact.
In the manager's terms it never reaches a *published* block as such: a published block is a full block and a 15-row tail pass
makes a full block only when it ends one, in which case the block mixes two classes (§5.5's "records both and the row where they
meet" is the right shape and is needed).
WebGPU: no class. Every arm was bit-identical, including pinning the row-count selector `w4MultiFor` both ways.
Metal: `TestMC5_prefillChunkInvariance` (the registered original; not re-run here, no Mac).

### 2.3 The continuation effect on CUDA (§5.6 property 3, measured)

50 trials, np = 600 prompt tokens then d = 32 teacher-forced tokens, the real 1.5B, text from this repository's `decoder/*.go`.
A = prefill 600, decode 32 rows. B = prefill all 632 in one pass. Both fast class.

| | f32 K/V | f16 K/V (serve's default) |
|---|---|---|
| decode-written K/V elements that differ from the prefill-written ones | 22,140,567 of 22,937,600 (96.5%) | 21,711,615 of 22,937,600 (94.7%) |
| max \|diff\| (largest \|value\| 360) | 4.58 | 3.74 |
| rms difference ÷ rms value, mean over trials (worst) | 0.00676 (0.00965) | 0.00662 (0.00959) |
| prompt rows [0, 600): a 600-row pass against the first 600 rows of a 632-row pass | 0 differ | 0 differ |
| next-token logits A against B: cosine min / mean | 0.994166 / 0.997914 | 0.992742 / 0.997952 |
| argmax of A equals argmax of B | 49 of 50 | 49 of 50 |
| trials whose rows and logits are bit-identical | 0 of 50 | 0 of 50 |
| B's top-1 minus top-2 logit: min / median | 0.00876 / 2.48 | 0.00876 / 2.48 |
| **control A′ (exact class) against C (exact class, whole prefill)** | **0 elements differ, cosine 1.000000, argmax 50 of 50, 0 logits differ** | **the same** |

**Reading.** A row decode writes is not the row a fast-class prefill writes: about 0.7% rms apart, with a few activations several
units off in a layer whose values reach 360. A reply that continues a conversation from its own decode-written rows can
therefore differ from the one a cold prefill of the same text would give: in 1 of 50 trials of this sample the argmax moved. The control pins the
source: the exact class equals decode to the bit, so the difference is the fast prefill's own numerics (it is the cosine-gated
lever, not a bug). The continuation effect exists today (resident reuse carries it) and is what §5.6 property 3 accepts. The f16
store narrows the fraction of elements that differ by about two points and does not change the effect's size. The tiny fixture's
random weights make its argmax meaningless; its line in `cuda-f32-f16.log` is a plumbing check.

### 2.4 What this does and does not support

- **Supports §5.6 property 2 (sharing invariance) on CUDA and WebGPU, with the producer class in the key.** Chunk invariance
  holds on both backends for every chunk width and ragged tail tried at or above 16 rows, and a split at a block-aligned
  offset is invariant exactly when the pass that wrote the prefix was in the same class the whole prompt would have used.
  Sharing across conversations on CUDA is therefore safe only if a hit's rows came from a pass of the requester's own class: the
  existing 512-token rule, plus the 16-row rule above, which no code states today.
- **Does not measure:** a windowed layer (the gate refuses a windowed load); any model but the llama-attnfa-tiny fixture and the
  Qwen2.5-Coder-1.5B; WebGPU on a backend other than Vulkan on this card; Metal. The CUDA census's INFERRED point that a
  windowed fused prefill is chunk-invariant only for 64-aligned chunk starts is untested and stays a KV1 test obligation.
- **Doc consequence:** none of the three non-invariant CUDA arms is a defect; each is a case the class key must keep apart.
  **No backend measured here needs a fix for chunk invariance.** The §5.6 sentence "A non-invariant backend keeps sharing within
  one conversation until it is fixed" does not trigger for CUDA or WebGPU.

### 2.5 Amendment, 2026-10-13 (KV3): the 8-row tail of `chunk = 496` no longer happens

The chunker (`cuda/prefill.go:cudaResident.prefillChunked`) now never ends a multi-pass prefill in a pass of fewer than 16 rows: it moves the
missing rows from the previous pass to the last. So the arm `chunk=496` (1,000 tokens = 496 + 496 + 8) now runs as 496 + 488 + 16, every pass
fused, and is invariant; the non-invariant readings above (starts at position 992) are the behaviour before that change and stay as measured.
The splits (`split@985` and the others) are two separate calls and are unchanged, which is why the 16-row condition is also asked of the
reuse guard and the block key (`docs/measurements/kv3-cuda-2026-10-13.md` §3.2).
