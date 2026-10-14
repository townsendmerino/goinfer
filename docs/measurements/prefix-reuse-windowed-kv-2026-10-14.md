# Prefix reuse under CUDA windowed KV: why Codex x Mellum2.1 re-prefilled most turns, and the fix (2026-10-14)

- **Box:** nobara-pc, RTX 2070 SUPER 8 GB, driver 595.91.07. **Data:** the I3 night of 2026-10-13, job B (`mellum-16k`: `serve --backend cuda
  --ctx 16384 --moe-cache-experts`, serve binary `81791568`), committed under
  `docs/measurements/integration-harness-2026-10/i3-night-2026-10-13/job-b/` (`records.jsonl.gz`, `serve-mellum-16k.log.gz`). No request
  bodies were recorded; everything below is read from request sizes, the server's time to first token, Codex's own session files, and the code.
- **Fix built on:** `origin/main` at `e9dae735`, branch `kvwindow-ring-2026-10`.
- **Files beside this record:** `sim.py` (the replay, Python; the same replay is the Go unit test `TestWindowedKV_i3Replay`),
  `memtable.py` (the memory table below), and the night speed guard of section 8 (`speed_guard.py`, `run-speed-guard.sh`).

## 1. The finding

In the Codex x Mellum2.1 cell, 35 of the 59 `/v1/responses` requests had a time to first token of 61 to 98 s: a cold prefill of the whole
6,100 to 9,600-token prompt. The other 24 took 0.9 to 4.6 s. The cold ones cost about 2,430 s (their time to first token, less what a warm
prefill takes): 77% of the cell's 3,151 s of run wall time, 27% of job B's 150 minutes and about 11% of the whole 364-minute night.

## 2. The cause

**Classification: a guard declines the reuse (class c), triggered by a client-side divergence late in the prompt (class a). The prompt
does not differ early, and the cache is not evicted.**

1. **The client drops the previous reply.** `/v1/responses` returns no reasoning items (`internal/serveapp/responses.go:responseReq`: "reasoning
   items are out of scope"), so Codex has nothing to send back: its session file holds only the function call and its output for each turn.
   The KV slot holds the prompt plus every generated token, `<think>` block included. The next prompt therefore diverges at the previous
   prompt's end, and reusing the cache means rolling back the whole previous reply. The sizes show it: a 6,116-token prompt and 319 generated
   tokens are followed by a 6,208-token prompt (+92, the call and its output). opencode on `/v1/chat/completions` sends the reasoning back
   (each prompt grows by the reply plus about 18) and every follow-up of its runs was warm.
2. **The window cannot roll back that far, most of the time.** Mellum2.1 has 21 sliding-window layers (window 1,024), and with CUDA windowed
   KV (on by default) each held `window + 512` positions. `cuda/kvwindow.go:cudaResident.kvEnsureSlot` compacted when a write passed
   `base + 1536` and moved the base to the new row's window start (`p0 - 1023`). `cuda/kvwindow.go:cudaResident.ReusableKV` (called through
   `decoder/resident_reuse.go:Model.residentReuseFloor`) returns 0, a cold prefill, when the reused prefix's window starts below the base. So
   how far back a follow-up could reuse **cycled between 0 and 512 positions**, a sawtooth set by where in the compaction cycle the last
   reply's final token landed. A rollback of 248 tokens could miss and one of 258 hit. This is why it was 35 of 59, not all of them.

**Evidence: the replay.** `sim.py` runs the compaction and `ReusableKV` arithmetic over the 59 recorded requests (prompt and completion
sizes; one KV slot, as the serve log says; a follow-up inside a run diverges at the previous prompt's end; prefill in 512-row passes; one
write per generated token but the last). It predicts cold or warm correctly for **59 of 59** requests, and stays 59 of 59 with the
divergence point moved 3 tokens either way.

**Control.** Codex x Qwen3-4B on the same night: same client, same Responses route, no sliding-window layers, so no windowed KV. None of
its 48 requests was cold, and a new run even reused the last run's prefix (time to first token 28 ms on a 5,770-token prompt).

**Not this cause:** eviction (one conversation, one slot; nothing else ran), block alignment or the 16-row chunk rule (the replay needs
neither), a family guard (Mellum2.1 has no recurrent state). A run's first request was cold under every rule, because the previous run's
reply was over 512 tokens. Without the window it would have reused the shared instructions and tools, about 6,000 tokens.

**Claude Code x Mellum2.1 (32K context):** 9 of 12 requests were cold. Five were run starts, two followed a Claude Code compaction, and two
inside a run (18,923 to 20,323 and 19,317 to 20,717 tokens, after replies of 104 and 230 tokens) cannot be explained from the logs.
Claude Code sends thinking back, so those divergences must be earlier in the prompt. That needs the request bodies.

## 3. Scope

- **CUDA, every family with sliding-window layers past its window:** Mellum, Gemma 2, 3 and 4, Mistral, Command-R7B (cohere2), gpt-oss, Phi-3.
  Any client whose follow-up does not carry the previous reply as generated: Codex (no reasoning items on `/v1/responses`), or a client that
  edits near the end of the prompt. opencode, and Claude Code within a turn, send the reply back.
- **Metal and WebGPU** keep the whole context for a window layer, so they are unaffected.
- **The CPU backend (checked here, not fixed):** `decoder/kvcache.go:ring` holds exactly `window` slots, addressed `p % window`, and
  `ring.truncate` reports a rewind exact only when the ring never wrapped or the rewind is a single position. `decoder/session.go:Session.rewindForReuse`
  turns an inexact rewind into a cold prefill. So the CPU budget does not cycle. It is one position, always: on a context past the window,
  every Codex follow-up on a windowed family re-prefills. The CUDA fix does not carry over. The ring is modular with a mirror copy for
  contiguous reads (`ring.mirrored`, `ring.window`), and holding a rollback means changing its slot count and its read path on the attention
  hot path in `decoder/`. Left as a finding.

## 4. The fix: a rollback floor at every point of the cycle, plus what free VRAM buys

The owner's constraint (2026-10-14): the reuse budget must not cycle down to 0 with where the last decode landed, and the fix must not cost
residency, context, a KV slot or an expert-cache slot. A first build (a fixed slack of 1,536 positions, commit `c3802931`) met the first
half and failed the second, measured by loading each windowed model in real `serve` (main against that build, 2026-10-14): Mellum2.1 at
serve's defaults fell off the GPU to the CPU path (its 4,096-token floor needed 337 MB against 253 MB free), Gemma-3-4B's context fell from
6,722 to 4,096, and the expert cache shrank from 57 to 56 slots on Mellum2.1 at the I3 settings and from 41 to 35 on Gemma-4-26B, because
an expert-cache load sizes its cache from what the KV leaves without asking the planner. That build is replaced by two parts.

**Part 1, always: a small-step compaction at the planned slack** (`cuda/kvwindow.go:kvWindowNextBase`, `kvWindowStep = 64`). The slack stays
512, so the plan, the price (`decoder.WindowedKVSlack`, unchanged) and the allocation equal main's. A compaction now moves the base only so
far that `slack - step` = **448 positions of rollback** stay behind the end of the write, so the budget lies between 448 and 512 whatever point
of the cycle the last decode stopped at. The price is copying: a compaction copies the live tail (window - 1 + rollback positions of every
windowed layer, through the scratch buffer, so twice), and it now comes every 65 positions of decode instead of every 513:

| model | today: 1,023 positions every 513 tokens | step 64: 1,470 positions every 65 tokens |
|---|---|---|
| Mellum2.1 (21 x 4,096 B a position) | 88 MB a compaction, 0.17 MB a token | 126 MB a compaction, 1.9 MB a token |
| Gemma-4-26B (25 x 16,384 B a position) | 419 MB, 0.8 MB a token | 602 MB, 9.3 MB a token |

Doubled for the scratch round trip, Mellum's step-64 copying is about 3.9 MB a token, some 17 µs at this card's bandwidth against a token
of about 25 ms. The night guard (section 8) measures it. Nothing else assumed whole-window jumps: the room check (`kvRoomFor`) holds for any
rows that fit, CUDA graphs are already off under windowed KV, and a compaction is a plain copy, so the kernel-class rules (64-aligned
starts, the 16-row tail) do not apply to it.

**Part 2, when the card has room: extra rollback from leftover VRAM** (`cudaResident.kvGrantRollback`, called last in the build, after the
context, the KV slots and the expert cache are fixed). Free VRAM, less what is held back (the companion and vision-tower reserve the build
was given, the 384 MB decode margin, and two default 512-row prefill passes' activation scratch, which `prefillCore` allocates per
request), buys whole steps of extra slack in every slot's windowed layers, up to a rollback of 1,024 (`kvWindowRollbackMax`). The windowed
buffers are reallocated (nothing is written yet); a failed allocation keeps the planned ones. **An expert-cache load gets none:** its cache
was sized to what was free. **So the rollback a load gets depends on the memory state at load time**, between 448 and 1,024, and the
serve log says which: `cuda: windowed KV: N more positions from free VRAM`.

Measured the same way, with each windowed model loaded in real `serve` on this card (the new build against main, 2026-10-14):

| model, flags | main | this build |
|---|---|---|
| Mellum2.1, serve defaults | resident, ctx 4,096, 1 slot | the same; no grant (448) |
| Mellum2.1, `--ctx 16384 --moe-cache-experts` (I3) | expert cache 57 slots | 57 slots; no grant (448) |
| Gemma-4-26B, `--moe-cache-experts` | expert cache 41 slots, ctx 4,096 | 41 slots; no grant (448) |
| Gemma-3-4B, defaults | ctx 6,722, 3 slots | the same; +448 granted (306 MB, beside the 2.4 GB vision-tower reserve): 896 |
| Gemma 4 E2B, defaults | ctx 16,384, 4 slots | the same; +576 granted (55 MB): 1,024 |
| Gemma 4 E4B, defaults | ctx 16,384, 4 slots | the same; +576 granted (181 MB): 1,024 |

**What the I3 replay says** (`TestWindowedKV_i3Replay`; the I3 load was an expert-cache one, so it got the floor): of the 54 follow-ups
inside a run, 30 were cold under the old rule, **16 at the 448 floor** (each rolling back 544 or more), and 5 at the most the grant gives
(rollback 1,024: three replies of over 1,300 tokens and two of Codex's own compactions). Step 32 gives the same 16 as step 64.

**Caveat.** A prefill pass of 448 rows or more moves the base to that pass's own window start, so until decode has written past it, a
follow-up that diverges inside that last pass (an edit, not a dropped reply) has `rows of that pass + tokens decoded since` of budget, not
448. A dropped reply, Codex's shape, is always covered: its divergence is the prompt's end.

**Memory**, per windowed layer per KV slot. Windowed layers stay f32 (`cuda/kvpolicy.go` gives the sliding-window ring no f16 form), so a
layer holds `2 x kvDim x 4 x min(ctx, window + slack)` bytes (`cuda/resident.go:kvBytesForCap`; `TestWindowedKV_allocation` proves it equals
the bytes allocated before the grant). The planned figure is main's. The last column is the most the grant can take, and it takes it only
from VRAM nothing else was given. Real geometry from each checkpoint's `config.json` in `~/models`, by `memtable.py`; the load table above is
the measurement on the device.

| checkpoint | family | window W | windowed layers (own KV) / all | kvDim | bytes per position per layer | per layer per slot, planned (W+512, as main) | per layer per slot at the most the grant gives (W+1088) | most the grant can take, all windowed layers, per slot |
|---|---|---|---|---|---|---|---|---|
| Mellum2.1-thinking | mellum | 1024 | 21 / 28 | 512 | 4096 | 6.0 MiB | 8.2 MiB | 47 MiB |
| gemma-2-2b-it | gemma2 | 4096 | 13 / 26 | 1024 | 8192 | 36.0 MiB | 40.5 MiB | 58 MiB |
| gemma-3-4b-it | gemma3_text | 1024 | 29 / 34 | 1024 | 8192 | 12.0 MiB | 16.5 MiB | 130 MiB |
| Gemma 4 E2B | gemma4_text | 512 | 12 / 35 | 256 | 2048 | 2.0 MiB | 3.1 MiB | 14 MiB |
| Gemma 4 E4B | gemma4_text | 512 | 20 / 42 | 512 | 4096 | 4.0 MiB | 6.2 MiB | 45 MiB |
| gemma-4-26b-a4b-it | gemma4_text | 1024 | 25 / 30 | 2048 | 16384 | 24.0 MiB | 33.0 MiB | 225 MiB |
| gemma-4-31B-it | gemma4_text | 1024 | 50 / 60 | 4096 | 32768 | 48.0 MiB | 66.0 MiB | 900 MiB |
| command-r7b | cohere2 | 4096 | 24 / 32 | 1024 | 8192 | 36.0 MiB | 40.5 MiB | 108 MiB |
| gpt-oss-20b | gpt_oss | 128 | 12 / 24 | 512 | 4096 | 2.5 MiB | 4.8 MiB | 27 MiB |
| Phi-3-mini-4k | phi3 | 2047 | 32 / 32 | 3072 | 24576 | 60.0 MiB | 73.5 MiB | 432 MiB |
| Mistral-7B-v0.1 (8 KV heads x 128, 32 layers; no config on disk) | mistral | 4096 | 32 / 32 | 1024 | 8192 | 36.0 MiB | 40.5 MiB | 144 MiB |

## 5. Gates

Day tier, `cuda` and `cuda goinfer_testhooks` on the RTX 2070 SUPER, each read with `-v` for `--- PASS` (none skipped):

- `TestWindowedKV_i3Replay` (no device): the 59 recorded requests through `kvWindowNextBase` and `kvWindowReusable`. The old rule must
  reproduce the server's cold or warm for every request and its 35 cold; the 448 floor must leave exactly 16 of 54 in-run follow-ups cold,
  none with a rollback within 448; the largest grant (1,024) 5, none within 1,024. It assumes no grant at the floor (the I3 load was an
  expert-cache one) and every in-run follow-up diverging at the previous prompt's end.
- `TestWindowedKV_rollbackReuse` (three tiny windowed fixtures, slack 24, step 8: rollback 16): reuse after a dropped reply of the rollback's
  length, bit-identical to the full-KV resident and to a cold prefill of the whole prompt. Against main it failed (`ReusableKV(42) = 0` after
  a 16-position rollback); it passes.
- `TestWindowedKV_stepIdentity64` (three fixtures, step 64 on a slack of 80): 600 decode steps through 8 compactions, every step's logits and
  every held K and V position bit-identical to the full-KV resident. A planted defect (the compaction copying one position short) fails it.
- `TestWindowedKV_expertCacheGetsNoGrant`: an expert-cache load keeps the planned slack and the 448-equivalent floor, while the same load
  without the expert cache is granted (so the test can go red); and the decoder still prices a 1,024-window layer at 1,536 positions, main's
  figure the expert cache is sized beside. On the first build it could not hold: that build allocated 1,024 more positions per windowed
  layer to every load, expert cache or not. A planted defect (the grant not skipping the expert cache) fails it.
- `TestWindowedKV_grant`: a load with spare VRAM gets the full grant in every slot, one whose free VRAM is only the reserve gets none, the
  context and slot count are the same either way, and decode under the grant is bit-identical to full KV with the granted rollback
  reusable. `TestWindowedKV_grantArithmetic` pins `kvWindowGrant`.
- The existing `TestWindowedKV_*` suite (decode, prefill, verify, slots, step, upload, reuse floor, agent turns, offsets) passes under the
  new rule, with the grant off so the fixtures still compact.

## 6. For KV2 (passed on by the owner)

KV2's design (`docs/tasks/task-kv-architecture-2026-10.md`, section 5) keeps a window group to `ceil(W/B) + 1` blocks and plans "one block of
slack behind the window, so a rollback inside a verify width stays valid": a rollback budget of 64 positions. **That budget does not serve
an agent turn.** The rollback a follow-up needs is the previous reply: median 258 tokens and 90th percentile 1,328 in the I3 cell. Replayed
with 64 positions of slack and none kept, 53 of 54 in-run Codex follow-ups would have prefilled cold, against 30 under main's rule.

What this record found that the KV2 design has to keep:

- **A rollback floor that does not cycle.** Blocks make it cheaper than here: releasing a block costs no copy, so a window group can keep
  `ceil((W + R) / B) + 1` blocks with a floor of R behind the newest position at no copying cost. R = 448 reproduces this build's 16 of 54;
  R = 1,024 gives 5 of 54 (a 1,024 + 64 replay).
- **Its memory comes from somewhere priced.** Here the floor costs nothing because it fits inside main's 512 slack, and anything above it is
  taken only from VRAM left over after the plan, never from the context, a slot or an expert cache. KV2's planner sizes pools per group
  (option A); the floor belongs in the window group's priced need, and any larger rollback in what is left over.
- **Or keep released window blocks reattachable.** A block released behind the window that stays in the prefix cache at refcount 0 (the
  design's "or the cache") can be re-attached by a rollback, which makes a smaller floor enough.
- **A gate on recorded traffic.** `TestWindowedKV_i3Replay`'s shape (recorded sizes in, the server's warm or cold out) can be pointed at the
  KV manager's window-group retention once it replaces `kvEnsureSlot`; its 35-of-59 half pins the old behaviour as the reference.
- `decoder/resident_reuse.go` and the KV manager were not touched: the floor is still `ResidentReusableKV`, answered by `cuda/kvwindow.go`.

## 7. Where the constants live

The step (64) and the rollback cap (1,024) live in `cuda/kvwindow.go`, and the planned slack is still `decoder.WindowedKVSlack` (512),
unchanged. So nothing under `decoder/` changes and the parity manifest does not go stale. The first build put a new constant in
`decoder/model.go`, a file every parity family's hash covers, and a CUDA-only figure staled 42 families; that is gone with it.

## 8. Speed guard, registered 2026-10-14 before any run (night tier)

**Question.** Does the small-step compaction cost decode speed? The added copying is about 0.1% of a token by arithmetic; the guard measures
it served.

- **Instrument:** `run-speed-guard.sh` beside this record (driver `speed_guard.py`), queued on nobara with `night.py`, which holds the timing
  lock. Two goinfer arms, pinned binaries in `~/goinfer-bench/kvwindow-ring-2026-10-14/` (`serve-main` built from `e9dae735`, `serve-new`
  from this branch; `binaries.sha256` is checked first). No peer arm (TE5(a): the decision is new / main).
- **Cells:** Mellum2.1 (`--ctx 16384 --moe-cache-experts`, the I3 settings) and Gemma-4-26B-A4B (`gemma4-26b-int4.giw --moe-cache-experts`,
  ctx 4,096). Both are expert-cache loads, so both arms hold the same planned KV and the new arm gets no grant: the cells isolate the step.
- **Workload:** `/v1/completions`, greedy, a prompt that counts from 1 to 400 (about 1,950 tokens on Mellum2.1), 640 tokens generated. Decode
  runs from about position 1,950 to 2,590, past window + slack (1,536), so the new arm compacts about 10 times a completion and main about
  once. Decode rate = (tokens - 1) / (last token - first token), timed by the client from the first streamed token, so prefill is excluded.
- **Design and N:** per model, four sessions in the order main, new, new, main (one ABBA quad), each a fresh server; 3 completions a session.
  N = 6 completions per arm per model. An idle gate before each session (1-minute load under 1.0, at most 300 s).
- **Rule:** per model, R = median(new rates) / median(main rates). **PASS** if R >= 0.99 on both models. **FAIL** if R < 0.97 on either.
  Anything else is **AMBIGUOUS, parked**: the owner decides with the numbers. Fixed N, no early stop.
- **Void:** a session not `cuda-resident`, a session without the windowed-KV line, or a completion under 90% of the 640 tokens.
- **Read beside the verdict, not graded:** whether the texts are identical within each arm and across arms (the tiny-fixture identity gate
  is the correctness gate; a difference here is investigated, not graded).
- **Prediction:** PASS at 0.7 (the copying is about 0.1% of a token), AMBIGUOUS at 0.25 (two sessions an arm resolve about 1%), FAIL at 0.05.
- **What each result does:** PASS, the step stays at 64. AMBIGUOUS, the owner decides; a larger step halves the copying for a smaller
  floor (step 128: floor 384). FAIL, the step is raised or the copy goes direct (no scratch round trip), and the guard re-runs.
- **Cost estimate:** a session is the idle gate (about 30 s, measured in the smoke), the load (Mellum2.1 20 s, Gemma-4-26B 50 s) and 3 x
  (prefill of the prompt, about 20 s, plus 640 tokens at about 65 tokens a second on Mellum2.1 and 40 on the 26B). Mellum2.1 about 137 s a
  session, the 26B about 200 s: 4 x 137 + 4 x 200 s, about 22 minutes; queued at 25.
- **Smoke, exploratory and not graded** (2026-10-14, `SMOKE=1`, one 64-token completion per arm on Mellum2.1, into an empty directory): the
  script ran end to end, both sessions resident and windowed, texts identical across arms, 65.9 against 63.5 tokens a second (one
  completion each, which resolves nothing).

**Amendment 1, 2026-10-14, before any run: N raised by a power check.** The owner asked for `scripts/power.py` on both cells at N = 6 per
arm before the run. The answer is no. The paired unit in `power.py` is one restart per arm. The registry entry covering both cells is
`seed-nobara-cuda-goinfer-all-S` (direct A/A, sd 0.0081 log units per restart, floor 0). There is no Mellum2.1- or 26B-specific A/A, so both
cells fall back to it:
`power.py served-decode nobara/cuda/<model>/d2048 --bar 0.97 --expect 0.99 --paired` needs **4 paired restarts** to resolve 0.99 against
0.97 at power 0.80 (exact t power 0.905 at 4). One ABBA quad gives 2 a model.
- **Changed:** per model, two quads in the order main, new, new, main, new, main, main, new (ABBA then BAAB), so 8 sessions, 4 restart pairs
  and 12 completions an arm. `speed_guard.py`'s `ORDER` is the only code change, and the pinned copy the night job runs is the same file.
- **Unchanged:** the cells, the workload, K = 3 completions a session, the rule (medians, PASS >= 0.99 on both, FAIL < 0.97 on either,
  otherwise parked), the void conditions and the prediction.
- **Disclosed:** the registry's sd was measured on restarts of 16 to 24 completions, and these sessions hold 3. Within-session spread adds
  to it in proportion to 1/sqrt(3). Decode rate within a served session is steady, so this is taken as small, but it is not measured here,
  so 4 pairs is a floor, not a margin.
- **Cost:** 8 sessions x (137 s Mellum2.1 + 200 s 26B), about 45 minutes; re-queued at 50.

**Result, 2026-10-14: PASS on both models**, graded as registered with amendment 1. It ran on nobara under the night runner, started by
the owner at 11:20 PDT, 11:34 to 12:34. The binaries were checked against `binaries.sha256` first (both OK), on driver 595.91.07 with
the GPU idle at the start. All 8 sessions per model ran, with 12 completions per arm, each 640 of 640 tokens. Every session was
resident and windowed, so none was void.

| model | median main (tok/s) | median new (tok/s) | R = new / main | range main | range new | verdict |
|---|---|---|---|---|---|---|
| Mellum2.1 (`--ctx 16384 --moe-cache-experts`) | 67.10 | 66.90 | **0.9971** | 66.70–67.76 | 66.68–67.26 | PASS |
| Gemma-4-26B-A4B (`--moe-cache-experts`) | 37.25 | 37.13 | **0.9970** | 36.95–37.54 | 36.56–37.46 | PASS |

Read beside the verdict and not graded: the per-restart-pair ratios (mean new over mean main within each adjacent main/new pair) were
1.0031, 0.9995, 0.9897 and 0.9979 on Mellum2.1, and 0.9965, 0.9883, 0.9989 and 0.9962 on the 26B. The texts are identical within each
arm and across arms. The prediction (PASS 0.7) holds. The step stays at 64. Files: `speed-guard-2026-10-14/` beside this record
(`summary.json`, `results.json`, `provenance.txt`, `binaries-check.txt`); the server logs are in the run directory under
`~/goinfer-logs/kvwindow-ring-2026-10-14/`.
