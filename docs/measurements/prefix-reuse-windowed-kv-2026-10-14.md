# Prefix reuse under CUDA windowed KV: why Codex x Mellum2.1 re-prefilled most turns, and the fix (2026-10-14)

- **Box:** nobara-pc, RTX 2070 SUPER 8 GB, driver 595.91.07. **Data:** the I3 night of 2026-10-13, job B (`mellum-16k`: `serve --backend cuda
  --ctx 16384 --moe-cache-experts`, serve binary `81791568`), committed under
  `docs/measurements/integration-harness-2026-10/i3-night-2026-10-13/job-b/` (`records.jsonl.gz`, `serve-mellum-16k.log.gz`). No request
  bodies were recorded; everything below is read from request sizes, the server's time to first token, Codex's own session files, and the code.
- **Fix built on:** `origin/main` at `e9dae735`, branch `kvwindow-ring-2026-10`.
- **Files beside this record:** `sim.py` (the replay, Python; the same replay is the Go unit test `TestWindowedKV_i3Replay`) and
  `memtable.py` (the memory table below).

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

## 4. The fix: a fixed rollback, kept at every point of the compaction cycle

**Chosen: keep a rollback reserve in the ring**, not recompute the window tail. Recomputing would start re-prefills at arbitrary positions
and run into the kernel-class rules (starts not aligned to 64, the 16-row tail rule). Keeping the positions changes no kernel and no read.

- **Rule** (`cuda/kvwindow.go:kvWindowNextBase`): a compaction moves the base to `min(first row's window start, end of the write - window + 1 -
  rollback)`, never back. Every compaction therefore leaves `rollback` positions behind the newest one. A follow-up that diverges up to
  `rollback` positions before the furthest position written since the last compaction reuses its prefix, wherever the last reply ended. The
  budget lies between `rollback` and `rollback + 512` and never drops to 0.
- **Sizes** (`decoder/model.go`): `WindowedKVRollback = 1024`, `WindowedKVSlack = WindowedKVRollback + 512`. The 512 is how far a slot writes
  between compactions, and it bounds the rows of one pass, as before. The compaction scratch grows to `window - 1 + rollback` positions of the
  widest windowed layer.
- **Why 1,024.** The rollback a Codex follow-up needs is the previous reply. Over the 59 I3 replies the median is 258 tokens, the 75th
  percentile 584, the 90th 1,328 and the largest 4,096; 53 of 59 are at most 1,024. Replayed at each budget (rollback + 512 of slack):

  | rollback kept | in-run follow-ups cold, of 54 | rollbacks of those |
  |---|---|---|
  | 0, the old rule (512 slack) | 30 | (35 with the run starts: the server's figure) |
  | 512 | 13 | 544 to 5,412 |
  | **1,024 (chosen)** | **5** | 1,327, 1,397, 2,085 (long replies); 5,412 and 2,616 (Codex's own compactions, a new conversation) |
  | 1,536 | 3 | 2,085, 5,412, 2,616 |
  | 2,048 | 2 | 5,412, 2,616 |

  At 1,024 the misses left are replies of over 1,300 tokens, whose own decode (about 33 s at Mellum's 40 tokens a second) is already of
  the same order as the re-prefill. Going to 2,048 saves three more misses in this cell and doubles the added memory, which on an 8 GB card
  running a C′ expert cache comes out of the expert slots. One constant changes it.
- **Memory.** The added cost is 1,024 positions per windowed layer per KV slot. Windowed layers stay f32, because `cuda/kvpolicy.go` gives
  the sliding-window ring no f16 form. So a layer holds `2 x kvDim x 4 x min(ctx, window + slack)` bytes (`cuda/resident.go:kvBytesForCap`,
  which `TestWindowedKV_allocation` proves equal to the bytes allocated, and which decoder's planner prices the same way, so a fitted context
  shrinks to match). The table below is that formula at each checkpoint's real geometry, read from its `config.json` in `~/models` by
  `memtable.py`. No real checkpoint was loaded to read the figures off a device.

| checkpoint | family | window | windowed layers with their own KV / all | kvDim | bytes per position per layer | per layer per slot, window+512 (before) | per layer per slot, window+1536 (after) | added, all windowed layers, per slot |
|---|---|---|---|---|---|---|---|---|
| Mellum2.1-thinking | mellum | 1024 | 21 / 28 | 512 | 4,096 | 6.0 MiB | 10.0 MiB | +84 MiB |
| gemma-2-2b-it | gemma2 | 4096 | 13 / 26 | 1024 | 8,192 | 36.0 MiB | 44.0 MiB | +104 MiB |
| gemma-3-4b-it | gemma3 | 1024 | 29 / 34 | 1024 | 8,192 | 12.0 MiB | 20.0 MiB | +232 MiB |
| Gemma 4 E2B | gemma4 | 512 | 12 / 35 | 256 | 2,048 | 2.0 MiB | 4.0 MiB | +24 MiB |
| Gemma 4 E4B | gemma4 | 512 | 20 / 42 | 512 | 4,096 | 4.0 MiB | 8.0 MiB | +80 MiB |
| gemma-4-26b-a4b-it | gemma4 | 1024 | 25 / 30 | 2048 | 16,384 | 24.0 MiB | 40.0 MiB | +400 MiB |
| gemma-4-31B-it | gemma4 | 1024 | 50 / 60 | 4096 | 32,768 | 48.0 MiB | 80.0 MiB | +1,600 MiB |
| command-r7b | cohere2 | 4096 | 24 / 32 | 1024 | 8,192 | 36.0 MiB | 44.0 MiB | +192 MiB |
| gpt-oss-20b | gpt_oss | 128 | 12 / 24 | 512 | 4,096 | 2.5 MiB | 6.5 MiB | +48 MiB |
| Phi-3-mini-4k | phi3 | 2047 | 32 / 32 | 3072 | 24,576 | 60.0 MiB | 84.0 MiB | +768 MiB |
| Mistral-7B-v0.1 (8 KV heads x 128, 32 layers; no config on disk) | mistral | 4096 | 32 / 32 | 1024 | 8,192 | 36.0 MiB | 44.0 MiB | +256 MiB |

  Phi-3-mini's context is 4,096, so its layers now hold 3,583 of 4,096 positions and windowing saves almost nothing on it. A context at or
  below `window + slack` is held whole, as before.

## 5. Gates

- `TestWindowedKV_i3Replay` (`cuda/kvwindow_replay_test.go`, tag `cuda`, no device): the 59 recorded requests through
  `kvWindowNextBase` and `kvWindowReusable`. The old rule (slack 512, rollback 0) must reproduce the server's cold or warm for every request
  and its 35 cold. Under the production constants, every in-run follow-up whose rollback is within `WindowedKVRollback` must be warm.
  **Before the fix: FAIL**: the old-rule half matched 35 of 59 exactly, and 30 in-run follow-ups with rollbacks of 90 to 1,008 were cold.
  **After: PASS**, with 5 of 54 in-run follow-ups cold, each rolling back more than 1,024.
- `TestWindowedKV_rollbackReuse` (`cuda/kvwindow_test.go`, tags `cuda goinfer_testhooks`, on `mistral-tiny-window`, `gemma3-vl-tiny` and
  `cohere2-tiny` at a test slack of 24 with a rollback of 16). Ten turns. Each prefills a suffix at the previous prompt's end, then decodes
  `rollback` tokens that the next prompt never contains, and the next turn must find its prefix reusable. At least 5 of the 9 reuses must follow a
  compaction. Every prefill and decode is bit-identical to the full-KV resident under the same schedule, and each reused prefill's logits are
  bit-identical to a cold full-KV prefill of the whole prompt. A rollback of `slack + 2` must report 0. **Before: FAIL** on all three
  fixtures (`ReusableKV(42) = 0` after a 16-position rollback, base 33). **After: PASS** on all three.
- The existing `TestWindowedKV_*` suite (decode, prefill, verify, slots, step, upload, reuse floor, agent turns, offsets) passes unchanged
  under the new rule. Its test knob now sets the rollback with the slack.

## 6. For KV2 (passed on by the owner)

KV2's design (`docs/tasks/task-kv-architecture-2026-10.md`, section 5) keeps a window group to `ceil(W/B) + 1` blocks, and plans "one block of
slack behind the window, so a rollback inside a verify width stays valid": a 64-position rollback budget. **That budget does not serve an
agent turn.** The rollback a follow-up needs is the previous reply: median 258 tokens, 90th percentile 1,328 in this cell. Replayed with 64 positions of slack and no rollback reserve, 53
of 54 in-run Codex follow-ups would have prefilled cold. That is worse than the shipped rule's 30.

What this record found that the KV2 design has to keep:

- **A fixed rollback reserve, not a cycling one.** Keeping 1,024 positions (16 blocks of 64) behind the newest position in a window
  group gives the same 5 of 54 as here. With blocks there is no compaction copy, so the extra 512 positions of stride CUDA holds are not
  needed: a 1,024 + 64 replay also gives 5 of 54.
- **Or keep released window blocks reattachable.** If a window block released behind the window stays in the prefix cache at refcount 0
  (the design's "or the cache") and a rollback re-attaches it by hash, the reserve can be smaller. The gate is the same.
- **A gate on recorded traffic.** `TestWindowedKV_i3Replay`'s shape (recorded sizes in, the server's warm or cold out) can be pointed at the
  KV manager's window-group retention once it replaces `kvEnsureSlot`. Its 35-of-59 half pins the old behaviour as the reference.
- `decoder/resident_reuse.go` and the KV manager were not touched by this fix: the floor is still `ResidentReusableKV`, answered by
  `cuda/kvwindow.go`.

## 7. A modularity note

`WindowedKVRollback` and `WindowedKVSlack` live in `decoder/model.go`, a file every parity family's hash covers. Changing a CUDA-only
allocation figure staled 42 families' parity hashes and needed `scripts/refresh_parity_hashes.sh` (36 goldens passed, 0 failed). Moving the
two constants to a file outside the parity core would end that coupling.
