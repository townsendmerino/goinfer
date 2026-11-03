# Review notes: writeup 09, "Turn nine in under half a second" (drafted 2026-09-29)

Draft, `reviewed:` empty. Body about 840 words (tables and code included). Site build with `-drafts` passes. Nothing was run: no server, tests or benchmarks. Everything was read from files.

## Claims and sources

| Claim | Source |
|---|---|
| 10-turn agent replay: turn 1 cold 727 prompt tokens, 0 reused, 651.2 ms; turn 9 1922 prompt, 1668 reused, 319.5 ms; turns 2-10 191.2-319.5 ms (all ten rows in the table) | `docs/benchmarks.md` "W4 (Qwen2.5-7B-Instruct, GGUF Q4_K_M) - agent-turn transcript replay" (base variant table); raw `docs/measurements/peer-matrix-2026-09/nobara-w4-transcript-d7_2026-09-09.json` (I diffed the table against `variants.base[].probe`) |
| Machine/date: nobara-pc, RTX 2070 SUPER, driver 595.91.07, 2026-09-09, goinfer `b22e080e` dirty, greedy, 7B Q4_K_M as int4, CUDA resident | same section "Provenance"; JSON `header` |
| "Time to first token" = wall clock of a `max_tokens:1` non-streaming request, because forced tool calls do not stream | same section ("Why TTFT here is from two non-streaming requests"); `scripts/bench_peer_transcript.py` docstring |
| Without reuse turns 1-3 8.86, 9.01, 9.13 s; with reuse 8.80, 0.58, 0.42 s; 25 tool schemas, 7B, RTX 2070 SUPER, verified 2026-09-02 | `docs/integrations/claude-code.md`, "What to expect" table and top "Verified end to end 2026-09-02" |
| CPU path: LRU of sessions, longest shared prefix, beats what it shares with other sessions, rewound and suffix prefilled | `internal/serveapp/sessions.go` (`acquire` comment, `bestExtend`, `pickSession`) |
| GPU path: `--kv-sessions` = KV slots, binds slot holding the prefix, new conversation takes empty slot else LRU; memory guard clamps; ~117 MB (1.5B, 4k, Metal) | `docs/server.md` "On Metal, CUDA and WebGPU, several conversations stay resident" |
| `usage.prefill_reused_tokens` | `docs/server.md`; `docs/tasks/task-turn-telemetry-2026-09.md` section 1 |
| Default 4, 0 disables, `--session-dir` persists warm sessions | `internal/serveapp/main.go` flag definitions (`kv-sessions`, `session-dir`); `docs/server.md` |
| Banner strings | `internal/serveapp/banner.go` lines ~176-193 |
| Tokens prefilled turns 2-6: 26, 27, 24, 20, 27 at 1/2/4 clients (new), 187, 342, 494, 642, 797 (old, Metal 4 clients) | `docs/measurements/concurrency-mc1-2026-09-26.md` Results table (M1 Pro 16 GB, macOS 26.6.2, qwen2.5-coder-1.5b q4_k_m int4) |
| Same on CUDA; old 4-client 176, 331, 483, 631, 786 (the record prints these for the 2-client old cell, and the 4-client old cell prints the same list) | `docs/measurements/concurrency-mc1-cuda-2026-09-27.md` Results table |
| CPU thrash: `e1c867f6` 2026-09-23, 7 tokens reused per turn, 0.69× at 2 clients, live 3 days, fix 27.4 → 43.3 tok/s | `docs/measurements/concurrency-mc0-2026-09-26.md` header bullets and "The CPU bug" |
| Metal at 2 clients reused 7 tokens per turn after the first (one slot) | same, header bullets |
| `--spec ngram` trailing-token: 156 vs 157 reused positions, batched-prefill arithmetic differs, later turns diverge, fixed by `finishTrailing` | `docs/measurements/spec-vs-batching-metal-2026-09-27.md` ("The fix" section and the note near line 82); the same story is in writeup 14 |
| Limit: 7B on 8 GB CUDA at 8192 got 2 of 4 slots, 55.8 against 55.5 tok/s, 896 MB per slot | `docs/measurements/concurrency-mc1-cuda-2026-09-27.md` header and "The 7B: the clamp binds" |
| Limit: recurrent/hybrid keep one slot, reuse only an exact strict extension | `docs/server.md` ("One slot: the recurrent families"); `decoder/resident_reuse.go` comment at the `hasRecurrentState` rule |
| Limit: CPU fast prefill attention, suffix >= 512 tokens not bit-identical, `GOINFER_CPU_FAST_ATTENTION=0` makes it exact, 1.43x on a cold 2048-token turn | `docs/server.md` "Prompt-prefix KV caching" |
| Edited early message = prefilled again after the change | `docs/integrations/claude-code.md` ("Reuse is skipped whenever the prompt diverges..."); `internal/serveapp/sessions.go` (`acquire` doc) |
| `--embed-int4=false` on Metal | `docs/quantization.md` "Known issue" (line ~134) |
| Command line, replay script | `docs/server.md` line ~513; `scripts/bench_peer_transcript.py` argparse (`out`, `--model`, `--backend`) |

## Title

"Under half a second" is supported for the W4 shape: every warm turn 2-10 read 191.2-319.5 ms, turn 9 319.5 ms. It is not supported in general. The earlier 25-schema Claude Code record has its warm turn 2 at 0.58 s (turn 3 at 0.42 s). The page says so in the last "It doesn't" item. Only CUDA has a first-token record; Metal records have only tokens prefilled and whole-turn time.

## Conflicts and how I resolved them

- `docs/server.md` ("Prompt-prefix KV caching") says a session is reused only if its whole history is a prefix of the new prompt, and that a session diverging anywhere is not reused. The code (`sessions.go`, `acquire`/`bestExtend`, L-15/P-18) reuses the longest common prefix and rewinds the session. The code is newer; I described the code. `server.md` looks stale here and should be corrected (not touched: only two files are mine).
- `docs/integrations/claude-code.md` says reuse is "single-conversation: two interleaved conversations will each cold-prefill". MC1 (2026-09-26 Metal, 2026-09-27 CUDA/WebGPU) superseded that with per-conversation slots. I used the MC1 records and `server.md`, and only used claude-code.md for its 2026-09-02 numbers. It is stale too.
- The MC1 CUDA record prints the old-build turn 2-6 counts as "176, 331, 483, 631, 786" for the 2- and 4-client cells; I quoted them as the 4-client CUDA row.

## Left out as unverified

- A cold (no-reuse) time for turn 9 of the W4 replay: the record has no reuse-off arm for it. The page shows turn 1 cold (727 tokens) and does not claim a ratio for turn 9.
- Any first-token time on Metal or CPU with reuse, and any repeat or spread for the W4 numbers (one run each, base and edited_turn6 differ by a few ms per turn).
- Whether `--session-dir` works with GPU-resident slots: `server.md` says a resident model has no session cache; I wrote "the docs describe that flag for the CPU session cache". Please check.
- `GOINFER_NO_RESIDENT_REUSE` is named in `banner.go` but I could not find it in `testdata/env_reads.txt`; not mentioned on the page.

## Questions for the owner

1. Title: keep "Turn nine in under half a second", given it holds on one CUDA replay (319.5 ms) but not on the 25-schema turn (0.58 s)? Alternative: "Turn nine in a third of a second" is more exact for the W4 number but reads oddly; or "A long conversation, without re-reading it".
2. The W4 row is a single dirty-tree run from 2026-09-09. Is that enough for the headline, or queue a night re-run (with 3+ repeats and a Metal first-token cell) before publishing?
3. The "one token short" paragraph is `--spec ngram` history and overlaps writeup 14. Keep it here as a reuse-correctness example, or cut it and rely on 14?
