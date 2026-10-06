# Review notes: writeup 08 "A 26B model on an 8 GB card" (drafted 2026-09-29)

Draft only (`reviewed:` empty). Body about 810 words. The site build passes with `-drafts`. Nothing was run or measured; every figure is copied from a record.

## Claims and sources

| Claim | Source |
|---|---|
| 40.2 tok/s (runs 40.2 / 40.2 / 40.1), Ollama 22.2 (22.2 x3), llama.cpp 27.6, ratio 1.815, ctx 2048, greedy, 2026-09-25 | `peer-claim-2026-09-25.md`, "Cell c" table (line ~257) and `peer-claim-2026-09-25/c-26b.json` |
| RTX 2070 SUPER 8 GB, driver 595.91.07, Ryzen 7 3700X, Nobara 44; Ollama v0.32.5; llama-server 427291b | same, "Cells a-f in detail" Provenance; Configuration section |
| Cell c is an architecture comparison, never in a family claim; goinfer int4 `.giw`, peers Q4_K_M GGUF, CPU offload | same, cell table note (line ~96) |
| Run = mean of 8 completions of 64 tokens, timed from first streamed token; 3 runs | same, Part 1 "The ratio and the bars"; `c-26b.json` (24 completion rates, `ngen` 64) |
| Peak RSS 25.5 GB vs 17.2 (Ollama) and 17.0 (llama.cpp) | same, "Cell c, read with care"; `c-26b.json` `rss_peak_kb` 25536724 |
| Repeated-prompt cache-warmth caution; overlap took 30.4 to 38.7 on same checkpoint | same, "Cell c, read with care" |
| Figure is dated, re-run queued at pinned commit `754f12d3` | `peer-sweep-2026-09-29.md` (status, "Pinned commit", cell c inherited as "the 26B at BENCH_CTX=2048") |
| 128 experts, top-8, ~11.4 GB int4 experts, ~714 MB/token, PCIe the wall, 76.1% / 82.2% hit rate, 16.12 / 17.62 pre-overlap | `benchmarks.md` B4 and B4.1 (tables, 2026-08-27, test harness with synchronous H2D) |
| Ollama ~24.5 tok/s on Google QAT q4_0 GGUF (older, different file) | `benchmarks.md` B4 "RE-VERIFIED" block; the "Superseded 2026-09-25" note says that reading no longer holds |
| Mechanism: full expert stack in pinned host memory, small device slot block, per-layer LRU, slot count capped to free VRAM (64 requested, 29 granted, 3.4 GB free in one load) | `cuda/resident.go` `cacheWQ` / `upExperts`; `moe-streaming-decode-overlap-ab-2026-09-22.log` first lines |
| Every expert executes on the GPU, "bit-identical to fully-resident", off by default, declines to CPU without it | `internal/loadflags/loadflags.go` `moeCacheExpertsHelp` |
| DMA overlap: second queue, dense branch and hit experts run under the copy; 1.271x paired median, 30.36 to 38.66 pooled, bit-identical; off under graphs; prefill untouched | `moe-streaming-decode-overlap-ceiling-2026-09-22.md` "Result 2" and "What this does not cover"; `cuda/backend.go` line 1865 |
| Serve command: `-model bench=<path> -backend cuda -moe-cache-experts -ctx 2048` | `scripts/bench_peer.py` (the Popen at ~line 1080, `GOINFER_MOE_PATH["M26"]`, `moe_args`, `CTX_PIN`) |
| Flags `-moe-cache-experts`, `-moe-cache-slots`, `-ctx`, `-require-backend`; in chat too | `loadflags.go` (lines 63-66), `internal/serveapp/main.go:429`, `CHANGELOG.md` line 57 |
| Halving ctx 4096 to 2048 gave 30 to 40 slots | `benchmarks.md` B4.1 table |
| Metal: three attempts (N=64, 32, 8) on 16 GB M1 Pro, swap spirals, killed; no served rate | `metal-moe-autopager-m26-2026-09-20.md` (result paragraph, "Out of scope") |
| Metal later: 5.98 / 6.19 / 6.07 / 6.04 tok/s, N=8, ctx 512, 32 tokens, n=2 per arm, "not a registered bench" | `m26-alias-fork-collapse-2026-09-24.md` section 1 (cell) and 7a (table) |
| Metal streams experts by pread into fixed device slots | `docs/tasks/task-never-swap-2026-09.md`, memory-budget table row for "Metal resident, .giw source" (line ~111) |
| "capacity-bound, not a kernel or MoE deficiency" | `docs/capability-matrix.json` gemma-4-26b-a4b `needs` (line 353) |
| Code/tests exist for Gemma 4, gpt-oss, Qwen3.6-35B-A3B under the cache | `cuda/gemma4_26b_cache_test.go`, `cuda/gptoss_real20b_test.go`, `cuda/qwen35moe_35b_cache_test.go` (headers) |
| Only int4-kind experts take the slot path | `cuda/resident.go` `upExperts` |

## Conflicts between records, and how I resolved them

- **Ollama's rate on this model.** `benchmarks.md` B4 still has the block "faster than goinfer's 16.98 tok/s, ~24.5 tok/s" (Ollama, Google QAT q4_0 file, 42% GPU / 58% CPU). Cell c has 22.2 on the Q4_K_M file. Used cell c as the newest, and named the 24.5 only in "doesn't" as a different file, to show the gap depends on the checkpoint.
- **Ratio wording.** The record prints "1.82x" in prose and 1.815 in the table (median). I quoted 1.815 (no rounding of mine).
- **The Metal record contradicts itself.** `m26-alias-fork-collapse-2026-09-24.md` section 8 says "M26 with the fix: not run", but section 7a, same day, has four M26 arms with the fix. I used 7a. Section 8 looks stale.
- **The "1.2-1.4 tok/s" figure** (the retracted Zeno/35B number) is not quoted. The Metal 26B record itself never obtained a rate in the 09-20/09-22 attempts.
- The brief said "keeps every expert on the GPU". The experts live in host memory and execute on the GPU; the page says both, so "on the GPU" is not read as "in VRAM".
- The README (and a comment in `bench_peer.py`) call the CLI example a GGUF; the 40.2 was measured on the `.giw`. The page says so.

## Left out as unverified

- How `~/models/gemma4-26b-int4.giw` was built (which `prequant` command, or whether it is the sidecar). `cmd/prequant` accepts `-quant int4`, but I found no record naming the command for this file. The page does not say how to make it.
- Which slot count the 40.2 run actually got (the served run's log is not in the record). The "29 of 64" figure is from the 2026-09-22 A/B log, and is labelled as one logged load.
- Whether Ollama's placement in cell c was the same 42/58 split (not recorded for the Q4_K_M file).
- Any measured rate for gpt-oss-20b or Qwen3.6-35B-A3B under `-moe-cache-experts` on CUDA (README says gpt-oss is "not yet as thoroughly measured"). Only "tests exist" is said.
- Host RAM of nobara-pc, and whether 25.5 GB peak RSS is stable across the 9-29 re-run.
- Whether the 09-29 night job has actually run: I only read the pre-registration, which says "queued". I did not run `night.py list`.
- The Metal machine in the 09-24 record is named only "MacBook darwin/arm64, 16 GB"; the page implies it is the same M1 Pro as the 09-20 runs.
- The `GOINFER_MOE_DMA_OVERLAP=0` knob is read from `cuda/backend.go`; it is not on `testdata/env_reads.txt` (I did not check how `knobValue` resolves it, env or Options), so the "Use it" bullet may need trimming.
- Prefill for the 26B (not covered by the overlap; `prefill-moe-m26-2026-09-04.md` exists, not read).

## Things to check

- The peer-claim record's own provenance says `goinfer_tree_dirty: true` for the cell c run (commit 411e7fc4). The page names the commit, not the dirty flag. Decide if it should.
- "Metal pages experts from disk" rests on one table row in the never-swap task doc; the wording is mine.

## Questions for the owner

1. Should the page wait for tonight's re-run of cell c (`peer-sweep-2026-09-29.md`) and quote the new figure, or publish with 2026-09-25 and the "may be replaced" note?
2. The Mac "doesn't" quotes 5.98 to 6.19 tok/s from a non-registered n=2 run on 2026-09-24. Is that fit for a public page, or should it say only "swap spirals, then a slow bounded run"?
3. Should the page show how to build the `.giw` bundle so a reader can reproduce 40.2? Right now the README's GGUF command is the only runnable form, and it is a different file.
