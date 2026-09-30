# Review notes, batch 1: writeups 13, 14, 16, 18 (drafted 2026-09-29)

All four are drafts (`reviewed:` empty). Every figure listed in each file's `figures` appears in its body and in the named record (checked by the site's own `CheckWriteups`). Nothing was measured or run for these; the two example commands (13, 14) are illustrative, not recorded runs.

Build note: the real `site` build fails on 13 only, at `verify.go`'s "its page never names it". The title has an apostrophe ("doesn't") and the page HTML-escapes it to `&#39;`, so the raw-title `strings.Contains` misses. It is a site bug, not a writeup one (compare against `html.EscapeString(w.Title)`). With that one line patched in a scratch copy, all four build with `-drafts`.

Body lengths (words, tables and code included): 13 = 977, 14 = 925, 16 = 894, 18 = 771. 13 and 14 run over the 900 target.

## 13 Batching that doesn't change the answer (measured 2026-09-27)

| Claim | Source (file, heading) |
|---|---|
| Metal 1.5B, 4 clients: 1.593×, 76.2 to 121.5 tok/s, p99 turn 7.09 to 4.67 s, lone request 1.002× / 1.004×, macOS 26.6.2 | `concurrency-mc3-2026-09-26.md`, "Result" and "The W7 grading" |
| Metal 7B, 4 clients: 1.785×, 45.5 vs 25.4–25.6 tok/s, p99 0.592×; 2 clients 1.018× | `concurrency-mc3-7b-w7-2026-09-27.md`, "Result", gate table |
| 7B 2 clients later 1.121× (over the previous build) | `concurrency-mc3-s4-2026-09-27.md`, "Result" |
| 1.5B later step polish: 153.1 vs 117.8–121.6 tok/s, 1.261× (vs the previous batched build) | `concurrency-mc3-s3-2026-09-27.md`, "Result" |
| CUDA 1.5B 1.380× (218.0 to 300.6), 7B 1.826× (74.03 to 135.16, one pair, not gated), 99.8% of tokens in 4-row steps, driver 595.91.07 | `concurrency-mc3-cuda-2026-09-27.md`, gate table, "W7 cells" |
| CPU 7B 2.19× the step-1 workers (5.44 to 11.92), lone 1.000×, 0.5B forced-on 0.73× | `concurrency-mc3c-step2-2026-09-27.md`, "Result" |
| Bit-identity test: 4 sequences × 12 steps, 0 logits differ; gates registered before timing | mc3 record, "Identity and correctness gates" and "The W7 grading" |
| Defaults `-max-concurrent` 4, `-kv-sessions` 4, `-prefill-chunk` 512, `-max-queue` 8 (429); `-cpu-batch auto` at 2 GiB; ~117 MB slot | `docs/server.md` (concurrency paragraphs); `internal/serveapp/main.go` flag definitions |
| Banner line text | `internal/serveapp/banner.go` |
| Not continuous batching or paged attention; bit-identity is within machine and OS | `docs/positioning.md` |
| Which families stay one at a time | mc3 record "Scope"; cuda record "Reading"; server.md |

**Conflicts and how they were resolved**
- The brief's "Metal 1.59×" is the 1.5B, MC3-only build (2026-09-26). The newer Metal 7B figure (1.785×, 2026-09-27) is on a later stack (S2, S3, chunked prefill). I show both with their models and dates and say the builds differ.
- The 1.5B has a later gain (S3, 153.1 tok/s) measured against the previous batched build, not the serialized one. I did not chain it into a new ratio against serialized (76.2 to 153.1 would be a cross-record ratio). The newest direct 1.5B-vs-serialized ratio stays 1.593×.
- `server.md` rounds to 1.79× and 5.4 to 11.9; the page quotes the records' 1.785× and 5.44 to 11.92.
- The CPU 2.19× is against the step-1 workers (already concurrent), not one-at-a-time serving. The page says so.

**Not verified, left out**
- The CUDA batched-head change (2026-09-28) has an in-process gain in `task-concurrency-2026-09.md`, but its W7 rerun is only "queued". No record found, so CUDA rows are the pre-change figures.
- Whether CUDA and WebGPU are affected by the `--embed-int4` default flip (`docs/quantization.md` says unchecked). Metal is stated plainly in Use it and in "doesn't".

**Questions**
1. The Metal example needs `--embed-int4=false` until the known issue is fixed. Publish with that in the text, or hold the page?
2. Queue a fresh 1.5B full-stack-vs-serialized Metal run for the night, so the headline Metal number is not two builds old?

## 14 Faster, with the same words (measured 2026-09-28)

| Claim | Source |
|---|---|
| Copy 2.082×, chat 1.068× (1.5B); 7B 1.846× / 1.014×; earlier build 0.977× / 0.956×; 109 vs 52.7 tok/s; chat rounds 1.066–1.317 outlier; bars (1.25×, 0.97×, identity hard) | `metal-spec-step-verify-2026-09-27.md`, headline table, "The served grading" |
| Extra verify row 0.108–0.235 of a token at M = 8; old verify 0.71–0.96 | same record, §1 gate 2; `spec-vs-batching-metal-2026-09-27.md` "Result" |
| Earlier build 0.979× copy, 0.928× chat; under 4 clients 0.610× / 0.489× | `spec-vs-batching-metal-2026-09-27.md`, §2 table |
| Later turns differed: 3 / 18 and 15 / 72 equal, 30 / 30 single-turn; cause (one token short, 156 vs 157); fix commit 97615930; 24 / 24 after | same record, §4 and "Update: §4's divergence found and fixed" |
| CUDA block drafter same gap; 14/14 with fix, 11/14 reverted (nobara, 2026-09-28) | same record, "the other speculation loops checked" |
| Mechanism (prompt-lookup index, adaptive depth) | `docs/spec/02-cache-ngram.md`; metal record §2 |
| `--spec ngram`, greedy identical, sampled "in-distribution", per-request fallback | `internal/serveapp/main.go` flag help; `docs/server.md` |
| Use for one user or agent loop, not a shared server | `docs/server.md`; metal record "Reading" |

**Conflicts**: the brief's "0.98×" matches two records (0.979× the morning measurement, 0.977× the graded run's old arm). Both are quoted and explained. `server.md`'s "1.85×" for the 7B is 1.846× in the record; the record is used.

**Not verified, left out**: the CUDA 2.14× on a verbatim-copy prompt (`spec-decode-lane-2026-09-21.md`) was not opened. Any CPU or CUDA speed claim for n-gram speculation is absent. Whether "speculate when alone, batch under load" is still unbuilt rests on the task doc header ("registered, owner to decide").

**Questions**
1. Keep the CUDA block-drafter sentence in "What we got wrong"? It is true and in the record, but the page is a Metal page.
2. The measurements predate the `--embed-int4` default flip. `--embed-int4=false` reproduces the measured configuration (the earlier default), but nobody re-ran it with the flag. Acceptable?

## 16 Numbers with their receipts (measured 2026-09-28)

| Claim | Source |
|---|---|
| Row rules (same machine, checkpoint and quant, greedy, pinned versions, date, thermal note, local disk, engine `usage`, idle box with state recorded) | `docs/benchmarks.md`, "Methodology" |
| 1.78× retired 2026-08-09: in-process kernel number divided by peer end-to-end; 320.1 vs 269.4 tok/s server to server | `docs/legacy-benchmarks.md`, "RETIRED (2026-08-09)" box under §B2 |
| 1587.1 s vs ~350 s, 4.5× | `docs/benchmarks.md`, Methodology, "Verified-idle box" bullet |
| tokens per chunk 1.0000–1.0456 vs 1.0000–1.0909 across 130 cells; 7 of 24 cells voided | `docs/benchmarks.md`, Methodology |
| Harness guarantees (own HTTP, decode-only from first token, interleaved with restarts, provenance header, idle refusal, per-tensor same-weights check) | `scripts/bench_peer.py` docstring |
| Pre-registered bars, pair-decided outcomes, void rules | `peer-claim-2026-09-25.md`, "Part 1" |
| The example row and the sweep result table | same record, "Cells a–f in detail", Part 2 table, Part 3 "The claim" |
| 0.81%, 1.64% (0.978–0.996), 466 pairs, 0.15%, 0.74% / 3.12% / 5.03% (426 CUDA pairs), 112.4 → 116.5, 16 gaps, three overstated records | `docs/measurements/noise-registry.md` §2, §3, §5, §7 |

**Conflicts and choices**
- `docs/benchmarks.md` "Reproduce it" still names `bench_compare.sh` as the way to reproduce, but `bench_peer.py`'s docstring, the retired-1.78× box and `CLAUDE.md` say it drives no peer. The page follows the latter. That section of benchmarks.md is worth fixing.
- The "3.5%" cross-session drift: `CLAUDE.md` says 3.6%, the registry says ~3.5% and reads it as one observation. The page quotes the two raw readings (112.4, 116.5) and no percentage.
- The void rule: the peer-claim record used 95% of requested tokens; `benchmarks.md` now recommends "fewer than half". The page says "too few tokens" with no number.
- The phrase "a cross-session ratio is not a ratio" is in `CLAUDE.md` and in the registry's suggested wording, not as a heading in the registry.

**Not verified, left out**
- "More than three machines": `site/data/machines.json` lists three keys (Mac, CUDA, CPU) but they are two computers (nobara has both the CPU and the GPU). The page says two computers.
- No actual thermal-state value is quoted. The harness records load and GPU temperature per cell; I did not open a cell's temperature.
- Whether newer records supersede the 2026-09-25 sweep numbers (later CPU and Metal work exists: e.g. an avx2quant gate, R18). The table is dated and matches the record. `claims.json` uses the same record.

**Questions**
1. Show the sweep table at all? It includes losses (Metal 0.58–0.75×, CPU 0.80×), which suits the page but may be stale against newer CPU/Metal work.
2. Is the 1587.1 s story fine to publish (it reads as an internal error)? It is the sharpest example of a rule that passed and was still wrong.

## 18 Starts in a hundredth of a second (measured 2026-09-24)

| Claim | Source |
|---|---|
| Load 0.00 s / 0.01 s vs 5.61 s / 16.49 s; heap +0 / +1 MB vs +1262 / +4961 MB; RssAnon, RssFile | `cpu-giw-vs-direct-2026-09-24.md`, "Reported, not decided on" |
| Decode 1.0007 / 1.0016, A/A 0.9995 / 1.0035; hard gate identical streams; pre-registration | same record, "Results", "Pre-registration" |
| Machine, date and time (08:52–09:02, load < 0.6), v12 weights / v3 bundle, 1.29 / 5.18 GB, transcode 20.9 / 102.1 s | same record, "Results" |
| CUDA: 2.6–2.8 s vs 8.0–8.2 s (1.5B), 10.0–11.0 s vs 26.2–26.8 s (7B); first start 29.7 s / 110.4 s | same record, "CUDA, and the linux default flip" |
| macOS memory: 327 MB vs 1572 MB (1.5B, before v12) | `sidecar-default-2026-09-22.md`, "Results" |
| Default on darwin and linux; `-direct-load`, `GOINFER_GGUF_DIRECT=1`; refusal when disk is short; transcode message; `cmd/prequant` flags; a directly passed `.giw` is not a cache | `internal/prequant/prequant.go`; `internal/loadflags/loadflags.go`; `cmd/prequant/main.go`; `CHANGELOG.md` |
| v12 aligned-scale change on 2026-09-24 | commit `cdae727d` (message, 07:14 -0700) |

**Re-measured after the scale-alignment change?** The figures were measured after it, on v12 / bundle v3 files (the record's own header line), the same morning. They were not re-measured on later formats: v13 and v14 (Metal alias, opt-in then default) and the unreleased v15 (CPU f16 scales), after which an older int4 sidecar is rebuilt once (`CHANGELOG.md`, Unreleased). The page says so in the last "doesn't" entry. The only post-change CPU figure I found is peak RSS in the CHANGELOG (1.5B: 1,123–1,127 MB before, 1,045–1,053 MB after rebuild); I did not use it, since it is a different quantity.

**Conflicts and choices**
- The brief's "0.01 s against 5.6–16.5 s, 1.3–5 GB" is `DefaultToSidecar`'s code comment. The record prints 0.00 s (1.5B) and 0.01 s (7B), and Go heap +1262 / +4961 MB. The page uses the record's figures.
- The brief says the conversion "takes minutes". The record says 20.9 s (1.5B) and 102.1 s (7B) on CPU, and 29.7 s / 110.4 s for a CUDA first start. The page quotes those, and the "minutes" only inside serve's own message.
- "About the model's size on disk": the record shows 1.29 GB and 5.18 GB sidecars; the code's comment gives 1.02–1.16× the q4_k_m source for int4. I did not restate the ratio.

**Not verified, left out**: a cold-cache first load (the record says not measured); any Metal or WebGPU load-time figure.

**Questions**
1. The title overstates the scope: 0.01 s is the CPU `Load` call with a warm cache. On CUDA the same 7B load is 10.0–11.0 s (against 26.2–26.8 s). Reword the title (e.g. "Loads in a hundredth of a second, after one conversion") or keep it and rely on the body?
2. Queue a night re-measure of load time and heap on the v15 format, so the figures match what ships?
