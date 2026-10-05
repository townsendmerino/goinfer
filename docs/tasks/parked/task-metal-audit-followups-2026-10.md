# Task (parked): what was left when the Metal audit program closed (2026-10)

> **Filed 2026-10-05**, when [`task-metal-audit-2026-10.md`](../../completed/task-metal-audit-2026-10.md) (the program
> that ran `docs/audit-metal-2026-09-30.md`'s §10 items) was archived. Everything it scoped is done, graded, or listed
> here with a trigger. The audit itself, [`audit-metal-2026-09-30.md`](../../audit-metal-2026-09-30.md), stays live until
> every finding in it is closed; this file is the program's remainder, not the audit's. M26 work, D-P01's batched expert
> phase 2 included, is live in [`task-m26-mac-2026-10.md`](../task-m26-mac-2026-10.md).

| # | item | state | trigger or owner |
|---|---|---|---|
| 1 | Short-prompt Metal TTFT rows (T1.11's other half, A-D01) | **Queued** as `metal-audit-shortk-ttft` (`docs/measurements/metal-audit-2026-10/run-shortk-ttft.sh`, pinned `93d586af`) for the night of 2026-10-05; pre-registered below | The morning after it runs: grade it against the rule below and replace the 2026-09-20 table in `docs/benchmarks.md` ("Metal short-prompt prefill floor — R3") |
| 2 | The int8 MoE rows form's speed (`gemv_w8a8_moe_rows`, on, bit-identical) | **Unread.** The Qwen1.5-MoE slice at int8int8 needs 5.43 GB resident; the guard declined it twice, and a bypass with the owner's one-off OK was killed by the kill-watch at +1.2 GB of swap in 19 s (2026-10-05). The owner declined both workarounds offered (a 2-layer slice, a Qwen3-30B-A3B slice) | A resident int8 MoE that fits a 16 GB Mac, or a Mac with more memory. Until then it stays on, unread |
| 3 | C-N01, the `Encoder.Dispatch` binding-length check | In aikit's tree (`a143d8d`, in aikit v1.56.0), but the `gpu` module is tagged separately and `gpu/v0.33.3` predates it | The next aikit `gpu/` tag: bump `metal/go.mod`'s `aikit/gpu` require |
| 4 | T1.13, gpt-oss's int4 requantisation error | Not run: the checkpoint is not on the Mac and G20 is off-limits here | nobara. Its numbers decide O5 |
| 5 | The owner decisions O1–O7 (table below) | None answered. O2, O4 and O6 have their preconditions in | The owner |
| 6 | D-P03, the k-slot expert dispatch | Parked at 1.019× on the Qwen1.5-MoE slice (k = 4), just under the bar; bit-identical, off | A k = 8 model resident (OLMoE, Qwen3-MoE, DeepSeek), where 20 dispatches go instead of 8. None fits resident on this Mac |
| 7 | E-P02's calibration follow-up (adjacent per-row up to B = 3) | Read 1.000 on the 7B at B = 3; not kept | A model where the per-row form wins at 3-4 clients in the whole step, not only the kernel |
| 8 | A-P03, small-M prefill attention at depth | Parked: T1.1's paired delta 15.5 ms, between the 10 ms close and the 25 ms build lines | The owner, or a served short-prompt TTFT at depth that names attention as the gap |
| 9 | B-P08, the block kernel's depth staircase | Stands on the 7B, parked by the owner 2026-10-03 (low value) | The owner |
| 10 | M26's lever 3 prefetch and C-B03's phase-1 fence | Built, gated bit-exact, off (0.894× / 0.947× and 0.878× / 0.848×); recorded in the M26 doc | A better guess that needs no phase-2 wait; a cheaper way to publish a layer's writes |

#### The owner decisions (the audit's Track 4)

| ID | Question | Needed before | Recommendation | Status (2026-10-05) |
|---|---|---|---|---|
| O1 (A-P04) | Embeddings: exact or batched | A-P04 | Pipeline first: bit-identical, 1.05–1.5×. Decide batched after measuring it | **Open.** A-P04 not built |
| O2 (D-B05) | `delta_rule`'s summation order | D-B05 (after D-B01) | Decide once D-B01 shows the recurrence share is large | **Open, now askable.** D-B01 shipped (8.06× on the 0.8B prefill); the recurrence share has not been read |
| O3 (B-N01) | An int4 head on Metal | none in this plan | Defer: lossy, about 2.3 points top-1, and its quality re-eval is parked | **Open, deferred** (recommendation stands) |
| O4 (B-P05) | Move decode's down projection off the per-word chain | phase 3 step 4 | Decide after B-P04 and B-P06, with the fidelity gate's result | **Open, now askable.** Both preconditions are in: B-P04 killed 2026-10-03, B-P06 killed 2026-10-04 |
| O5 (D-B03) | Native MXFP4 for gpt-oss | after T1.13 | Decide on T1.13's error numbers | **Open, waiting on T1.13** (nobara) |
| O6 (E-P04) | Device top-k/top-p/min-p | after phase 3 | Decide after the batched-step items | **Open, now askable.** The batched-step items are done (E-P02/E-P05/E-P08 shipped, E-P03 killed) |
| O7 (F-B01) | Keep P in f32 in steel | after F-G01 | A fidelity question; run the pooled gate first | **Open** |

#### Item 1's pre-registration, moved verbatim from the archived doc (written 2026-10-05, before it runs; owner: "lets work the 'Builds still open' in order")

`docs/benchmarks.md`'s "Metal short-prompt prefill floor — R3, 2026-09-20" table (1.5B, K = 32 / 64 / 128 / 256) predates
R16's prefill GEMM, R19's prefill attention, A-P01's tile selector and A-P02's floor of 16, and its 2026-10-01 note says
the ratios to Ollama are not current (A-D01). This re-measures it.

| | |
|---|---|
| Instrument | `docs/measurements/metal-audit-2026-10/run-shortk-ttft.sh` on the night queue: the table's own command, `scripts/bench_peer_prefill.py --backend metal --models 1.5B --depths 32,64,128,256 --n 6`, on a pinned serve binary and scripts worktree. Per cell the engines run interleaved, each server restarted: goinfer_exact (`--exact-prefill`), goinfer (the default, which batches from 16 tokens, so no floor override as the 09-20 run needed), Ollama 0.32.5 and mlx-lm 0.31.3 on its local 4-bit checkpoint. 6 unique-prefix prompts a cell, plus the harness's cache check (repeat ÷ fresh) and its depth-scaling check. Estimate about 30 minutes. |
| Reading | A record, not a gate. Per K: goinfer ÷ Ollama, goinfer ÷ mlx-lm, and goinfer ÷ goinfer_exact, each from the cell's TTFT medians, graded by the peer sweep's bands (level 0.97–1.03, ahead > 1.03, behind < 0.97; all six pairs decide, never the mean; an engine's spread over 5% caps the cell to AMBIGUOUS). The cells replace the 09-20 table's, which stays as history. |
| Void if | Any goinfer cell's serve log shows a decode path other than `metal-resident (int4)`, the harness's cache check fails (an engine answering fresh prompts from a cache), or TTFT does not rise with K for an engine (the harness refuses then). |
| Reported | The 09-20 table's ratios beside the new ones; the batched default against `--exact-prefill` at every K. |

**Not carried:** the audit's Minor findings were never swept one by one against the program. They are in the audit doc,
which is live.

<!-- doc-reviewed: 2026-10-05 -->
