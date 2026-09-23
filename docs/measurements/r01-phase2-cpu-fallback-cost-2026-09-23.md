# R-01 Phase 2 / R-04 decision input — the real cost of today's CPU fallback

**Purpose.** `docs/tasks/task-recompute-audit.md`'s R-01 Phase 2 (resident-GPU state parking)
pre-registered a decision rule (added 2026-09-23, alongside the L-15/P-18 fix): the honest
do-nothing arm is not a cold prefill, it's today's actual fallback when a conversation doesn't
hold the single shared resident GPU slot — the staged CPU path. This measures that path's real
decode speed for the model the doc's own R-01/R-04 rows name as the one this hits
(Qwen3.6-35B-A3B, Gated-DeltaNet MoE), so the comparison is a number, not a projection.

**Machine/provenance.** nobara-pc, commit `e1c867f6` (2026-09-23), driver `595.91.07`. Model:
`~/models/qwen3.6-35b-a3b-int4.giw` (local NVMe, 21 GiB — the same goinfer-native kind-4 bundle
`docs/benchmarks.md`'s M35 row uses). `serve` built fresh from that commit
(`go build ./cmd/serve`), `-backend cpu`, no `-moe-cache-experts` (CUDA-only flag), no
`-stream-weights` (the model fits this box's RAM outright — 21 GiB against 52 GiB available at
the time of the run, so this is goinfer's normal fully-resident-in-RAM CPU path, not the
disk-paging fallback that produced the Mac's catastrophic M35/M26 result recorded elsewhere in
`docs/benchmarks.md`). Greedy (`temperature=0`), `scripts/prompts.json`-style filler prompt (76
prompt tokens after tokenization), `max_tokens=64`. Load average 0.5–1.6 across the two runs
(idle box, not the zero-load Linux floor but not contending either). Raw log:
[`m35-cpu-decode-2026-09-23.log`](m35-cpu-decode-2026-09-23.log).

**Two runs.** The first (non-streaming) request measured end-to-end wall time only (32.65s for
76+64 tokens), which conflates prefill and decode — not a clean decode number by this repo's own
convention (`docs/benchmarks.md`'s W3 note: "tok/s is decode-only by construction, timed from the
first streamed token"). Re-run streaming, timing from the first SSE chunk to the last, to get a
clean separation:

| | value |
|---|---|
| load time (mmap, no read-through) | 5s |
| prefill (76 tokens, send → first chunk) | 13.85s |
| **decode (64 tokens, first → last chunk)** | **15.05s → 4.19 tok/s** |

**The comparison.** `docs/benchmarks.md`'s M35 row (W1, depth 128, nobara CUDA, same model, same
box): **23.5 tok/s** resident. Ratio: **23.5 / 4.19 ≈ 5.6× — CPU decode is real, not
catastrophic**, for this model on this box (RAM headroom is what makes the difference from the
Mac's case, where the model didn't fit RAM at all and the same fallback never completed in
2h10min). This is markedly less severe than the ~11× this doc's L01 funding-cell measurement found
for CPU-offloaded *per-expert* compute inside an otherwise-resident pass
(`docs/measurements/l01-funding-cell-2026-09-21.md`) — a different mechanism (partial offload vs.
whole-turn CPU decode), so the two numbers are not expected to match, but they triangulate in the
same direction: CPU compute for this architecture's experts is a mid-single-digit-to-low-double-digit
multiple slower than resident GPU, not two or three orders of magnitude slower.

**What this means for R-01 Phase 2 / R-04, and what it does NOT settle.**

For a 500-token turn (a plausible agent-loop response length): CPU ≈ 500/4.19 ≈ 119s; GPU-resident
≈ 500/23.5 ≈ 21s. A ~98s difference against a park/restore round-trip priced at low hundreds of ms
(spec/09's own 62.8 MiB recurrent-state figure for this exact model, plus the KV cache — R-04's own
row cites ~43ms to park 257 MiB of KV for a 2.3k-token conversation) looks like an overwhelming win
for parking, arithmetically.

**But this arithmetic answers a narrower question than R-01 Phase 2 actually needs answered.**
Parking (Download the resident state to host RAM when a conversation loses the slot; Upload it back
when it reclaims the slot) only makes the *eventual restore* cheap — it does not, by itself, change
what a conversation does *while it does not hold the slot*. Today that's the graceful CPU fallback
this measurement just priced (§ R-04's 2026-09-23 note on `decoder/model.go`'s `resBusy` CAS loser).
Parking a conversation's device state does not let it skip generating its current turn — it still
needs `resBusy` free to run resident at all, at some point. So the 98s/turn saved above is only
realized if the conversation is willing to **wait** for the slot rather than fall back to CPU
immediately, which today's code does not do (the CAS is non-blocking, per `decoder/model.go`'s own
comment) — and turning it into a bounded wait is a **scheduling/concurrency policy question**,
genuinely separate from the snapshot/restore mechanism Phase 2 was scoped to build. Given CPU
fallback is 5.6×, not catastrophically slow, a losing conversation blocking for even a few seconds
hoping for the slot could easily lose to just running on CPU immediately, depending on how long the
current holder's own turn runs — this needs its own measurement (turn-length distribution vs.
wait-vs-CPU crossover), not an assumption either way.

**Conclusion: this is a real, useful number for R-01 Phase 2's decision rule, but it exposes a
second, smaller, and possibly more valuable question underneath it that the original Phase 2 scope
did not separate out.** Recorded in `task-recompute-audit.md`'s R-01 section as two distinct
not-yet-funded candidates rather than folded into one. Neither is built here.
