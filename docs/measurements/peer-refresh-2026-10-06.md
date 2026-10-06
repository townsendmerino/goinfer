# Mac peer rows that predate the Metal audit: phi3-mini decode and the 1.5B's long-prompt TTFT (pre-registered 2026-10-05)

`docs/benchmarks.md`'s Metal decode row was refreshed after the Metal audit (2026-10-04, `a845af18`, ahead of Ollama in
all 12 cells), and its short-prompt TTFT table on 2026-10-05 (`93d586af`). Two Mac peer readings were not:

- **phi3-mini Metal decode, 0.94× Ollama (2026-09-17).** It predates every Metal decode change since: R17's block
  decode attention, R18's rows GEMVs, the audit's B-P01/B-P02 and C-B01's greedy chain, and native int8 (on since
  `9e51f882`; the harness loads phi3-mini at int8int8). It is stale.
- **The 1.5B's long-prompt TTFT, K = 512 and 3900 (2026-09-30, `754f12d3`, cell h):** AMBIGUOUS-HIGH 1.037 and LEVEL
  0.983 against Ollama. Nothing on its path has changed since (D-B02 and A-P01 change the GEMM tile only up to 128 rows;
  the two critical prefill fixes in `de1c7f17` are off the default path), so this re-run gives the row a post-audit
  build and adds mlx-lm, which cell h did not have. It is expected to hold.

## Instrument, written before the run

`docs/measurements/peer-refresh-2026-10-06/run-mac.sh` on the Mac night queue: the serve binary built from `52024b37`
(main, a clean worktree; `--version` `v0.20.1-0.20261006042738-52024b37f344`) and a scripts worktree at the same rev.
MacBook Pro M1 Pro 16 GB, checkpoints from `~/models`, Ollama 0.32.5, mlx-lm 0.31.3 on mlx-community 4-bit.

1. **phi3-mini decode:** one `bench_peer.py` sweep on Metal, goinfer against Ollama (`p3m-local`, the same weights as the
   GGUF goinfer loads, 195/195 tensors), greedy at depth 128, 2048 and 3900 and `temp1.0_notrunc` at depth 128, 3 runs
   per cell, the harness's defaults otherwise (phi3-mini at int8int8, serve at its defaults). The same cell shape as the
   2026-10-04 refresh. No mlx-lm arm: there is no MLX phi3-mini checkpoint on this machine.
2. **The 1.5B's long-prompt TTFT:** `bench_peer_prefill.py --backend metal --models 1.5B --depths 512,3900 --n 6`, the
   same command as the 2026-10-05 short-prompt table at the longer depths: goinfer (default), goinfer `--exact-prefill`,
   Ollama and mlx-lm, interleaved per cell, each server restarted.

Tier: night (timed peer runs). Stopping rule: fixed N, no early stop. Cost basis: the 2026-10-04 refresh took 26 minutes
for 3 models x 3 engines, so about 8 minutes for one model x 2 engines; the short-prompt table took 18 minutes for 4
depths, and these two depths are longer prompts; queued at 45 minutes.

## Reading, written before the result

**A record, not a gate.** Each cell gets the harness's own grade (AHEAD / LEVEL / BEHIND / AMBIGUOUS-HIGH /
AMBIGUOUS-LOW, with the 5% spread cap), as the 2026-09-29 sweep and the 2026-10-04 refresh used. The cells replace the
phi3-mini clause of the Metal decode row and the long-prompt half of the Metal prefill row in `docs/benchmarks.md`, with
this provenance.

- **Void if** goinfer's DecodePath is not `metal-resident` in a decode cell, a reply ends under the harness's 95% token
  gate, or a step exits non-zero. A void cell is reported as void and the row keeps its old reading, marked stale.
- **Reported against the old readings:** phi3-mini against 0.94× (2026-09-17); K = 512 against 1.037 and K = 3900
  against 0.983 (2026-09-30).

## Read 2026-10-06 (the Mac night queue, 21:54 PDT start on AC power; the job ran 21:58–22:14)

Provenance: goinfer `52024b37` (a clean tree, serve sha256 `02e7cce6fc11db99…`), Ollama 0.32.5, mlx-lm 0.31.3. Both
steps exited 0. Raw: `phi3-decode.json`, `prefill-long.json` and `prefill-long-cells.txt` in
`docs/measurements/peer-refresh-2026-10-06/`.

**1. phi3-mini decode: VOID, all four goinfer cells.** goinfer never ran on Metal. The harness loads phi3-mini at
int8int8 (its `BENCH_QUANT_OVERRIDE` default), goinfer then uses per-32 activation scales for Phi-3 ("Phi-3's activation
outliers are rounded to zero by the per-row int8 activation scale", queue-engineering.md H2), and the Metal resident
declined: every goinfer cell's decode path was `cpu (int8int8)`, and the harness refused it as not `metal-resident`,
which is this record's void rule. Ollama's cells ran (41.9 / 35.8 / 31.5 tok/s greedy at 128 / 2048 / 3900, 42.0 at
T = 1.0, spreads under 0.3%). Per the rule, the row keeps its 2026-09-17 reading, marked stale. What the void shows:
**at the quantization the harness treats as Phi-3's default, phi3-mini is not served by Metal at all on this build**,
so the 0.94× describes a configuration a default load no longer takes. Whether a Metal phi3-mini row should be read at
another quantization (int4, which the 2026-09-17 row may have used) is a question for the owner, not this record.

**2. The 1.5B's long-prompt TTFT: VALID.** Every engine started in both cells, and every cache check was healthy (fresh
prompts miss each engine's prefix cache). Graded per pair as the 2026-10-05 short-prompt table (level 0.97–1.03, ahead
> 1.03, behind < 0.97, all pairs decide, an engine spread over 5% caps the cell):

| K | goinfer tok/s (spread) | ÷ Ollama, median (pairs) | ÷ mlx-lm, median (pairs) | grade vs Ollama | grade vs mlx-lm |
|---|---|---|---|---|---|
| 512 | 1012.8 (11.3%) | 1.092 (0.987–1.097) | 1.048 (0.956–1.060) | AMBIGUOUS-HIGH (spread) | AMBIGUOUS-HIGH (spread) |
| 3900 | 922.3 (0.2%) | 0.978 (0.977–0.979) | 0.901 (0.893–0.917) | **LEVEL** | **BEHIND** |

Against the old readings: K = 512 1.037 → 1.092, still AMBIGUOUS-HIGH, and still capped by goinfer's own spread (its
first two requests read 567 and 569 ms, the other four 511–514); K = 3900 0.983 → 0.978, LEVEL both times. The
prediction (it holds) held. New to this row: mlx-lm is ahead of goinfer at K = 3900 by about 10%. goinfer's batched
prefill is 9.7–9.8× its own `--exact-prefill` (104 and 94 tok/s), which ran in both cells this time.
