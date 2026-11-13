# RF2 post-kill confirmation read, 2026-11-09 (nobara, CUDA)

Machine nobara, RTX 2070 SUPER, driver 595.112.07, 16 cores. Ran 19:10 to 19:45 PST, then a re-run 19:46 to 19:51 PST. Greedy (temperature 0, seed 1), essay-v2 prompts, ABBA per cell in one session, every cell under the exclusive timing lock with the load gate at 0.9. Models from `~/models` (NVMe): qwen2.5-coder 1.5B and qwen2.5 7B q4_k_m, gemma3-1b. Thermal: the card idled at 48 C before the run.

- NEW = `62649f78e` (main after #869).
- OLD = `895fd8165`, branch `g-rf2-s-old-confirm-2026-11-09`: 62649f78e with RF2's PRs 787 799 804 811 823 826 869 reverted (`make-old-arm.sh`).
- Phases: decode (1.5B and 7B at depths 128, 2048, 3900, ctx 4096), deep (depth 8000, ctx 12288), gemma (gemma3-1b at 128, 2048, 3900). TTFT, MC3 and verify were not re-measured.

## Result (score.txt, score-rerun.txt; `score_speed.py`)

G2 decode, bar >= 0.98 at every depth: every cell reads 0.999 to 1.001 (median of block ratios) and passes. One cell was VOID: 7B depth 128, ctx 4096, goinfer block 0, "swap grew +0.25 MB during the cell (> 0 MB)". It stays in `decode-qwen.json`/`.log` and is not averaged in. It was re-run in the same ABBA order and environment (`rerun-7b128.sh`): 1.000 PASS (`decode-qwen-rerun-7b128.*`). The 7B depth-128 cell at ctx 12288 (decode-deep) is a different cell and also reads 1.000.

G4 never-swap, from procmem.jsonl: 32 server pairs, peak VmSwap 0.0 MB on every server, new minus old VmHWM -45.3 to +27.5 MB: PASS. The re-run's 4 pairs: peak VmSwap 0.0, VmHWM -8.2 to +6.7 MB: PASS. System swap was 1358 MB throughout (context only).

The score is rf1s's. `serve-logs/` stays on nobara.
