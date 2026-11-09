# G-RF2-S, first working build: read 2026-11-09 (nobara)

The RF2 speed gate and never-swap (`docs/tasks/task-forward-and-memory-2026-11.md`, "RF2: registration" and "RF2: addendum 1"),
read once on RF2's first working build. It is read again on the final build, once RF2-3b moves the solo shapes onto `StepRagged`.

## Provenance

- **Machine:** nobara, RTX 2070 SUPER, NVIDIA driver 595.112.07, run by nobara-sb01.
- **When:** 2026-11-09, 05:05–05:48 PST, by day. Every cell ran under the exclusive timing lock and started idle (load1 < 0.9,
  re-read inside the lock). There were no voids.
- **Arms**, A B B A per cell group in one session:
  - **A, old:** `rf2-s-old-e6a43d77` at ab3a6ad8e, which is e6a43d77 with RF2's four merges reverted (#811, #804, #799, #787).
    `serve-cuda` sha256 99541c53bf50938b.
  - **B, new:** e6a43d77, `main` after #811. sha256 0e232a46521c0b41.
  - Old to new differs by exactly RF2's code.
- **Checkpoints,** from `~/models` on nobara:
  - Qwen2.5-Coder-1.5B-Instruct Q4_K_M and Qwen2.5-7B-Instruct Q4_K_M;
  - gemma3-1b (`bench_peer.py`'s entry).
- **Settings:** `bench_peer.py` defaults, int4, greedy (temperature 0, seed 1).
- **Thermal:** the GPU was at 47 °C at the first cell's start (the per-record `machine` block).
- **Swap:** growth 0 on every record, and every record's `error` is null.
- **Script:** `run-g-rf2-s.sh`, phases decode, deep, gemma, ttft and mc3.
- **Not here:** attempt 1 (04:32) refused before measuring (no `go` on PATH for the host-memory reading) and stays on nobara with
  `serve-logs/`.

## Readings

Decode tok/s, B/A per block, then the median per cell (KV4's `score_speed.py`):

| model | depth | B/A | verdict |
|---|---|---|---|
| 1.5B | 128 | 1.000 | PASS |
| 1.5B | 2,048 | 0.999 | PASS |
| 1.5B | 3,900 | 1.000 | PASS |
| 1.5B | 8,000 | 1.000 | PASS |
| 7B | 128 | 1.000 | PASS |
| 7B | 2,048 | 0.999 | PASS |
| 7B | 3,900 | 0.999 | PASS |
| 7B | 8,000 | 1.001 | PASS |
| gemma3-1b | 128 | 1.000 | PASS |
| gemma3-1b | 2,048 | 1.000 | PASS |
| gemma3-1b | 3,900 | 1.000 | PASS |

Time to first token, A/B over three pairs (`ttft_ab.py`):

| model | K = 512 | K = 3,900 |
|---|---|---|
| 1.5B | 0.982 (0.982, 0.976, 1.000) | 1.012 (1.034, 1.012, 1.000) |
| 7B | 0.998 (0.998, 0.997, 1.001) | 1.000 (1.000, 0.998, 1.000) |

MC3 aggregate at 4 clients on the 1.5B, B/A per pair (`score_mc3.py`):

| pair | old tok/s | new tok/s | B/A |
|---|---|---|---|
| 1 | 355.4 | 434.7 | 1.223 |
| 2 | 357.2 | 434.7 | 1.217 |
| 3 | 356.6 | 432.6 | 1.213 |

The median is **1.217**. `content_sha` is equal in all 24 turns of each pair.

## Verdict, against the registered bands

- **Decode: PASS** at every registered depth, 128 included, on all three models (0.999–1.001). The kill is not triggered.
- **TTFT: PASS** (bar: at least 0.97); the worst cell is 0.982 (1.5B, K = 512).
- **MC3: PASS** at 1.217, with identical replies. MC3's step goes through RF2-3a's ragged launches: one `rope_kv_ragged` and one
  `attn_ragged` per layer, in place of each row's own decode launches.
- **Never-swap (KV4's G4): PASS as registered.** Peak VmSwap is 0 on every server of both arms. Taken cell by cell as the median of
  the cell's runs, the VmHWM difference is −24.5 to +25.6 MB, within the 64 MB bar (`g4_by_cell.py`).
  - KV4's `score_speed.py` reports FAIL. It applies the bar to each server pair rather than to the cell median: its pair 32 reads
    +68.5 MB, and the next pair −35.7.
  - Across all 41 pairs the median difference is +0.3 MB. This is host-memory spread, not memory RF2 holds: CUDA's pool takes
    VRAM, and peak VmSwap is 0 on every server.
  - The registered rule decides. Both readings are recorded here.
- **Verify: PASS** (bar: at least 0.98), read separately below.

## Verify cells (2026-11-09, nobara, after #816)

`bench_spec_lane.py --build-b`: exact attention with `-spec ngram`, A = the old build and B = the new, on the same two binaries as
above (sha256 99541c53bf50938b and 0e232a46521c0b41). Three pairs per cell, arm order alternating, plus an A/A floor. Raw data is in
`verify/`.

| cell | prompt tokens | B/A per pair | median | A/A floor |
|---|---|---|---|---|
| 1.5B:3900 | 1,072 | 1.0019, 1.0030, 1.0060 | 1.0030 | 1.0003 |
| 7B:7500 | 1,969 | 0.9999, 1.0008, 1.0004 | 1.0004 | 1.0004 |

- **Verify: PASS.** On this build, verify keeps today's call: the solo shapes do not move onto `StepRagged` until RF2-3b-3, so this
  is the do-nothing reading the final build is compared against.
- The cell names are characters of `decoder/model.go` in the prompt; the token counts above are what the engine reported.

## Files

The raw files are in this directory:
- `decode-*.json` and `.log`;
- `ttft-*.json` and `.log`;
- `w7.json`, `w7-servers.log` and `mc3-cells.log`;
- `run.log`;
- `procmem.jsonl.gz`.

Scorers, run from the repository root:

    python3 docs/measurements/kv4-cuda-2026-10-21/score_speed.py <procmem.jsonl> docs/measurements/rf2-2026-11/g-rf2-s-2026-11-09/decode-{qwen,deep,gemma3}.json
    python3 docs/measurements/kv4-cuda-2026-10-21/score_mc3.py docs/measurements/rf2-2026-11/g-rf2-s-2026-11-09/w7.json
    (cd docs/measurements/rf2-2026-11/g-rf2-s-2026-11-09 && python3 g4_by_cell.py <procmem.jsonl>)
