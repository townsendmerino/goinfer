# S3 swap tripwire — the two negative controls (2026-09-24)

`docs/tasks/task-never-swap-2026-09.md` S3's registered rule has a positive control (a direct gpt-oss-20b load must
trip and abort before baseline + 1 GB — run 2026-09-22, `swap-tripwire-2026-09-22.md`: tripped at +0.72–0.80 GB, then
a burst carried swap to +1.69 GB before the external kill switch fired; recorded as a limit) and two negative controls,
which this record runs: **a sidecar 1.5B load and a 7B Metal load must never trip across 100 completions.** Raw data:
`swap-tripwire-negative-controls-2026-09-24/`.

**Setup.** MacBook (M1 Pro, 16 GB), `metal/cmd/serve` built from `6b14c19e`, `GOINFER_SWAP_GUARD` unset (the default:
serving guard armed after startup, +512 MB over its own baseline, 2 s poll). Each control: start serve, 100 sequential
`/v1/completions`, `max_tokens` 64, `temperature` 0, five prompts rotated. Swap-used also sampled **externally** every
1 s (`sysctl -n vm.swapusage` in a separate process), so the guard's reading is checked against an independent one.
Models from `~/models`.

| control | load | completions | guard | swap-used (external, 1 s) |
|---|---|---|---|---|
| 1.5B `.gguf`, `-backend cpu -quant int4` | darwin default sidecar: one-time `cpu-arm64` transcode (21 s, 1,233 MB), then the `.giw` in 9 ms | **100/100** HTTP 200 | armed, baseline 1.41 GB; **0 trips** | 1,346.12 MB, flat |
| 7B `.gguf`, `-backend metal -quant int4` | its v14 metal sidecar, weights aliased (4,020 MB bound in place, 1 MB copied); 54 ms | **100/100** HTTP 200 | armed, baseline 1.41 GB; **0 trips** | 1,346.12 MB, flat |

External sampler: 467 readings from 21:49:20 to 21:57:30, every one 1,346.12 MB — zero growth across both controls,
agreeing with the guard's own 1.41 GB (= 1,346 MiB) baseline. **Both negative controls PASS.**

What this does and does not show: a normal load and a normal serving loop on this machine do not make the guard fire —
the false-positive half of the rule. It says nothing new about the positive control's limit (a fast burst outrunning a
2 s poll), which stands as recorded on 2026-09-22.
