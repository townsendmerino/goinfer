# R17 `-backend auto`: smoke runs, 2026-10-01

**Exploratory, not a measurement.** These are single functional runs on a busy desktop (the Mac's 1-minute load average
was 3–5). They show which backend `-backend auto` chose and what each load reported. No figure here is a result, and
none should be quoted as a speed.

The binaries were built from the R17 working tree (base `6148638a` plus the R17 change, which landed in the commit that
added this record) with go1.27.0, `CGO_ENABLED=0`:

- Mac (M1 Pro, 16 GB): `demo/chat/build-embed.sh --name r17-smoke ~/models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf`,
  the model-included 0.5B with its int8int8 bundle, built from `./metal/cmd/chat` (darwin/arm64).
- nobara-pc (RTX 2070 SUPER): the same script for linux/amd64 (`./cuda/cmd/chat`, `-tags prequant,cuda`), and the plain
  `goinfer-chat` from `./cuda/cmd/chat` (`-tags cuda`). Both were cross-compiled on the Mac and copied over.

Every run is greedy (`--temp 0 --max 48`), with the prompt "Write a Go function that reverses a string."; the models are
read from `~/models` on each machine.

## nobara-pc (`nobara-smoke.log`, `nobara-smoke.sh`)

Run under `scripts/timing_lock.py` with the GPU otherwise idle.

| run | startup line | load banner |
| --- | --- | --- |
| plain `goinfer-chat`, 0.5B q4_k_m `.gguf`, `--direct-load` | `backend: cuda (auto: a CUDA device answered; -backend cpu to use the CPU)` | `decode=cuda-resident (int4)` |
| the same with `CUDA_VISIBLE_DEVICES=-1` | `backend: cpu (auto: cuda is built in, but no CUDA device answered)` | `decode=cpu (int4)` |
| model-included 0.5B (int8int8) | `backend: cuda (auto: a CUDA device answered; -backend cpu to use the CPU)` | `decode=cuda-resident (int8int8)`, loaded in 800 ms |

## Mac (`mac-smoke.log`)

One discarded warm-up run first, so every listed run reads the embedded weights from the page cache. An earlier
sequence without it had the first run load in 1.7 s on the CPU path: the bundle is aliased from the binary image, so a
cold run pages it in during load and the first decode steps.

| run | load banner | load |
| --- | --- | --- |
| `-backend auto` (twice) | `decode=cpu (int8int8) — requested auto (metal) → running on cpu: -backend auto keeps this int8int8 model on the CPU: …` | 176 ms, 193 ms |
| `-backend cpu` (twice) | `decode=cpu (int8int8)` | 167 ms, 167 ms |
| `-backend metal` | `decode=metal-resident (int8int8→int4, no Metal int8 GEMV kernel)` | 1.691 s |

All runs gave the same greedy text, except `-backend metal`, which runs at int4. Before the precision guard
(`decoder/residency.go` `autoMetalPrecision`), `-backend auto` took the `-backend metal` row: the int8int8 bundle went
resident re-quantized to int4 (seen at 16:10 that day, loading in 2.6 s; that run's output was not saved). The decode
rates printed in the log (89.7–96.9 tok/s on the CPU path across four runs, 105.7 in the one Metal run) are single
runs under that load, too few to say whether Metal was faster.
