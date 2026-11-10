# #873 (B9 a1): `chat fit` at the PR's merge-base and at its head, over every ~/models entry on nobara, 2026-11-09

Untimed (shared timing lock), CPU only; nothing here is a measurement. Machine nobara, 62 GB RAM. 149 entries: `~/models/*/`, `~/models/*.gguf`, `~/models/*/*.gguf`; 22 of them resolve into `/srv/models` (config reads off the archive disk).

- A = `589c6ebd9`, #873's merge-base (not today's main: a moved main would add unrelated differences). B = `55128eca3`, the PR's head (A plus its two commits). Both `chat` binaries built from `./demo/chat`.
- `rc-A.txt`, `rc-B.txt`: exit status per entry. `out.tar.gz`: per-entry output files (`out/A`, `out/B`), the first attempts of the heads the watchdog killed (`first-attempt/`), and the three direct checks (`direct/`). `compare.txt`: the result below; `compare.py`, `sweep*.sh`: how.

## Result

- Direct checks (deepseek-v2-lite, moonlight-16b, ministral3-3b-bf16): rc 0 at both heads on all three. Output not byte-identical: only host-memory figures differ.
- 145 of 149 entries compared: the exit status is equal at both heads on every one (94 rc 0, 51 rc 1; rc 1 is entries that are not a loadable model directory, equal at both heads), and the plan lines (dense, experts, KV, RESIDENT) are identical. 49 are byte-identical; 96 are identical once the host-memory advisories ("fit is tight", "context capped", and the "fits N GB free" or "currently has N GB" figures) are masked: whether those print follows the memory available when the fit ran (16 to 60 GB across the sweep), not the head.
- 0 entries differ in exit status or plan. No shipped config in `~/models` is refused or planned differently by #873 here.
- NOT comparable on this box, 4 entries: `laguna-xs2` (both heads), `qwen3.8-27b` (both), the Qwen3.8-27B UD-Q4_K_M GGUF file inside the qwen38-gguf model directory (both), `qwen3.6-35b-a3b` (B). Their fits swap the box past a watchdog of 512 MB over the fit's start (reproduced alone on a retry), so the sweep killed them (rc 137, a `MEMKILL` line). The watchdog's threshold was not changed.

## What went wrong in the run, so it is not repeated

A first run did four fits at a time and swapped the box to 32 GB; it was stopped and resumed strictly serial. One `chat fit` peaks at 32 GB resident for a 9B model and 53 GB for a 30B (measured with `/usr/bin/time`), so a fit of a large checkpoint is not a light check on a 62 GB box.
