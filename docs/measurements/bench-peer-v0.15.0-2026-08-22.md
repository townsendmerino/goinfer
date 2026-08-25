# v0.15.0 peer benchmark — goinfer vs Ollama across CPU, CUDA and WebGPU

> **Status: COMPLETE — 33/33 cells, zero failures.** Ran 2026-08-22 on `linux-62gb`
> (RTX 2070 SUPER, 8 GB, Ryzen 7 3700X) against goinfer `5b040f2`, peer Ollama **v0.32.6**.
> Results below.
>
> **These numbers are already partly superseded.** They predate the A1 attention restructure
> (2.4–2.8×), the arm64 W4A8 row4 repack and the int4-mode LM head, all of which landed after and
> move the goinfer column — the CPU rows especially. Kept because the METHOD, the fairness proof
> and the CUDA depth curve are what the next run has to reproduce or beat, and because the CPU
> deficit recorded here is what prompted the diagnosis in
> `mac-cpu-decode-vs-ollama-2026-08-22.md`.

## Where everything is

| what | where |
|---|---|
| run dir (binaries, Modelfiles, logs) | `/home/francis/bench-v0.15.0/` |
| incremental results (written per cell) | `/home/francis/bench-v0.15.0/results.json` |
| stdout log | `/home/francis/bench-v0.15.0/bench.log` |
| relaunch / resume | `/home/francis/bench-v0.15.0/run.sh` |

Detached with `setsid` (PPID 1, own SID, no TTY), so it is not a child of any Claude session and
survives disconnect. Check it with:

```sh
ps -eo pid,etime,cmd | grep '[b]ench_peer'
tail -5 /home/francis/bench-v0.15.0/bench.log
python3 -c "import json;d=json.load(open('/home/francis/bench-v0.15.0/results.json'));print(len(d),'cells')"
```

**To resume after any interruption, re-run `/home/francis/bench-v0.15.0/run.sh`.** It reloads
`results.json` and skips every cell already recorded, so a kill costs at most one cell.

## What it measures

33 cells. Phase A is the backend table (0.5B/1.5B/7B x {goinfer-cpu, ollama-cpu, goinfer-cuda,
ollama-cuda, goinfer-webgpu} at depth 128). Phase B is the depth curve (512/2048/3900, CUDA only,
both engines). Greedy, decode-only, interleaved, server restarted between cells, sampling sent
explicitly to both sides. `NGEN=64`, `NCOMP=8`, `NRUNS=2`.

**WebGPU has no ollama counterpart.** Its row is scored against ollama's CUDA row and must be
written up as CROSS-BACKEND, never as a like-for-like peer cell. Metal is not measurable here —
it belongs to the Mac, where it can ride with the C3 consumer window this release already triggers.

## The fairness check, and why the old one could not pass

`bench_peer.py` promised "the same GGUF file on both sides (verify by md5)". **That is
unsatisfiable through Ollama's import path for any model.** `ollama create` repacks the container
in a different tensor ORDER: metadata keys and values identical, tensor names/shapes/types
identical, all 339 offsets different, whole-file md5 different — and every tensor bit-identical.
A tail-slice md5 fails for the same reason.

Replaced by `scripts/gguf_same_weights.py`, which reads each tensor at its own offset in its own
file. **Verified 2026-08-22 before this run, all three models built from the goinfer GGUF via a
`FROM` Modelfile:**

| model | ollama tag | tensors identical |
|---|---|---|
| qwen2.5-coder-0.5b-instruct-q4_k_m | `q05` | **291/291** |
| qwen2.5-coder-1.5b-instruct-q4_k_m | `q15` | **339/339** |
| qwen2.5-7b-instruct-q4_k_m | `q7b` | **339/339** |

## Why a 7B was added

0.5B and 1.5B are both tiny and this repo has already been burned by publishing off that: CUDA
graphs measured 1.4–1.7x on a small model and **1.01x at real size**, because CPU dispatch
overlaps GPU compute once the model is big enough. The two small rows are kept for continuity
with the currently published figures; the 7B is the row a claim should rest on.

The 7B is `qwen2.5-7b-instruct`, not `-coder` — no coder 7B GGUF is on this box. Same architecture
and tokenizer family; say so rather than implying one family across all three rows.

## Results (2026-08-22, `5b040f2`, Ollama v0.32.6, greedy, decode-only)

**Phase A — backend table, depth 128 (tok/s)**

| model | goinfer-cpu | ollama-cpu | ratio | goinfer-cuda | ollama-cuda | ratio | goinfer-webgpu |
|---|---|---|---|---|---|---|---|
| 0.5B | 27.4 | 50.7 | 0.54× | 284.3 | 262.5 | 1.08× | 121.1 |
| 1.5B | 10.6 | 24.2 | 0.44× | 220.8 | 196.1 | 1.13× | 89.7 |
| 7B | 4.2 | 5.9 | 0.71× | 73.3 | 73.1 | 1.00× | 46.0 |

**Phase B — depth curve, CUDA (goinfer ÷ ollama)**

| model | 128 | 512 | 2048 | 3900 |
|---|---|---|---|---|
| 0.5B | 1.08× | 1.11× | 0.94× | **0.78×** |
| 1.5B | 1.13× | 1.15× | 0.88× | **0.71×** |
| 7B | 1.00× | 0.96× | 0.82× | **0.70×** |

**Two findings worth carrying forward.** The CUDA advantage is **1.08–1.13× at shallow depth and a
dead tie at 7B**, not the 1.19× the README's v0.14.0-era table shows for 0.5B @128 — one cell moving
while its neighbours reproduce, unconfirmed by a second run. And **goinfer degrades with context
depth while Ollama stays flat**: Ollama holds 262→259 / 196→174 / 73→70 across 128→3900 where
goinfer falls 284→202 / 220→123 / 73→49. The depth curve is the more consequential of the two and
is not addressed by this release's CPU work.

The WebGPU column has **no ollama counterpart** (ollama has no WebGPU build); it is scored against
ollama's CUDA row and must be labelled cross-backend, never presented as a like-for-like peer cell.
