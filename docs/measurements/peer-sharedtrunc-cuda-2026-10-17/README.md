# The pre-registered shared truncated preset, CUDA (2026-10-17)

The cells pre-registered in `docs/measurements/peer-notrunc-2026-10-17.md` ("The shared truncated preset: pre-registration"): CUDA, depth 128, config `temp0.8_topk40_topp0.9` (temperature 0.8, top_k 40, top_p 0.9, seed 1 on every arm),
3 runs × 8 completions × 64 tokens, models 0.5B, gemma3-1b and phi3-mini, goinfer against Ollama and llama.cpp, same session, interleaved by the harness. Tier 1, by day, through the timing lock, in **two halves** (owner, 2026-10-17).
Graded by that record's bands and void rules (`docs/measurements/peer-notrunc-2026-10-17/grade.py`, unchanged). The published docs were not edited here.

## Provenance

| | |
|---|---|
| harness | `scripts/bench_peer.py` at main `e8e8c95d` (the like-for-like fix: seed, context 4096, `min_p`, quant class on every arm; a run whose quants differ is refused). Its CI (`ci`) was green before the run; the `coverage` workflow was still running when half 1 started |
| goinfer binary | `serve-cuda-e8e8c95d`, built from that commit with `CGO_ENABLED=0 go build -tags cuda ./cuda/cmd/serve`; **sha256 `f7e2b747b8eee969610f322e1ac42ce94f731ec5f44e5b7e1cdd09173e98eff7`** |
| peers | Ollama 0.32.5 (tags `q05`, `g31b`, `p3m-local`), llama-server 0.4.0-dev `427291b` (`-ngl 99`); weights are the same files (`docs/measurements/peer-claim-2026-09-25.md`, "Same weights") |
| machine | nobara-pc, RTX 2070 SUPER 8 GB, driver 595.91.07, Nobara 44; loadavg at the cells' starts in each cell's `machine` field (gate 1.0) |
| checkpoints | local `~/models`: `qwen2.5-coder-0.5b-instruct-q4_k_m.gguf`, `gemma3-1b-q4_k_m.gguf`, `Phi-3-mini-4k-instruct-q4.gguf` (its own directory) |
| times | half 1 (0.5B, gemma3-1b; 12 cells): announced 7 min, took **258 s**. Half 2 (phi3-mini; 6 cells): announced 3 to 4 min, took **221 s**. Each half also carries the harness's unskippable greedy depth-128 phase A (six and three cells) as same-session controls |
| first attempt | a launch before the harness fix failed at its build step (no `go.work` in the worktree) and measured nothing: `run-attempt0-nogowork-build-failed.log` |
| raw | `half1.json`, `half2.json` (every cell: per-run rates, `sent`, `ctx`, `quant`, `quant_class`, `quant_matched`, token counts, machine state), `half1.log`, `half2.log`, `grade-half1.jsonl`, `grade-half2.jsonl`, `half.sh`, `preflight.py`, `preflight.md` |

## Before the first half: every arm's request fields and quant (from the harness's own functions)

Seed 1, temperature 0.8, top_p 0.9, top_k 40 on every arm; context 4096 on every arm (goinfer `-ctx`, Ollama `num_ctx`, llama-server `--ctx-size`); `min_p` 0 on Ollama and llama.cpp (goinfer has no `min_p` request field);
the quant classes match for all three models (`quant_mismatch` empty). The check passed ("CHECK OK" at the end of `preflight.md`):

context every arm: cell_ctx() = 4096

### 0.5B
- goinfer `{"max_tokens": 64, "model": "bench", "seed": 1, "stream": true, "stream_options": {"include_usage": true}, "temperature": 0.8, "top_k": 40, "top_p": 0.9}`
  cmd `<serve> -model bench=/home/francis/models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf -backend cuda -addr 127.0.0.1:8099 -ctx 4096`
- Ollama `{"model": "q05", "options": {"min_p": 0.0, "num_ctx": 4096, "num_predict": 64, "seed": 1, "temperature": 0.8, "top_k": 40, "top_p": 0.9}, "stream": true}`
- llama.cpp `{"max_tokens": 64, "min_p": 0.0, "model": "bench", "seed": 1, "stream": true, "stream_options": {"include_usage": true}, "temperature": 0.8, "top_k": 40, "top_p": 0.9}`
  cmd `llama-server --model /home/francis/models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf --port 8098 --host 127.0.0.1 --ctx-size 4096 -ngl 99`
- ollama quant: ('Q4_K_M', '4-bit')
- llamacpp quant: ('Q4_K_M', '4-bit')
- goinfer quant asked: ('int4', '4-bit')  (override table {'M35': 'int4mix', 'phi3-mini': 'q4k'})
- quant_mismatch: ''

### gemma3-1b
- goinfer `{"max_tokens": 64, "model": "bench", "seed": 1, "stream": true, "stream_options": {"include_usage": true}, "temperature": 0.8, "top_k": 40, "top_p": 0.9}`
  cmd `<serve> -model bench=/home/francis/models/gemma3-1b-q4_k_m.gguf -backend cuda -addr 127.0.0.1:8099 -ctx 4096`
- Ollama `{"model": "g31b", "options": {"min_p": 0.0, "num_ctx": 4096, "num_predict": 64, "seed": 1, "temperature": 0.8, "top_k": 40, "top_p": 0.9}, "stream": true}`
- llama.cpp `{"max_tokens": 64, "min_p": 0.0, "model": "bench", "seed": 1, "stream": true, "stream_options": {"include_usage": true}, "temperature": 0.8, "top_k": 40, "top_p": 0.9}`
  cmd `llama-server --model /home/francis/models/gemma3-1b-q4_k_m.gguf --port 8098 --host 127.0.0.1 --ctx-size 4096 -ngl 99`
- ollama quant: ('Q4_K_M', '4-bit')
- llamacpp quant: ('Q4_K_M', '4-bit')
- goinfer quant asked: ('int4', '4-bit')  (override table {'M35': 'int4mix', 'phi3-mini': 'q4k'})
- quant_mismatch: ''

### phi3-mini
- goinfer `{"max_tokens": 64, "model": "bench", "seed": 1, "stream": true, "stream_options": {"include_usage": true}, "temperature": 0.8, "top_k": 40, "top_p": 0.9}`
  cmd `<serve> -model bench=/home/francis/models/phi3-mini-4k-gguf/Phi-3-mini-4k-instruct-q4.gguf -backend cuda -addr 127.0.0.1:8099 -ctx 4096`
- Ollama `{"model": "p3m-local", "options": {"min_p": 0.0, "num_ctx": 4096, "num_predict": 64, "seed": 1, "temperature": 0.8, "top_k": 40, "top_p": 0.9}, "stream": true}`
- llama.cpp `{"max_tokens": 64, "min_p": 0.0, "model": "bench", "seed": 1, "stream": true, "stream_options": {"include_usage": true}, "temperature": 0.8, "top_k": 40, "top_p": 0.9}`
  cmd `llama-server --model /home/francis/models/phi3-mini-4k-gguf/Phi-3-mini-4k-instruct-q4.gguf --port 8098 --host 127.0.0.1 --ctx-size 4096 -ngl 99`
- ollama quant: ('Q4_K_M', '4-bit')
- llamacpp quant: ('Q4_K_M', '4-bit')
- goinfer quant asked: ('q4k', '4-bit')  (override table {'M35': 'int4mix', 'phi3-mini': 'q4k'})
- quant_mismatch: ''

CHECK OK: temperature, top_p, top_k and seed on every arm; min_p on Ollama and llama.cpp (goinfer has no min_p field)

## Quant each arm loaded (the shared-preset cells; the greedy controls carry the same labels)

| model | arm | quant loaded | class | matched | context | decode path (goinfer) |
|---|---|---|---|---|---:|---|
| 0.5B | goinfer | int4 | 4-bit | True | 4096 | cuda-resident (int4) |
| 0.5B | ollama | Q4_K_M | 4-bit | True | 4096 | — |
| 0.5B | llamacpp | Q4_K_M | 4-bit | True | 4096 | — |
| gemma3-1b | goinfer | int4 | 4-bit | True | 4096 | cuda-resident (int4) |
| gemma3-1b | ollama | Q4_K_M | 4-bit | True | 4096 | — |
| gemma3-1b | llamacpp | Q4_K_M | 4-bit | True | 4096 | — |
| phi3-mini | goinfer | q4k | 4-bit | True | 4096 | cuda-resident (q4k) |
| phi3-mini | ollama | Q4_K_M | 4-bit | True | 4096 | — |
| phi3-mini | llamacpp | Q4_K_M | 4-bit | True | 4096 | — |

## Readings: `temp0.8_topk40_topp0.9`

Ratio r = goinfer ÷ peer, pair i = goinfer run i against peer run i; AHEAD all pairs above 1.03, LEVEL all in [0.97, 1.03], BEHIND all below 0.97, AMBIGUOUS on a straddle; either engine's spread over 5% caps the cell at AMBIGUOUS. No cell was void.

| model | peer | goinfer tok/s | peer tok/s | r = goinfer ÷ peer (median; pairs) | verdict | spreads goinfer / peer |
|---|---|---:|---:|---|---|---|
| 0.5B | ollama | 343.4 | 262.2 | **1.312** (1.318, 1.298, 1.312) | **AHEAD** | 1.0% / 0.9% |
| 0.5B | llamacpp | 343.4 | 375.5 | **0.917** (0.917, 0.903, 0.924) | **BEHIND** | 1.0% / 1.3% |
| gemma3-1b | ollama | 195.7 | 153.6 | **1.272** (1.279, 1.271, 1.272) | **AHEAD** (token gate: ok/short) | 0.0% / 0.6% |
| gemma3-1b | llamacpp | 195.7 | 213.8 | **0.91** (0.906, 0.91, 0.931) | **BEHIND** | 0.0% / 2.7% |
| phi3-mini | ollama | 112.6 | 125.9 | **0.895** (0.895, 0.895, 0.893) | **BEHIND** | 0.3% / 0.1% |
| phi3-mini | llamacpp | 112.6 | 127.8 | **0.881** (0.881, 0.881, 0.882) | **BEHIND** | 0.3% / 0.3% |

## Greedy depth 128, same sessions (controls, not part of the claim)

| model | peer | goinfer tok/s | peer tok/s | r = goinfer ÷ peer (median; pairs) | verdict | spreads goinfer / peer |
|---|---|---:|---:|---|---|---|
| 0.5B | ollama | 350.9 | 264.0 | **1.33** (1.324, 1.33, 1.333) | **AHEAD** | 0.7% / 0.0% |
| 0.5B | llamacpp | 350.9 | 374.3 | **0.937** (0.936, 0.937, 0.94) | **BEHIND** | 0.7% / 0.3% |
| gemma3-1b | ollama | 200.6 | 151.2 | **1.329** (1.316, 1.335, 1.329) | **AHEAD** | 1.2% / 0.1% |
| gemma3-1b | llamacpp | 200.6 | 212.5 | **0.947** (0.932, 0.953, 0.947) | **BEHIND** | 1.2% / 1.0% |
| phi3-mini | ollama | 113.8 | 125.9 | **0.904** (0.904, 0.904, 0.904) | **BEHIND** | 0.1% / 0.0% |
| phi3-mini | llamacpp | 113.8 | 127.9 | **0.891** (0.891, 0.889, 0.891) | **BEHIND** | 0.1% / 0.2% |

## Notes

- The greedy and shared-preset ratios agree on every model and peer to within about 4% (0.5B vs Ollama 1.330 / 1.312; vs llama.cpp 0.937 / 0.917; gemma3-1b 1.329 / 1.272 and 0.947 / 0.910; phi3-mini 0.904 / 0.895 and 0.891 / 0.881): the preset costs the goinfer arm nothing the greedy control does not show.
- phi3-mini at q4k is BEHIND both peers at every config (0.88 to 0.90). Its goinfer rate is 112.6 tok/s against 90.4 at `int8int8` in `docs/measurements/peer-sampled-cuda-2026-10-17/README.md`.
- Ollama's gemma3-1b arm is `short` in the shared-preset cell (shortest completion under 64 of the 64 tokens, above the void threshold of 32).
- One session, three runs per cell, the pre-registered shape; not the order-reversed design. llama-server applies its own `min_p` when none is sent; here 0 was sent to it and to Ollama.
