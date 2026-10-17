# CUDA sampled peer cells re-measured with every arm sent temperature, top_p and top_k (2026-10-17)

> **Run 2 (the shared truncated preset below) is NOT pre-registered: temperature 1.0.** The pre-registered shared preset is the Mac's `temp0.8_topk40_topp0.9` (temperature 0.8, `docs/measurements/peer-notrunc-2026-10-17.md`, "The shared truncated preset"); run 2 sent temperature 1.0, top_k 40, top_p 0.9 on an owner message that gave 1.0, later corrected to 0.8. Its readings are a record, not the pre-registered cells.
>
> **Run 1 went over the day budget:** 24 cells, 12 minutes of wall time (12:22–12:34 PDT), where the day rule is nothing estimated over about 10 minutes (CLAUDE.md, "Run budget"). The run's duration was not stated to the owner before it started (CLAUDE.md asks for that); the owner's by-day instruction was taken as the override. Run 2 took 6 minutes.

Tier 1, by day, through the timing lock (`bench_peer.py` holds it for its whole run), nobara-pc. **Exploratory and ungraded**: no decision rule was registered for these runs, so no AHEAD/BEHIND grade is given, only ratios.
Made for the Mac session, which is updating the published figures; the published docs were not edited here. Question asked (owner, 2026-10-17): re-measure the eight CUDA peer cells of
`docs/benchmarks.md`, "Sampled configurations, depth 128, CUDA", with `bench_peer.py`'s current sampled configs (temperature, top_p and top_k sent explicitly to every arm), then once more with one shared truncated preset.

## Provenance

| | |
|---|---|
| goinfer binary | `~/goinfer-bench/serve-cuda-63038121`, built from main `63038121f28464af09247ae7f4382c05dacab7ba` with `CGO_ENABLED=0 go build -tags cuda ./cuda/cmd/serve`; **sha256 `3740363400b63adb337b45862484eaf30719fa6394f3bbf4f1a7fd912409677f`** |
| peer | Ollama 0.32.5 (`~/ollama-0325/bin/ollama`, the models under `~/ollama-0325/models`), one server restart per cell |
| machine | nobara-pc, RTX 2070 SUPER 8 GB, driver 595.91.07, Nobara 44, kernel 7.2.0-202; GPU 48–50 °C and loadavg 0.56–0.98 at the cells' starts (the harness's idle gate, 1.0) |
| checkpoints (all local, `~/models`) | `qwen2.5-coder-0.5b-instruct-q4_k_m.gguf` (Ollama tag `q05`), `gemma3-1b-q4_k_m.gguf` (`g31b`), `Phi-3-mini-4k-instruct-q4.gguf` (in its own directory under `~/models`) (`p3m-local`). Same weights on both sides: `docs/measurements/peer-claim-2026-09-25.md`, "Same weights"; the files and tags have not changed since (not re-checked) |
| goinfer decode path | 0.5B and gemma3-1b `cuda-resident (int4)`; phi3-mini `cuda-resident (int8int8)` (the harness default for phi3-mini, `BENCH_QUANT_OVERRIDE`; goinfer's int4 phi3-mini output was junk when last checked, `peer-claim-2026-09-25.md`) |
| method | `scripts/bench_peer.py` as of the commit in the provenance header: depth 128 (129-token essay-v2 prompts), 64 tokens per completion, 8 completions per run, 2 runs per cell, decode-only inter-token rate, cells interleaved goinfer/Ollama with a server restart between |
| dates | run 1 12:22–12:34 PDT, run 2 12:34–12:40 PDT, 2026-10-17 |
| raw | `results-b51.json`, `results-trunc.json` (every cell's per-run rates, `sent`, token counts, machine state), `run-b51.log`, `run-trunc.log`, `run-b51.sh`, `run-trunc.sh`, `make_record.py` (the tables below) |

The harness always plans its phase A (greedy, depth 128, every model, both engines) and cannot skip it, so each run also carries six greedy cells: a same-session control, taken twice (run 1 and run 2).
Run 2 used the config `temp1.0_topk40_topp0.9` (`scripts/bench_peer.py` `CONFIGS`, added in this change; its results header stamps the pre-rebase commit of this change, which no longer exists; it was then rebased onto the main that also carries the Mac session's own `temp0.8_topk40_topp0.9`, which was not run here). A second bench_peer run on this box (`peer-notrunc-2026-10-17`, another session, three engines, 3 runs) held the lock first; this run waited for it and is independent of it.

## Every arm was sent temperature, top_p and top_k explicitly

Confirmed from the harness's own payload functions (below) and from each cell's `sent` field in the results: in every sampled config (`temp1.0_notrunc`, `temp0.8_topp0.95`, `temp0.8_topk40`, `temp1.0_topk40_topp0.9`) the goinfer request carries `temperature`, `top_p` and `top_k`
at top level, and the Ollama request carries them in `options`, with equal values on both sides. `greedy` sends only `temperature: 0` (Ollama also `seed: 1`), by design. Differences between the arms that remain: Ollama always gets `seed: 1`, goinfer gets a seed only in the configs that define one
(`temp0.8_topp0.95`, `temp0.8_topk40`, `temp1.0_topk40_topp0.9`), not in `temp1.0_notrunc`; Ollama gets `num_ctx: 4096`; goinfer asks for `stream_options.include_usage`. `min_p` is not sent to either (`docs/benchmarks.md`, "Sampling changes that move comparability").

### Run 1: each arm's exact request fields (built by bench_peer.py's own payload functions; the prompt is named by its SHA-256)

#### 0.5B (prompt 497 chars, sha256 9b23da994ef03cf2, 129 tokens, format essay-v2; GGUF /home/francis/models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf, Ollama tag q05)

- `greedy` goinfer `{"max_tokens": 64, "model": "bench", "stream": true, "stream_options": {"include_usage": true}, "temperature": 0}`
  Ollama `{"model": "q05", "options": {"num_ctx": 4096, "num_predict": 64, "seed": 1, "temperature": 0}, "stream": true}`
- `temp1.0_notrunc` goinfer `{"max_tokens": 64, "model": "bench", "stream": true, "stream_options": {"include_usage": true}, "temperature": 1.0, "top_k": 0, "top_p": 1.0}`
  Ollama `{"model": "q05", "options": {"num_ctx": 4096, "num_predict": 64, "seed": 1, "temperature": 1.0, "top_k": 0, "top_p": 1.0}, "stream": true}`
- `temp0.8_topp0.95` goinfer `{"max_tokens": 64, "model": "bench", "seed": 1, "stream": true, "stream_options": {"include_usage": true}, "temperature": 0.8, "top_k": 0, "top_p": 0.95}`
  Ollama `{"model": "q05", "options": {"num_ctx": 4096, "num_predict": 64, "seed": 1, "temperature": 0.8, "top_k": 0, "top_p": 0.95}, "stream": true}`
- `temp0.8_topk40` goinfer `{"max_tokens": 64, "model": "bench", "seed": 1, "stream": true, "stream_options": {"include_usage": true}, "temperature": 0.8, "top_k": 40, "top_p": 1.0}`
  Ollama `{"model": "q05", "options": {"num_ctx": 4096, "num_predict": 64, "seed": 1, "temperature": 0.8, "top_k": 40, "top_p": 1.0}, "stream": true}`
#### gemma3-1b (prompt 493 chars, sha256 32e2b91af738ceb6, 129 tokens, format essay-v2; GGUF /home/francis/models/gemma3-1b-q4_k_m.gguf, Ollama tag g31b)

- `greedy` goinfer `{"max_tokens": 64, "model": "bench", "stream": true, "stream_options": {"include_usage": true}, "temperature": 0}`
  Ollama `{"model": "g31b", "options": {"num_ctx": 4096, "num_predict": 64, "seed": 1, "temperature": 0}, "stream": true}`
- `temp1.0_notrunc` goinfer `{"max_tokens": 64, "model": "bench", "stream": true, "stream_options": {"include_usage": true}, "temperature": 1.0, "top_k": 0, "top_p": 1.0}`
  Ollama `{"model": "g31b", "options": {"num_ctx": 4096, "num_predict": 64, "seed": 1, "temperature": 1.0, "top_k": 0, "top_p": 1.0}, "stream": true}`
- `temp0.8_topp0.95` goinfer `{"max_tokens": 64, "model": "bench", "seed": 1, "stream": true, "stream_options": {"include_usage": true}, "temperature": 0.8, "top_k": 0, "top_p": 0.95}`
  Ollama `{"model": "g31b", "options": {"num_ctx": 4096, "num_predict": 64, "seed": 1, "temperature": 0.8, "top_k": 0, "top_p": 0.95}, "stream": true}`
- `temp0.8_topk40` goinfer `{"max_tokens": 64, "model": "bench", "seed": 1, "stream": true, "stream_options": {"include_usage": true}, "temperature": 0.8, "top_k": 40, "top_p": 1.0}`
  Ollama `{"model": "g31b", "options": {"num_ctx": 4096, "num_predict": 64, "seed": 1, "temperature": 0.8, "top_k": 40, "top_p": 1.0}, "stream": true}`
#### phi3-mini (prompt 501 chars, sha256 46979ffcae1e1bbb, 129 tokens, format essay-v2; GGUF /home/francis/models/phi3-mini-4k-gguf/Phi-3-mini-4k-instruct-q4.gguf, Ollama tag p3m-local)

- `greedy` goinfer `{"max_tokens": 64, "model": "bench", "stream": true, "stream_options": {"include_usage": true}, "temperature": 0}`
  Ollama `{"model": "p3m-local", "options": {"num_ctx": 4096, "num_predict": 64, "seed": 1, "temperature": 0}, "stream": true}`
- `temp1.0_notrunc` goinfer `{"max_tokens": 64, "model": "bench", "stream": true, "stream_options": {"include_usage": true}, "temperature": 1.0, "top_k": 0, "top_p": 1.0}`
  Ollama `{"model": "p3m-local", "options": {"num_ctx": 4096, "num_predict": 64, "seed": 1, "temperature": 1.0, "top_k": 0, "top_p": 1.0}, "stream": true}`
- `temp0.8_topp0.95` goinfer `{"max_tokens": 64, "model": "bench", "seed": 1, "stream": true, "stream_options": {"include_usage": true}, "temperature": 0.8, "top_k": 0, "top_p": 0.95}`
  Ollama `{"model": "p3m-local", "options": {"num_ctx": 4096, "num_predict": 64, "seed": 1, "temperature": 0.8, "top_k": 0, "top_p": 0.95}, "stream": true}`
- `temp0.8_topk40` goinfer `{"max_tokens": 64, "model": "bench", "seed": 1, "stream": true, "stream_options": {"include_usage": true}, "temperature": 0.8, "top_k": 40, "top_p": 1.0}`
  Ollama `{"model": "p3m-local", "options": {"num_ctx": 4096, "num_predict": 64, "seed": 1, "temperature": 0.8, "top_k": 40, "top_p": 1.0}, "stream": true}`


### Run 2: each arm's exact request fields (built by bench_peer.py's own payload functions; the prompt is named by its SHA-256)

#### 0.5B (prompt 497 chars, sha256 9b23da994ef03cf2, 129 tokens, format essay-v2; GGUF /home/francis/models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf, Ollama tag q05)

- `greedy` goinfer `{"max_tokens": 64, "model": "bench", "stream": true, "stream_options": {"include_usage": true}, "temperature": 0}`
  Ollama `{"model": "q05", "options": {"num_ctx": 4096, "num_predict": 64, "seed": 1, "temperature": 0}, "stream": true}`
- `temp1.0_topk40_topp0.9` goinfer `{"max_tokens": 64, "model": "bench", "seed": 1, "stream": true, "stream_options": {"include_usage": true}, "temperature": 1.0, "top_k": 40, "top_p": 0.9}`
  Ollama `{"model": "q05", "options": {"num_ctx": 4096, "num_predict": 64, "seed": 1, "temperature": 1.0, "top_k": 40, "top_p": 0.9}, "stream": true}`
#### gemma3-1b (prompt 493 chars, sha256 32e2b91af738ceb6, 129 tokens, format essay-v2; GGUF /home/francis/models/gemma3-1b-q4_k_m.gguf, Ollama tag g31b)

- `greedy` goinfer `{"max_tokens": 64, "model": "bench", "stream": true, "stream_options": {"include_usage": true}, "temperature": 0}`
  Ollama `{"model": "g31b", "options": {"num_ctx": 4096, "num_predict": 64, "seed": 1, "temperature": 0}, "stream": true}`
- `temp1.0_topk40_topp0.9` goinfer `{"max_tokens": 64, "model": "bench", "seed": 1, "stream": true, "stream_options": {"include_usage": true}, "temperature": 1.0, "top_k": 40, "top_p": 0.9}`
  Ollama `{"model": "g31b", "options": {"num_ctx": 4096, "num_predict": 64, "seed": 1, "temperature": 1.0, "top_k": 40, "top_p": 0.9}, "stream": true}`
#### phi3-mini (prompt 501 chars, sha256 46979ffcae1e1bbb, 129 tokens, format essay-v2; GGUF /home/francis/models/phi3-mini-4k-gguf/Phi-3-mini-4k-instruct-q4.gguf, Ollama tag p3m-local)

- `greedy` goinfer `{"max_tokens": 64, "model": "bench", "stream": true, "stream_options": {"include_usage": true}, "temperature": 0}`
  Ollama `{"model": "p3m-local", "options": {"num_ctx": 4096, "num_predict": 64, "seed": 1, "temperature": 0}, "stream": true}`
- `temp1.0_topk40_topp0.9` goinfer `{"max_tokens": 64, "model": "bench", "seed": 1, "stream": true, "stream_options": {"include_usage": true}, "temperature": 1.0, "top_k": 40, "top_p": 0.9}`
  Ollama `{"model": "p3m-local", "options": {"num_ctx": 4096, "num_predict": 64, "seed": 1, "temperature": 1.0, "top_k": 40, "top_p": 0.9}, "stream": true}`


## Readings

Ratio = goinfer ÷ Ollama on the cell's mean tok/s (below 1: Ollama is faster). The token gate is the harness's: a completion under 32 of 64 tokens voids the cell (`void`); `short` means every completion was at least 32 but not all reached 64.

### Run 1: the eight published cells, new readings against the published rows (ratio = goinfer / Ollama; below 1 means Ollama is faster)

| config | model | goinfer tok/s (runs) | Ollama tok/s (runs) | **new ratio** | token gate (goinfer / Ollama) | published goinfer / Ollama | **published ratio** |
|---|---|---|---|---|---|---|---|
| temp1.0_notrunc | 0.5B | 354.4 (352.4503790881874, 356.27659828902597) | 214.9 (214.9836531030438, 214.7250214964768) | **1.649** | void / ok | 237.7 / 268.7 | 0.885 |
| temp1.0_notrunc | gemma3-1b | 204.1 (203.91454541708055, 204.22533887785553) | 123.1 (123.12234595009562, 123.14525097659238) | **1.658** | ok / ok | 143.7 / 148.2 | 0.970 |
| temp1.0_notrunc | phi3-mini | 90.5 (90.45664491787782, 90.48382787203876) | 123.5 (123.61056045193597, 123.47249683224442) | **0.733** | ok / ok | 109.8 / 125.6 | 0.874 |
| temp0.8_topp0.95 | 0.5B | 343.5 (344.71815573746113, 342.1993810600502) | 173.7 (173.27336546496952, 174.1495440582535) | **1.978** | ok / ok | 227.2 / 265.2 | 0.857 |
| temp0.8_topp0.95 | gemma3-1b | 195.4 (196.4643694900721, 194.33530364291028) | 114.0 (114.05001960510909, 113.94802212814346) | **1.714** | ok / short | 124.7 / 148.2 | 0.841 |
| temp0.8_topp0.95 | phi3-mini | 89.8 (89.78116706484813, 89.79923563027141) | 119.3 (119.37651701631077, 119.2275032383715) | **0.753** | ok / ok | 102.1 / 125.7 | 0.812 |
| temp0.8_topk40 | gemma3-1b | 194.7 (193.64586789218515, 195.79433868988122) | 153.0 (153.08426822497876, 152.84610819322552) | **1.273** | ok / short | 156.4 / 148.0 | 1.057 |
| temp0.8_topk40 | phi3-mini | 89.7 (89.73002320486084, 89.66576012545335) | 125.9 (125.93924839224681, 125.87094631456422) | **0.712** | ok / ok | 112.1 / 125.7 | 0.892 |
| temp0.8_topk40 | 0.5B | 342.6 (341.3251565820293, 343.78999476723675) | 261.8 (262.6899643060581, 260.9251918528312) | **1.309** | ok / ok | — (not a published row: a control) | — |


**Cells that fail the harness's gate:** 0.5B `temp1.0_notrunc` is **VOID** on the goinfer arm (shortest completion 10 of 64 tokens: a sampled end of sequence, the same cause as `peer-claim-2026-09-25.md` cell f); its ratio above is not a reading. The `short` cells (Ollama's gemma3-1b arm, shortest completion 63 of 64 in `temp0.8_topp0.95` and 55 of 64 in `temp0.8_topk40`) pass the gate.
The published rows' phi3-mini figures were taken with Ollama tag `p3m`, a different revision of the weights (`peer-claim-2026-09-25.md`), and goinfer on a build and quant path that are not recorded in the row: the phi3-mini rows are not continuous with the published ones.

### Greedy depth 128, same session (phase A; controls)

| model | goinfer tok/s | Ollama tok/s | ratio |
|---|---|---|---|
| 0.5B | 350.1 | 263.4 | 1.329 |
| gemma3-1b | 203.4 | 150.9 | 1.348 |
| phi3-mini | 90.4 | 126.0 | 0.717 |


### Run 2: the shared truncated preset (temperature 1.0, top_k 40, top_p 0.9)

| config | model | goinfer tok/s (runs) | Ollama tok/s (runs) | **new ratio** | token gate (goinfer / Ollama) | published goinfer / Ollama | **published ratio** |
|---|---|---|---|---|---|---|---|
| temp1.0_topk40_topp0.9 | 0.5B | 338.3 (337.98340648356196, 338.70300024754306) | 262.7 (262.361654839422, 263.1119154852779) | **1.288** | ok / ok | — (not a published row: a control) | — |
| temp1.0_topk40_topp0.9 | gemma3-1b | 195.2 (195.86715675676356, 194.44605996554844) | 153.3 (153.3279229188476, 153.29888240623077) | **1.273** | ok / short | — (not a published row: a control) | — |
| temp1.0_topk40_topp0.9 | phi3-mini | 89.7 (89.65807571752578, 89.6743801241875) | 125.9 (126.0523783786211, 125.81731908740176) | **0.712** | ok / ok | — (not a published row: a control) | — |


Gate: Ollama's gemma3-1b arm `short` (shortest completion 57 of 64); the rest `ok`.

### Greedy depth 128, same session (phase A; controls)

| model | goinfer tok/s | Ollama tok/s | ratio |
|---|---|---|---|
| 0.5B | 353.9 | 264.5 | 1.338 |
| gemma3-1b | 200.0 | 150.7 | 1.327 |
| phi3-mini | 90.4 | 126.1 | 0.717 |


## What moved (observations, not causes)

- **The Ollama arm is slower under the explicit "off" values than in the published rows** for the two untruncated and the top-p configs (0.5B temp 1.0: 268.7 → 214.9; 0.5B temp 0.8 + top_p 0.95: 265.2 → 173.7; gemma3-1b temp 1.0: 148.2 → 123.1), and unchanged where top_k 40 is sent (gemma3-1b 148.0 → 153.0; phi3-mini 125.7 → 125.9; 0.5B 261.8). The published Ollama figures were taken before the harness sent top_k 0 and top_p 1.0 (`docs/benchmarks.md`, "Sampling changes that move comparability").
- **The goinfer arm is faster than the published rows on the 0.5B and gemma3-1b** (0.5B temp 1.0: 237.7 → 354.4; gemma3-1b temp 0.8 + top_p 0.95: 124.7 → 195.4) and **slower on phi3-mini** (109.8 → 90.5; its greedy cell reads 90.4, so the sampling configs cost it nothing here). The published goinfer rows for the top_p and top_k configs are marked stale in the docs (the R7 addendum).
- **Same-session controls agree across the two runs:** greedy ratios 1.329 / 1.348 / 0.717 (run 1) and 1.338 / 1.327 / 0.717 (run 2) for 0.5B / gemma3-1b / phi3-mini. The shared truncated preset reads 1.288 / 1.273 / 0.712, within about 4% of the greedy ratios on each model.
- Two runs of eight completions per cell: the per-run rates in the tables are the spread; this is not the 3-pair, order-reversed design of the pre-registered sweeps.

## Limits

One session each, two runs per cell. The goinfer and Ollama arms differ in seed handling in `temp1.0_notrunc` (above). No `min_p`. The 0.5B `temp1.0_notrunc` goinfer cell is void. phi3-mini is not comparable with its published rows (peer tag, quant path).
