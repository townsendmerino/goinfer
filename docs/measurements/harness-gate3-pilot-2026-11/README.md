# Gate 3's by-day pilot: one harness request, turn 1 teacher-forced, on Gemma 4 26B-A4B

**For** `docs/tasks/task-harness-reliability-2026-10.md`, row Gate-3. Francis's choice of 2026-11-09: a 5-10 minute pilot
first, by day, to pin the per-request time the night-cost estimate rests on; he then picks Gate 3's size. Not a gate: it
measures time only, and nothing in it is graded.

**Box and model.** nobara first (CUDA, the RTX 2070 SUPER): Gemma 4 26B-A4B runs there through the expert cache, and Gate 3's
CUDA cells are its own. Model: `~/models/gemma4-26b-gguf/gemma-4-26B_q4_0-it.gguf` (the vetted QAT file, `bench_peer.py`'s
G26Q). The Mac's Metal cells get the same pilot when its session is up, with the Mac's copy of the same file under
`~/models`; its number is the Mac's, not nobara's.

**1. Record the request** (once, seconds; no model answers, so the harness runs no tool):
```
python3 docs/measurements/integration-harness-2026-10/harness_capture.py <dir holding the claude binary> ~/goinfer-bench/gate3-pilot/capture claude
```
The first Claude Code request lands in `~/goinfer-bench/gate3-pilot/capture/claude/req-1.json`: its system prompt and tools,
the harness-scale prompt Gate 3's cells carry.

**2. Build serve and check the fit** (a minute; it must not refuse):
```
(cd backend/cuda && go build -tags cuda -o ~/goinfer-bench/gate3-pilot/serve ./cmd/serve)
~/goinfer-bench/gate3-pilot/serve --model ~/models/gemma4-26b-gguf/gemma-4-26B_q4_0-it.gguf --backend cuda \
  --moe-cache-experts --ctx 32768 --addr 127.0.0.1:18600
```
Read the `fit:` and `KV plan:` banner lines and the decode path. A refusal or a CPU decode path stops the pilot; report it
with the banner. Stop the server once the banner is read.

**3. Run, under the exclusive timing lock** (about 5-10 minutes: 31 requests):
```
python3 scripts/timing_lock.py run --label gate3-pilot -- bash -c '
  ~/goinfer-bench/gate3-pilot/serve --model ~/models/gemma4-26b-gguf/gemma-4-26B_q4_0-it.gguf --backend cuda \
    --moe-cache-experts --ctx 32768 --addr 127.0.0.1:18600 > ~/goinfer-bench/gate3-pilot/serve.log 2>&1 &
  pid=$!; until curl -sf http://127.0.0.1:18600/health >/dev/null; do sleep 2; done
  python3 docs/measurements/harness-gate3-pilot-2026-11/pilot_turn1.py \
    ~/goinfer-bench/gate3-pilot/capture/claude/req-1.json http://127.0.0.1:18600 | tee ~/goinfer-bench/gate3-pilot/pilot.txt
  kill $pid'
```
It prints each request's time and tokens, then `t_cold` (the first request, the whole prompt's prefill), `t_warm` (the
median of the 30 sampled requests, which reuse the prefix), and the per-cell estimates. Commit `pilot.txt` and the serve
banner beside this file. The Messages route takes no seed, so the sampled requests are not reproducible; for a timing that
does not matter.

**4. The estimate.** Per cell, with a trace of T = 10 turns, C = 4 trace-and-dialect combinations, free-running sessions of
12 turns:
- full (30 seeds, 10 sessions): C × [T × (t_cold + 30 × t_warm) + 10 × 12 × t_warm]
- reduced (10 seeds, 5 sessions): C × [T × (t_cold + 10 × t_warm) + 5 × 12 × t_warm]

`t_cold` per turn is an upper bound: a later turn's prefill is its delta, not the whole prompt. Gemma 4 26B is the most
expensive of nobara's cells. The others take, per request, about 0.3 (phi3-mini), 0.25 (granite-4.0-h-tiny), 0.5 (a
Qwen3.5 dense) and 0.8 (gpt-oss-20b) of its time, from the 2026-11-09 sizing sent to the coordinator (decode rates on
nobara). So nobara's Gate 3 is about 2.85 × the pilot's per-cell figure, full or reduced; phi3-mini's 4k context does not
hold the Claude Code trace, so its share applies only to the traces that fit. The Mac's total comes from its own pilot.
