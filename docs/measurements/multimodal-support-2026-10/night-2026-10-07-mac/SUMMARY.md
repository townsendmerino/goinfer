# Night run 2026-10-07 — macbookpro.lan

Runner finished. Started 2026-10-07 22:23 PDT, deadline 06:30, ended 23:27 (1 h 04 min).

5 of 7 job(s) ok.

| # | job | result | ran | est | queued at rev | log |
|---|---|---|---|---|---|---|
| 1 | s2-tower-speed | ok | 11 min | 20 min | 36fe0d11 +dirty | `/Users/francistownsend-merino/goinfer-logs/night/runs/2026-10-07/s2-tower-speed.log` |
| 2 | s3-rootcause | ok | 9 min | 30 min | 6a5d4efb | `/Users/francistownsend-merino/goinfer-logs/night/runs/2026-10-07/s3-rootcause.log` |
| 3 | s9-speed | ok | 3 min | 15 min | fbd687d2 | `/Users/francistownsend-merino/goinfer-logs/night/runs/2026-10-07/s9-speed.log` |
| 4 | s6-e4b | FAILED rc=1 | 0 min | 1 h 00 min | ca874e46 | `/Users/francistownsend-merino/goinfer-logs/night/runs/2026-10-07/s6-e4b.log` |
| 5 | s7-mac | ok | 5 min | 40 min | af773dbd | `/Users/francistownsend-merino/goinfer-logs/night/runs/2026-10-07/s7-mac.log` |
| 6 | s13lite-mac | ok | 6 min | 40 min | af773dbd | `/Users/francistownsend-merino/goinfer-logs/night/runs/2026-10-07/s13lite-mac.log` |
| 7 | peer-vetted-mac | FAILED rc=1 | 2 min | 40 min | 575b4a8c | `/Users/francistownsend-merino/goinfer-logs/night/runs/2026-10-07/peer-vetted-mac.log` |

## Not ok — last 25 lines of each log

### s6-e4b — FAILED rc=1

```
=== cwd      /Users/francistownsend-merino/tmcode/goinfer
=== rev      queued at ca874e46, now c4e7435d
=== host     macbookpro.lan
=== est      60.0 min, timeout 120.0 min
=== by       Claude, S6
=== doc      docs/tasks/task-multimodal-support-2026-10.md
=== START    2026-10-07 23:04:09 PDT  loadavg 0.97
binaries: /Users/francistownsend-merino/goinfer-bench/s6 (main fb14b00b, serve 6e4449ae)
started:  2026-10-07 23:04:09 PDT
ProductName:		macOS ProductVersion:		26.6.2 BuildVersion:		25G83 
Now drawing from 'AC Power'
/dev/disk3s5   460Gi   414Gi    21Gi    96%    3.8M  216M    2%   /System/Volumes/Data
=== 1. sidecar 23:04:09
!! prequant failed
prequant: load /Users/francistownsend-merino/models/gemma-4-E4B-it (int4): decoder: gemma-4-E4B-it needs ~8.8 GB resident at quant int4 + 7.1 GB KV = 15.9 GB; this machine currently has 7.5 GB of memory available (budget 5.3 GB = 70% of that).
  Loading it would page to swap rather than run, so it was NOT loaded.
  This is a safetensors checkpoint — -stream-weights only helps a .gguf source. If you believe this machine can actually hold it (this guard's 70% margin is deliberately conservative), set GOINFER_NO_FIT_GUARD=1 and re-run; cmd/prequant can then build a streamable .giw from it once, for every later run.
  Or run a smaller model.
  Set GOINFER_NO_FIT_GUARD=1 to load anyway if this machine really fits it
  This is a safetensors checkpoint — -stream-weights only helps a .gguf source. If you believe this machine can actually hold it (this guard's 70% margin is deliberately conservative), set GOINFER_NO_FIT_GUARD=1 and re-run; cmd/prequant can then build a streamable .giw from it once, for every later run.
  Or run a smaller model.
  Set GOINFER_NO_FIT_GUARD=1 to load anyway if this machine really fits it
finished: 2026-10-07 23:04:10 PDT rc=1

=== END      2026-10-07 23:04:10 PDT  rc=1
```

### peer-vetted-mac — FAILED rc=1

```
  File "/Users/francistownsend-merino/goinfer-bench/peer-vetted/wt-65b2c22a/scripts/bench_peer.py", line 1824, in main
    rates, err, counts = run_cell(engine, mk, depth, cfg, backend)
                         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/francistownsend-merino/goinfer-bench/peer-vetted/wt-65b2c22a/scripts/bench_peer.py", line 1119, in run_cell
    prompt = prompt_for_depth(depth, model_key)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/francistownsend-merino/goinfer-bench/peer-vetted/wt-65b2c22a/scripts/bench_peer.py", line 880, in prompt_for_depth
    return _PROMPTS[f"{model_key}:{depth}"]["text"]
           ~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^
KeyError: 'G26Q:128'
=== 2026-10-08T06:25:43Z END rc=1; swap total = 4096.00M  used = 3069.69M  free = 1026.31M  (encrypted)
=== 2026-10-08T06:25:43Z START gpt-oss on Metal
# preflight OK: gate instant · loadavg [2.4599609375, 2.0966796875, 2.4521484375] · gpu None · compute apps []
# 2 cells planned, 0 already done
#   tokens/chunks = 4.0000  (1536 tok / 384 chunks)
#   token gate = ok  (min 64 / threshold 32, warm-up 64)
#   rss_peak = 7.380 GiB (7738208 KiB)
{"phase": "A", "engine": "goinfer", "backend": "metal", "model": "G20", "depth": 128, "prompt_tokens": 130, "prompt_format": "essay-v2", "config": "greedy", "sent": {"temperature": 0}, "note": "temperature=0 both sides (deterministic)", "runs": [28.838289086182233, 29.026920551882966, 29.0183594447717], "error": null, "machine": {"loadavg": [2.5029296875, 2.111328125, 2.455078125], "gpu": null, "therm": "Note: No thermal warning level has been recorded / Note: No performance warning level has been recorded / Note: No CPU power status has been recorded", "gate": {"kind": "instant", "wait_s": 3.4, "busy_pct": 2.1}}, "secs": 72.0, "counts": {"swap": {"swap_base_mb": 3069.69, "swap_growth_mb": 0.0, "swap_killed": null}, "tokens": 1536, "chunks": 384, "tokens_per_chunk": 4.0, "completion_rates": [27.59650536355641, 28.96602275641429, 29.036540513908186, 28.973419446829986, 28.913295512251473, 29.001674891766747, 29.129615751314727, 29.089238453416037, 28.805636580419964, 28.981109291680074, 29.02937408394816, 29.14389574464619, 29.02335098673297, 29.375980153540258, 28.84491509980263, 29.011102474293498, 28.974424379985745, 29.001401750557683, 29.019606570892805, 29.023785552495244, 28.98794902373639, 28.96068658641037, 29.099580454318016, 29.07944123977735], "ngen": 64, "completion_tokens": [64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64], "warmup_tokens": 64, "token_gate": {"verdict": "ok", "min": 64, "threshold": 32, "rule": "void if any completion < 32 of 64 tokens or unreported"}, "decode_path": "metal-resident (int4mix\u2192int4, no Metal int8 GEMV kernel)", "rss_peak_kb": 7738208, "phases": {"start_to_listening_s": 2.05, "warmup_s": 10.04, "runs_wall_s": 56.81, "decode_timed_s": 52.21, "prefill_and_request_s": 4.6, "teardown_s": 3.06}}, "mean": 29.0, "spread": 0.2}
{"phase": "A", "engine": "ollama", "backend": "metal", "model": "G20", "depth": 128, "prompt_tokens": 130, "prompt_format": "essay-v2", "config": "greedy", "sent": {"temperature": 0, "seed": 1}, "note": "temperature=0 both sides (deterministic)", "runs": null, "error": "warmup failed: HTTP Error 500: Internal Server Error", "machine": {"loadavg": [1.640625, 1.9228515625, 2.35107421875], "gpu": null, "therm": "Note: No thermal warning level has been recorded / Note: No performance warning level has been recorded / Note: No CPU power status has been recorded", "gate": {"kind": "instant", "wait_s": 3.4, "busy_pct": 1.8}}, "secs": 15.1}
=== 2026-10-08T06:27:21Z END rc=0; swap total = 4096.00M  used = 3069.69M  free = 1026.31M  (encrypted)
## metal-g20.json
{"file": "metal-g20.json", "backend": "metal", "model": "G20", "depth": 128, "config": "greedy", "goinfer": [28.8, 29.0, 29.0], "ollama": [], "decode_path": "metal-resident (int4mix\u2192int4, no Metal int8 GEMV kernel)", "swap_mb": [0.0, null], "ollama_version": "0.32.5", "ctx_pin": 2048, "outcome": "GOINFER-ALONE", "why": "Ollama: warmup failed: HTTP Error 500: Internal Server Error", "median_goinfer": 29.0}
## metal-g26q.json

=== END      2026-10-07 23:27:21 PDT  rc=1
```


## Who picks these up

- s2-tower-speed: Claude, S2 · docs/tasks/task-multimodal-support-2026-10.md
- s3-rootcause: Claude, S3 root cause · docs/tasks/task-multimodal-support-2026-10.md
- s9-speed: Claude, S9 · docs/tasks/task-multimodal-support-2026-10.md
- s6-e4b: Claude, S6 · docs/tasks/task-multimodal-support-2026-10.md
- s7-mac: Claude, S7 · docs/tasks/task-multimodal-support-2026-10.md
- s13lite-mac: Claude, S13-lite · docs/tasks/task-multimodal-support-2026-10.md
- peer-vetted-mac: Claude, peer-vetted (M3 26B + M2 gpt-oss) · docs/measurements/peer-vetted-2026-10-07-macbook.md
