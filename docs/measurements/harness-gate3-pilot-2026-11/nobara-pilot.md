# Gate 3 pilot, nobara, 2026-11-09

Raw output `nobara-pilot.txt`; server banner `nobara-serve-banner.txt`. Time only, nothing graded.

- Machine nobara (RTX 2070 SUPER, 8 GB, driver 595.112.07, 16 cores, 62 GB RAM). Date 2026-11-09, 19:58 to 20:04 PST, exclusive timing lock, no other timed work. 1-minute load at the server's start 4.68 (the model load, not a cell); not a void-gated run.
- Build: main `20503ed33`, `backend/cuda` `go build -tags cuda`. Model `~/models/gemma4-26b-gguf/gemma-4-26B_q4_0-it.gguf` (local NVMe), `--moe-cache-experts --ctx 32768`.
- Fit (the server's own banner): decode path cuda-resident (int4); 32768-token cap needs 1.9 GB KV (weights 13.3 GB); KV plan 1 conversation (4 asked for). Expert cache 29 slots per layer.
- Request: the recorded Claude Code turn 1 (`req-1.json`, 18,635 prompt tokens), sampled at temperature 0.7 after a greedy first, 512 max tokens, Messages route, non-streaming. Not reproducible (no seed on the route); for timing that does not matter.
- Result: t_cold 284.7 s; t_warm median 2.4 s (30 requests, 75 prompt tokens each after prefix reuse); mean output 15 tokens, so t_warm is for short replies (5 tokens in 22 of 30; the 79 to 81 token replies took 5.7 s).
- Estimates as the script printed them: per M26-class cell, full 4.3 h, reduced 3.6 h. By the README's 2.85 factor nobara's whole Gate 3 is about 12.3 h full or 10.3 h reduced.
- A first launch the same evening (19:57) got HTTP 404 on request 1: the server was registered by its file id, not `pilot`. No timing was taken; the README's run command now names `pilot=<path>`.
