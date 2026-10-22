# D8b-build: q4k's Q4_K tensors as .giw kind 8 (the CPU side), 2026-10-21

Gates: `docs/tasks/task-memory-campaign-2026-10.md`, §4 MEM13, "D8b-build gates" (D8B-G), registered before the code and
amended before any verdict (the kind is 8; the version question is D9). Machine: MacBook Pro, M1 Pro, 16 GB, macOS 26.6,
Go 1.27.2, darwin/arm64; the owner's desktop open.

| gate | reading | result |
|---|---|---|
| D8B-G1 round trip | a one-layer llama `.gguf` with seven Q4_K projections, written by the test: through `TranscodeGIW(q4k)` all seven come back as Q4_K, aliased from the mapping; token-by-token logits and 8 greedy tokens byte-equal to the direct q4k load; the bundle reports `q4k` and runs per-32 activations with no `Quant` set | **PASS** |
| D8B-G2 routing | `decoder.Load`'s default opens the q4k sidecar and computes what the direct load does; `modelload.routeFor` sends a q4k `.gguf` to its sidecar, keeps `-direct-load` direct, refuses `-stream-weights`, and keeps a directory at q4k direct | **PASS** |
| D8B-G3 version | at v17: the version and sidecar tests green; a kind-8 bundle written on this branch and loaded by main's build (`97adc2b1`) is refused naming "encoding 8", the error a sidecar rebuild keys on | **PASS** |
| D8B-G4 tier 1 | `~/models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf`, `Quant: "q4k"`, `Backend: "cpu"`: Go heap in use 583.6 MiB direct → 1.7 MiB through the sidecar (bar ≤ 0.25 ×); 16 greedy tokens identical; the transcode took 1 s and wrote 587 MB (`g4-q4kload-mac.txt`) | **PASS** |

Plants, each run once on the committed tree: kind 8 written with an empty payload (G1 and G2 red); the routing left
direct (G2 red).

Not run here: the CUDA cell (ledger row D8b-cuda, nobara).
