# D8a-build: one int4 copy under Backend "cpu" on arm64, 2026-10-21

Gates: `docs/tasks/task-memory-campaign-2026-10.md`, §4 MEM13, "D8a-build gates" (D8A-G), registered before the code.
Machine: MacBook Pro, M1 Pro, 16 GB, macOS 26.6, Go 1.27.2, darwin/arm64; the owner's desktop open.

| gate | reading | result |
|---|---|---|
| D8A-G1 bytes | 40 families load (15 MoE); every row4-eligible int4 tensor is row4 alone under `Backend: "cpu"`; the int4 bytes halve on every family whose tensors are all eligible (`g1-g3.log`) | **PASS** |
| D8A-G2 bit-identity | token-by-token logits and 8 greedy tokens through `Generate` byte-equal between `Backend: "cpu"` and `Backend: ""` on all 40 | **PASS** |
| D8A-G3 writer | `SerializeWeights` of a `Backend: "cpu"` int4 llama-tiny refuses, naming the row4 layout, and returns no bytes | **PASS** |
| D8A-G4 tier 1 | `~/models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf`, `DirectLoad`, `Backend: "cpu"`, int4: Go heap in use 522.0 MiB (main) → 455.3 MiB (branch), −66.7 MiB against the ≥ 50 MiB bar (arithmetic ≈ 63); 16 greedy tokens identical (`g4-cpuload-mac.txt`) | **PASS** |

Planted defects, run once each on the committed tree and reverted: the load keeping the canonical copy (G1 red, G2
green), and the single copy quantized from a perturbed weight (G2 red, G1 green).

The G1 byte figures include Q/K/V and gate/up, which `Backend: "cpu"` already built as one copy before this change; the
gate's own change check counts layer 0's `o_proj` and `down_proj` moving from canonical plus row4 to row4 alone.
