# MM1's MoE reserve (Francis's option A): the census diff, 2026-10-27 (nobara)

Gate G-MA-1 in `docs/tasks/task-subsystems-2026-10.md`, "MM1's MoE reserve". The planner reserves an MoE placement's experts
before its context grows: resident first, with every expert and its host copy, and then the expert cache, with the top-k.

## How it was measured
- `go test ./decoder -run 'TestDecisionCensus$' -args -census.dump=<file>`, at the base (#320's head, ddf85d869) and with the
  change, on nobara (amd64).
- `diff.py <before> <after> <rows.json>` compares the two dumps. The census records only the readings where a group's outcome
  class changes (its placement, or for the Metal fit: floor, above the floor), plus the zoo's readings. So:
  - placements and classes are compared at every reading either run records;
  - contexts are compared only at readings both runs record, because a context between two recorded readings is not known.
    A first pass that compared contexts between readings reported Metal contexts falling by up to 12,224 positions. Those
    readings were ones the new curve does not record (it climbs in 8-position steps past the census's change-point cap), so
    that pass was wrong.
- `groups.tsv.gz` lists every changed group, with its moves by direction. Regenerate the dumps from the two commits.

## Result
| | groups |
|---|---:|
| in the census | 29,637 |
| changed | 3,534 (D3 3,366; D4 168) |
| with a placement or class move | 1,068 (D3 984; D4 84) |
| text only (a reason line, or a context at a reading only one run records) | 2,466 |

The placement and class moves, by reading:

| move | readings |
|---|---:|
| declined → expert-cached | 9,676 |
| weight-paged → expert-cached | 5,028 |
| declined → resident | 732 |
| expert-cached → resident | 92 |
| Metal fit: floor → above the floor | 1,232 |
| **any move down the order, or a Metal fit back to the floor** | **0** |

- Every moved group is an MoE fixture: qwen35, qwen3_5_moe, qwen3next, qwen3moe (and -k3), qwen3vlmoe, mixtral, deepseek,
  gemma4-moe-kv (and -32), glm (safetensors, .gguf, split .gguf, -bias), gptoss, bailing_hybrid, laguna (m1, xs2, xs21),
  llama4 and nemotron3nano. No dense group moves.
- The D4 moves (Metal's context fit, 84 groups over 12 MoE fixtures) are the fix to the Metal fall: an MoE load's fit now
  leaves the floor where all of its experts fit beside a longer context, instead of only at the candidate.
- At readings both runs record, the context differs only on declines (108, the attempted context a refusal echoes).

## Real MoE checkpoints on nobara (not in the census)
`goinfer fit <model> -backend cuda` built from the base and from the change, back to back on nobara (RTX 2070 SUPER, the
desktop's context present), each model from `~/models`:

| model | before | after |
|---|---|---|
| `gemma4-26b-q4_k_m.gguf` | declined: "expert-cached needs at least top-k=8 slots/layer (0.75 GB) but only 0.01 GB is free beside dense+KV" | expert-cached, 8/128 experts, context 11,264 |
| `gemma4-26b-q4_k_m.int4.e4h.cuda.v17.giw` | declined (same reason) | expert-cached, 8/128 experts, context 11,264 |
| `qwen15-moe-a27b` (int4) | declined: "top-k=4 slots/layer (0.44 GB) but only 0.02 GB is free" | expert-cached, 4/60 experts, context 15,104 |

Before the change, `fit` told a user that the 26B does not fit this card, while the CUDA build (its own context fit) runs it
expert-cached. The host guard's verdict on the recorded profile (`fit -profile testdata/machines/nobara-desktop.json`) is
byte-identical before and after for all three.
