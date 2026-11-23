- **Changes behavior: on Metal, Phi-3 mini (head dimension 96) decodes on the split-KV flash-decode lane by default, at a context of at
  most 4,096,** the rule CUDA took the same day. `--flash-decode-hd96` now applies on Metal: its hd-96 kernel runs once a layer attends
  1,024 keys or more. Its readings, on Phi-3-mini-4k at q4k on an M1 Pro:
  - served decode 1.29× faster at depth 3,900 (28.5 against 22.1 tok/s), and level at 128;
  - about 3.6 MB of attention scratch when the lane is set (401 KB partial buffer, up to 3.2 MB for the batched step), priced in the fit
    check; the longest context that fits does not change;
  - outputs differ from the exact path's, inside the lane's fidelity gate: agreement equal to exact's (99.84%), no hard flips.

  A longer context keeps the exact path unless `--flash-decode-hd96` is given. `--flash-decode-hd96=false` turns the lane off. Decided
  by Francis, 2026-11-23.
