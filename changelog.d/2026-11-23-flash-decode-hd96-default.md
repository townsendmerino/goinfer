- **Changes behavior: on CUDA, Phi-3 mini (head dimension 96) decodes on the flash-decode lane by default, at a context of at most
  4,096.** `--flash-decode-hd96` (new) is on by default for such a load and runs the lane's new hd-96 kernels once a layer attends
  2,048 keys or more. Its readings, on Phi-3-mini-4k at q4k:
  - served decode 1.15× faster at depth 3,900 (67.9 against 59.1 tok/s), and level at 128;
  - no change in device or host memory, or in the longest context that fits;
  - outputs differ from the exact path's, inside the lane's fidelity gate: agreement 0.78 pt below exact, no hard flips.

  A longer context keeps the exact path unless `--flash-decode-hd96` is given, because the gates read only a 4,096 context.
  `--flash-decode-hd96=false` turns the lane off. Decided by Francis, 2026-11-22 and 2026-11-23.
