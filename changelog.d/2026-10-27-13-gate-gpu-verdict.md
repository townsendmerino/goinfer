- **`gate gpu`'s verdict is a pure function of its tally, and its record holds that tally by name** (SB12, step 7).
  `gradeGPU` (`cmd/gate/grade_gpu.go`) decides NO_GATE, VACUOUS, EMPTY, FAIL, INCONCLUSIVE or PASS from the check groups'
  tally and the host, dirty lines and time the runner gathers first. Each `docs/measurements/gpu-gate` record now also
  holds the groups' names, the vacuous and empty cells and the skip notes, so it can be regraded in full. No verdict
  changes.
