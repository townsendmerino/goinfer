- **`gate census` and `gate heavy` write a run record, and `-replay FILE` regrades it** (SB12, step 3). Each run leaves
  `gate_<name>_run.json` beside its streams in `-logdir`: the gate's policy, the provenance, and each cell's exit code,
  duration and stream. `-replay` prints and exits exactly as the run did (shown byte for byte on mbp2). The grading
  (`gradeTallyRun`, `reportCensus`, `reportTally`) lives in `grade_*.go` files that read no machine state. No verdict
  changes.
