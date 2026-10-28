- **`gate identity` writes a run record, and `gate identity -replay FILE` regrades a run made with `-keep`** (SB12,
  step 6). `identity_run.json` in the run directory holds what identity printed before its verdict, every cell, and
  the times, host, HEAD and date its report reads. A replay judges each cell again from the kept dumps and prints
  exactly what the run printed. The run records of `census`, `heavy`, `parity` and `quick` now refuse a cell with no
  stream, or a stream that cannot be read, instead of grading it as empty. No verdict changes.
