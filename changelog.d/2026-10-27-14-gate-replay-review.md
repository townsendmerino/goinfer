- **`gate parity -replay` uses the gate lists the sweep ran with** (SB12, Gemini's review). A parity run record now
  carries `neverConfirmed`, `assetNeverBuilt` and `realckptNotRequired` as they were, so editing those lists does not
  change an old record's verdict. Each cell's counts are printed from its stream, and the summary-record line from the
  record. `gate quick -replay` keeps wall-clock times only, and grades a job that failed before it ran by its error.
