- **`TestNemotronResidentCUDA_tier1` defaults `NEMOTRON_TIER1` to `"4b"` when unset, so `gate gpu` runs the 4B whole-model
  tier-1 cell without an extra environment variable.** When `NEMOTRON_TIER1` is unset and the Qwen2.5-Coder 1.5B control or
  Nemotron-H-4B-Instruct-128K checkpoint is absent on disk, the test skips naming the checkpoint path so `gate gpu` grades
  the `nemotron-h` heavy group as a named all-skip group and the 2c skip census lists it. Setting `NEMOTRON_TIER1`
  explicitly (`4b`, `9bslice`, `9b`) still fails with `VOID` when a requested checkpoint is missing or when a token names
  an unknown cell. Approved by Francis on 2026-10-29.
