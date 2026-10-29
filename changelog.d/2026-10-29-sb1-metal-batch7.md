- **SB1 step 1, Metal batch 7 (`internal/mamba2`, `metal/kernels`, `docs/tasks/task-subsystems-2026-10.md` §8)**: the
  Mamba-2 mixer between its projections (causal conv, selective scan, grouped gated RMSNorm) moves out of `decoder` into
  `internal/mamba2`, with `decoder`'s `mamba2Step` calling `mamba2.Mix` so there is a single implementation (`decoder`
  bit-identical on its Mamba-2, Granite and Nemotron tests and on hashed tiny-fixture logits). `TestMambaGNorm_groups` and
  `TestActQuant_relu2` move to `metal/kernels`; the two Mamba chain tests stay in `metal`, which still loads the fixtures
  through `decoder` for their inputs.
