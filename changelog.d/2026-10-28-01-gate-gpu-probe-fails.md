- **`gate gpu` fails the WebGPU group when its adapter probe never answers.** A probe that fails to build, crashes, or
  exits non-zero before reporting now counts as a failure of the WebGPU group, and the gate's verdict goes red. Before,
  the gate read it as "no WebGPU adapter" and skipped the group, which let a broken `-tags gpu` test build pass for
  missing hardware. Set `GOINFER_GATE_SKIP_WEBGPU` to skip the group on purpose. A probe that runs and finds no adapter
  still skips the group as before. This is a verdict change, approved by Francis on 2026-10-28.
