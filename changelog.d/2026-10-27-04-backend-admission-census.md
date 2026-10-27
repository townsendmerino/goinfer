- **Each GPU backend's own resident admission is pinned by a census** (SB2). `cuda`'s `TestBackendAdmissionCensus_cuda` and
  `gpu`'s `TestBackendAdmissionCensus_webgpu` take every tiny fixture and option cell that the decoder admits to a backend.
  For each, they record what that backend's build decides from the model alone: CUDA's plan-only build, and WebGPU's static
  gates. No device is opened. WebGPU's static gates now live in one function that `BuildResident` calls first. Its answers
  are unchanged: the real build gives the same answer on every censused cell before and after the move. No model's
  admission changes.
