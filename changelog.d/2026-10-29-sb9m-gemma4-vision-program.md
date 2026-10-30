- **SB9 step M block 3, Gemma 4 vision (`internal/vision`, `metal`, `docs/tasks/task-subsystems-2026-10.md` §8)**:
  `vision.Gemma4Program` describes Gemma 4 vision's sandwich-norm block, and the CPU tower (`Gemma4Encoder.Forward`)
  and the Metal tower (`g4vAccel.Hidden`) walk it in place of their own layer loops. The vocabulary gains the residual
  add, the GELU-tanh product, axial RoPE, the unscaled RMSNorm form and the post-attention and post-MLP norm weights,
  and the CPU walker applies a tower's linear clamps (Gemma 4's ClippableLinear). The CPU and Metal tower goldens are
  unchanged, and the four program plants turn exactly `gemma4-vision` red on both. CUDA and WebGPU walk it next
  (nobara-mm01).
