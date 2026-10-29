- **SB9 step M block 2 on Metal (`metal`, `docs/tasks/task-subsystems-2026-10.md` §8)**: Metal's Qwen3-VL (with DeepStack)
  and GLM-OCR towers walk `vision.Qwen3Program` and `vision.GlmOcrProgram` through `gridVAccel.walkBlock`, which gains the
  RMSNorm form, the per-head norm, the fused qkv (the three split linears), RoPE and the SiLU product. The shared block
  loop keeps Qwen2.5-VL alone. The Metal tower golden is unchanged, and all four program plants turn exactly each
  program's families red on Metal.
  `walkBlock` calls the shared `vision.TowerOp.CheckRegs` before every op, as the other walkers do.
