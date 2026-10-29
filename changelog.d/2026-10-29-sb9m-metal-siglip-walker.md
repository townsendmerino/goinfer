- **SB9 step M block 1 on Metal (`metal`, `docs/tasks/task-subsystems-2026-10.md` §8)**: Metal's SigLIP tower walks
  `vision.SiglipProgram` (`gridVAccel.walkBlock`, through the tower's existing LayerNorm, linear, attention and GELU-tanh
  operations) in place of the grid towers' shared block loop, which keeps Qwen3, GLM-OCR and Qwen2.5-VL
  (`gridVAccel.loopBlock`). The Metal tower golden is unchanged, and `TestTowerGolden_metal` now runs
  `towerid.CheckProgramPlants`: each of the three program plants turns the `siglip` cases red on Metal (its golden has no
  `siglip2-naflex` case). `CheckProgramPlants` now fails a golden with no case of a family that walks the program, and
  `walkBlock` refuses an op routed to registers its operation does not use or whose widths do not match its weight.
