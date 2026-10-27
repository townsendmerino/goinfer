- **gpu: the WebGPU kernel layer moved to `gpu/internal/kernels`** (SB1; goes out in gpu v0.24.0). `gpu`'s twelve exported
  types (`Context`, `DecodeRunner`, `DeviceBuffer`, `GEMVRunner`, `AttnWeights`, `LayerW`, `ModelW`, `ResidentMatrix`,
  `ResidentW8A8`, `ResidentW4A8`, `ResidentStackedW8A8`, `VisionEncoder`) are aliases of that package's types and its
  three functions (`New`, `GPUEverAvailable`, `LiveBufferBytes`) forward to it, so a program written against `gpu`
  compiles unchanged and the WebGPU backend still registers through `import _ ".../gpu"`. What changes: apidiff reports
  the twelve types' identity (and `New`'s signature, which names `*Context`) as incompatible, and `%T` and reflection
  print `kernels.Context` and so on. No new public package: the kernel layer is internal, and no name became exported on
  a public type. Every kernel's code is the same (`docs/measurements/sb1-gpu-split-2026-10-27/`).
