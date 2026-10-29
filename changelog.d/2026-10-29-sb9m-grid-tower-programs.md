- **The Qwen3-VL and GLM-OCR vision blocks are programs too** (SB9 step M, no change in output). On the CPU, CUDA and WebGPU,
  these towers now run each block by walking one description of its operations (`vision.Qwen3Program`,
  `vision.GlmOcrProgram`), as SigLIP's already did. All three CPU towers share one walker. Every tower output is
  bit-identical to its golden, and a defect planted in one program turns exactly that program's towers red.
