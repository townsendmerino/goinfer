- **Qwen2.5-VL's vision block is a program too** (SB9 step M, no change in output). On the CPU, CUDA and WebGPU, the
  Qwen2.5-VL tower now runs each block by walking one description of its operations (`vision.Qwen25Program`). Each block
  still attends over its own windows or, in the full-attention blocks, over the whole frame. Every tower output is
  bit-identical to its golden, and a defect planted in the program turns exactly the Qwen2.5-VL cases red.
