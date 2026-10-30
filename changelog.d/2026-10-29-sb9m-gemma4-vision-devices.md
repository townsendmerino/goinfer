- **Gemma 4's vision block runs as a program on CUDA and WebGPU too** (SB9 step M, no change in output). The CUDA and
  WebGPU Gemma 4 vision towers now walk `vision.Gemma4Program` each block, as the CPU and Metal do. Every tower output
  is bit-identical to its golden, and a defect planted in the program turns exactly the Gemma 4 vision cases red.
