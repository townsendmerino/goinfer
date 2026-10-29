- **SigLIP's vision block is one program on every backend** (SB9 step M, no change in output). The CPU, CUDA and WebGPU
  SigLIP towers, and SigLIP2 NaFlex on the CPU, now run each block by walking one description of its operations
  (`vision.SiglipProgram`) instead of a loop of their own. Every tower output is bit-identical to its golden, and a defect
  planted in the program turns the SigLIP towers red on each of those backends.
