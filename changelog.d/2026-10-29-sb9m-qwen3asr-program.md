- **Qwen3-ASR's encoder layer is a program** (SB9 step M, no change in output). The Qwen3-ASR encoder runs each layer by
  walking one description of its operations (`audio.QwenASRProgram`) on the CPU, its only backend, with the exact (erf)
  GELU Whisper's program added. The CPU tower golden is unchanged, and each defect planted in the program turns exactly
  the Qwen3-ASR case red.
