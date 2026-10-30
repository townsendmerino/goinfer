- **Whisper's audio encoder runs as a program** (SB9 step M, no change in output). The Whisper encoder, which Voxtral
  shares, now runs each layer by walking one description of its operations (`audio.WhisperProgram`, on the vision
  tower's program type), on the CPU and on CUDA. Every tower output is bit-identical to its golden, and a defect planted
  in the program turns exactly the Whisper and Voxtral cases red.
