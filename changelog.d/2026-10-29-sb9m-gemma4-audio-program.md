- **Gemma 4 audio's conformer block is a program** (SB9 step M shape 4, no change in output). The CPU and Metal Gemma 4
  audio towers walk one description of the conformer block (`audio.Gemma4AudioProgram`): the two half-step
  feed-forwards, local attention with relative-position keys and a softcap, and the GLU and depthwise-conv module. The
  tower program vocabulary gains SiLU, the GLU, the causal depthwise conv, local attention, the scaled residual add and a
  2·hidden register. Both towers' goldens are unchanged, and each defect planted in the program turns exactly the Gemma 4
  audio cases red on the CPU and on Metal.
