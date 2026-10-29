- **Qwen2.5-VL's vision block is a program on Metal too** (SB9 step M block 4, no change in output). Metal's Qwen2.5-VL
  tower walks `vision.Qwen25Program` through `gridVAccel.walkBlock`, and the last hand-ordered grid block loop on Metal is
  gone. The Metal tower golden is unchanged, and each defect planted in the program turns exactly the Qwen2.5-VL cases
  red on Metal.
