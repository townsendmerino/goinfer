- **Fixed: Metal's batched `ForwardN` ran each Gated-DeltaNet layer on the resident's own residual, not the row's.** A batch of N tokens through a
  DeltaNet model (Qwen3.5, Qwen3-Next) read the wrong input in those layers, so its logits differed from N sequential steps. Each row now
  runs on its own residual, bit-identical to the sequential steps (N of 2, 4 and 8, from position 0 and 4).
