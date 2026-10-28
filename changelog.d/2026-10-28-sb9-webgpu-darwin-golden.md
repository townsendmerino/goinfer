- **Per-tower goldens for WebGPU on macOS** (SB9, tests only). The WebGPU backend's 15 tower cases on an Apple M1 Pro are now
  compared bit for bit with a committed golden. On CI, whose runners have a virtual GPU with no golden of its own, a
  different adapter skips with a named reason; on any real machine it still fails until the golden is re-recorded.
