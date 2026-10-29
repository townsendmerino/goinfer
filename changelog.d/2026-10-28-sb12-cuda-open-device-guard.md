- **The CUDA test suite names any device a test leaves open** (SB12, tests only). Every device a CUDA test creates is
  counted until it is released. The VRAM ledger test fails at its start, naming the open devices, when an earlier test
  left one open, because an open device shares its small-buffer pool with later ones and makes the measured memory read
  short. The suite fails at the end on any device never released. Seven tests that leaked a device now release it.
