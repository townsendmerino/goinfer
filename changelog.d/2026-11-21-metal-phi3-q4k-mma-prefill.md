- **Faster: Phi-3's prompt processing on Metal runs through the f16 matrix-multiply prefill.** The q4k lane used to
  process a prompt one step at a time. On an M1 Pro, Phi-3 mini 4k q4 now prefills 6.04x as fast at 512 tokens and 4.73x
  at 3,900 (111.9 s to 23.7 s), inside its registered fidelity gate (agreement and KL level, no hard flips) (#1162).
