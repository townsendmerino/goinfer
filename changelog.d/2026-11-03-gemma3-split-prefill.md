- **Gemma 3 on Metal prefills with the split attention kernel.** It is the default at head dim 256, and outputs change. Checked on Gemma 3 1B
  against the CPU at int4: 91.7% top-1 agreement over 96 positions, where the exact kernel it replaces reads 94.8%, and every miss is a
  near-tie. Its speed on Gemma 3 has not been measured; on Gemma 4 E2B it made a 4,096-token prefill 4.61× faster.
