- **Faster: min-p sampling scans the vocabulary on all cores.** With `min_p` set, the candidate scan over the logits is split across cores
  and gives the same candidates in the same order: 2.7x faster at a 262k vocabulary (about 110 us a token), 1.9x at 152k and 1.05x at 32k
  (M1 Pro; `docs/measurements/p42-r15-2026-11-09/README.md`).
