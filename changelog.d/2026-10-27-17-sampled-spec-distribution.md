- **Sampled n-gram speculation is checked lossless in distribution** (SB5). On a fake model, `decoder`'s
  `TestNgramSpecLoop_sampledIsLosslessOnTheFake` compares 30,000 seeded speculative runs against plain sampling's
  distribution, computed exactly. It holds the total variation under 0.03, both when drafts are accepted and when they
  are rejected. No token stream changes.
