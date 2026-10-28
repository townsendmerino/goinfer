- **Engine policies get a `Forward` interface, a scripted fake model and a census of their token streams** (SB5).
  `decoder`'s `TestPolicyStreams_census` holds 38 token streams of greedy, n-gram, draft-model, grammar, session and
  sampled decoding on tiny fixtures, each exact policy's stream equal to plain decoding's. Grammar-constrained and
  draft-model speculation now run over `Forward` (CPU and resident) and are shown exact on a fake model whose logits are
  scripted from the token history (339 cases, every depth of rejection, a context cap). No token stream changes.
