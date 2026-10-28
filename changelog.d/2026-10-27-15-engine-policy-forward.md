- **Engine policies get a `Forward` interface, a scripted fake model and a census of their token streams** (SB5).
  `decoder`'s `TestPolicyStreams_census` holds 38 token streams of greedy, n-gram, draft-model, grammar, session and
  sampled decoding on tiny fixtures, each exact policy's stream equal to plain decoding's. Grammar-constrained
  speculation now runs over `Forward` and is shown exact against constrained greedy on a fake model whose logits are
  scripted from the token history (216 cases, every depth of rejection). No token stream changes.
