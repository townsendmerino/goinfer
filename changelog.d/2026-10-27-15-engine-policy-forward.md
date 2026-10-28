- **Engine policies get a `Forward` interface, a scripted fake model and a census of their token streams** (SB5).
  `decoder`'s `TestPolicyStreams_census` holds 38 token streams of greedy, n-gram, draft-model, grammar, session and
  sampled decoding on tiny fixtures, each exact policy's stream equal to plain decoding's. Grammar-constrained,
  draft-model and n-gram speculation and the session reuse policy each run as one function, shown exact on a fake model
  whose logits are scripted from the token history (every depth of rejection, argmax-only verifies, a context cap, ring
  and recurrent KV caches). No token stream changes.
