- **`fit` and the serve banner price every KV slot Metal keeps.** A load without `--kv-sessions` keeps two KV slots on Metal where
  they fit. `goinfer-chat fit`'s Metal line now prices both, and says how many it keeps; before, it priced one at a pinned `-ctx` and
  at the default context. The banner's `fit:` line, on a resident with two or more slots, now gives the KV the pool holds and the room
  left after all of it. On the Phi-3 mini q4 at `-ctx 2048`, that is 1.5 GB, against the 0.8 GB of one conversation it printed.
