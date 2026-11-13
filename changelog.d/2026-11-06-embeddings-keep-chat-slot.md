- **`/v1/embeddings` on a GPU-resident decoder model no longer evicts a parked chat's cached prefix.** An embeddings request
  (and `PromptHidden`, `PromptHiddenAll`) wrote the resident KV slot the last chat used, so that conversation's next turn
  prefilled from scratch. It now runs in an empty slot, else the least recently used one; with one slot the old cost remains.
