- **`goinfer fit` and the context plan count a hybrid model's recurrent state.** Gated DeltaNet, Mamba-2, KDA and short-conv layers
  hold a window and a state per KV slot. The plan now prices them on their own line and names them in its reason, so a hybrid's planned
  context can come out slightly lower. A model without such layers plans exactly as before.
