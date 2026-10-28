- **`serve`'s text generation runs through `engine.Engine`** (SB7b). The shell builds an `engine.Request` and starts
  it on the model's engine. No behavior changes: the wire census reproduces byte for byte.
