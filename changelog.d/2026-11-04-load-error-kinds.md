- **`decoder.Load`'s refusals can be told apart with `errors.Is`:** `ErrUnsupportedModel`, `ErrUnknownQuant` and `ErrUnknownBackend`. Their
  text is unchanged. A preprocess config that exists but cannot be read is now an error, where the Qwen, LFM2-VL and Pixtral loaders used to
  fall back to default image sizes without a word.
