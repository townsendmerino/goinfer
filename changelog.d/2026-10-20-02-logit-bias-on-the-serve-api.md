- **`logit_bias` on the serve API.** `/v1/chat/completions`, `/v1/completions` and `/v1/responses` take OpenAI's `logit_bias`
  (token id to a value in [-100,100], added to the logit), and `/v1/messages` takes the same field as an extension. Before, the
  field was ignored. The sampler already applied a bias, and its on-device fast paths already stand aside for one. A malformed
  bias is a 400.
