package decoder

// qwen35VLPerToken is a test seam: true forces Qwen3.5 multimodal prefill's per-token loop so tests can compare
// batched and per-token paths over the same prompt. Production never sets it.
var qwen35VLPerToken bool
