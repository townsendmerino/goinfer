- **On `/v1/messages`, `input_tokens` now counts only the tokens prefilled for that request; reused tokens are reported as
  `cache_read_input_tokens`, matching Anthropic's definitions.** The prompt is their sum. Before, `input_tokens` was the
  whole prompt and no reuse was reported, so a cold request reads as before. A client that took `input_tokens` as the prompt
  size adds `cache_read_input_tokens`, as Claude Code already does. A stream's `message_delta` carries the split, and its
  `message_start` the whole prompt (`docs/server.md`, "usage on /v1/messages").
