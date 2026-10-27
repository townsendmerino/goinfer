- **A `tool_choice` of `required` (OpenAI) or `any` (Anthropic) on a family with no JSON call form is an error when the reply
  holds no call,** instead of prose. Those families (Gemma 4, gpt-oss, a model under `-tool-format template`) are parsed, not
  constrained, so the check runs once the reply is parsed, and no prose streams before it. Each model's `/v1/models` and `/health`
  entry reports how a forced call is enforced: `forced_tool_call` is `grammar`, `post-check` or `off` (`GOINFER_TOOL_UNION=0`).
