- **Every chat template's prompts, token IDs and parsed tool calls are pinned by a census** (SB8). `chat`'s
  `TestCodecCensus` records what the serving routes render and encode for eleven fixed conversations (tool calls and
  results, a replayed reasoning turn, image and audio placeholders, a user message carrying control-token text) for 55
  template sources, per think mode, tool format and Harmony effort; `TestToolCallParseCensus` records what each family's
  parser reads from 42 model outputs, malformed ones included. No prompt, token or parsed call changes in this release
  because of it; it is the before-picture the conversation codec's move to one package must reproduce byte for byte.
