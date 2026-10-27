- **Every chat template's prompts, token IDs and parsed tool calls are pinned by a census** (SB8). `chat`'s
  `TestCodecCensus` records what the serving routes render for 55 template sources, per think mode, tool format and
  Harmony effort: nine fixed conversations (tool calls and results, a replayed reasoning turn, a user message carrying
  control-token text, among them), plus an image or audio placeholder where the family has one, so 9 to 11 per source.
  Token IDs are recorded for the sources that come with a tokenizer (the tracked tokenizers and local models), not for a
  bare template or a constructor. `TestToolCallParseCensus` records what each family's parser reads from 43 model
  outputs, malformed ones included, and checks the parsing contract directly. No prompt, token or parsed call changes
  in this release because of it; it is the before-picture the conversation codec's move to one package must reproduce
  byte for byte.
