- **The conversation codec has one home, `internal/codec`** (SB8, C1). It holds the path from a conversation to the
  token IDs a model reads: template detection from a tokenizer, a prompt's segments with or without tools, the image and
  audio block splice, and the encoding. `chat`'s codec census now also runs through it, against the same goldens. No
  caller uses it yet, and no prompt, token or public API changes: `chat` and `tokenizer` are as they were.
