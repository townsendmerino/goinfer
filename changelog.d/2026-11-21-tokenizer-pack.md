- **Tokenizer packs.** `tokenizer.BuildPack` turns a Gemma-family (SentencePiece-style BPE) `tokenizer.json` into a
  deterministic binary pack, and `tokenizer.LoadPack` loads it in tens of milliseconds instead of parsing the JSON, with
  identical token ids. Byte-level tokenizers are refused.
