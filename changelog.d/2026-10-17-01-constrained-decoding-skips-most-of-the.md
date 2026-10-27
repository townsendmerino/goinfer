- **Constrained decoding skips most of the vocabulary at structural positions.** Before walking the grammar for a token, the
  masker checks its first byte, and where few bytes are legal its second, against what the grammar accepts next. The mask is
  unchanged. With opencode's 19 tools, the tokens before a call's tool name cost about 0.5 s per call on nobara; that cost
  is now a few milliseconds.
