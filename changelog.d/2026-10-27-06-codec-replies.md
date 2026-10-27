- **Replies are read through `internal/codec` too** (SB8, C3). Splitting a reply into reasoning and answer, settling a
  finished turn's tool calls (the inline parse, a gpt-oss router's calls, and a call read from the reasoning of a turn
  with no answer) and reconciling streamed prose moved from `goinfer-serve` into the codec; `goinfer-serve` and
  `goinfer-chat` call it. No reply is read differently: the census's reply and parse tables hold it, run through the codec.
