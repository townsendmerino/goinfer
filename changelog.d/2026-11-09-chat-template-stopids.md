- **`chat.Template.StopIDs(tok)` returns the token ids that end a generation.** The tokenizer's end-of-sequence and
  end-of-turn ids where it has them, then each of the template's stop strings that is a single token, each id once: what
  `decoder.SamplingParams.StopIDs` and `constrain.NewMasker` take, without the loop every caller wrote. Hard tier.
