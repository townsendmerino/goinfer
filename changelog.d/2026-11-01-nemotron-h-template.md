- **Nemotron-H 4B Instruct renders its own chat template.** `serve` and `chat` now write its turn markers and stop at
  `<SPECIAL_11>`, byte for byte with HuggingFace's render, instead of a raw prompt whose reply ran on past the answer.
  Nemotron-Nano-9B-v2 has a different template and still renders raw.
