- **Tokenizing a prompt takes about half the time and allocates far less, with identical token ids.** On an M1 Pro, in process:
  1.96–1.98× on byte-level tokenizers and 1.79× on Gemma's SentencePiece path, with 25–44× fewer allocations.
