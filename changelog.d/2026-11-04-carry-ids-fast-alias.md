- **Exact token carry (`--carry-ids`, `--ids-key-file`), token-ID logging (`--log-token-ids`), and the fast model alias (`--fast-alias`).**
  `--carry-ids` (on by default) carries signed generated token IDs in `/v1/responses` reasoning items (`encrypted_content`) and
  `/v1/messages` thinking blocks (`signature`) so an agent turn that echoes prior assistant output reuses the exact generated token IDs
  instead of re-tokenizing text; `--ids-key-file` persists the signing key across restarts, and `--log-token-ids` appends per-generation
  JSON lines with prompt length, SHA-256, reuse/slot positions, and generated token IDs for measurement runs. `--fast-alias` (on by
  default) registers `<model>-fast` on the same weights and KV slot with thinking disabled (`chat_template_kwargs.enable_thinking: false`)
  for lightweight agent side requests such as title generation, and `goinfer-serve connect` points Claude Code and opencode's small-model
  settings at it.
