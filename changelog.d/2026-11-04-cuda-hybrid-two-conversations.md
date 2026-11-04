- **On CUDA a hybrid model can hold two conversations, and CUDA and Metal save end-of-prompt recurrent checkpoints.** On CUDA, each KV
  slot now allocates its own recurrent state (`DeltaNet` conv and recurrent state, and `Mamba-2` conv and SSM state), so a hybrid model
  with `--kv-sessions 2` or more holds two concurrent conversations warm instead of evicting the other client's state on every turn (on
  Qwen3.5-9B with two alternating clients, follow-up median TTFT drops from `6.93 s` to `0.05 s`, for `+562 MiB` on the card). On both
  CUDA and Metal, a hybrid model also checkpoints its recurrent state at the end of the prompt (and at the generation tail on CUDA), so
  a follow-up turn that strips or edits the assistant's reasoning or reply restores from the checkpoint instead of re-prefilling from
  token 0.
