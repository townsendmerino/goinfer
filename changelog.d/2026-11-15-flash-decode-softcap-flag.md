- **`--flash-decode-softcap` (chat and serve, off by default) lets a Gemma 2 model decode on CUDA's flash-decode lane.**
  The lane runs in its capped variant; its outputs differ from the exact split-KV lane Gemma 2 decodes on by default, and
  the registered fidelity gate (Gemma 2 2B, 3,900 keys) passes. A model without an attention softcap, and any other
  backend, ignore it.
