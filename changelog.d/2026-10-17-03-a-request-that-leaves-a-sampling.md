- **A request that leaves a sampling field out now gets the model's recommended value, where the checkpoint carries one.**
  Before, every omitted field fell to temperature 1.0 over the whole vocabulary. The values come from a Hugging Face
  directory's `generation_config.json` or a GGUF's `general.sampling.*` keys; a bare `.gguf` never reads a
  `generation_config.json` from its folder. A field the request sends always wins, a greedy request is unchanged, and
  `repetition_penalty` is named on the load banner and not applied (`docs/server.md`, "Sampling defaults"). Default output
  changes for the checkpoints that carry values, among those checked: Qwen2.5 0.5B Instruct, Qwen3 0.6B, Qwen3-VL 2B and
  Qwen2.5-VL 3B, Gemma 3 4B (the Hugging Face directory), Gemma 4 E2B and E4B (GGUF and directory), LFM2.5 2.6B, GLM-OCR and
  Spark 2.5 1.7B. Unchanged, because they carry none: the Qwen2.5 and Qwen2.5-Coder GGUFs, Phi-3 mini, Gemma 2 2B, Gemma 3
  4B's GGUF, DeepSeek-V2-Lite, InternLM2, Ministral 3, LFM2.5-VL and Qwen3.5 0.8B. A client that wants the old default
  sends `temperature: 1, top_p: 1, top_k: 0`. `goinfer-chat --batch` follows the same rule, under its explicit flags.
