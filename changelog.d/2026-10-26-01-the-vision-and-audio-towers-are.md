- **The vision and audio towers are goinfer's own code** (`internal/vision`, `internal/audio`: aikit v1.75.0's, unchanged).
  `multimodal` and `embeddinggemma2`'s Experimental functions that take a tower (`RegisterGemma4Tower`, `Qwen3TowerFeatures`,
  `RegisterWhisperDevice`, `NewAudioAccelerator` and the rest) now take goinfer's tower types instead of aikit's, which a program
  outside goinfer cannot name; `multimodal.LoadSiglipTower` and `multimodal.LoadGlmOcrTower` load the two towers the examples
  use. Every tower's output is bit-identical on the CPU, Metal and WebGPU (`docs/measurements/ax2-2026-10/`).
