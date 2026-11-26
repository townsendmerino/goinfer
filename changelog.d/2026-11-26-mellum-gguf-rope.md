- **Fix: Mellum `.gguf` files now decode correctly.** goinfer un-permuted a Mellum GGUF's q/k projections as if llama.cpp had
  converted it llama-style, but llama.cpp converts Mellum as a NEOX-rope architecture and leaves them in Hugging Face order. Every
  Mellum GGUF therefore gave fluent-looking but wrong output, in every release that read them: JetBrains' own Mellum 2.1 files
  (Q4_K_M, Q8_0) and the community Mellum 2 files alike. The safetensors checkpoints were never affected. Fixing it bumps the
  `.giw` sidecar format to v18, so each cached model of any family transcodes once on its first load after upgrading
  (`<model>.<quant>[.e4h].<target>.v18.giw` beside the old `.v17.giw`, which you can delete to free its disk space).
