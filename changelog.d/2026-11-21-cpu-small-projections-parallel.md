- **Faster: Qwen2.5 0.5B's attention projections use every core on the CPU.** The int4 parallel threshold now sits below
  their size, so the q, k, v and o projections fan out: about 2.6% per token, bit-identical, 10 of 10 timed pairs faster
  (#1121).
