- **Faster q4k prefill on Metal.** A model loaded at `--quant q4k` (Phi-3) prefills eight rows a step on the batched
  step's kernels instead of one token at a time, with bit-identical results: 1.99x on Phi-3 mini at 200 tokens.
