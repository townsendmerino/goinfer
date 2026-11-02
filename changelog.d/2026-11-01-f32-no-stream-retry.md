- **Weight streaming is refused at a quant it cannot page.** Weight paging reads only int8 and int4 weights. At `--quant f32`
  a `.gguf` the fit guard refuses stays refused rather than retrying, and an explicit `-stream-weights` is refused by name, as at
  q4k. A tight or refused load at those quants is told what can help instead (a smaller model, `--quant int4`, a smaller `-ctx`).
