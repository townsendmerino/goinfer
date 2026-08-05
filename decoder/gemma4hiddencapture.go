package decoder

// gemma4HiddenBuf accumulates a COPY of the residual stream after each layer (index 0 =
// post-embedding, from g4traceHidden's layer -1; index i+1 = after layer i) on every
// runLayersGemma4 call, when capture is on. It backs the per-layer LOCALIZATION a resident-vs-CPU
// gate uses (metal/cuda Step 4): diffing the resident hidden after layer N against this attributes
// a whole-forward miss to a specific layer's geometry seam or K=V forward in ONE run, instead of
// staring at a final-logit cosine that only says "something is wrong". Observe-only: with capture
// off, g4traceHidden is nil and the forward is byte-identical.
var gemma4HiddenBuf [][]float32

