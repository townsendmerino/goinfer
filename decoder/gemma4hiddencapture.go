package decoder

// gemma4HiddenBuf accumulates a COPY of the residual stream after each layer (index 0 =
// post-embedding, from g4traceHidden's layer -1; index i+1 = after layer i) on every
// runLayersGemma4 call, when capture is on. It backs the per-layer LOCALIZATION a resident-vs-CPU
// gate uses (metal/cuda Step 4): diffing the resident hidden after layer N against this attributes
// a whole-forward miss to a specific layer's geometry seam or K=V forward in ONE run, instead of
// staring at a final-logit cosine that only says "something is wrong". Observe-only: with capture
// off, g4traceHidden is nil and the forward is byte-identical.
var gemma4HiddenBuf [][]float32

// SetGemma4HiddenCaptureForTest toggles per-layer hidden capture, clearing the buffer when enabled.
// Exported so a resident test in another package (metal/cuda) can drive a CPU gemma4 forward and
// read the per-layer residual stream — it cannot touch the unexported g4traceHidden directly.
// Capture only a SINGLE token's forward (drive one ForwardForTest between on and off): the buffer
// accumulates across calls, so a multi-token pass concatenates every token's per-layer trace.
func SetGemma4HiddenCaptureForTest(on bool) {
	if on {
		gemma4HiddenBuf = nil
		g4traceHidden = func(_ int, h []float32) {
			gemma4HiddenBuf = append(gemma4HiddenBuf, append([]float32(nil), h...))
		}
	} else {
		g4traceHidden = nil
		gemma4HiddenBuf = nil
	}
}

// Gemma4HiddenCaptureForTest returns the captured residual stream per layer, in call order:
// index 0 = post-embedding, index i+1 = the stream after layer i (pre-final-norm, matching what a
// resident runner's trunk-to-N-layers produces).
func Gemma4HiddenCaptureForTest() [][]float32 { return gemma4HiddenBuf }
