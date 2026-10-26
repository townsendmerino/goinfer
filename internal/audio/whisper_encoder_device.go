package audio

// WhisperDevice runs the whole Whisper-style encoder (the two convolutions, the position add, the layer stack, the final LayerNorm) on another device, from the weights the
// encoder exports. It is the one hook a backend plugs into: Whisper's transcriber and Voxtral's tower both call WhisperEncoder.Forward, so setting a device moves both.
type WhisperDevice interface {
	// Forward returns the [1500][DModel] last hidden state for feats [mels][3000], as WhisperEncoder.Forward does; feats is not modified.
	Forward(feats []float32) ([]float32, error)
	Close() error
}

// WhisperLayerWeights is one encoder layer's tensors, row-major as the checkpoint stores them ([out][in] for the projections). k_proj has no bias.
type WhisperLayerWeights struct {
	LN1W, LN1B, LN2W, LN2B     []float32
	QW, QB, KW, VW, VB, OW, OB []float32
	FC1W, FC1B, FC2W, FC2B     []float32
}

// WhisperWeights is a read-only view of the loaded encoder for a device to upload: Conv1W is [DModel][MelBins][3], Conv2W [DModel][DModel][3], Pos [1500][DModel]. The slices
// alias the encoder's own and must not be written; from an F32 checkpoint they are views into its mapping, readable until the encoder's Close.
type WhisperWeights struct {
	Cfg                WhisperEncoderConfig
	Conv1W, Conv1B     []float32
	Conv2W, Conv2B     []float32
	Pos                []float32
	Layers             []WhisperLayerWeights
	FinalLNW, FinalLNB []float32
	LayerNormEps       float32
}

// Weights exports the loaded tensors.
func (e *WhisperEncoder) Weights() WhisperWeights {
	w := WhisperWeights{Cfg: e.Cfg, Conv1W: e.conv1W, Conv1B: e.conv1B, Conv2W: e.conv2W, Conv2B: e.conv2B, Pos: e.pos, FinalLNW: e.lnW, FinalLNB: e.lnB, LayerNormEps: whisperLayerNormEps}
	for i := range e.layers {
		l := &e.layers[i]
		w.Layers = append(w.Layers, WhisperLayerWeights{LN1W: l.ln1W, LN1B: l.ln1B, LN2W: l.ln2W, LN2B: l.ln2B,
			QW: l.qW, QB: l.qB, KW: l.kW, VW: l.vW, VB: l.vB, OW: l.oW, OB: l.oB, FC1W: l.fc1W, FC1B: l.fc1B, FC2W: l.fc2W, FC2B: l.fc2B})
	}
	return w
}

// SetDevice routes Forward through d (nil returns to the CPU). A planted defect (SetDefectForTest) always runs on the CPU: the defects are the CPU computation's. The caller owns d and closes it.
func (e *WhisperEncoder) SetDevice(d WhisperDevice) { e.device = d }

// whisperLayerNormEps is the encoder's LayerNorm epsilon (torch's default, which Whisper's modules use).
const whisperLayerNormEps = 1e-5
