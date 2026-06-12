package vision

import "fmt"

// GPU export: the resident GPU encoder lives in the goinfer/gpu module (cgo,
// -tags gpu). It reads the loaded tower through these exported types so vision
// stays pure-Go. The matmul weights must be int8 (the GPU path uses the tiled
// W8A8 kernel) — load the tower with quant=true.

// GPUMat is one matmul weight in W8A8 form: int8 rows [Rows,Cols] + per-row scales.
type GPUMat struct {
	Q          []int8
	Scales     []float32
	Rows, Cols int
}

// GPULayer is one SigLIP transformer layer's weights for the GPU encoder.
type GPULayer struct {
	LN1w, LN1b     []float32
	Qw, Kw, Vw, Ow GPUMat
	Qb, Kb, Vb, Ob []float32
	LN2w, LN2b     []float32
	FC1w, FC2w     GPUMat
	FC1b, FC2b     []float32
}

// GPUWeights is the whole tower, flattened for the GPU encoder.
type GPUWeights struct {
	Hidden, Inter, NumLayers, NumHeads, HeadDim int
	NumPatches, Grid, PatchSize, NumChannels    int
	Eps                                         float32
	PatchW, PatchB, PosEmb                      []float32 // patch-embed conv (f32) + positional
	PostLNw, PostLNb                            []float32
	Layers                                      []GPULayer
}

// GPUWeights exports the tower for the GPU resident encoder. Requires int8
// matmul weights (LoadEncoder quant=true) — errors otherwise.
func (e *Encoder) GPUWeights() (GPUWeights, error) {
	c := e.Cfg
	gm := func(m qmat) (GPUMat, error) {
		if m.q == nil {
			return GPUMat{}, fmt.Errorf("vision: GPU encoder needs int8 weights — load with quant=true")
		}
		return GPUMat{Q: m.q, Scales: m.scales, Rows: m.rows, Cols: m.cols}, nil
	}
	w := GPUWeights{
		Hidden: c.HiddenSize, Inter: c.IntermediateSize, NumLayers: c.NumHiddenLayers,
		NumHeads: c.NumAttentionHeads, HeadDim: c.HiddenSize / c.NumAttentionHeads,
		NumPatches: e.numPatches, Grid: e.grid, PatchSize: c.PatchSize, NumChannels: c.NumChannels,
		Eps:    float32(c.LayerNormEps),
		PatchW: e.patchW, PatchB: e.patchB, PosEmb: e.posEmb,
		PostLNw: e.postLNw, PostLNb: e.postLNb,
		Layers: make([]GPULayer, len(e.layers)),
	}
	for i := range e.layers {
		l := &e.layers[i]
		var err error
		gl := GPULayer{
			LN1w: l.ln1w, LN1b: l.ln1b, LN2w: l.ln2w, LN2b: l.ln2b,
			Qb: l.qb, Kb: l.kb, Vb: l.vb, Ob: l.ob, FC1b: l.fc1b, FC2b: l.fc2b,
		}
		for _, p := range []struct {
			dst *GPUMat
			src qmat
		}{{&gl.Qw, l.qw}, {&gl.Kw, l.kw}, {&gl.Vw, l.vw}, {&gl.Ow, l.ow}, {&gl.FC1w, l.fc1w}, {&gl.FC2w, l.fc2w}} {
			if *p.dst, err = gm(p.src); err != nil {
				return GPUWeights{}, err
			}
		}
		w.Layers[i] = gl
	}
	return w, nil
}

// GridPatches runs the CPU im2col patch extraction (the pure-Go preprocess step
// that stays on the host) and returns patches [NumPatches * (C*P*P)] for the GPU
// patch-embed matmul.
func (e *Encoder) GridPatches(pixels []float32) ([]float32, error) {
	c := e.Cfg
	if want := c.NumChannels * c.ImageSize * c.ImageSize; len(pixels) != want {
		return nil, fmt.Errorf("vision: pixels len %d, want %d", len(pixels), want)
	}
	P, W, cpp := c.PatchSize, c.ImageSize, c.NumChannels*c.PatchSize*c.PatchSize
	patches := make([]float32, e.numPatches*cpp)
	for gh := 0; gh < e.grid; gh++ {
		for gw := 0; gw < e.grid; gw++ {
			dst := patches[(gh*e.grid+gw)*cpp:]
			for ch := 0; ch < c.NumChannels; ch++ {
				for kh := 0; kh < P; kh++ {
					for kw := 0; kw < P; kw++ {
						dst[(ch*P+kh)*P+kw] = pixels[ch*W*W+(gh*P+kh)*W+(gw*P+kw)]
					}
				}
			}
		}
	}
	return patches, nil
}
