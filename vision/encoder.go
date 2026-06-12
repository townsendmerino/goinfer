package vision

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"

	"github.com/townsendmerino/aikit/linalg"
)

// SigLIP / ViT vision encoder (the Gemma 3 vision tower) as a pure-Go forward —
// the P2 piece of docs/multimodal.md. It maps preprocessed pixel_values to a
// last_hidden_state, the sequence of patch embeddings the projector turns into
// image tokens. The attention/FFN projections run f32 or int8 W8A8 (LoadEncoder's
// quant flag; the patch-embed conv stays f32); parity is cosine vs the HF
// SiglipVisionModel golden (scripts/pin_siglip_vision.py) — 1.0 for f32, ~0.9999
// for int8 — the standard the rest of the f32-SIMD attention path meets.
//
// Structure (all reused from the text side's primitives): Conv2d patch embedding
// (as im2col + matmul), learned position embeddings, N pre-LN transformer blocks
// (BIDIRECTIONAL multi-head attention — no causal mask, this is an image — plus a
// gelu-tanh MLP), and a final post-layernorm.

// EncoderConfig mirrors the SiglipVisionConfig fields the forward needs.
type EncoderConfig struct {
	HiddenSize        int     `json:"hidden_size"`
	IntermediateSize  int     `json:"intermediate_size"`
	NumHiddenLayers   int     `json:"num_hidden_layers"`
	NumAttentionHeads int     `json:"num_attention_heads"`
	NumChannels       int     `json:"num_channels"`
	ImageSize         int     `json:"image_size"`
	PatchSize         int     `json:"patch_size"`
	LayerNormEps      float64 `json:"layer_norm_eps"`
}

type encLayer struct {
	ln1w, ln1b     []float32
	qw, kw, vw, ow qmat      // [hidden,hidden] matmul weights (f32 or int8)
	qb, kb, vb, ob []float32 // biases stay f32
	ln2w, ln2b     []float32
	fc1w, fc2w     qmat // [inter,hidden] / [hidden,inter] matmul weights
	fc1b, fc2b     []float32
}

// Encoder is a loaded SigLIP vision tower.
type Encoder struct {
	Cfg              EncoderConfig
	grid, numPatches int
	patchW           []float32 // [hidden, C*P*P] (Conv2d weight, kept f32 — input embedding)
	patchB           []float32 // [hidden]
	posEmb           []float32 // [numPatches, hidden]
	layers           []encLayer
	postLNw, postLNb []float32
	resident         ResidentEncoder // device-resident GPU forward (EnableResident); nil = CPU path
}

// LoadEncoder reads a SigLIP vision checkpoint (config.json + model.safetensors)
// and returns a ready Encoder. Weights are copied out, so the safetensors file is
// closed before return (no retained mmap).
func LoadEncoder(dir string, quant bool) (*Encoder, error) {
	raw, err := os.ReadFile(filepath.Join(dir, "config.json"))
	if err != nil {
		return nil, fmt.Errorf("vision: read config: %w", err)
	}
	// The tiny pinned tower's config.json IS the SigLIP EncoderConfig (flat); a
	// real HF VL checkpoint nests it under "vision_config". Prefer the nested one.
	var wrap struct {
		EncoderConfig
		VisionConfig *EncoderConfig `json:"vision_config"`
	}
	if err := json.Unmarshal(raw, &wrap); err != nil {
		return nil, fmt.Errorf("vision: parse config: %w", err)
	}
	cfg := wrap.EncoderConfig
	if wrap.VisionConfig != nil {
		cfg = *wrap.VisionConfig
	}
	if cfg.LayerNormEps == 0 {
		cfg.LayerNormEps = 1e-6
	}
	if cfg.NumChannels == 0 {
		cfg.NumChannels = 3 // SigLIP is RGB; real vision_config omits num_channels
	}
	st, err := openWeights(dir)
	if err != nil {
		return nil, fmt.Errorf("vision: open safetensors: %w", err)
	}
	defer st.Close()

	e := &Encoder{Cfg: cfg}
	e.grid = cfg.ImageSize / cfg.PatchSize
	e.numPatches = e.grid * e.grid
	// "" for the tiny stripped tower, "vision_tower.vision_model." inside a real
	// gemma-3-4b-it (where the SigLIP tower lives in the model shards).
	pfx := tensorPrefix(st, "embeddings.patch_embedding.weight", "vision_tower.vision_model.")
	get := func(name string) []float32 {
		if err != nil {
			return nil
		}
		var v []float32
		v, err = st.TensorF32(pfx + name)
		return append([]float32(nil), v...) // copy out so st can close
	}
	hidden, inter := cfg.HiddenSize, cfg.IntermediateSize
	// qm wraps a matmul weight as f32 or int8 (W8A8). Attention/FFN projections
	// quantize under -vision-quant; the patch-embed conv stays f32 (input
	// embedding — quant error there propagates through every layer).
	qm := func(name string, rows, cols int) qmat {
		w := get(name)
		if err != nil {
			return qmat{}
		}
		return newQMat(w, rows, cols, quant)
	}
	e.patchW = get("embeddings.patch_embedding.weight") // [hidden, C*P*P], f32
	e.patchB = get("embeddings.patch_embedding.bias")
	e.posEmb = get("embeddings.position_embedding.weight")
	e.layers = make([]encLayer, cfg.NumHiddenLayers)
	for l := range e.layers {
		p := fmt.Sprintf("encoder.layers.%d.", l)
		lw := &e.layers[l]
		lw.ln1w, lw.ln1b = get(p+"layer_norm1.weight"), get(p+"layer_norm1.bias")
		lw.qw, lw.qb = qm(p+"self_attn.q_proj.weight", hidden, hidden), get(p+"self_attn.q_proj.bias")
		lw.kw, lw.kb = qm(p+"self_attn.k_proj.weight", hidden, hidden), get(p+"self_attn.k_proj.bias")
		lw.vw, lw.vb = qm(p+"self_attn.v_proj.weight", hidden, hidden), get(p+"self_attn.v_proj.bias")
		lw.ow, lw.ob = qm(p+"self_attn.out_proj.weight", hidden, hidden), get(p+"self_attn.out_proj.bias")
		lw.ln2w, lw.ln2b = get(p+"layer_norm2.weight"), get(p+"layer_norm2.bias")
		lw.fc1w, lw.fc1b = qm(p+"mlp.fc1.weight", inter, hidden), get(p+"mlp.fc1.bias")
		lw.fc2w, lw.fc2b = qm(p+"mlp.fc2.weight", hidden, inter), get(p+"mlp.fc2.bias")
	}
	e.postLNw, e.postLNb = get("post_layernorm.weight"), get("post_layernorm.bias")
	if err != nil {
		return nil, fmt.Errorf("vision: load weights: %w", err)
	}
	return e, nil
}

// Forward runs the encoder on pixel_values [NumChannels*ImageSize*ImageSize]
// (a single image, CHW order — the preprocess output) and returns last_hidden_state
// [numPatches * HiddenSize], row-major over patches in (row, col) grid order.
func (e *Encoder) Forward(pixels []float32) ([]float32, error) {
	c := e.Cfg
	want := c.NumChannels * c.ImageSize * c.ImageSize
	if len(pixels) != want {
		return nil, fmt.Errorf("vision: pixels len %d, want %d (%d×%d×%d)", len(pixels), want, c.NumChannels, c.ImageSize, c.ImageSize)
	}
	// Device-resident GPU path (EnableResident): im2col on the host, the whole
	// transformer on the device. Same numerics (W8A8), ~9× faster on a real model.
	if e.resident != nil {
		patches, err := e.GridPatches(pixels)
		if err != nil {
			return nil, err
		}
		return e.resident.ForwardPatches(patches)
	}
	hidden, np, P, W := c.HiddenSize, e.numPatches, c.PatchSize, c.ImageSize
	cpp := c.NumChannels * P * P

	// 1. im2col patch extraction in the Conv2d weight's (c,kh,kw) order, patches in
	// (gh,gw) row-major — matching HF's embeddings.flatten(2).transpose.
	patches := make([]float32, np*cpp)
	for gh := 0; gh < e.grid; gh++ {
		for gw := 0; gw < e.grid; gw++ {
			dst := patches[(gh*e.grid+gw)*cpp:]
			for ch := 0; ch < c.NumChannels; ch++ {
				for kh := range P {
					for kw := range P {
						dst[(ch*P+kh)*P+kw] = pixels[ch*W*W+(gh*P+kh)*W+(gw*P+kw)]
					}
				}
			}
		}
	}
	// patch embed: h[np,hidden] = patches[np,cpp] · patchW[hidden,cpp]ᵀ + bias, + posEmb
	h := make([]float32, np*hidden)
	linalg.MatmulBT(patches, e.patchW, h, np, cpp, hidden)
	addBias(h, e.patchB, np, hidden)
	for i := range h {
		h[i] += e.posEmb[i]
	}

	for l := range e.layers {
		lw := &e.layers[l]
		// attention block (pre-LN, residual)
		n1 := layerNorm(h, lw.ln1w, lw.ln1b, np, hidden, c.LayerNormEps)
		att := e.attention(n1, lw, np)
		o := make([]float32, np*hidden)
		lw.ow.matmul(att, o, np)
		addBias(o, lw.ob, np, hidden)
		for i := range h {
			h[i] += o[i]
		}
		// MLP block (pre-LN, residual): fc2(geluTanh(fc1(x)))
		n2 := layerNorm(h, lw.ln2w, lw.ln2b, np, hidden, c.LayerNormEps)
		inter := c.IntermediateSize
		mid := make([]float32, np*inter)
		lw.fc1w.matmul(n2, mid, np)
		addBias(mid, lw.fc1b, np, inter)
		geluTanh(mid)
		mlp := make([]float32, np*hidden)
		lw.fc2w.matmul(mid, mlp, np)
		addBias(mlp, lw.fc2b, np, hidden)
		for i := range h {
			h[i] += mlp[i]
		}
	}
	return layerNorm(h, e.postLNw, e.postLNb, np, hidden, c.LayerNormEps), nil
}

// attention runs bidirectional multi-head self-attention (no causal mask) over
// the np patches. Per head, QKᵀ and scores·V run on the f32 SIMD A·Bᵀ kernel
// (MatmulBT) — f32 is ample here (HF runs SigLIP in bf16/f16, far less precise),
// and the f64-accumulate the text path uses for the discrete MoE router is just
// dead weight on a vision tower, where it dominated the CPU prefill time. At
// SigLIP sizes (≈4096 patches) this is the difference between minutes and seconds
// per image vs the old scalar triple-loop.
func (e *Encoder) attention(x []float32, lw *encLayer, np int) []float32 {
	hidden, nH := e.Cfg.HiddenSize, e.Cfg.NumAttentionHeads
	hd := hidden / nH
	scale := float32(1.0 / math.Sqrt(float64(hd)))
	q := make([]float32, np*hidden)
	k := make([]float32, np*hidden)
	v := make([]float32, np*hidden)
	lw.qw.matmul(x, q, np)
	addBias(q, lw.qb, np, hidden)
	lw.kw.matmul(x, k, np)
	addBias(k, lw.kb, np, hidden)
	lw.vw.matmul(x, v, np)
	addBias(v, lw.vb, np, hidden)

	out := make([]float32, np*hidden)
	// Per-head scratch: contiguous q/k [np,hd], vᵀ [hd,np], scores [np,np], out [np,hd].
	qh := make([]float32, np*hd)
	kh := make([]float32, np*hd)
	vt := make([]float32, hd*np)
	scores := make([]float32, np*np)
	oh := make([]float32, np*hd)
	for head := range nH {
		off := head * hd
		for i := range np {
			copy(qh[i*hd:(i+1)*hd], q[i*hidden+off:i*hidden+off+hd])
			copy(kh[i*hd:(i+1)*hd], k[i*hidden+off:i*hidden+off+hd])
			vrow := v[i*hidden+off : i*hidden+off+hd]
			for d := range hd {
				vt[d*np+i] = vrow[d] // vᵀ so scores·V = MatmulBT(scores, vᵀ)
			}
		}
		// scores[np,np] = qh · khᵀ, scaled, row-softmax.
		linalg.MatmulBT(qh, kh, scores, np, hd, np)
		for i := range np {
			row := scores[i*np : (i+1)*np]
			for j := range row {
				row[j] *= scale
			}
			softmaxRow(row)
		}
		// out_head[np,hd] = scores[np,np] · v_head[np,hd] = MatmulBT(scores, vᵀ).
		linalg.MatmulBT(scores, vt, oh, np, np, hd)
		for i := range np {
			copy(out[i*hidden+off:i*hidden+off+hd], oh[i*hd:(i+1)*hd])
		}
	}
	return out
}

// --- small f32 helpers (LayerNorm is standard — mean/var — not RMS) ---

func layerNorm(x, w, b []float32, rows, dim int, eps float64) []float32 {
	out := make([]float32, rows*dim)
	for r := range rows {
		xr := x[r*dim : r*dim+dim]
		var mean float64
		for _, val := range xr {
			mean += float64(val)
		}
		mean /= float64(dim)
		var variance float64
		for _, val := range xr {
			d := float64(val) - mean
			variance += d * d
		}
		variance /= float64(dim)
		inv := 1.0 / math.Sqrt(variance+eps)
		dst := out[r*dim : r*dim+dim]
		for d := range dim {
			dst[d] = float32((float64(xr[d])-mean)*inv)*w[d] + b[d]
		}
	}
	return out
}

func geluTanh(x []float32) {
	const c = 0.7978845608028654 // sqrt(2/π)
	for i, val := range x {
		v := float64(val)
		x[i] = float32(0.5 * v * (1.0 + math.Tanh(c*(v+0.044715*v*v*v))))
	}
}

func addBias(x, bias []float32, rows, dim int) {
	for r := range rows {
		dst := x[r*dim : r*dim+dim]
		for d := range dim {
			dst[d] += bias[d]
		}
	}
}

func softmaxRow(s []float32) {
	maxv := s[0]
	for _, v := range s {
		if v > maxv {
			maxv = v
		}
	}
	var sum float64
	for i, v := range s {
		e := math.Exp(float64(v) - float64(maxv))
		s[i] = float32(e)
		sum += e
	}
	inv := 1.0 / sum
	for i := range s {
		s[i] = float32(float64(s[i]) * inv)
	}
}
