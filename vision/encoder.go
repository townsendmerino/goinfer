package vision

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"

	"github.com/townsendmerino/aikit/embed"
	"github.com/townsendmerino/aikit/linalg"
)

// SigLIP / ViT vision encoder (the Gemma 3 vision tower) as a pure-Go forward —
// the P2 piece of docs/multimodal.md. It maps preprocessed pixel_values to a
// last_hidden_state, the sequence of patch embeddings the projector turns into
// image tokens. f32 throughout (the tower is ~0.4B; int8 is a follow-on);
// parity is cosine vs the HF SiglipVisionModel golden (scripts/pin_siglip_vision.py),
// the same standard the rest of the f32-SIMD attention path meets.
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
	ln1w, ln1b             []float32
	qw, qb, kw, kb, vw, vb []float32 // each [hidden,hidden] / [hidden]
	ow, ob                 []float32
	ln2w, ln2b             []float32
	fc1w, fc1b             []float32 // [inter,hidden] / [inter]
	fc2w, fc2b             []float32 // [hidden,inter] / [hidden]
}

// Encoder is a loaded SigLIP vision tower.
type Encoder struct {
	Cfg              EncoderConfig
	grid, numPatches int
	patchW           []float32 // [hidden, C*P*P] (Conv2d weight flattened per out channel)
	patchB           []float32 // [hidden]
	posEmb           []float32 // [numPatches, hidden]
	layers           []encLayer
	postLNw, postLNb []float32
}

// LoadEncoder reads a SigLIP vision checkpoint (config.json + model.safetensors)
// and returns a ready Encoder. Weights are copied out, so the safetensors file is
// closed before return (no retained mmap).
func LoadEncoder(dir string) (*Encoder, error) {
	raw, err := os.ReadFile(filepath.Join(dir, "config.json"))
	if err != nil {
		return nil, fmt.Errorf("vision: read config: %w", err)
	}
	var cfg EncoderConfig
	if err := json.Unmarshal(raw, &cfg); err != nil {
		return nil, fmt.Errorf("vision: parse config: %w", err)
	}
	if cfg.LayerNormEps == 0 {
		cfg.LayerNormEps = 1e-6
	}
	st, err := embed.OpenSafetensorsMmap(filepath.Join(dir, "model.safetensors"))
	if err != nil {
		return nil, fmt.Errorf("vision: open safetensors: %w", err)
	}
	defer st.Close()

	e := &Encoder{Cfg: cfg}
	e.grid = cfg.ImageSize / cfg.PatchSize
	e.numPatches = e.grid * e.grid
	get := func(name string) []float32 {
		if err != nil {
			return nil
		}
		var v []float32
		v, err = tensorF32(st, name)
		return append([]float32(nil), v...) // copy out so st can close
	}
	e.patchW = get("embeddings.patch_embedding.weight") // [hidden,C,P,P] → flat [hidden, C*P*P]
	e.patchB = get("embeddings.patch_embedding.bias")
	e.posEmb = get("embeddings.position_embedding.weight")
	e.layers = make([]encLayer, cfg.NumHiddenLayers)
	for l := range e.layers {
		p := fmt.Sprintf("encoder.layers.%d.", l)
		lw := &e.layers[l]
		lw.ln1w, lw.ln1b = get(p+"layer_norm1.weight"), get(p+"layer_norm1.bias")
		lw.qw, lw.qb = get(p+"self_attn.q_proj.weight"), get(p+"self_attn.q_proj.bias")
		lw.kw, lw.kb = get(p+"self_attn.k_proj.weight"), get(p+"self_attn.k_proj.bias")
		lw.vw, lw.vb = get(p+"self_attn.v_proj.weight"), get(p+"self_attn.v_proj.bias")
		lw.ow, lw.ob = get(p+"self_attn.out_proj.weight"), get(p+"self_attn.out_proj.bias")
		lw.ln2w, lw.ln2b = get(p+"layer_norm2.weight"), get(p+"layer_norm2.bias")
		lw.fc1w, lw.fc1b = get(p+"mlp.fc1.weight"), get(p+"mlp.fc1.bias")
		lw.fc2w, lw.fc2b = get(p+"mlp.fc2.weight"), get(p+"mlp.fc2.bias")
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
		linalg.MatmulBT(att, lw.ow, o, np, hidden, hidden)
		addBias(o, lw.ob, np, hidden)
		for i := range h {
			h[i] += o[i]
		}
		// MLP block (pre-LN, residual): fc2(geluTanh(fc1(x)))
		n2 := layerNorm(h, lw.ln2w, lw.ln2b, np, hidden, c.LayerNormEps)
		inter := c.IntermediateSize
		mid := make([]float32, np*inter)
		linalg.MatmulBT(n2, lw.fc1w, mid, np, hidden, inter)
		addBias(mid, lw.fc1b, np, inter)
		geluTanh(mid)
		mlp := make([]float32, np*hidden)
		linalg.MatmulBT(mid, lw.fc2w, mlp, np, inter, hidden)
		addBias(mlp, lw.fc2b, np, hidden)
		for i := range h {
			h[i] += mlp[i]
		}
	}
	return layerNorm(h, e.postLNw, e.postLNb, np, hidden, c.LayerNormEps), nil
}

// attention runs bidirectional multi-head self-attention (no causal mask) over the
// np patches. Scalar per-(head,query) — correctness-first for v1; at real SigLIP
// sizes (≈4096 patches) the QKᵀ / scores·V terms should move onto linalg.MatmulBT
// like the text path's attendBatchedHeads (a noted follow-on).
func (e *Encoder) attention(x []float32, lw *encLayer, np int) []float32 {
	hidden, nH := e.Cfg.HiddenSize, e.Cfg.NumAttentionHeads
	hd := hidden / nH
	scale := 1.0 / math.Sqrt(float64(hd))
	q := make([]float32, np*hidden)
	k := make([]float32, np*hidden)
	v := make([]float32, np*hidden)
	linalg.MatmulBT(x, lw.qw, q, np, hidden, hidden)
	addBias(q, lw.qb, np, hidden)
	linalg.MatmulBT(x, lw.kw, k, np, hidden, hidden)
	addBias(k, lw.kb, np, hidden)
	linalg.MatmulBT(x, lw.vw, v, np, hidden, hidden)
	addBias(v, lw.vb, np, hidden)

	out := make([]float32, np*hidden)
	scores := make([]float32, np)
	for head := range nH {
		off := head * hd
		for i := range np {
			qi := q[i*hidden+off : i*hidden+off+hd]
			for j := range np {
				kj := k[j*hidden+off : j*hidden+off+hd]
				var dot float64
				for d := range hd {
					dot += float64(qi[d]) * float64(kj[d])
				}
				scores[j] = float32(dot * scale)
			}
			softmaxRow(scores)
			oi := out[i*hidden+off : i*hidden+off+hd]
			for j := range np {
				w := scores[j]
				vj := v[j*hidden+off : j*hidden+off+hd]
				for d := range hd {
					oi[d] += w * vj[d]
				}
			}
		}
	}
	return out
}

// --- small f32 helpers (LayerNorm is standard — mean/var — not RMS) ---

func tensorF32(st *embed.SafetensorsFile, name string) ([]float32, error) {
	t, err := st.Tensor(name)
	if err != nil {
		return nil, fmt.Errorf("vision: tensor %q: %w", name, err)
	}
	switch t.DType {
	case "F32":
		return t.Float32s()
	case "BF16":
		return t.BFloat16sToF32()
	case "F16":
		return t.Float16sToF32()
	}
	return nil, fmt.Errorf("vision: tensor %q dtype %q unsupported (want F32/BF16/F16)", name, t.DType)
}

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
