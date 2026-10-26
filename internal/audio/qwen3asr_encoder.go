package audio

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"

	"github.com/townsendmerino/aikit/embed"
	"github.com/townsendmerino/aikit/linalg"
)

// Qwen3-ASR's audio encoder ("AuT") and projector, float32 on the CPU.
//
// Parity reference: transformers 5.15.0's Qwen3ASREncoder and Qwen3ASRMultiModalProjector, read in goinfer's docs/tasks/task-multimodal-support-2026-10.md (G-S14b). The mel features arrive as
// [128][T] with T a multiple of 100 (QwenASRFeatures). Each 100-frame chunk goes through three Conv2d (k3 s2 p1, exact-erf GELU after each), a bias-free Linear over the flattened
// (channel, frequency) axes and a sinusoidal position that restarts in every chunk; the real post-CNN positions of all chunks are packed into one sequence; 18 pre-LayerNorm layers attend within
// windows of (max post-CNN length of a chunk) * (n_window_infer / 100) positions; then ln_post and the projector (Linear, GELU, Linear).
//
// A limit inherited from the reference: its length function hard-codes 13 post-CNN steps per full chunk, which is true only for n_window = 50 (100-frame chunks). This port computes the true
// lengths, which are equal for 50 and are what the reference would need for any other value; only 50 is validated.

// QwenASRConfig is the audio_config fields the encoder reads.
type QwenASRConfig struct {
	DModel       int `json:"d_model"`
	Layers       int `json:"encoder_layers"`
	Heads        int `json:"encoder_attention_heads"`
	FFN          int `json:"encoder_ffn_dim"`
	Downsample   int `json:"downsample_hidden_size"`
	MelBins      int `json:"num_mel_bins"`
	NWindow      int `json:"n_window"`
	NWindowInfer int `json:"n_window_infer"`
	OutputDim    int `json:"output_dim"`
}

func (c QwenASRConfig) validate() error {
	switch {
	case c.DModel <= 0 || c.Layers <= 0 || c.Heads <= 0 || c.DModel%c.Heads != 0 || c.FFN <= 0 || c.Downsample <= 0 || c.OutputDim <= 0:
		return fmt.Errorf("audio: qwen3-asr d_model %d, %d layers, %d heads, ffn %d, downsample %d, output %d", c.DModel, c.Layers, c.Heads, c.FFN, c.Downsample, c.OutputDim)
	case c.MelBins != QwenASRMels:
		return fmt.Errorf("audio: qwen3-asr takes %d mel bins, config has %d", QwenASRMels, c.MelBins)
	case c.NWindow != 50:
		return fmt.Errorf("audio: n_window %d (only 50, the checkpoint's value, is validated: the reference hard-codes 13 steps per chunk)", c.NWindow)
	case c.NWindowInfer < 2*c.NWindow || c.NWindowInfer%(2*c.NWindow) != 0:
		return fmt.Errorf("audio: n_window_infer %d is not a multiple of the %d-frame chunk", c.NWindowInfer, 2*c.NWindow)
	}
	return nil
}

type qwenASRLayer struct {
	ln1W, ln1B, ln2W, ln2B         []float32
	qW, qB, kW, kB, vW, vB, oW, oB []float32
	fc1W, fc1B, fc2W, fc2B         []float32
}

// QwenASREncoder is the encoder with its projector.
type QwenASREncoder struct {
	Cfg                            QwenASRConfig
	conv                           [3]struct{ w, b []float32 }
	convOutW                       []float32
	layers                         []qwenASRLayer
	lnPostW, lnPostB               []float32
	proj1W, proj1B, proj2W, proj2B []float32
	hf                             int // frequency rows after the three convs
	defect                         int
	st                             *embed.SafetensorsFile // held until Close: F32 weights are views into its mapping (see WhisperEncoder.st)
	closed                         bool
}

// Planted defects for G-S14b2/3's red runs. Zero is the shipped computation.
const (
	qwenASREncDefectNone = iota
	qwenASREncDefectNoPos
	qwenASREncDefectGlobalPos
	qwenASREncDefectFullAttention
	qwenASREncDefectTanhGELU
	qwenASREncDefectKeepPadding
	qwenASREncDefectNoProjGELU
)

// QwenASREncDefectCountForTest is the number of planted defects SetDefectForTest accepts (1..N).
const QwenASREncDefectCountForTest = qwenASREncDefectNoProjGELU

// SetDefectForTest plants one defect (0 clears it). Never call it in production.
func (e *QwenASREncoder) SetDefectForTest(d int) { e.defect = d }

// LoadQwenASREncoder reads the encoder and projector from a Qwen3-ASR checkpoint directory (config.json with thinker_config.audio_config, or a bare audio_config; model.safetensors with the
// tensors under "thinker.audio_tower.", "audio_tower." or no prefix, the projector as proj1/proj2 beside them).
func LoadQwenASREncoder(dir string) (*QwenASREncoder, error) {
	raw, err := os.ReadFile(filepath.Join(dir, "config.json"))
	if err != nil {
		return nil, err
	}
	var top struct {
		Thinker *struct {
			Audio json.RawMessage `json:"audio_config"`
		} `json:"thinker_config"`
		Audio json.RawMessage `json:"audio_config"`
	}
	if err := json.Unmarshal(raw, &top); err != nil {
		return nil, fmt.Errorf("audio: %s/config.json: %w", dir, err)
	}
	ac := top.Audio
	if top.Thinker != nil && len(top.Thinker.Audio) > 0 {
		ac = top.Thinker.Audio
	}
	if len(ac) == 0 {
		return nil, fmt.Errorf("audio: %s/config.json has no audio_config", dir)
	}
	e := &QwenASREncoder{}
	if err := json.Unmarshal(ac, &e.Cfg); err != nil {
		return nil, err
	}
	if err := e.Cfg.validate(); err != nil {
		return nil, err
	}
	st, err := openSafetensorsDir(dir)
	if err != nil {
		return nil, err
	}
	prefix := ""
	for _, p := range []string{"thinker.audio_tower.", "audio_tower.", ""} {
		if _, err := st.Tensor(p + "conv2d1.weight"); err == nil {
			prefix = p
			break
		}
	}
	c := e.Cfg
	get := func(name string, want ...int) []float32 {
		if err != nil {
			return nil
		}
		var t []float32
		t, err = st.TensorF32(prefix+name, want...)
		return t
	}
	D, H := c.DModel, c.Downsample
	e.conv[0].w, e.conv[0].b = get("conv2d1.weight", H, 1, 3, 3), get("conv2d1.bias", H)
	e.conv[1].w, e.conv[1].b = get("conv2d2.weight", H, H, 3, 3), get("conv2d2.bias", H)
	e.conv[2].w, e.conv[2].b = get("conv2d3.weight", H, H, 3, 3), get("conv2d3.bias", H)
	e.hf = ((((c.MelBins+1)/2+1)/2 + 1) / 2)
	e.convOutW = get("conv_out.weight", D, H*e.hf)
	for i := range c.Layers {
		p := fmt.Sprintf("layers.%d.", i)
		e.layers = append(e.layers, qwenASRLayer{
			ln1W: get(p+"self_attn_layer_norm.weight", D), ln1B: get(p+"self_attn_layer_norm.bias", D),
			qW: get(p+"self_attn.q_proj.weight", D, D), qB: get(p+"self_attn.q_proj.bias", D),
			kW: get(p+"self_attn.k_proj.weight", D, D), kB: get(p+"self_attn.k_proj.bias", D),
			vW: get(p+"self_attn.v_proj.weight", D, D), vB: get(p+"self_attn.v_proj.bias", D),
			oW: get(p+"self_attn.out_proj.weight", D, D), oB: get(p+"self_attn.out_proj.bias", D),
			ln2W: get(p+"final_layer_norm.weight", D), ln2B: get(p+"final_layer_norm.bias", D),
			fc1W: get(p+"fc1.weight", c.FFN, D), fc1B: get(p+"fc1.bias", c.FFN),
			fc2W: get(p+"fc2.weight", D, c.FFN), fc2B: get(p+"fc2.bias", D),
		})
	}
	e.lnPostW, e.lnPostB = get("ln_post.weight", D), get("ln_post.bias", D)
	e.proj1W, e.proj1B = get("proj1.weight", D, D), get("proj1.bias", D)
	e.proj2W, e.proj2B = get("proj2.weight", c.OutputDim, D), get("proj2.bias", c.OutputDim)
	if err != nil {
		st.Close()
		return nil, fmt.Errorf("audio: loading %s: %w", dir, err)
	}
	e.st = st
	return e, nil
}

// Close releases the checkpoint mapping the weights may alias; Forward and Tower return an error afterwards. Calling Close again does nothing.
func (e *QwenASREncoder) Close() error {
	if e.closed {
		return nil
	}
	st := e.st
	*e = QwenASREncoder{Cfg: e.Cfg, closed: true} // drop every view into the mapping before unmapping it
	if st == nil {
		return nil
	}
	return st.Close()
}

// QwenASRTokenCount is the number of audio tokens for `valid` real mel frames (the processor's _get_audio_token_length): full 100-frame chunks give 13 each, the last partial chunk
// three stride-2 steps of its own length.
func QwenASRTokenCount(valid int) int {
	return valid/QwenASRChunkFrames*13 + postCNN(valid%QwenASRChunkFrames)
}

// postCNN is the length after three (k=3, s=2, p=1) convolutions; zero stays zero.
func postCNN(l int) int {
	for range 3 {
		if l > 0 {
			l = (l-1)/2 + 1
		}
	}
	return l
}

// Forward runs the encoder and projector over feats [128][T] (T a multiple of 100) with `valid` real frames, and returns the audio embeddings [n][OutputDim] and n.
func (e *QwenASREncoder) Forward(feats []float32, T, valid int) ([]float32, int, error) {
	h, n, err := e.Tower(feats, T, valid)
	if err != nil {
		return nil, 0, err
	}
	return e.project(h, n), n, nil
}

// Tower is the encoder without the projector (the reference's audio_tower output): [n][DModel].
func (e *QwenASREncoder) Tower(feats []float32, T, valid int) ([]float32, int, error) {
	if e.closed {
		return nil, 0, errEncoderClosed
	}
	c := e.Cfg
	chunk := 2 * c.NWindow
	switch {
	case T <= 0 || T%chunk != 0:
		return nil, 0, fmt.Errorf("audio: %d frames is not a positive multiple of %d", T, chunk)
	case len(feats) != c.MelBins*T:
		return nil, 0, fmt.Errorf("audio: %d feature values for %d mels x %d frames", len(feats), c.MelBins, T)
	case valid <= 0 || valid > T:
		return nil, 0, fmt.Errorf("audio: %d valid frames of %d", valid, T)
	}
	D := c.DModel
	nChunks := T / chunk
	steps := postCNN(chunk) // 13
	var packed []float32
	n := 0
	maxLen := 0
	for ch := range nChunks {
		cl := min(max(valid-ch*chunk, 0), chunk)
		keep := postCNN(cl)
		maxLen = max(maxLen, keep)
		if keep == 0 && e.defect != qwenASREncDefectKeepPadding {
			continue
		}
		// This chunk's [128][chunk] image, HWC with one channel.
		x := make([]float32, c.MelBins*chunk)
		for m := range c.MelBins {
			copy(x[m*chunk:(m+1)*chunk], feats[m*T+ch*chunk:m*T+(ch+1)*chunk])
		}
		cur, hh, ww := x, c.MelBins, chunk
		cin := 1
		for i := range 3 {
			cur, hh, ww = e.conv3x3s2(cur, cin, hh, ww, i)
			cin = c.Downsample
		}
		// cur is [hf][steps][ch]; the reference flattens (channel, frequency) per time step: index ch*hf + f.
		rows := make([]float32, steps*c.Downsample*e.hf)
		for t := range ww {
			for f := range hh {
				for k := range c.Downsample {
					rows[t*c.Downsample*e.hf+k*e.hf+f] = cur[(f*ww+t)*c.Downsample+k]
				}
			}
		}
		emb := make([]float32, steps*D)
		linalg.MatmulBT(rows, e.convOutW, emb, steps, c.Downsample*e.hf, D)
		for t := range steps {
			pos := t
			if e.defect == qwenASREncDefectGlobalPos {
				pos = ch*steps + t
			}
			if e.defect != qwenASREncDefectNoPos {
				pe := sinusoid(pos, D)
				for j := range D {
					emb[t*D+j] += pe[j]
				}
			}
			if t < keep || e.defect == qwenASREncDefectKeepPadding {
				packed = append(packed, emb[t*D:(t+1)*D]...)
				n++
			}
		}
	}
	if n == 0 {
		return nil, 0, fmt.Errorf("audio: no audio positions")
	}
	// Attention windows over the packed sequence.
	window := maxLen * (c.NWindowInfer / chunk)
	if e.defect == qwenASREncDefectFullAttention || window <= 0 {
		window = n
	}
	h := packed
	for i := range e.layers {
		e.layer(&e.layers[i], h, n, window)
	}
	layerNorm(h, e.lnPostW, e.lnPostB, n, D)
	return h, n, nil
}

func (e *QwenASREncoder) gelu(v float32) float32 {
	if e.defect == qwenASREncDefectTanhGELU {
		x := float64(v)
		return float32(0.5 * x * (1 + math.Tanh(math.Sqrt(2/math.Pi)*(x+0.044715*x*x*x))))
	}
	return float32(0.5 * float64(v) * (1 + math.Erf(float64(v)/math.Sqrt2)))
}

// conv3x3s2 is Conv2d(k=3, s=2, p=1) with bias and an exact GELU over x [H][W][cin] (HWC) with the conv i's weights [cout, cin, 3, 3]: output [Ho][Wo][cout].
func (e *QwenASREncoder) conv3x3s2(x []float32, cin, H, W, i int) ([]float32, int, int) {
	cout := e.Cfg.Downsample
	Ho, Wo := (H+1)/2, (W+1)/2
	K := cin * 9
	col := make([]float32, Ho*Wo*K)
	for hh := range Ho {
		for ww := range Wo {
			dst := col[(hh*Wo+ww)*K : (hh*Wo+ww+1)*K]
			for kh := range 3 {
				hi := 2*hh - 1 + kh
				if hi < 0 || hi >= H {
					continue
				}
				for kw := range 3 {
					wi := 2*ww - 1 + kw
					if wi < 0 || wi >= W {
						continue
					}
					src := x[(hi*W+wi)*cin : (hi*W+wi+1)*cin]
					for ci, v := range src {
						dst[ci*9+kh*3+kw] = v
					}
				}
			}
		}
	}
	out := make([]float32, Ho*Wo*cout)
	linalg.MatmulBT(col, e.conv[i].w, out, Ho*Wo, K, cout)
	b := e.conv[i].b
	for r := range Ho * Wo {
		row := out[r*cout : (r+1)*cout]
		for j := range row {
			row[j] = e.gelu(row[j] + b[j])
		}
	}
	return out, Ho, Wo
}

// sinusoid is row `pos` of SinusoidsPositionEmbedding: sin of pos * inv_timescales in the first half, cos in the second, with the table built in float32 as torch does.
func sinusoid(pos, channels int) []float32 {
	half := channels / 2
	inc := float32(math.Log(10000) / float64(half-1))
	out := make([]float32, channels)
	for i := range half {
		inv := float32(math.Exp(float64(-inc * float32(i))))
		st := float64(float32(pos) * inv)
		out[i] = float32(math.Sin(st))
		out[half+i] = float32(math.Cos(st))
	}
	return out
}

func layerNorm(x, w, b []float32, rows, dim int) {
	const eps = 1e-5
	for r := range rows {
		row := x[r*dim : (r+1)*dim]
		var mean float64
		for _, v := range row {
			mean += float64(v)
		}
		mean /= float64(dim)
		var vr float64
		for _, v := range row {
			d := float64(v) - mean
			vr += d * d
		}
		inv := 1 / math.Sqrt(vr/float64(dim)+eps)
		for i, v := range row {
			row[i] = float32((float64(v)-mean)*inv)*w[i] + b[i]
		}
	}
}

func linear(x, w, b []float32, rows, in, out int) []float32 {
	y := make([]float32, rows*out)
	linalg.MatmulBT(x, w, y, rows, in, out)
	for r := range rows {
		row := y[r*out : (r+1)*out]
		for j := range row {
			row[j] += b[j]
		}
	}
	return y
}

// layer runs one pre-LayerNorm encoder layer on h [n][D] in place, attending within consecutive windows of `window` positions.
func (e *QwenASREncoder) layer(ly *qwenASRLayer, h []float32, n, window int) {
	c := e.Cfg
	D := c.DModel
	x := append([]float32(nil), h...)
	layerNorm(x, ly.ln1W, ly.ln1B, n, D)
	q, k, v := linear(x, ly.qW, ly.qB, n, D, D), linear(x, ly.kW, ly.kB, n, D, D), linear(x, ly.vW, ly.vB, n, D, D)
	hd := D / c.Heads
	scale := float32(1 / math.Sqrt(float64(hd)))
	att := make([]float32, n*D)
	for lo := 0; lo < n; lo += window {
		hi := min(lo+window, n)
		m := hi - lo
		sc := make([]float32, m)
		for hh := range c.Heads {
			off := hh * hd
			for i := lo; i < hi; i++ {
				qi := q[i*D+off : i*D+off+hd]
				mx := float32(math.Inf(-1))
				for j := lo; j < hi; j++ {
					kj := k[j*D+off : j*D+off+hd]
					var s float32
					for d := range hd {
						s += qi[d] * kj[d]
					}
					s *= scale
					sc[j-lo] = s
					if s > mx {
						mx = s
					}
				}
				var sum float32
				for j := range m {
					sc[j] = float32(math.Exp(float64(sc[j] - mx)))
					sum += sc[j]
				}
				o := att[i*D+off : i*D+off+hd]
				for j := lo; j < hi; j++ {
					w := sc[j-lo] / sum
					vj := v[j*D+off : j*D+off+hd]
					for d := range hd {
						o[d] += w * vj[d]
					}
				}
			}
		}
	}
	a := linear(att, ly.oW, ly.oB, n, D, D)
	for i := range h {
		h[i] += a[i]
	}
	y := append([]float32(nil), h...)
	layerNorm(y, ly.ln2W, ly.ln2B, n, D)
	f := linear(y, ly.fc1W, ly.fc1B, n, D, c.FFN)
	for i, vv := range f {
		f[i] = e.gelu(vv)
	}
	g := linear(f, ly.fc2W, ly.fc2B, n, c.FFN, D)
	for i := range h {
		h[i] += g[i]
	}
}

// project is the multimodal projector: Linear, GELU, Linear.
func (e *QwenASREncoder) project(h []float32, n int) []float32 {
	D := e.Cfg.DModel
	p := linear(h, e.proj1W, e.proj1B, n, D, D)
	if e.defect != qwenASREncDefectNoProjGELU {
		for i, v := range p {
			p[i] = e.gelu(v)
		}
	}
	return linear(p, e.proj2W, e.proj2B, n, D, e.Cfg.OutputDim)
}
