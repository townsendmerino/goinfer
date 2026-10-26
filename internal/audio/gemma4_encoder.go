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

// Gemma4AudioConfig is the audio_config fields the tower reads.
type Gemma4AudioConfig struct {
	HiddenSize        int     `json:"hidden_size"`
	NumHiddenLayers   int     `json:"num_hidden_layers"`
	NumAttentionHeads int     `json:"num_attention_heads"`
	SubsampleChannels []int   `json:"subsampling_conv_channels"`
	ConvKernelSize    int     `json:"conv_kernel_size"`
	ResidualWeight    float64 `json:"residual_weight"`
	ChunkSize         int     `json:"attention_chunk_size"`
	ContextLeft       int     `json:"attention_context_left"`
	ContextRight      int     `json:"attention_context_right"`
	LogitCap          float64 `json:"attention_logit_cap"`
	RMSNormEps        float64 `json:"rms_norm_eps"`
	OutputProjDims    int     `json:"output_proj_dims"`
	UseClippedLinears bool    `json:"use_clipped_linears"`
}

func (c Gemma4AudioConfig) validate() error {
	switch {
	case c.HiddenSize <= 0 || c.NumHiddenLayers <= 0 || c.NumAttentionHeads <= 0 || c.HiddenSize%c.NumAttentionHeads != 0:
		return fmt.Errorf("audio: hidden %d, %d layers, %d heads", c.HiddenSize, c.NumHiddenLayers, c.NumAttentionHeads)
	case len(c.SubsampleChannels) != 2 || c.SubsampleChannels[0] != gemma4Mels:
		// proj_input_dim is (channels[0]//4)*channels[1]: it equals the flattened width only when channels[0] is the
		// mel count (spec §2.1), so anything else would be a different model.
		return fmt.Errorf("audio: subsampling channels %v, want [%d, c]", c.SubsampleChannels, gemma4Mels)
	case c.ContextRight != 0:
		return fmt.Errorf("audio: attention_context_right %d (only 0, causal, is implemented)", c.ContextRight)
	case c.ContextLeft < 1 || c.ConvKernelSize < 1 || c.OutputProjDims <= 0:
		return fmt.Errorf("audio: context_left %d, conv kernel %d, output dims %d", c.ContextLeft, c.ConvKernelSize, c.OutputProjDims)
	}
	return nil
}

// Window is how far back a query sees: keys at distance 0..Window-1 (spec §2.4: context_left - 1 positions; 12 here,
// one fewer than Gemma 3n).
func (c Gemma4AudioConfig) Window() int { return c.ContextLeft - 1 }

// Gemma4Proj is one projection: nn.Linear's [Out, In] weight, its bias (nil when none) and, for a ClippableLinear,
// its bounds (±Inf when the checkpoint does not clip).
type Gemma4Proj struct {
	W                            []float32
	B                            []float32
	Out, In                      int
	InMin, InMax, OutMin, OutMax float32
}

// Gemma4AudioLayer is one conformer block's weights.
type Gemma4AudioLayer struct {
	FF1, FF2      Gemma4AudioFF
	NormPreAttn   []float32
	NormPostAttn  []float32
	NormOut       []float32
	Q, K, V, Post Gemma4Proj
	RelK          []float32 // [Window, hidden]: relative_k_proj applied to the position table at distances 0..Window-1
	PerDimScale   []float32 // [head dim], raw (softplus in forward)
	ConvPreNorm   []float32
	ConvStart     Gemma4Proj // hidden -> 2*hidden, then GLU
	ConvW         []float32  // depthwise [hidden, kernel]
	ConvNorm      []float32
	ConvEnd       Gemma4Proj
}

// Gemma4AudioFF is one half-step feed-forward.
type Gemma4AudioFF struct {
	PreNorm, PostNorm []float32
	Up, Down          Gemma4Proj
}

// Gemma4AudioEncoder is a loaded Gemma 4 audio tower with its embedder (embed_audio) to the text width.
type Gemma4AudioEncoder struct {
	Cfg            Gemma4AudioConfig
	TextHiddenSize int
	conv0W, conv1W []float32 // [c0,1,3,3], [c1,c0,3,3]
	norm0, norm1   []float32 // LayerNorm weights (no bias)
	inputProj      Gemma4Proj
	Layers         []Gemma4AudioLayer
	outputProj     Gemma4Proj // with bias
	embedProj      Gemma4Proj // OutputProjDims -> text hidden, after an unweighted RMSNorm
}

func inf32(s float64) float32 { return float32(math.Inf(int(s))) }

// LoadGemma4AudioEncoder reads the audio tower and embed_audio from an HF checkpoint directory whose config.json has
// an audio_config (EmbeddingGemma 2, Gemma 4 E2B/E4B) and text_config.hidden_size. Weights are copied out of the
// mmap as float32.
func LoadGemma4AudioEncoder(dir string) (*Gemma4AudioEncoder, error) {
	raw, err := os.ReadFile(filepath.Join(dir, "config.json"))
	if err != nil {
		return nil, fmt.Errorf("audio: read config: %w", err)
	}
	var wrap struct {
		Audio *Gemma4AudioConfig `json:"audio_config"`
		Text  *struct {
			HiddenSize int `json:"hidden_size"`
		} `json:"text_config"`
	}
	if err := json.Unmarshal(raw, &wrap); err != nil {
		return nil, fmt.Errorf("audio: parse config: %w", err)
	}
	if wrap.Audio == nil {
		return nil, fmt.Errorf("audio: config.json has no audio_config")
	}
	c := *wrap.Audio
	if c.RMSNormEps == 0 {
		c.RMSNormEps = 1e-6
	}
	if c.LogitCap == 0 {
		c.LogitCap = 50
	}
	if c.ResidualWeight == 0 {
		c.ResidualWeight = 0.5
	}
	if err := c.validate(); err != nil {
		return nil, err
	}
	if wrap.Text == nil || wrap.Text.HiddenSize <= 0 {
		return nil, fmt.Errorf("audio: text_config.hidden_size missing (the embedder's output width)")
	}
	var st *embed.SafetensorsFile
	if idx := filepath.Join(dir, "model.safetensors.index.json"); fileExists(idx) {
		st, err = embed.OpenSafetensorsShardedMmap(idx)
	} else {
		st, err = embed.OpenSafetensorsMmap(filepath.Join(dir, "model.safetensors"))
	}
	if err != nil {
		return nil, fmt.Errorf("audio: open safetensors: %w", err)
	}
	defer st.Close()
	pfx := ""
	if _, e := st.Tensor("audio_tower.output_proj.weight"); e != nil {
		pfx = "model."
	}
	H, c0, c1 := c.HiddenSize, c.SubsampleChannels[0], c.SubsampleChannels[1]
	get := func(name string, want ...int) []float32 {
		if err != nil {
			return nil
		}
		var v []float32
		v, err = st.TensorF32(pfx+name, want...)
		if err != nil {
			return nil
		}
		return append([]float32(nil), v...)
	}
	scalar := func(name string) float32 {
		v := get(name)
		if err == nil && len(v) != 1 {
			err = fmt.Errorf("audio: %s has %d elements, want a scalar", name, len(v))
		}
		if err != nil {
			return 0
		}
		return v[0]
	}
	plain := func(name string, out, in int) Gemma4Proj {
		return Gemma4Proj{W: get(name, out, in), Out: out, In: in, InMin: inf32(-1), InMax: inf32(1), OutMin: inf32(-1), OutMax: inf32(1)}
	}
	clipped := func(base string, out, in int) Gemma4Proj {
		p := plain(base+".linear.weight", out, in)
		if c.UseClippedLinears {
			p.InMin, p.InMax = scalar(base+".input_min"), scalar(base+".input_max")
			p.OutMin, p.OutMax = scalar(base+".output_min"), scalar(base+".output_max")
		}
		return p
	}
	e := &Gemma4AudioEncoder{Cfg: c, TextHiddenSize: wrap.Text.HiddenSize}
	sp := "audio_tower.subsample_conv_projection."
	e.conv0W = get(sp+"layer0.conv.weight", c0, 1, 3, 3)
	e.norm0 = get(sp+"layer0.norm.weight", c0)
	e.conv1W = get(sp+"layer1.conv.weight", c1, c0, 3, 3)
	e.norm1 = get(sp+"layer1.norm.weight", c1)
	flat := (gemma4Mels / 4) * c1
	e.inputProj = plain(sp+"input_proj_linear.weight", H, flat)
	pe := relPositionTable(c.ContextLeft, H) // distances 0..ContextLeft-1
	nH := c.NumAttentionHeads
	hd := H / nH
	e.Layers = make([]Gemma4AudioLayer, c.NumHiddenLayers)
	for l := range e.Layers {
		p := fmt.Sprintf("audio_tower.layers.%d.", l)
		ff := func(n string) Gemma4AudioFF {
			return Gemma4AudioFF{PreNorm: get(p+n+".pre_layer_norm.weight", H), PostNorm: get(p+n+".post_layer_norm.weight", H),
				Up: clipped(p+n+".ffw_layer_1", 4*H, H), Down: clipped(p+n+".ffw_layer_2", H, 4*H)}
		}
		ly := &e.Layers[l]
		ly.FF1, ly.FF2 = ff("feed_forward1"), ff("feed_forward2")
		ly.NormPreAttn, ly.NormPostAttn, ly.NormOut = get(p+"norm_pre_attn.weight", H), get(p+"norm_post_attn.weight", H), get(p+"norm_out.weight", H)
		ly.Q, ly.K, ly.V = clipped(p+"self_attn.q_proj", H, H), clipped(p+"self_attn.k_proj", H, H), clipped(p+"self_attn.v_proj", H, H)
		ly.Post = clipped(p+"self_attn.post", H, H)
		ly.PerDimScale = get(p+"self_attn.per_dim_scale", hd)
		if rk := get(p+"self_attn.relative_k_proj.weight", H, H); err == nil {
			w := c.Window()
			ly.RelK = make([]float32, w*H)
			linalg.MatmulBT(pe[:w*H], rk, ly.RelK, w, H, H)
		}
		ly.ConvPreNorm = get(p+"lconv1d.pre_layer_norm.weight", H)
		ly.ConvStart = clipped(p+"lconv1d.linear_start", 2*H, H)
		ly.ConvW = get(p+"lconv1d.depthwise_conv1d.weight", H, 1, c.ConvKernelSize)
		ly.ConvNorm = get(p+"lconv1d.conv_norm.weight", H)
		ly.ConvEnd = clipped(p+"lconv1d.linear_end", H, H)
	}
	e.outputProj = plain("audio_tower.output_proj.weight", c.OutputProjDims, H)
	e.outputProj.B = get("audio_tower.output_proj.bias", c.OutputProjDims)
	e.embedProj = plain("embed_audio.embedding_projection.weight", e.TextHiddenSize, c.OutputProjDims)
	if err != nil {
		return nil, fmt.Errorf("audio: load weights: %w", err)
	}
	return e, nil
}

func fileExists(p string) bool { _, err := os.Stat(p); return err == nil }

// relPositionTable is Gemma4AudioRelPositionalEncoding for distances 0..n-1: row d = [sin(d·inv) ‖ cos(d·inv)] with
// inv[i] = exp(-i ln(10000) / (H/2 - 1)) (spec §2.2), computed in float32 as the reference's buffer is.
func relPositionTable(n, H int) []float32 {
	half := H / 2
	inv := make([]float32, half)
	for i := range inv {
		inv[i] = float32(math.Exp(-float64(i) * math.Log(10000) / float64(half-1)))
	}
	pe := make([]float32, n*H)
	for d := range n {
		for i, f := range inv {
			a := float32(d) * f
			pe[d*H+i] = float32(math.Sin(float64(a)))
			pe[d*H+half+i] = float32(math.Cos(float64(a)))
		}
	}
	return pe
}

// Gemma4AudioStages is the tower's intermediate outputs, for per-stage differencing against the reference.
type Gemma4AudioStages struct {
	Sub    []float32   // after the subsampler and its projection, [n, hidden]
	Blocks [][]float32 // each block's output, [n, hidden]
	Tower  []float32   // after output_proj, [n, OutputProjDims]
}

// Forward runs the tower and the embedder on T valid log-mel frames [T, 128] (Gemma4Features) and returns the soft
// tokens [n, TextHiddenSize], n = Gemma4SoftTokens(T).
func (e *Gemma4AudioEncoder) Forward(feats []float32, T int) ([]float32, error) {
	out, _, err := e.forward(feats, T, false)
	return out, err
}

// ForwardStages is Forward that also returns the intermediate stages.
func (e *Gemma4AudioEncoder) ForwardStages(feats []float32, T int) ([]float32, *Gemma4AudioStages, error) {
	return e.forward(feats, T, true)
}

func (e *Gemma4AudioEncoder) forward(feats []float32, T int, keep bool) ([]float32, *Gemma4AudioStages, error) {
	if T <= 0 || len(feats) != T*gemma4Mels {
		return nil, nil, fmt.Errorf("audio: %d feature values for %d frames of %d", len(feats), T, gemma4Mels)
	}
	h, n := e.subsample(feats, T)
	var st *Gemma4AudioStages
	if keep {
		st = &Gemma4AudioStages{Sub: append([]float32(nil), h...)}
	}
	for l := range e.Layers {
		e.block(&e.Layers[l], h, n)
		if keep {
			st.Blocks = append(st.Blocks, append([]float32(nil), h...))
		}
	}
	tw := e.outputProj.apply(h, n)
	if keep {
		st.Tower = append([]float32(nil), tw...)
	}
	rmsNorm(tw, nil, n, e.Cfg.OutputProjDims, e.Cfg.RMSNormEps)
	return e.embedProj.apply(tw, n), st, nil
}

// Subsample is Forward's first stage, the conv stack and its projection, for a device-resident tower that runs the
// blocks: [n, hidden] and n.
func (e *Gemma4AudioEncoder) Subsample(feats []float32, T int) ([]float32, int, error) {
	if T <= 0 || len(feats) != T*gemma4Mels {
		return nil, 0, fmt.Errorf("audio: %d feature values for %d frames of %d", len(feats), T, gemma4Mels)
	}
	h, n := e.subsample(feats, T)
	return h, n, nil
}

// FinishBlocks is Forward's tail after the last block: output_proj, the unweighted RMSNorm and the embedder.
func (e *Gemma4AudioEncoder) FinishBlocks(h []float32, n int) ([]float32, error) {
	if n <= 0 || len(h) != n*e.Cfg.HiddenSize {
		return nil, fmt.Errorf("audio: %d values for %d rows of %d", len(h), n, e.Cfg.HiddenSize)
	}
	tw := e.outputProj.apply(h, n)
	rmsNorm(tw, nil, n, e.Cfg.OutputProjDims, e.Cfg.RMSNormEps)
	return e.embedProj.apply(tw, n), nil
}

// apply runs the projection over rows of x: input clamp, the matmul, bias, output clamp (spec §2.3).
func (p *Gemma4Proj) apply(x []float32, rows int) []float32 {
	in := x
	if !math.IsInf(float64(p.InMin), 0) || !math.IsInf(float64(p.InMax), 0) {
		in = make([]float32, len(x))
		for i, v := range x {
			in[i] = clamp(v, p.InMin, p.InMax)
		}
	}
	out := make([]float32, rows*p.Out)
	linalg.MatmulBT(in, p.W, out, rows, p.In, p.Out)
	if p.B != nil {
		for r := range rows {
			row := out[r*p.Out : (r+1)*p.Out]
			for j := range row {
				row[j] += p.B[j]
			}
		}
	}
	if !math.IsInf(float64(p.OutMin), 0) || !math.IsInf(float64(p.OutMax), 0) {
		for i, v := range out {
			out[i] = clamp(v, p.OutMin, p.OutMax)
		}
	}
	return out
}

func clamp(v, lo, hi float32) float32 {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}

// rmsNorm is Gemma4RMSNorm in place over rows of dim: x · (mean(x²) + eps)^-½ · w, w applied directly (nil: no
// weight).
func rmsNorm(x, w []float32, rows, dim int, eps float64) {
	for r := range rows {
		row := x[r*dim : (r+1)*dim]
		var ss float32
		for _, v := range row {
			ss += v * v
		}
		inv := float32(1 / math.Sqrt(float64(ss/float32(dim))+eps))
		for i, v := range row {
			v *= inv
			if w != nil {
				v *= w[i]
			}
			row[i] = v
		}
	}
}

// subsample is the two-conv stack and input_proj_linear (spec §2.1): [T, 128] -> [n, hidden].
func (e *Gemma4AudioEncoder) subsample(feats []float32, T int) ([]float32, int) {
	c0, c1 := e.Cfg.SubsampleChannels[0], e.Cfg.SubsampleChannels[1]
	x0, t0, f0 := conv3x3s2HWC(feats, 1, T, gemma4Mels, e.conv0W, c0) // [T, 128] is HWC with one channel
	layerNormReLU(x0, e.norm0, t0*f0, c0, e.Cfg.RMSNormEps)
	x1, t1, f1 := conv3x3s2HWC(x0, c0, t0, f0, e.conv1W, c1)
	layerNormReLU(x1, e.norm1, t1*f1, c1, e.Cfg.RMSNormEps)
	// x1 is [t1, f1, c1] already: the flatten index f*c1 + c (spec §2.1).
	return e.inputProj.apply(x1, t1), t1
}

// conv3x3s2HWC is Conv2d(k=3, s=2, p=1, no bias) over x [T, F, cin] (HWC) with w [cout, cin, 3, 3], as an im2col
// matmul: output [To, Fo, cout].
func conv3x3s2HWC(x []float32, cin, T, F int, w []float32, cout int) ([]float32, int, int) {
	To, Fo := (T+1)/2, (F+1)/2
	K := cin * 9
	col := make([]float32, To*Fo*K) // per output position, (ci, kt, kf) to match w's layout
	for t := range To {
		for f := range Fo {
			dst := col[(t*Fo+f)*K : (t*Fo+f+1)*K]
			for kt := range 3 {
				ti := 2*t - 1 + kt
				if ti < 0 || ti >= T {
					continue
				}
				for kf := range 3 {
					fi := 2*f - 1 + kf
					if fi < 0 || fi >= F {
						continue
					}
					src := x[(ti*F+fi)*cin : (ti*F+fi+1)*cin]
					for ci, v := range src {
						dst[ci*9+kt*3+kf] = v
					}
				}
			}
		}
	}
	out := make([]float32, To*Fo*cout)
	linalg.MatmulBT(col, w, out, To*Fo, K, cout)
	return out, To, Fo
}

// layerNormReLU is the subsampler's bias-free LayerNorm over the channel axis (mean-subtracted, biased variance,
// eps 1e-6) and a ReLU, in place over rows of dim.
func layerNormReLU(x, w []float32, rows, dim int, eps float64) {
	for r := range rows {
		row := x[r*dim : (r+1)*dim]
		var mean float32
		for _, v := range row {
			mean += v
		}
		mean /= float32(dim)
		var vr float32
		for _, v := range row {
			d := v - mean
			vr += d * d
		}
		inv := float32(1 / math.Sqrt(float64(vr/float32(dim))+eps))
		for i, v := range row {
			row[i] = max(0, (v-mean)*inv*w[i])
		}
	}
}

func silu(v float32) float32 { return v / (1 + float32(math.Exp(float64(-v)))) }

// block runs one conformer block on h [n, hidden] in place (spec §2.3).
func (e *Gemma4AudioEncoder) block(ly *Gemma4AudioLayer, h []float32, n int) {
	c := e.Cfg
	H := c.HiddenSize
	rw := float32(c.ResidualWeight)
	ffw := func(ff *Gemma4AudioFF) {
		y := append([]float32(nil), h...)
		rmsNorm(y, ff.PreNorm, n, H, c.RMSNormEps)
		u := ff.Up.apply(y, n)
		for i, v := range u {
			u[i] = silu(v)
		}
		y = ff.Down.apply(u, n)
		rmsNorm(y, ff.PostNorm, n, H, c.RMSNormEps)
		for i, v := range y {
			h[i] += rw * v
		}
	}
	ffw(&ly.FF1)
	// attention
	y := append([]float32(nil), h...)
	rmsNorm(y, ly.NormPreAttn, n, H, c.RMSNormEps)
	y = e.attention(ly, y, n)
	rmsNorm(y, ly.NormPostAttn, n, H, c.RMSNormEps)
	for i, v := range y {
		h[i] += v
	}
	// light conv
	y = append(y[:0], h...)
	rmsNorm(y, ly.ConvPreNorm, n, H, c.RMSNormEps)
	g := ly.ConvStart.apply(y, n) // [n, 2H]
	glu := make([]float32, n*H)
	for t := range n {
		for j := range H {
			a, b := g[t*2*H+j], g[t*2*H+H+j]
			glu[t*H+j] = a * float32(1/(1+math.Exp(float64(-b))))
		}
	}
	k := c.ConvKernelSize
	conv := make([]float32, n*H)
	for t := range n {
		for j := range H {
			var s float32
			for kk := range k {
				ti := t - (k - 1) + kk // causal: left pad k-1
				if ti >= 0 {
					s += ly.ConvW[j*k+kk] * glu[ti*H+j]
				}
			}
			conv[t*H+j] = s
		}
	}
	rmsNorm(conv, ly.ConvNorm, n, H, c.RMSNormEps)
	for i, v := range conv {
		conv[i] = silu(v)
	}
	y = ly.ConvEnd.apply(conv, n)
	for i, v := range y {
		h[i] += v
	}
	ffw(&ly.FF2)
	rmsNorm(h, ly.NormOut, n, H, c.RMSNormEps)
}

var (
	gemma4QScaleBase = float32(1 / math.Ln2) // times head_dim^-½
	gemma4KScale     = float32(math.Log(1+math.E) / math.Ln2)
)

// attention is the chunked local attention as a direct causal sliding window (spec §2.4): key j is seen by query t
// iff 0 <= t - j < Window. Logits are q·k + q·relk[t-j], softcapped, then softmax.
func (e *Gemma4AudioEncoder) attention(ly *Gemma4AudioLayer, y []float32, n int) []float32 {
	c := e.Cfg
	H, nH := c.HiddenSize, c.NumAttentionHeads
	hd := H / nH
	q, k, v := ly.Q.apply(y, n), ly.K.apply(y, n), ly.V.apply(y, n)
	qs := make([]float32, hd)
	base := gemma4QScaleBase / float32(math.Sqrt(float64(hd)))
	for d := range hd {
		qs[d] = base * float32(math.Log1p(math.Exp(float64(ly.PerDimScale[d])))) // softplus
	}
	for t := range n {
		for hh := range nH {
			for d := range hd {
				q[t*H+hh*hd+d] *= qs[d]
				k[t*H+hh*hd+d] *= gemma4KScale
			}
		}
	}
	W := c.Window()
	capv := float32(c.LogitCap)
	ctx := make([]float32, n*H)
	lg := make([]float32, W)
	for t := range n {
		for hh := range nH {
			qv := q[t*H+hh*hd : t*H+hh*hd+hd]
			mx := float32(math.Inf(-1))
			lo := max(0, t-W+1)
			for j := lo; j <= t; j++ {
				kv := k[j*H+hh*hd : j*H+hh*hd+hd]
				rk := ly.RelK[(t-j)*H+hh*hd : (t-j)*H+hh*hd+hd]
				var s float32
				for d, a := range qv {
					s += a*kv[d] + a*rk[d]
				}
				s = capv * float32(math.Tanh(float64(s/capv)))
				lg[j-lo] = s
				mx = max(mx, s)
			}
			var z float32
			for i := range t - lo + 1 {
				lg[i] = float32(math.Exp(float64(lg[i] - mx)))
				z += lg[i]
			}
			out := ctx[t*H+hh*hd : t*H+hh*hd+hd]
			for j := lo; j <= t; j++ {
				p := lg[j-lo] / z
				vv := v[j*H+hh*hd : j*H+hh*hd+hd]
				for d := range hd {
					out[d] += p * vv[d]
				}
			}
		}
	}
	return ly.Post.apply(ctx, n)
}
