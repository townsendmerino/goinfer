// Package embeddinggemma2 runs the text encoder of Google's EmbeddingGemma 2 (`model_type` embedding_gemma2,
// google/embeddinggemma-2): a 24-layer bidirectional Gemma 4-style encoder whose last hidden states are projected to
// the embedding width, mean-pooled over every token (the prompt included) and L2-normalised, the
// sentence-transformers pipeline the checkpoint ships (Transformer -> Pooling(mean) -> Normalize).
//
// It is an encoder, not a decoder: attention is bidirectional on every layer (a symmetric window of radius
// `sliding_window` on the sliding layers, the whole sequence on the full ones), there is no KV cache and no decode
// step, so it does not go through decoder.Load. Only the text half of the checkpoint is read; the vision and audio
// towers stay on disk. The forward runs on the CPU in float32. docs/tasks/task-embeddinggemma2.md has the design.
package embeddinggemma2

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"sync"

	"github.com/townsendmerino/aikit/embed"
	"github.com/townsendmerino/aikit/linalg"
)

// Config is the text half of an EmbeddingGemma 2 config.json (its text_config).
type Config struct {
	VocabSize      int      `json:"vocab_size"`
	Hidden         int      `json:"hidden_size"`
	Intermediate   int      `json:"intermediate_size"`
	Layers         int      `json:"num_hidden_layers"`
	Heads          int      `json:"num_attention_heads"`
	KVHeads        int      `json:"num_key_value_heads"`
	HeadDim        int      `json:"head_dim"`
	SlidingWindow  int      `json:"sliding_window"` // an inclusive radius: |q - k| <= SlidingWindow
	PLEDim         int      `json:"hidden_size_per_layer_input"`
	EmbeddingDim   int      `json:"embedding_dim"`
	RMSNormEps     float64  `json:"rms_norm_eps"`
	HiddenAct      string   `json:"hidden_activation"`
	LayerTypes     []string `json:"layer_types"`
	MaxPositions   int      `json:"max_position_embeddings"`
	PerLayerConfig map[string]struct {
		HeadDim int `json:"head_dim"`
		KVHeads int `json:"num_key_value_heads"`
	} `json:"per_layer_config"`
	RopeParameters map[string]struct {
		Theta float64 `json:"rope_theta"`
		Type  string  `json:"rope_type"`
	} `json:"rope_parameters"`
}

type composite struct {
	ModelType  string `json:"model_type"`
	TextConfig Config `json:"text_config"`
}

// layer holds one encoder layer's weights, row-major as nn.Linear stores them ([out, in]).
type layer struct {
	full             bool
	headDim, kvH     int
	theta            float64
	inNorm           []float32
	q, k, v, o       []float32
	qNorm, kNorm     []float32
	postAttnNorm     []float32
	preFFNorm        []float32
	gate, up, down   []float32
	postFFNorm       []float32
	pleGate, pleProj []float32
	plePostNorm      []float32
	scalar           float32
}

// Model is a loaded EmbeddingGemma 2 text encoder. Its weights are immutable after Load, so concurrent Embed calls
// are safe.
type Model struct {
	cfg        Config
	embed      []float32 // [vocab, hidden]
	pleProj    []float32 // [layers*ple, hidden]
	pleNorm    []float32 // [ple]
	layers     []layer
	finalNorm  []float32
	projection []float32 // [embedding_dim, hidden]
}

// Config returns the loaded text config.
func (m *Model) Config() Config { return m.cfg }

// Dim is the embedding width (embedding_dim, 768 for google/embeddinggemma-2).
func (m *Model) Dim() int { return m.cfg.EmbeddingDim }

// LoadConfig reads dir/config.json and returns its text half, refusing anything that is not embedding_gemma2.
func LoadConfig(dir string) (Config, error) {
	raw, err := os.ReadFile(filepath.Join(dir, "config.json"))
	if err != nil {
		return Config{}, fmt.Errorf("embeddinggemma2: %w", err)
	}
	var c composite
	if err := json.Unmarshal(raw, &c); err != nil {
		return Config{}, fmt.Errorf("embeddinggemma2: config.json: %w", err)
	}
	if c.ModelType != "embedding_gemma2" {
		return Config{}, fmt.Errorf("embeddinggemma2: model_type %q, want embedding_gemma2", c.ModelType)
	}
	cfg := c.TextConfig
	if cfg.Hidden <= 0 || cfg.Layers <= 0 || cfg.Heads <= 0 || cfg.KVHeads <= 0 || cfg.HeadDim <= 0 || cfg.EmbeddingDim <= 0 ||
		cfg.PLEDim <= 0 || cfg.VocabSize <= 0 || len(cfg.LayerTypes) != cfg.Layers {
		return Config{}, fmt.Errorf("embeddinggemma2: text_config incomplete (hidden %d, layers %d, heads %d/%d, head_dim %d, embedding_dim %d, ple %d, vocab %d, %d layer_types)",
			cfg.Hidden, cfg.Layers, cfg.Heads, cfg.KVHeads, cfg.HeadDim, cfg.EmbeddingDim, cfg.PLEDim, cfg.VocabSize, len(cfg.LayerTypes))
	}
	if cfg.HiddenAct != "gelu_pytorch_tanh" {
		return Config{}, fmt.Errorf("embeddinggemma2: hidden_activation %q, only gelu_pytorch_tanh is implemented", cfg.HiddenAct)
	}
	if cfg.RMSNormEps <= 0 {
		return Config{}, fmt.Errorf("embeddinggemma2: rms_norm_eps %v", cfg.RMSNormEps)
	}
	for _, t := range cfg.LayerTypes {
		if t != "sliding_attention" && t != "full_attention" {
			return Config{}, fmt.Errorf("embeddinggemma2: layer type %q", t)
		}
		if p, ok := cfg.RopeParameters[t]; !ok || p.Theta <= 0 || (p.Type != "" && p.Type != "default") {
			return Config{}, fmt.Errorf("embeddinggemma2: rope_parameters[%q] missing or not a default rope", t)
		}
	}
	return cfg, nil
}

// Load reads the text encoder from an HF checkpoint directory (config.json + model.safetensors). The vision and audio
// tensors in the same file are not read.
func Load(dir string) (*Model, error) {
	cfg, err := LoadConfig(dir)
	if err != nil {
		return nil, err
	}
	st, err := embed.OpenSafetensorsMmap(filepath.Join(dir, "model.safetensors"))
	if err != nil {
		return nil, fmt.Errorf("embeddinggemma2: %w", err)
	}
	defer st.Close()
	get := func(name string, shape ...int) ([]float32, error) {
		v, err := st.TensorF32("language_model."+name, shape...)
		if err != nil {
			return nil, fmt.Errorf("embeddinggemma2: %s: %w", name, err)
		}
		// An F32 tensor comes back aliasing the mapping, which Close unmaps; own a copy.
		return append([]float32(nil), v...), nil
	}
	H, I, P, L := cfg.Hidden, cfg.Intermediate, cfg.PLEDim, cfg.Layers
	m := &Model{cfg: cfg, layers: make([]layer, L)}
	if m.embed, err = get("embed_tokens.weight", cfg.VocabSize, H); err != nil {
		return nil, err
	}
	if m.pleProj, err = get("ple.per_layer_model_projection.weight", L*P, H); err != nil {
		return nil, err
	}
	if m.pleNorm, err = get("ple.per_layer_projection_norm.weight", P); err != nil {
		return nil, err
	}
	if m.finalNorm, err = get("norm.weight", H); err != nil {
		return nil, err
	}
	if m.projection, err = get("embedding_projection.weight", cfg.EmbeddingDim, H); err != nil {
		return nil, err
	}
	for i := range m.layers {
		ly := &m.layers[i]
		ly.full = cfg.LayerTypes[i] == "full_attention"
		ly.headDim, ly.kvH = cfg.HeadDim, cfg.KVHeads
		for k, pc := range cfg.PerLayerConfig {
			if n, e := strconv.Atoi(k); e == nil && n == i {
				if pc.HeadDim > 0 {
					ly.headDim = pc.HeadDim
				}
				if pc.KVHeads > 0 {
					ly.kvH = pc.KVHeads
				}
			}
		}
		if cfg.Heads%ly.kvH != 0 || ly.headDim%2 != 0 {
			return nil, fmt.Errorf("embeddinggemma2: layer %d: %d heads over %d KV heads, head_dim %d", i, cfg.Heads, ly.kvH, ly.headDim)
		}
		ly.theta = cfg.RopeParameters[cfg.LayerTypes[i]].Theta
		qd, kvd := cfg.Heads*ly.headDim, ly.kvH*ly.headDim
		p := fmt.Sprintf("layers.%d.", i)
		for _, w := range []struct {
			dst   *[]float32
			name  string
			shape []int
		}{
			{&ly.inNorm, "input_layernorm.weight", []int{H}},
			{&ly.q, "self_attn.q_proj.weight", []int{qd, H}},
			{&ly.k, "self_attn.k_proj.weight", []int{kvd, H}},
			{&ly.v, "self_attn.v_proj.weight", []int{kvd, H}},
			{&ly.o, "self_attn.o_proj.weight", []int{H, qd}},
			{&ly.qNorm, "self_attn.q_norm.weight", []int{ly.headDim}},
			{&ly.kNorm, "self_attn.k_norm.weight", []int{ly.headDim}},
			{&ly.postAttnNorm, "post_attention_layernorm.weight", []int{H}},
			{&ly.preFFNorm, "pre_feedforward_layernorm.weight", []int{H}},
			{&ly.gate, "mlp.gate_proj.weight", []int{I, H}},
			{&ly.up, "mlp.up_proj.weight", []int{I, H}},
			{&ly.down, "mlp.down_proj.weight", []int{H, I}},
			{&ly.postFFNorm, "post_feedforward_layernorm.weight", []int{H}},
			{&ly.pleGate, "ple_block.per_layer_input_gate.weight", []int{P, H}},
			{&ly.pleProj, "ple_block.per_layer_projection.weight", []int{H, P}},
			{&ly.plePostNorm, "ple_block.post_per_layer_input_norm.weight", []int{H}},
		} {
			if *w.dst, err = get(p+w.name, w.shape...); err != nil {
				return nil, err
			}
		}
		s, err := get(p+"layer_scalar", 1)
		if err != nil {
			return nil, err
		}
		ly.scalar = s[0]
	}
	return m, nil
}

// Hidden runs the encoder over ids and returns the projected last hidden state, [len(ids), Dim()] row-major: the
// sentence-transformers Transformer module's token_embeddings, before pooling.
func (m *Model) Hidden(ids []int) ([]float32, error) {
	out, _, err := m.forward(ids, false)
	return out, err
}

// Embed runs the encoder over ids, mean-pools the projected hidden states over every position and L2-normalises the
// result: the checkpoint's sentence embedding. The ids are the whole tokenised input (BOS, any task prompt, the text).
func (m *Model) Embed(ids []int) ([]float32, error) {
	h, _, err := m.forward(ids, false)
	if err != nil {
		return nil, err
	}
	return poolNormalize(h, len(ids), m.cfg.EmbeddingDim), nil
}

// poolNormalize mean-pools T rows of width d and L2-normalises the mean (sentence-transformers' Pooling(mean) then
// Normalize; F.normalize's eps of 1e-12 guards a zero vector).
func poolNormalize(h []float32, T, d int) []float32 {
	mean := make([]float64, d)
	for t := range T {
		for j, v := range h[t*d : (t+1)*d] {
			mean[j] += float64(v)
		}
	}
	var ss float64
	for j := range mean {
		mean[j] /= float64(T)
		ss += mean[j] * mean[j]
	}
	n := math.Max(math.Sqrt(ss), 1e-12)
	out := make([]float32, d)
	for j := range mean {
		out[j] = float32(mean[j] / n)
	}
	return out
}

// forward is the encoder. With keepLayers it also returns every layer's input and the last layer's output (before the
// final norm), for per-layer differencing in tests.
func (m *Model) forward(ids []int, keepLayers bool) ([]float32, [][]float32, error) {
	x, err := m.EmbedTokens(ids)
	if err != nil {
		return nil, nil, err
	}
	return m.forwardEmbeds(x, len(ids), keepLayers)
}

// EmbedTokens is the encoder's input rows for ids: each token's embedding times sqrt(hidden), [len(ids), hidden]. An
// image's soft tokens replace their placeholder rows after this scale, unscaled (EmbedImage).
func (m *Model) EmbedTokens(ids []int) ([]float32, error) {
	c := m.cfg
	H := c.Hidden
	if len(ids) == 0 {
		return nil, fmt.Errorf("embeddinggemma2: no input ids")
	}
	x := make([]float32, len(ids)*H)
	scale := float32(math.Sqrt(float64(H)))
	for t, id := range ids {
		if id < 0 || id >= c.VocabSize {
			return nil, fmt.Errorf("embeddinggemma2: token id %d out of range [0, %d)", id, c.VocabSize)
		}
		row := m.embed[id*H : (id+1)*H]
		for j, v := range row {
			x[t*H+j] = v * scale
		}
	}
	return x, nil
}

// ForwardEmbedsCPU runs the encoder on T prepared input rows x [T, hidden] (EmbedTokens, with any image rows spliced
// in), returning the projected last hidden state as forward does. x is not modified.
func (m *Model) ForwardEmbedsCPU(x []float32, T int, keepLayers bool) ([]float32, [][]float32, error) {
	return m.forwardEmbeds(append([]float32(nil), x...), T, keepLayers)
}

// forwardEmbeds is the encoder from its input rows; it works in x in place.
func (m *Model) forwardEmbeds(x []float32, T int, keepLayers bool) ([]float32, [][]float32, error) {
	c := m.cfg
	H, P := c.Hidden, c.PLEDim
	if T == 0 || len(x) != T*H {
		return nil, nil, fmt.Errorf("embeddinggemma2: %d input values for %d rows of %d", len(x), T, H)
	}
	if c.MaxPositions > 0 && T > c.MaxPositions {
		return nil, nil, fmt.Errorf("embeddinggemma2: %d positions, over max_position_embeddings %d", T, c.MaxPositions)
	}
	// Projection-only per-layer inputs: RMSNorm(reshape(W · emb · hidden^-0.5, [T, L, P])), from the scaled embeddings.
	ple := make([]float32, T*c.Layers*P)
	linalg.MatmulBT(x, m.pleProj, ple, T, H, c.Layers*P)
	ps := float32(1 / math.Sqrt(float64(H)))
	for i := range ple {
		ple[i] *= ps
	}
	for r := 0; r < T*c.Layers; r++ {
		rmsNorm(ple[r*P:(r+1)*P], m.pleNorm, c.RMSNormEps)
	}
	var hidden [][]float32
	keep := func() {
		if keepLayers {
			hidden = append(hidden, append([]float32(nil), x...))
		}
	}
	keep()
	normed := make([]float32, T*H)
	for li := range m.layers {
		ly := &m.layers[li]
		// attention block
		copy(normed, x)
		rmsRows(normed, ly.inNorm, T, H, c.RMSNormEps)
		a := m.attention(ly, normed, T)
		rmsRows(a, ly.postAttnNorm, T, H, c.RMSNormEps)
		for i := range x {
			x[i] += a[i]
		}
		// MLP block
		copy(normed, x)
		rmsRows(normed, ly.preFFNorm, T, H, c.RMSNormEps)
		f := mlp(normed, ly.gate, ly.up, ly.down, T, H, c.Intermediate)
		rmsRows(f, ly.postFFNorm, T, H, c.RMSNormEps)
		for i := range x {
			x[i] += f[i]
		}
		// PLE block: x += RMSNorm(proj(gelu(gate(x)) * ple_i))
		g := make([]float32, T*P)
		linalg.MatmulBT(x, ly.pleGate, g, T, H, P)
		parallelRows(T, func(t0, t1 int) {
			for t := t0; t < t1; t++ {
				pi := ple[(t*c.Layers+li)*P : (t*c.Layers+li+1)*P]
				for j := range P {
					g[t*P+j] = geluTanh(g[t*P+j]) * pi[j]
				}
			}
		})
		pr := make([]float32, T*H)
		linalg.MatmulBT(g, ly.pleProj, pr, T, P, H)
		rmsRows(pr, ly.plePostNorm, T, H, c.RMSNormEps)
		for i := range x {
			x[i] = (x[i] + pr[i]) * ly.scalar
		}
		keep()
	}
	rmsRows(x, m.finalNorm, T, H, c.RMSNormEps)
	out := make([]float32, T*c.EmbeddingDim)
	linalg.MatmulBT(x, m.projection, out, T, H, c.EmbeddingDim)
	return out, hidden, nil
}

// attention is one bidirectional self-attention block on the normed rows xn [T, H]: q/k/v projections, per-head
// RMSNorm (q and k scaled, v unscaled), rotate-half RoPE at positions 0..T-1, scores scaled by 1.0 (the q/k norms
// take the place of 1/sqrt(d)), a symmetric window on sliding layers, softmax in float32, and o_proj.
//
// The scores and the weighted values are matmuls (linalg.MatmulBT, SIMD and parallel), over blocks of attnBlock query
// rows against only the keys a block can reach (the window plus the block on a sliding layer), so the score matrix is
// never T x T and a long input does not allocate T^2. Keys outside a row's own window get no weight.
func (m *Model) attention(ly *layer, xn []float32, T int) []float32 {
	c := m.cfg
	H, nH, hd, nKV := c.Hidden, c.Heads, ly.headDim, ly.kvH
	qd, kvd := nH*hd, nKV*hd
	q := make([]float32, T*qd)
	k := make([]float32, T*kvd)
	v := make([]float32, T*kvd)
	linalg.MatmulBT(xn, ly.q, q, T, H, qd)
	linalg.MatmulBT(xn, ly.k, k, T, H, kvd)
	linalg.MatmulBT(xn, ly.v, v, T, H, kvd)
	cos, sin := ropeTables(T, hd, ly.theta)
	// Normalise and rotate, and lay each head out contiguously: qh [nH][T][hd], kh and vh [nKV][T][hd].
	qh := make([]float32, nH*T*hd)
	kh := make([]float32, nKV*T*hd)
	vh := make([]float32, nKV*T*hd)
	parallelRows(T, func(t0, t1 int) {
		for t := t0; t < t1; t++ {
			cs, sn := cos[t*hd/2:(t+1)*hd/2], sin[t*hd/2:(t+1)*hd/2]
			for h := range nH {
				dst := qh[(h*T+t)*hd : (h*T+t+1)*hd]
				copy(dst, q[(t*nH+h)*hd:(t*nH+h+1)*hd])
				rmsNorm(dst, ly.qNorm, c.RMSNormEps)
				rope(dst, cs, sn)
			}
			for h := range nKV {
				dk := kh[(h*T+t)*hd : (h*T+t+1)*hd]
				copy(dk, k[(t*nKV+h)*hd:(t*nKV+h+1)*hd])
				rmsNorm(dk, ly.kNorm, c.RMSNormEps)
				rope(dk, cs, sn)
				dv := vh[(h*T+t)*hd : (h*T+t+1)*hd]
				copy(dv, v[(t*nKV+h)*hd:(t*nKV+h+1)*hd])
				rmsNorm(dv, nil, c.RMSNormEps)
			}
		}
	})
	ctx := make([]float32, T*qd)
	B := min(attnBlock, T)
	scores := make([]float32, B*T)
	vt := make([]float32, hd*T)
	outb := make([]float32, B*hd)
	group := nH / nKV
	W := c.SlidingWindow
	for g := range nKV {
		for i0 := 0; i0 < T; i0 += B {
			i1 := min(T, i0+B)
			rows := i1 - i0
			klo, khi := 0, T
			if !ly.full {
				klo, khi = max(0, i0-W), min(T, i1+W)
			}
			kc := khi - klo
			// vt is this block's values transposed, [hd][kc], so the weighted sum is an a·bᵀ matmul too.
			for j := klo; j < khi; j++ {
				row := vh[(g*T+j)*hd : (g*T+j+1)*hd]
				for d, x := range row {
					vt[d*kc+(j-klo)] = x
				}
			}
			for h := g * group; h < (g+1)*group; h++ {
				sc := scores[:rows*kc]
				linalg.MatmulBT(qh[(h*T+i0)*hd:(h*T+i1)*hd], kh[(g*T+klo)*hd:(g*T+khi)*hd], sc, rows, hd, kc)
				parallelRows(rows, func(r0, r1 int) {
					for r := r0; r < r1; r++ {
						i := i0 + r
						lo, hi := klo, khi
						if !ly.full {
							lo, hi = max(klo, i-W), min(khi, i+W+1)
						}
						softmaxWindow(sc[r*kc:(r+1)*kc], lo-klo, hi-klo)
					}
				})
				ob := outb[:rows*hd]
				linalg.MatmulBT(sc, vt[:hd*kc], ob, rows, kc, hd)
				for r := range rows {
					copy(ctx[((i0+r)*nH+h)*hd:((i0+r)*nH+h+1)*hd], ob[r*hd:(r+1)*hd])
				}
			}
		}
	}
	o := make([]float32, T*H)
	linalg.MatmulBT(ctx, ly.o, o, T, qd, H)
	return o
}

// attnBlock is how many query rows one attention matmul takes. A var only so a test can make blocks smaller than the
// tiny fixture's inputs and the sliding window (TestTiny_blockingIsInvisible).
var attnBlock = 128

// softmaxWindow turns row[lo:hi] into softmax weights (float32, max-subtracted) and zeroes the rest of row.
func softmaxWindow(row []float32, lo, hi int) {
	mx := float32(math.Inf(-1))
	for _, s := range row[lo:hi] {
		mx = max(mx, s)
	}
	var sum float32
	for j := lo; j < hi; j++ {
		e := float32(math.Exp(float64(row[j] - mx)))
		row[j] = e
		sum += e
	}
	inv := 1 / sum
	for j := lo; j < hi; j++ {
		row[j] *= inv
	}
	clear(row[:lo])
	clear(row[hi:])
}

// parallelRows splits [0, n) across GOMAXPROCS goroutines (serially when n is small).
func parallelRows(n int, f func(lo, hi int)) {
	w := runtime.GOMAXPROCS(0)
	if n < 2*w || w == 1 {
		f(0, n)
		return
	}
	var wg sync.WaitGroup
	for i := range w {
		lo, hi := n*i/w, n*(i+1)/w
		if lo == hi {
			continue
		}
		wg.Add(1)
		go func() {
			defer wg.Done()
			f(lo, hi)
		}()
	}
	wg.Wait()
}

// mlp is down(gelu_tanh(gate(x)) * up(x)) over T rows.
func mlp(x, gate, up, down []float32, T, H, I int) []float32 {
	g := make([]float32, T*I)
	u := make([]float32, T*I)
	linalg.MatmulBT(x, gate, g, T, H, I)
	linalg.MatmulBT(x, up, u, T, H, I)
	parallelRows(len(g), func(lo, hi int) {
		for i := lo; i < hi; i++ {
			g[i] = geluTanh(g[i]) * u[i]
		}
	})
	out := make([]float32, T*H)
	linalg.MatmulBT(g, down, out, T, I, H)
	return out
}

// geluTanh is PyTorch's gelu(approximate="tanh").
func geluTanh(x float32) float32 {
	const c = 0.7978845608028654 // sqrt(2/pi)
	xf := float64(x)
	return float32(0.5 * xf * (1 + math.Tanh(c*(xf+0.044715*xf*xf*xf))))
}

// rmsRows applies rmsNorm to each of T rows of width n.
func rmsRows(x, w []float32, T, n int, eps float64) {
	for t := range T {
		rmsNorm(x[t*n:(t+1)*n], w, eps)
	}
}

// rmsNorm is Gemma 4's RMSNorm in place: x * (mean(x^2) + eps)^-0.5, times w when w is non-nil (the weight is used
// as is, not as 1 + w as in Gemma 2 and 3).
func rmsNorm(x, w []float32, eps float64) {
	var ss float64
	for _, v := range x {
		ss += float64(v) * float64(v)
	}
	inv := float32(1 / math.Sqrt(ss/float64(len(x))+eps))
	for i := range x {
		x[i] *= inv
		if w != nil {
			x[i] *= w[i]
		}
	}
}

// ropeTables returns cos and sin for positions 0..T-1 over hd/2 frequencies, inv_freq[i] = theta^(-2i/hd).
func ropeTables(T, hd int, theta float64) (cos, sin []float32) {
	half := hd / 2
	cos, sin = make([]float32, T*half), make([]float32, T*half)
	for i := range half {
		inv := float32(1 / math.Pow(theta, float64(2*i)/float64(hd)))
		for t := range T {
			a := float32(t) * inv
			cos[t*half+i] = float32(math.Cos(float64(a)))
			sin[t*half+i] = float32(math.Sin(float64(a)))
		}
	}
	return cos, sin
}

// rope rotates one head in place, rotate-half style: (x1, x2) -> (x1 cos - x2 sin, x2 cos + x1 sin).
func rope(x, cos, sin []float32) {
	half := len(x) / 2
	for i := range half {
		x1, x2 := x[i], x[i+half]
		x[i] = x1*cos[i] - x2*sin[i]
		x[i+half] = x2*cos[i] + x1*sin[i]
	}
}
