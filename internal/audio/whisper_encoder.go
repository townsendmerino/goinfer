package audio

import (
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"sync"

	"github.com/townsendmerino/aikit/embed"
	"github.com/townsendmerino/aikit/linalg"
)

// The Whisper audio encoder, float32 on the CPU: transformers 5.15.0's WhisperEncoder, which Voxtral's audio tower also is (VoxtralEncoder, same modules; its projector is separate).
// Input is the [mels][3000] features of WhisperFeatures (the encoder refuses any other length: no attention mask exists); Conv1d(mels -> d, k3, p1) + GELU, Conv1d(d -> d, k3, s2, p1) + GELU,
// the checkpoint's learned positions ADDED to the 1500 frames, pre-LayerNorm layers (biased q/v/out, NO bias on k, GELU MLP, attention over all 1500 positions), a final LayerNorm.

// WhisperEncoderConfig is the encoder's shape from a Whisper config.json.
type WhisperEncoderConfig struct {
	DModel       int `json:"d_model"`
	Layers       int `json:"encoder_layers"`
	Heads        int `json:"encoder_attention_heads"`
	FFN          int `json:"encoder_ffn_dim"`
	MelBins      int `json:"num_mel_bins"`
	MaxPositions int `json:"max_source_positions"`
}

func (c WhisperEncoderConfig) validate() error {
	switch {
	case c.DModel <= 0 || c.Layers <= 0 || c.Heads <= 0 || c.DModel%c.Heads != 0 || c.FFN <= 0:
		return fmt.Errorf("audio: whisper d_model %d, %d layers, %d heads, ffn %d", c.DModel, c.Layers, c.Heads, c.FFN)
	case c.MelBins != 80 && c.MelBins != 128:
		return fmt.Errorf("audio: whisper takes 80 or 128 mel bins, config has %d", c.MelBins)
	case c.MaxPositions != WhisperFrames/2:
		return fmt.Errorf("audio: max_source_positions %d (the encoder is built for %d: 30 s of audio)", c.MaxPositions, WhisperFrames/2)
	}
	return nil
}

type whisperLayer struct {
	ln1W, ln1B, ln2W, ln2B     []float32
	qW, qB, kW, vW, vB, oW, oB []float32
	fc1W, fc1B, fc2W, fc2B     []float32
}

// WhisperEncoder is the loaded encoder.
type WhisperEncoder struct {
	Cfg            WhisperEncoderConfig
	conv1W, conv1B []float32
	conv2W, conv2B []float32
	pos            []float32 // [1500][d]
	layers         []whisperLayer
	lnW, lnB       []float32
	defect         int
	device         WhisperDevice
	// st is the checkpoint the weights came from. An F32 tensor is a view into its mapping, so the encoder holds the file until Close, the mapping's only release (see
	// embed.SafetensorsFile.Close).
	st     *embed.SafetensorsFile
	closed bool
}

// errEncoderClosed is what an encoder's forward returns after Close.
var errEncoderClosed = errors.New("audio: encoder is closed")

// Close releases the checkpoint mapping the weights may alias. Forward returns an error afterwards, and slices taken from Weights must no longer be read; a device set with
// SetDevice stays the caller's to close. Calling Close again does nothing.
func (e *WhisperEncoder) Close() error {
	if e.closed {
		return nil
	}
	st := e.st
	*e = WhisperEncoder{Cfg: e.Cfg, closed: true} // drop every view into the mapping before unmapping it
	if st == nil {
		return nil
	}
	return st.Close()
}

// Planted defects for G-S14d's red runs. Zero is the shipped computation.
const (
	whisperEncDefectNone = iota
	whisperEncDefectNoPos
	whisperEncDefectPosShifted
	whisperEncDefectNoScale
	whisperEncDefectTanhGELU
	whisperEncDefectNoFinalNorm
	whisperEncDefectWindowed
	whisperEncDefectPostNorm
)

// WhisperEncDefectCountForTest is the number of planted defects SetDefectForTest accepts (1..N).
const WhisperEncDefectCountForTest = whisperEncDefectPostNorm

// SetDefectForTest plants one defect (0 clears it). Never call it in production.
func (e *WhisperEncoder) SetDefectForTest(d int) { e.defect = d }

// parseWhisperEncoderConfig reads an encoder config in either spelling: Whisper's own (d_model, encoder_layers, encoder_attention_heads, encoder_ffn_dim) or the one transformers
// writes for Voxtral's audio tower (hidden_size, num_hidden_layers, num_attention_heads, intermediate_size: the FFN width, which Voxtral's projector then also reads as its input width).
func parseWhisperEncoderConfig(raw []byte) (WhisperEncoderConfig, error) {
	var c struct {
		WhisperEncoderConfig
		HiddenSize        int `json:"hidden_size"`
		NumHiddenLayers   int `json:"num_hidden_layers"`
		NumAttentionHeads int `json:"num_attention_heads"`
		IntermediateSize  int `json:"intermediate_size"`
	}
	if err := json.Unmarshal(raw, &c); err != nil {
		return WhisperEncoderConfig{}, err
	}
	out := c.WhisperEncoderConfig
	if out.DModel == 0 {
		out.DModel = c.HiddenSize
	}
	if out.Layers == 0 {
		out.Layers = c.NumHiddenLayers
	}
	if out.Heads == 0 {
		out.Heads = c.NumAttentionHeads
	}
	if out.FFN == 0 {
		out.FFN = c.IntermediateSize
	}
	return out, nil
}

// openSafetensorsDir opens a checkpoint directory's weights: a shard index when there is one, else model.safetensors.
func openSafetensorsDir(dir string) (*embed.SafetensorsFile, error) {
	var st *embed.SafetensorsFile
	var err error
	if idx := filepath.Join(dir, "model.safetensors.index.json"); fileExists(idx) {
		st, err = embed.OpenSafetensorsShardedMmap(idx)
	} else {
		st, err = embed.OpenSafetensorsMmap(filepath.Join(dir, "model.safetensors"))
	}
	if err == nil && testHookOpenedCheckpoint != nil {
		testHookOpenedCheckpoint(st)
	}
	return st, err
}

// testHookOpenedCheckpoint, set only by tests, sees each file openSafetensorsDir opens (the lifetime test holds a weak pointer to it).
var testHookOpenedCheckpoint func(*embed.SafetensorsFile)

// LoadWhisperEncoder reads the encoder from a Whisper checkpoint directory (config.json at the top, or under audio_config, in Whisper's key names or Voxtral's; model.safetensors or a
// shard index with the tensors under "model.encoder.", "encoder.", "audio_tower." (Voxtral) or no prefix).
func LoadWhisperEncoder(dir string) (*WhisperEncoder, error) {
	raw, err := os.ReadFile(filepath.Join(dir, "config.json"))
	if err != nil {
		return nil, err
	}
	var top struct {
		Audio json.RawMessage `json:"audio_config"`
	}
	if err := json.Unmarshal(raw, &top); err != nil {
		return nil, fmt.Errorf("audio: %s/config.json: %w", dir, err)
	}
	cfgRaw := raw
	if len(top.Audio) > 0 && string(top.Audio) != "null" {
		cfgRaw = top.Audio
	}
	cfg, err := parseWhisperEncoderConfig(cfgRaw)
	if err != nil {
		return nil, fmt.Errorf("audio: %s/config.json: %w", dir, err)
	}
	e := &WhisperEncoder{Cfg: cfg}
	if err := e.Cfg.validate(); err != nil {
		return nil, err
	}
	st, err := openSafetensorsDir(dir)
	if err != nil {
		return nil, err
	}
	prefix := ""
	for _, p := range []string{"model.encoder.", "encoder.", "audio_tower.", ""} {
		if _, err := st.Tensor(p + "conv1.weight"); err == nil {
			prefix = p
			break
		}
	}
	c := e.Cfg
	D := c.DModel
	get := func(name string, want ...int) []float32 {
		if err != nil {
			return nil
		}
		var t []float32
		t, err = st.TensorF32(prefix+name, want...)
		return t
	}
	e.conv1W, e.conv1B = get("conv1.weight", D, c.MelBins, 3), get("conv1.bias", D)
	e.conv2W, e.conv2B = get("conv2.weight", D, D, 3), get("conv2.bias", D)
	e.pos = get("embed_positions.weight", c.MaxPositions, D)
	for i := range c.Layers {
		p := fmt.Sprintf("layers.%d.", i)
		e.layers = append(e.layers, whisperLayer{
			ln1W: get(p+"self_attn_layer_norm.weight", D), ln1B: get(p+"self_attn_layer_norm.bias", D),
			qW: get(p+"self_attn.q_proj.weight", D, D), qB: get(p+"self_attn.q_proj.bias", D), kW: get(p+"self_attn.k_proj.weight", D, D),
			vW: get(p+"self_attn.v_proj.weight", D, D), vB: get(p+"self_attn.v_proj.bias", D), oW: get(p+"self_attn.out_proj.weight", D, D), oB: get(p+"self_attn.out_proj.bias", D),
			ln2W: get(p+"final_layer_norm.weight", D), ln2B: get(p+"final_layer_norm.bias", D),
			fc1W: get(p+"fc1.weight", c.FFN, D), fc1B: get(p+"fc1.bias", c.FFN), fc2W: get(p+"fc2.weight", D, c.FFN), fc2B: get(p+"fc2.bias", D),
		})
	}
	e.lnW, e.lnB = get("layer_norm.weight", D), get("layer_norm.bias", D)
	if err != nil {
		st.Close()
		return nil, fmt.Errorf("audio: loading %s: %w", dir, err)
	}
	e.st = st
	return e, nil
}

func (e *WhisperEncoder) gelu(v float32) float32 {
	if e.defect == whisperEncDefectTanhGELU {
		x := float64(v)
		return float32(0.5 * x * (1 + math.Tanh(math.Sqrt(2/math.Pi)*(x+0.044715*x*x*x))))
	}
	return float32(0.5 * float64(v) * (1 + math.Erf(float64(v)/math.Sqrt2)))
}

// geluInto applies the activation to buf in place: aikit's exact-GELU contract kernel (HF's "gelu", x times Phi(x)), spread over the cores in chunks, unless the planted
// tanh defect is on, which keeps its scalar form.
func (e *WhisperEncoder) geluInto(buf []float32) {
	if e.defect == whisperEncDefectTanhGELU {
		for i, v := range buf {
			buf[i] = e.gelu(v)
		}
		return
	}
	const chunk = 1 << 14
	parallelRows((len(buf)+chunk-1)/chunk, func(c0, c1 int) {
		lo, hi := c0*chunk, min(c1*chunk, len(buf))
		linalg.GELUContractInto(buf[lo:hi], buf[lo:hi])
	})
}

// parallelRows runs fn over [0,n) in contiguous blocks, one per core. The blocks are disjoint, so the result does not depend on the split. Small n runs inline.
func parallelRows(n int, fn func(lo, hi int)) {
	w := min(runtime.GOMAXPROCS(0), n)
	if w <= 1 {
		fn(0, n)
		return
	}
	var wg sync.WaitGroup
	for k := range w {
		lo, hi := k*n/w, (k+1)*n/w
		wg.Add(1)
		go func() {
			defer wg.Done()
			fn(lo, hi)
		}()
	}
	wg.Wait()
}

// Forward runs the encoder over feats [mels][3000] (WhisperFeatures' layout) and returns the [1500][DModel] last hidden state.
func (e *WhisperEncoder) Forward(feats []float32) ([]float32, error) {
	if e.closed {
		return nil, errEncoderClosed
	}
	c := e.Cfg
	const T = WhisperFrames
	if len(feats) != c.MelBins*T {
		return nil, fmt.Errorf("audio: %d feature values, want %d mels x %d frames", len(feats), c.MelBins, T)
	}
	if e.device != nil && e.defect == whisperEncDefectNone {
		return e.device.Forward(feats)
	}
	D := c.DModel
	// conv1: [mels][3000] -> [3000][D], k3 p1 s1
	col := make([]float32, T*c.MelBins*3)
	for t := range T {
		dst := col[t*c.MelBins*3 : (t+1)*c.MelBins*3]
		for ch := range c.MelBins {
			for k := range 3 {
				if ti := t - 1 + k; ti >= 0 && ti < T {
					dst[ch*3+k] = feats[ch*T+ti]
				}
			}
		}
	}
	x := make([]float32, T*D)
	linalg.MatmulBT(col, e.conv1W, x, T, c.MelBins*3, D)
	for t := range T {
		row := x[t*D : (t+1)*D]
		for j := range row {
			row[j] += e.conv1B[j]
		}
	}
	e.geluInto(x)
	// conv2: [3000][D] -> [1500][D], k3 p1 s2
	N := c.MaxPositions
	col2 := make([]float32, N*D*3)
	for t := range N {
		dst := col2[t*D*3 : (t+1)*D*3]
		for k := range 3 {
			ti := 2*t - 1 + k
			if ti < 0 || ti >= T {
				continue
			}
			src := x[ti*D : (ti+1)*D]
			for ch, v := range src {
				dst[ch*3+k] = v
			}
		}
	}
	h := make([]float32, N*D)
	linalg.MatmulBT(col2, e.conv2W, h, N, D*3, D)
	for t := range N {
		row := h[t*D : (t+1)*D]
		for j := range row {
			row[j] += e.conv2B[j]
		}
	}
	e.geluInto(h)
	switch e.defect {
	case whisperEncDefectNoPos:
	case whisperEncDefectPosShifted:
		for t := range N {
			p := e.pos[((t+1)%N)*D : ((t+1)%N+1)*D]
			for j := range D {
				h[t*D+j] += p[j]
			}
		}
	default:
		for i := range h {
			h[i] += e.pos[i]
		}
	}
	window := N
	if e.defect == whisperEncDefectWindowed {
		window = 400 // 8 s of 20 ms frames
	}
	for i := range e.layers {
		e.layer(&e.layers[i], h, N, window)
	}
	if e.defect != whisperEncDefectNoFinalNorm {
		layerNorm(h, e.lnW, e.lnB, N, D)
	}
	return h, nil
}

// layer runs one encoder layer in place. Attention is one GEMM per head: scores = Q_h K_h^T, a row softmax, then scores x V_h (V transposed so both products are the blocked A x B^T matmul).
func (e *WhisperEncoder) layer(ly *whisperLayer, h []float32, n, window int) {
	c := e.Cfg
	D := c.DModel
	post := e.defect == whisperEncDefectPostNorm
	var x []float32
	if post {
		x = h
	} else {
		x = append([]float32(nil), h...)
		layerNorm(x, ly.ln1W, ly.ln1B, n, D)
	}
	q := linear(x, ly.qW, ly.qB, n, D, D)
	// (A bias on k, the defect one might plant here, is a mathematical no-op: it adds the same q.b to every score of a query row and softmax is shift-invariant. That is why Whisper's k_proj has none.)
	k, v := linear(x, ly.kW, make([]float32, D), n, D, D), linear(x, ly.vW, ly.vB, n, D, D)
	hd := D / c.Heads
	scale := float32(1 / math.Sqrt(float64(hd)))
	if e.defect == whisperEncDefectNoScale {
		scale = 1
	}
	att := make([]float32, n*D)
	qh, kh, vt := make([]float32, n*hd), make([]float32, n*hd), make([]float32, hd*n)
	scBuf := make([]float32, min(window, n)*min(window, n)) // one score block, reused by every head and window
	for hh := range c.Heads {
		off := hh * hd
		for i := range n {
			copy(qh[i*hd:(i+1)*hd], q[i*D+off:i*D+off+hd])
			copy(kh[i*hd:(i+1)*hd], k[i*D+off:i*D+off+hd])
			for d := range hd {
				vt[d*n+i] = v[i*D+off+d]
			}
		}
		for lo := 0; lo < n; lo += window {
			hi := min(lo+window, n)
			m := hi - lo
			sc := scBuf[:m*m]
			linalg.MatmulBT(qh[lo*hd:hi*hd], kh[lo*hd:hi*hd], sc, m, hd, m)
			// The scale and the row softmax in one contract kernel (SoftmaxRowScaledContractInto), the rows independent and spread over the cores.
			parallelRows(m, func(r0, r1 int) {
				for i := r0; i < r1; i++ {
					row := sc[i*m : (i+1)*m]
					linalg.SoftmaxRowScaledContractInto(row, row, scale)
				}
			})
			// out[i][d] = sum_j sc[i][j] * v[lo+j][d]: with vt [hd][n], take the window's columns
			vtw := make([]float32, hd*m)
			for d := range hd {
				copy(vtw[d*m:(d+1)*m], vt[d*n+lo:d*n+hi])
			}
			ctx := make([]float32, m*hd)
			linalg.MatmulBT(sc, vtw, ctx, m, m, hd)
			for i := range m {
				copy(att[(lo+i)*D+off:(lo+i)*D+off+hd], ctx[i*hd:(i+1)*hd])
			}
		}
	}
	a := linear(att, ly.oW, ly.oB, n, D, D)
	for i := range h {
		h[i] += a[i]
	}
	if post {
		layerNorm(h, ly.ln1W, ly.ln1B, n, D)
	}
	y := h
	if !post {
		y = append([]float32(nil), h...)
		layerNorm(y, ly.ln2W, ly.ln2B, n, D)
	}
	f := linear(y, ly.fc1W, ly.fc1B, n, D, c.FFN)
	e.geluInto(f)
	g := linear(f, ly.fc2W, ly.fc2B, n, c.FFN, D)
	for i := range h {
		h[i] += g[i]
	}
	if post {
		layerNorm(h, ly.ln2W, ly.ln2B, n, D)
	}
}
