// Package whisper is a pure-Go Whisper decoder: the cross-attention transformer that turns aikit's Whisper encoder output into text (docs/tasks/task-multimodal-support-2026-10.md, S14.4c, G-S14f).
// float32 on the CPU, greedy, one 30 s window; the encoder, the log-mel features and the weights reader are aikit's. It follows transformers' WhisperDecoder.
package whisper

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"

	"github.com/townsendmerino/aikit/embed"
	"github.com/townsendmerino/aikit/linalg"
)

// Config is the decoder's shape, read from the checkpoint's config.json.
type Config struct {
	D         int `json:"d_model"`
	Layers    int `json:"decoder_layers"`
	Heads     int `json:"decoder_attention_heads"`
	FFN       int `json:"decoder_ffn_dim"`
	Vocab     int `json:"vocab_size"`
	MaxTarget int `json:"max_target_positions"`
	MaxSource int `json:"max_source_positions"`
}

func (c Config) validate() error {
	if c.D <= 0 || c.Layers <= 0 || c.Heads <= 0 || c.D%c.Heads != 0 || c.FFN <= 0 || c.Vocab <= 0 || c.MaxTarget <= 0 || c.MaxSource <= 0 {
		return fmt.Errorf("whisper: d_model %d, %d decoder layers, %d heads, ffn %d, vocab %d, %d target and %d source positions", c.D, c.Layers, c.Heads, c.FFN, c.Vocab, c.MaxTarget, c.MaxSource)
	}
	return nil
}

type layer struct {
	saLnW, saLnB                      []float32 // self_attn_layer_norm
	qW, qB, kW, vW, vB, oW, oB        []float32 // self_attn (k_proj has no bias)
	caLnW, caLnB                      []float32 // encoder_attn_layer_norm
	cqW, cqB, ckW, cvW, cvB, coW, coB []float32 // encoder_attn
	fLnW, fLnB                        []float32 // final_layer_norm
	fc1W, fc1B, fc2W, fc2B            []float32
}

// planted defects of decoder_test.go: the zero value is the correct decoder.
const (
	defectNone        = iota
	defectNoScale     // the query not multiplied by head_dim^-0.5
	defectPosShift    // learned positions looked up one position late
	defectCrossFromX  // cross-attention keys and values taken from the decoder's own hidden state
	defectNoFinalNorm // no layer_norm after the last layer
	defectNonCausal   // self-attention sees the future
	defectUntiedHead  // the head is not the token embedding transposed
)

// Decoder is a loaded Whisper decoder. It is immutable after Load; a State carries one request's caches.
type Decoder struct {
	Cfg    Config
	embed  []float32 // [vocab][D]; the head is its transpose
	pos    []float32 // [MaxTarget][D]
	layers []layer
	lnW    []float32
	lnB    []float32
	defect int
}

// Load reads the decoder from a Whisper checkpoint directory: config.json and the tensors under "model.decoder." (or "decoder.", or none) of model.safetensors or a shard index.
func Load(dir string) (*Decoder, error) {
	raw, err := os.ReadFile(filepath.Join(dir, "config.json"))
	if err != nil {
		return nil, err
	}
	var cfg Config
	if err := json.Unmarshal(raw, &cfg); err != nil {
		return nil, fmt.Errorf("whisper: %s/config.json: %w", dir, err)
	}
	if err := cfg.validate(); err != nil {
		return nil, err
	}
	var st *embed.SafetensorsFile
	if idx := filepath.Join(dir, "model.safetensors.index.json"); fileExists(idx) {
		st, err = embed.OpenSafetensorsShardedMmap(idx)
	} else {
		st, err = embed.OpenSafetensorsMmap(filepath.Join(dir, "model.safetensors"))
	}
	if err != nil {
		return nil, err
	}
	prefix := ""
	for _, p := range []string{"model.decoder.", "decoder.", ""} {
		if _, err := st.Tensor(p + "embed_tokens.weight"); err == nil {
			prefix = p
			break
		}
	}
	D := cfg.D
	get := func(name string, want ...int) []float32 {
		if err != nil {
			return nil
		}
		var t []float32
		t, err = st.TensorF32(prefix+name, want...)
		return t
	}
	d := &Decoder{Cfg: cfg}
	d.embed, d.pos = get("embed_tokens.weight", cfg.Vocab, D), get("embed_positions.weight", cfg.MaxTarget, D)
	for i := range cfg.Layers {
		p := fmt.Sprintf("layers.%d.", i)
		d.layers = append(d.layers, layer{
			saLnW: get(p+"self_attn_layer_norm.weight", D), saLnB: get(p+"self_attn_layer_norm.bias", D),
			qW: get(p+"self_attn.q_proj.weight", D, D), qB: get(p+"self_attn.q_proj.bias", D), kW: get(p+"self_attn.k_proj.weight", D, D),
			vW: get(p+"self_attn.v_proj.weight", D, D), vB: get(p+"self_attn.v_proj.bias", D), oW: get(p+"self_attn.out_proj.weight", D, D), oB: get(p+"self_attn.out_proj.bias", D),
			caLnW: get(p+"encoder_attn_layer_norm.weight", D), caLnB: get(p+"encoder_attn_layer_norm.bias", D),
			cqW: get(p+"encoder_attn.q_proj.weight", D, D), cqB: get(p+"encoder_attn.q_proj.bias", D), ckW: get(p+"encoder_attn.k_proj.weight", D, D),
			cvW: get(p+"encoder_attn.v_proj.weight", D, D), cvB: get(p+"encoder_attn.v_proj.bias", D), coW: get(p+"encoder_attn.out_proj.weight", D, D), coB: get(p+"encoder_attn.out_proj.bias", D),
			fLnW: get(p+"final_layer_norm.weight", D), fLnB: get(p+"final_layer_norm.bias", D),
			fc1W: get(p+"fc1.weight", cfg.FFN, D), fc1B: get(p+"fc1.bias", cfg.FFN), fc2W: get(p+"fc2.weight", D, cfg.FFN), fc2B: get(p+"fc2.bias", D),
		})
	}
	d.lnW, d.lnB = get("layer_norm.weight", D), get("layer_norm.bias", D)
	if err != nil {
		return nil, fmt.Errorf("whisper: loading %s: %w", dir, err)
	}
	return d, nil
}

func fileExists(p string) bool { _, err := os.Stat(p); return err == nil }

// State is one request's caches: the cross-attention keys and values (computed once from the encoder output) and the self-attention keys and values of every token so far.
type State struct {
	d              *Decoder
	nSrc           int
	crossK, crossV [][]float32 // per layer, [nSrc][D]
	selfK, selfV   [][]float32 // per layer, [pos][D], grown as tokens arrive
	pos            int
}

// NewState starts a request over the encoder's last hidden state enc ([nSrc][D], aikit's WhisperEncoder.Forward output).
func (d *Decoder) NewState(enc []float32) (*State, error) {
	D := d.Cfg.D
	if len(enc) == 0 || len(enc)%D != 0 {
		return nil, fmt.Errorf("whisper: %d encoder values for d_model %d", len(enc), D)
	}
	nSrc := len(enc) / D
	s := &State{d: d, nSrc: nSrc, selfK: make([][]float32, len(d.layers)), selfV: make([][]float32, len(d.layers))}
	for i := range d.layers {
		l := &d.layers[i]
		src := enc
		if d.defect == defectCrossFromX { // filled per token in Forward
			s.crossK, s.crossV = append(s.crossK, nil), append(s.crossV, nil)
			continue
		}
		s.crossK = append(s.crossK, linear(src, l.ckW, nil, nSrc, D, D))
		s.crossV = append(s.crossV, linear(src, l.cvW, l.cvB, nSrc, D, D))
	}
	return s, nil
}

// Pos is how many tokens the state holds.
func (s *State) Pos() int { return s.pos }

// Clone copies the state so a caller can branch (language detection runs over one token and the prompt then starts from scratch, so a fresh State is the usual choice; this exists for a test).
func (s *State) Clone() *State {
	c := *s
	c.selfK, c.selfV = make([][]float32, len(s.selfK)), make([][]float32, len(s.selfV))
	for i := range s.selfK {
		c.selfK[i], c.selfV[i] = append([]float32(nil), s.selfK[i]...), append([]float32(nil), s.selfV[i]...)
	}
	return &c
}

// Forward runs the next len(tokens) tokens through the decoder at positions pos, pos+1, ..., appends their self-attention keys and values to the state, and returns the logits [len(tokens)][vocab] (the last
// row only when all is false: the head is the largest single product, so a caller that wants the next token alone skips the rest).
func (s *State) Forward(tokens []int, all bool) ([][]float32, error) {
	d := s.d
	c := d.Cfg
	D, m := c.D, len(tokens)
	if m == 0 {
		return nil, fmt.Errorf("whisper: no tokens")
	}
	if s.pos+m > c.MaxTarget {
		return nil, fmt.Errorf("whisper: %d tokens at position %d exceed the %d target positions", m, s.pos, c.MaxTarget)
	}
	x := make([]float32, m*D)
	for i, t := range tokens {
		if t < 0 || t >= c.Vocab {
			return nil, fmt.Errorf("whisper: token id %d outside the %d-token vocabulary", t, c.Vocab)
		}
		p := s.pos + i
		if d.defect == defectPosShift {
			p++
			if p >= c.MaxTarget {
				p = c.MaxTarget - 1
			}
		}
		e, pe := d.embed[t*D:(t+1)*D], d.pos[p*D:(p+1)*D]
		for j := range D {
			x[i*D+j] = e[j] + pe[j]
		}
	}
	hd := D / c.Heads
	scale := float32(1 / math.Sqrt(float64(hd)))
	if d.defect == defectNoScale {
		scale = 1
	}
	for li := range d.layers {
		l := &d.layers[li]
		// causal self-attention over every token so far, this request's included
		h := append([]float32(nil), x...)
		layerNorm(h, l.saLnW, l.saLnB, m, D)
		q := linear(h, l.qW, l.qB, m, D, D)
		k, v := linear(h, l.kW, nil, m, D, D), linear(h, l.vW, l.vB, m, D, D)
		s.selfK[li], s.selfV[li] = append(s.selfK[li], k...), append(s.selfV[li], v...)
		scaleRows(q, scale)
		att := attend(q, s.selfK[li], s.selfV[li], m, s.pos+m, c.Heads, hd, s.pos, d.defect != defectNonCausal)
		add(x, linear(att, l.oW, l.oB, m, D, D))
		// cross-attention over the encoder's frames: no mask
		h = append(h[:0], x...)
		layerNorm(h, l.caLnW, l.caLnB, m, D)
		q = linear(h, l.cqW, l.cqB, m, D, D)
		ck, cv, n := s.crossK[li], s.crossV[li], s.nSrc
		if d.defect == defectCrossFromX {
			ck, cv, n = linear(h, l.ckW, nil, m, D, D), linear(h, l.cvW, l.cvB, m, D, D), m
		}
		scaleRows(q, scale)
		att = attend(q, ck, cv, m, n, c.Heads, hd, 0, false)
		add(x, linear(att, l.coW, l.coB, m, D, D))
		// feed-forward
		h = append(h[:0], x...)
		layerNorm(h, l.fLnW, l.fLnB, m, D)
		f := linear(h, l.fc1W, l.fc1B, m, D, c.FFN)
		for i := range f {
			f[i] = gelu(f[i])
		}
		add(x, linear(f, l.fc2W, l.fc2B, m, c.FFN, D))
	}
	if d.defect != defectNoFinalNorm {
		layerNorm(x, d.lnW, d.lnB, m, D)
	}
	s.pos += m
	rows := x
	nOut := m
	if !all {
		rows, nOut = x[(m-1)*D:], 1
	}
	head := d.embed
	if d.defect == defectUntiedHead {
		head = d.pos // a different matrix of the wrong size: only the shape-compatible rows are read, so the test sees it
		head = append(append([]float32(nil), head...), make([]float32, len(d.embed)-len(head))...)
	}
	flat := make([]float32, nOut*c.Vocab)
	linalg.MatmulBT(rows, head, flat, nOut, D, c.Vocab)
	out := make([][]float32, nOut)
	for i := range out {
		out[i] = flat[i*c.Vocab : (i+1)*c.Vocab]
	}
	return out, nil
}

// attend is multi-head attention of m query rows q [m][D] against n key rows k and value rows v ([n][D]). With causal, query i (at absolute position base+i) sees keys 0..base+i; otherwise every key.
// q arrives already multiplied by head_dim^-0.5 (transformers scales the projection, not the scores).
func attend(q, k, v []float32, m, n, heads, hd int, base int, causal bool) []float32 {
	D := heads * hd
	out := make([]float32, m*D)
	sc := make([]float32, n)
	for h := range heads {
		off := h * hd
		for i := range m {
			lim := n
			if causal {
				lim = base + i + 1
			}
			qi := q[i*D+off : i*D+off+hd]
			mx := float32(math.Inf(-1))
			for j := range lim {
				kj := k[j*D+off : j*D+off+hd]
				var dot float32
				for t := range hd {
					dot += qi[t] * kj[t]
				}
				sc[j] = dot
				if dot > mx {
					mx = dot
				}
			}
			var sum float64
			for j := range lim {
				e := math.Exp(float64(sc[j] - mx))
				sc[j] = float32(e)
				sum += e
			}
			inv := float32(1 / sum)
			o := out[i*D+off : i*D+off+hd]
			for j := range lim {
				w := sc[j] * inv
				vj := v[j*D+off : j*D+off+hd]
				for t := range hd {
					o[t] += w * vj[t]
				}
			}
		}
	}
	return out
}

// linear is x [n][in] times w [out][in] transposed, plus the bias when there is one.
func linear(x, w, b []float32, n, in, out int) []float32 {
	y := make([]float32, n*out)
	linalg.MatmulBT(x, w, y, n, in, out)
	if b != nil {
		for i := range n {
			row := y[i*out : (i+1)*out]
			for j := range row {
				row[j] += b[j]
			}
		}
	}
	return y
}

func scaleRows(q []float32, f float32) {
	if f != 1 {
		for i := range q {
			q[i] *= f
		}
	}
}

func add(x, y []float32) {
	for i := range x {
		x[i] += y[i]
	}
}

// layerNorm normalises each of the n rows of x [n][D] in place (epsilon 1e-5, torch's default).
func layerNorm(x, w, b []float32, n, D int) {
	for i := range n {
		row := x[i*D : (i+1)*D]
		var mean float64
		for _, v := range row {
			mean += float64(v)
		}
		mean /= float64(D)
		var vr float64
		for _, v := range row {
			d := float64(v) - mean
			vr += d * d
		}
		inv := 1 / math.Sqrt(vr/float64(D)+1e-5)
		for j, v := range row {
			row[j] = float32((float64(v)-mean)*inv)*w[j] + b[j]
		}
	}
}

// gelu is the exact (erf) form, the checkpoints' activation_function "gelu".
func gelu(v float32) float32 {
	return float32(0.5 * float64(v) * (1 + math.Erf(float64(v)/math.Sqrt2)))
}
