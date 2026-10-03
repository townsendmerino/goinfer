package clef

// The joint schema head (JointSchemaHead in the checkpoint's joint_schema_model.py, read in full for
// docs/measurements/decisions-d10-clef-2026-10-02.md): 121.8M parameters that read the backbone's post-final-norm hidden state at every
// position and return one logit per option of every question, jointly.
//
// This file is the head only. Its input is the backbone's last_hidden_state (decoder.Model.PromptHiddenAll, D11), the encoder's spans
// (Encode), and the lm_head rows of the option tokens. It runs in f32; the checkpoint stores bf16 and each weight is widened on load.
//
// What is ported and where the reference's structure shows through:
//   - nn.LayerNorm: eps 1e-5, biased variance, affine.
//   - nn.MultiheadAttention (batch_first, no mask): a fused in_proj_weight [3w,w] whose row blocks are q, k, v; 1/sqrt(head_dim) scaling;
//     softmax over keys per head; out_proj with bias.
//   - EvidenceRoutingLayer: pre-norm, and the MEMORY is normalised (memory_norm) before it is used as key and value.
//   - TransformerDecoderLayer(norm_first=True, activation="gelu"): the cross-attention memory is NOT normalised, unlike the routing
//     layers. x += self_attn(norm1(x)); x += cross_attn(norm2(x), memory); x += linear2(gelu(linear1(norm3(x)))).
//   - GELU is the exact erf form.
//   - The two logit scales are clamped at ln(100) before exp; F.normalize uses eps 1e-12 and F.cosine_similarity eps 1e-8.

import (
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
	"sync"

	"github.com/townsendmerino/aikit/embed"
	"github.com/townsendmerino/aikit/linalg"
)

// HeadConfig is joint_head_config.json. Every key is required and no other key is accepted.
type HeadConfig struct {
	HiddenSize    int `json:"hidden_size"`
	Width         int `json:"width"`
	RoutingLayers int `json:"routing_layers"`
	Layers        int `json:"layers"`
	Heads         int `json:"heads"`
	Feedforward   int `json:"feedforward"`
}

type layerNorm struct{ w, b []float32 }

type mha struct {
	inW, inB   []float32 // [3w, w], [3w]
	outW, outB []float32 // [w, w], [w]
}

type routingLayer struct {
	queryNorm, memoryNorm, ffNorm layerNorm
	attn                          mha
	ff1W, ff1B, ff2W, ff2B        []float32
}

type decoderLayer struct {
	norm1, norm2, norm3    layerNorm
	selfAttn, crossAttn    mha
	lin1W, lin1B, lin2W, b []float32 // b is linear2's bias
}

// Head is a loaded joint head.
type Head struct {
	Cfg HeadConfig

	hiddenNorm                                                                    layerNorm
	memoryProj, questionProj, optQuestionProj, globalProj, optCtxProj, optLexProj []float32 // [w, hidden]
	typeEmb                                                                       []float32 // [3, w]
	routing                                                                       []routingLayer
	optSummaryNorm, fieldNorm, optionNorm                                         layerNorm
	layers                                                                        []decoderLayer
	res0W, res0B, res3W                                                           []float32 // residual_scorer: [w, 4w], [w], [1, w]
	res3B                                                                         float32
	priorScale, jointScale, residualGate                                          float32
}

// LoadHead reads dir/joint_head_config.json and dir/joint_head.safetensors. It refuses a config with a missing or unknown key, a tensor
// the layout does not name, a missing tensor, and a tensor of the wrong shape: the module is version one of a custom file, and a
// head that loads with a silently different shape is worse than one that does not load.
func LoadHead(dir string) (*Head, error) {
	raw, err := os.ReadFile(filepath.Join(dir, "joint_head_config.json"))
	if err != nil {
		return nil, err
	}
	var cfg HeadConfig
	dec := json.NewDecoder(strings.NewReader(string(raw)))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&cfg); err != nil {
		return nil, fmt.Errorf("clef: joint_head_config.json: %w", err)
	}
	var keys map[string]json.RawMessage
	if err := json.Unmarshal(raw, &keys); err != nil || len(keys) != 6 {
		return nil, fmt.Errorf("clef: joint_head_config.json must carry exactly hidden_size, width, routing_layers, layers, heads and feedforward")
	}
	if cfg.HiddenSize <= 0 || cfg.Width <= 0 || cfg.RoutingLayers <= 0 || cfg.Layers <= 0 || cfg.Heads <= 0 || cfg.Feedforward <= 0 || cfg.Width%cfg.Heads != 0 {
		return nil, fmt.Errorf("clef: joint_head_config.json has a non-positive value or a width not divisible by heads: %+v", cfg)
	}
	st, err := embed.OpenSafetensorsMmap(filepath.Join(dir, "joint_head.safetensors"))
	if err != nil {
		return nil, err
	}
	defer st.Close()
	return buildHead(cfg, st)
}

// tensorReader tracks which tensors a layout consumed, so the leftovers can be refused.
type tensorReader struct {
	st   *embed.SafetensorsFile
	used map[string]bool
	err  error
}

func (r *tensorReader) get(name string, shape ...int) []float32 {
	if r.err != nil {
		return nil
	}
	t, err := r.st.Tensor(name)
	if err != nil {
		r.err = fmt.Errorf("clef: joint_head.safetensors has no tensor %q", name)
		return nil
	}
	if fmt.Sprint(t.Shape) != fmt.Sprint(shape) {
		r.err = fmt.Errorf("clef: tensor %q has shape %v, the layout wants %v", name, t.Shape, shape)
		return nil
	}
	data, err := r.st.TensorF32(name)
	if err != nil {
		r.err = fmt.Errorf("clef: tensor %q: %w", name, err)
		return nil
	}
	r.used[name] = true
	return data
}

func (r *tensorReader) ln(prefix string, n int) layerNorm {
	return layerNorm{r.get(prefix+".weight", n), r.get(prefix+".bias", n)}
}

func (r *tensorReader) mha(prefix string, w int) mha {
	return mha{
		inW: r.get(prefix+".in_proj_weight", 3*w, w), inB: r.get(prefix+".in_proj_bias", 3*w),
		outW: r.get(prefix+".out_proj.weight", w, w), outB: r.get(prefix+".out_proj.bias", w),
	}
}

func buildHead(cfg HeadConfig, st *embed.SafetensorsFile) (*Head, error) {
	r := &tensorReader{st: st, used: map[string]bool{}}
	w, hs, ff := cfg.Width, cfg.HiddenSize, cfg.Feedforward
	h := &Head{Cfg: cfg}
	h.hiddenNorm = r.ln("hidden_norm", hs)
	h.memoryProj = r.get("memory_projection.weight", w, hs)
	h.questionProj = r.get("question_projection.weight", w, hs)
	h.optQuestionProj = r.get("option_question_projection.weight", w, hs)
	h.globalProj = r.get("global_projection.weight", w, hs)
	h.optCtxProj = r.get("option_context_projection.weight", w, hs)
	h.optLexProj = r.get("option_lexical_projection.weight", w, hs)
	h.typeEmb = r.get("type_embedding.weight", 3, w)
	for i := 0; i < cfg.RoutingLayers; i++ {
		p := fmt.Sprintf("evidence_layers.%d", i)
		h.routing = append(h.routing, routingLayer{
			queryNorm: r.ln(p+".query_norm", w), memoryNorm: r.ln(p+".memory_norm", w), ffNorm: r.ln(p+".feedforward_norm", w),
			attn: r.mha(p+".attention", w),
			ff1W: r.get(p+".feedforward.0.weight", ff, w), ff1B: r.get(p+".feedforward.0.bias", ff),
			ff2W: r.get(p+".feedforward.3.weight", w, ff), ff2B: r.get(p+".feedforward.3.bias", w),
		})
	}
	h.optSummaryNorm = r.ln("option_summary_norm", w)
	for i := 0; i < cfg.Layers; i++ {
		p := fmt.Sprintf("layers.%d", i)
		h.layers = append(h.layers, decoderLayer{
			norm1: r.ln(p+".norm1", w), norm2: r.ln(p+".norm2", w), norm3: r.ln(p+".norm3", w),
			selfAttn: r.mha(p+".self_attn", w), crossAttn: r.mha(p+".multihead_attn", w),
			lin1W: r.get(p+".linear1.weight", ff, w), lin1B: r.get(p+".linear1.bias", ff),
			lin2W: r.get(p+".linear2.weight", w, ff), b: r.get(p+".linear2.bias", w),
		})
	}
	h.fieldNorm = r.ln("field_norm", w)
	h.optionNorm = r.ln("option_norm", w)
	h.res0W, h.res0B = r.get("residual_scorer.0.weight", w, 4*w), r.get("residual_scorer.0.bias", w)
	h.res3W = r.get("residual_scorer.3.weight", 1, w)
	b := r.get("residual_scorer.3.bias", 1)
	ps, js, rg := r.get("prior_logit_scale"), r.get("joint_logit_scale"), r.get("residual_gate")
	if r.err != nil {
		return nil, r.err
	}
	h.res3B, h.priorScale, h.jointScale, h.residualGate = b[0], ps[0], js[0], rg[0]
	var extra []string
	for _, name := range st.Names() {
		if !r.used[name] {
			extra = append(extra, name)
		}
	}
	if len(extra) > 0 {
		sort.Strings(extra)
		return nil, fmt.Errorf("clef: joint_head.safetensors carries %d tensor(s) this layout does not use (%s ...); refusing a head that differs from the one this port was verified against", len(extra), extra[0])
	}
	return h, nil
}

// LMHeadRows returns the raw lm_head row (the output-embedding row, hidden_size wide, NOT normalised) of a token id.
type LMHeadRows func(id int) ([]float32, error)

// Forward returns, per question, one logit per option in the encoder's option order. hidden is the backbone's last_hidden_state after
// its final norm, [L][hidden_size], and ids the encoder's input_ids (len L). Softmax each question's logits for the probabilities.
func (h *Head) Forward(hidden [][]float32, ids []int, qs []Question, lm LMHeadRows) ([][]float32, error) {
	cfg := h.Cfg
	L, w, hs := len(hidden), cfg.Width, cfg.HiddenSize
	if L == 0 || L != len(ids) {
		return nil, fmt.Errorf("clef: %d hidden rows for %d input ids", L, len(ids))
	}
	if len(qs) == 0 {
		return nil, errors.New("clef: no questions")
	}
	nOpt := 0
	for qi, q := range qs {
		if q.Type < 0 || q.Type > 2 || len(q.OptionIDs) == 0 || len(q.OptionSpans) != len(q.OptionIDs) {
			return nil, fmt.Errorf("clef: question %d is malformed", qi)
		}
		for _, sp := range append([][2]int{q.QuestionSpan}, q.OptionSpans...) {
			if sp[0] < 0 || sp[1] > L || sp[0] >= sp[1] {
				return nil, fmt.Errorf("clef: question %d has a span %v outside the %d-token sequence or empty", qi, sp, L)
			}
		}
		nOpt += len(q.OptionIDs)
	}
	flat := make([]float32, L*hs)
	for i, row := range hidden {
		if len(row) != hs {
			return nil, fmt.Errorf("clef: hidden row %d is %d wide, want %d", i, len(row), hs)
		}
		copy(flat[i*hs:], row)
	}

	// hidden_norm over every position, then memory = memory_projection(h).
	hn := make([]float32, L*hs)
	layerNormRows(hn, flat, h.hiddenNorm, L, hs)
	memory := matmulBT(hn, h.memoryProj, L, hs, w)
	global := hn[(L-1)*hs : L*hs]

	meanSpan := func(sp [2]int) []float32 {
		out := make([]float32, hs)
		for t := sp[0]; t < sp[1]; t++ {
			for j, v := range hn[t*hs : (t+1)*hs] {
				out[j] += v
			}
		}
		inv := 1 / float32(sp[1]-sp[0])
		for j := range out {
			out[j] *= inv
		}
		return out
	}
	qv := make([]float32, len(qs)*hs) // [Q, hs]
	ctx := make([]float32, nOpt*hs)   // [N, hs] the mean hidden over each option span
	lex := make([]float32, nOpt*hs)   // [N, hs] the mean lm_head row over each option's tokens
	qOf := make([]int, nOpt)          // option -> question
	first := make([]int, len(qs))     // question -> its first option
	rowCache := map[int][]float32{}
	n := 0
	for qi, q := range qs {
		copy(qv[qi*hs:], meanSpan(q.QuestionSpan))
		first[qi] = n
		for _, sp := range q.OptionSpans {
			copy(ctx[n*hs:], meanSpan(sp))
			dst := lex[n*hs : (n+1)*hs]
			for t := sp[0]; t < sp[1]; t++ {
				row, ok := rowCache[ids[t]]
				if !ok {
					var err error
					if row, err = lm(ids[t]); err != nil {
						return nil, err
					}
					if len(row) != hs {
						return nil, fmt.Errorf("clef: lm_head row for token %d is %d wide, want %d", ids[t], len(row), hs)
					}
					rowCache[ids[t]] = row
				}
				for j, v := range row {
					dst[j] += v
				}
			}
			inv := 1 / float32(sp[1]-sp[0])
			for j := range dst {
				dst[j] *= inv
			}
			qOf[n] = qi
			n++
		}
	}

	// option_query = ctx-projection + lexical-projection + question-projection(qv[q]).
	optQ := matmulBT(ctx, h.optCtxProj, nOpt, hs, w)
	addInPlace(optQ, matmulBT(lex, h.optLexProj, nOpt, hs, w))
	oq := matmulBT(qv, h.optQuestionProj, len(qs), hs, w)
	for i := 0; i < nOpt; i++ {
		addInPlace(optQ[i*w:(i+1)*w], oq[qOf[i]*w:(qOf[i]+1)*w])
	}

	// Evidence routing: pre-norm cross-attention over the (normalised) memory, then a feed-forward.
	x := optQ
	for _, l := range h.routing {
		nq := make([]float32, nOpt*w)
		layerNormRows(nq, x, l.queryNorm, nOpt, w)
		nm := make([]float32, L*w)
		layerNormRows(nm, memory, l.memoryNorm, L, w)
		addInPlace(x, h.attend(l.attn, nq, nOpt, nm, L))
		f := make([]float32, nOpt*w)
		layerNormRows(f, x, l.ffNorm, nOpt, w)
		addInPlace(x, h.feedForward(f, nOpt, l.ff1W, l.ff1B, l.ff2W, l.ff2B))
	}
	routed := x // [N, w]

	// Per question: softmax over its options of routed.field/sqrt(w), and the weighted sum of the routed rows.
	baseFields := matmulBT(qv, h.questionProj, len(qs), hs, w)
	summaries := make([]float32, len(qs)*w)
	for qi, q := range qs {
		no := len(q.OptionIDs)
		field := baseFields[qi*w : (qi+1)*w]
		sc := make([]float64, no)
		mx := math.Inf(-1)
		for o := 0; o < no; o++ {
			sc[o] = dot64(routed[(first[qi]+o)*w:(first[qi]+o+1)*w], field) / math.Sqrt(float64(w))
			mx = math.Max(mx, sc[o])
		}
		var z float64
		for o := range sc {
			sc[o] = math.Exp(sc[o] - mx)
			z += sc[o]
		}
		sum := summaries[qi*w : (qi+1)*w]
		for o := 0; o < no; o++ {
			wt := float32(sc[o] / z)
			for j, v := range routed[(first[qi]+o)*w : (first[qi]+o+1)*w] {
				sum[j] += wt * v
			}
		}
	}
	nsum := make([]float32, len(qs)*w)
	layerNormRows(nsum, summaries, h.optSummaryNorm, len(qs), w)
	gp := matmulBT(global, h.globalProj, 1, hs, w)
	fields := make([]float32, len(qs)*w)
	for qi, q := range qs {
		f := fields[qi*w : (qi+1)*w]
		for j := range f {
			f[j] = baseFields[qi*w+j] + nsum[qi*w+j] + gp[j] + h.typeEmb[q.Type*w+j]
		}
	}

	// Decoder layers: self-attention over the questions, cross-attention over the UN-normalised memory, feed-forward.
	nQ := len(qs)
	for _, l := range h.layers {
		t := make([]float32, nQ*w)
		layerNormRows(t, fields, l.norm1, nQ, w)
		addInPlace(fields, h.attend(l.selfAttn, t, nQ, t, nQ))
		layerNormRows(t, fields, l.norm2, nQ, w)
		addInPlace(fields, h.attend(l.crossAttn, t, nQ, memory, L))
		layerNormRows(t, fields, l.norm3, nQ, w)
		addInPlace(fields, h.feedForward(t, nQ, l.lin1W, l.lin1B, l.lin2W, l.b))
	}
	fn := make([]float32, nQ*w)
	layerNormRows(fn, fields, h.fieldNorm, nQ, w)
	on := make([]float32, nOpt*w)
	layerNormRows(on, routed, h.optionNorm, nOpt, w)

	// Logits.
	logCap := math.Log(100)
	priorScale := math.Exp(math.Min(float64(h.priorScale), logCap))
	jointScale := math.Exp(math.Min(float64(h.jointScale), logCap))
	gate := 1 / (1 + math.Exp(-float64(h.residualGate)))
	out := make([][]float32, nQ)
	for qi, q := range qs {
		anchor := make([]float64, hs)
		for j := 0; j < hs; j++ {
			anchor[j] = float64(qv[qi*hs+j] + global[j])
		}
		normalize64(anchor)
		field := fn[qi*w : (qi+1)*w]
		out[qi] = make([]float32, len(q.OptionIDs))
		for o := range q.OptionIDs {
			oi := first[qi] + o
			lexA := make([]float64, hs)
			for j, v := range lex[oi*hs : (oi+1)*hs] {
				lexA[j] = float64(v)
			}
			normalize64(lexA)
			var prior float64
			for j := range lexA {
				prior += lexA[j] * anchor[j]
			}
			prior *= priorScale
			opt := on[oi*w : (oi+1)*w]
			cosine := cosineSimilarity(field, opt)
			feat := make([]float32, 4*w)
			for j := 0; j < w; j++ {
				feat[j], feat[w+j] = field[j], opt[j]
				feat[2*w+j] = field[j] * opt[j]
				feat[3*w+j] = float32(math.Abs(float64(field[j] - opt[j])))
			}
			hid := matmulBT(feat, h.res0W, 1, 4*w, w)
			for j := range hid {
				hid[j] = geluExact(hid[j] + h.res0B[j])
			}
			residual := dot64(hid, h.res3W) + float64(h.res3B)
			joint := jointScale*cosine + residual
			out[qi][o] = float32(prior + gate*joint)
		}
	}
	return out, nil
}

// Softmax is the per-question softmax the reference applies to the logits, in float64.
func Softmax(logits []float32) []float64 {
	mx := math.Inf(-1)
	for _, v := range logits {
		mx = math.Max(mx, float64(v))
	}
	p := make([]float64, len(logits))
	var z float64
	for i, v := range logits {
		p[i] = math.Exp(float64(v) - mx)
		z += p[i]
	}
	for i := range p {
		p[i] /= z
	}
	return p
}

// attend is nn.MultiheadAttention(q, kv, kv) with no mask and need_weights=False: queries [nq, w] against keys and values [nk, w].
func (h *Head) attend(m mha, q []float32, nq int, kv []float32, nk int) []float32 {
	w, heads := h.Cfg.Width, h.Cfg.Heads
	hd := w / heads
	Q := matmulBT(q, m.inW[:w*w], nq, w, w)
	K := matmulBT(kv, m.inW[w*w:2*w*w], nk, w, w)
	V := matmulBT(kv, m.inW[2*w*w:], nk, w, w)
	addBias(Q, m.inB[:w], nq, w)
	addBias(K, m.inB[w:2*w], nk, w)
	addBias(V, m.inB[2*w:], nk, w)
	ctx := make([]float32, nq*w)
	scale := 1 / math.Sqrt(float64(hd))
	sc := make([]float64, nk)
	for i := 0; i < nq; i++ {
		for hh := 0; hh < heads; hh++ {
			qv := Q[i*w+hh*hd : i*w+(hh+1)*hd]
			mx := math.Inf(-1)
			for j := 0; j < nk; j++ {
				sc[j] = dot64(qv, K[j*w+hh*hd:j*w+(hh+1)*hd]) * scale
				mx = math.Max(mx, sc[j])
			}
			var z float64
			for j := range sc {
				sc[j] = math.Exp(sc[j] - mx)
				z += sc[j]
			}
			dst := ctx[i*w+hh*hd : i*w+(hh+1)*hd]
			for j := 0; j < nk; j++ {
				wt := float32(sc[j] / z)
				for d, v := range V[j*w+hh*hd : j*w+(hh+1)*hd] {
					dst[d] += wt * v
				}
			}
		}
	}
	out := matmulBT(ctx, m.outW, nq, w, w)
	addBias(out, m.outB, nq, w)
	return out
}

func (h *Head) feedForward(x []float32, n int, w1, b1, w2, b2 []float32) []float32 {
	w, ff := h.Cfg.Width, h.Cfg.Feedforward
	a := matmulBT(x, w1, n, w, ff)
	addBias(a, b1, n, ff)
	for i, v := range a {
		a[i] = geluExact(v)
	}
	out := matmulBT(a, w2, n, ff, w)
	addBias(out, b2, n, w)
	return out
}

// matmulBT is a[M,K] . b[N,K]^T, with the rows split across the cores (linalg.MatmulBTInto is serial by design).
func matmulBT(a, b []float32, M, K, N int) []float32 {
	dst := make([]float32, M*N)
	workers := min(runtime.GOMAXPROCS(0), max(1, M/8))
	if workers <= 1 {
		linalg.MatmulBTInto(dst, a, b, M, K, N)
		return dst
	}
	var wg sync.WaitGroup
	per := (M + workers - 1) / workers
	for lo := 0; lo < M; lo += per {
		hi := min(lo+per, M)
		wg.Add(1)
		go func() {
			defer wg.Done()
			linalg.MatmulBTInto(dst[lo*N:hi*N], a[lo*K:hi*K], b, hi-lo, K, N)
		}()
	}
	wg.Wait()
	return dst
}

func addInPlace(dst, src []float32) {
	for i := range dst {
		dst[i] += src[i]
	}
}

func addBias(x, bias []float32, n, w int) {
	for i := 0; i < n; i++ {
		row := x[i*w : (i+1)*w]
		for j := range row {
			row[j] += bias[j]
		}
	}
}

// layerNormRows is nn.LayerNorm (eps 1e-5, biased variance) over each row of src, with the statistics in float64.
func layerNormRows(dst, src []float32, ln layerNorm, rows, w int) {
	for i := 0; i < rows; i++ {
		row := src[i*w : (i+1)*w]
		var mean float64
		for _, v := range row {
			mean += float64(v)
		}
		mean /= float64(w)
		var vr float64
		for _, v := range row {
			d := float64(v) - mean
			vr += d * d
		}
		inv := 1 / math.Sqrt(vr/float64(w)+1e-5)
		out := dst[i*w : (i+1)*w]
		for j, v := range row {
			out[j] = float32((float64(v)-mean)*inv)*ln.w[j] + ln.b[j]
		}
	}
}

func geluExact(x float32) float32 {
	return float32(0.5 * float64(x) * (1 + math.Erf(float64(x)/math.Sqrt2)))
}

func dot64[A, B ~float32](a []A, b []B) float64 {
	var s float64
	for i := range a {
		s += float64(a[i]) * float64(b[i])
	}
	return s
}

// normalize64 is F.normalize(x, dim=-1): x / max(||x||, 1e-12).
func normalize64(x []float64) {
	var s float64
	for _, v := range x {
		s += v * v
	}
	n := math.Max(math.Sqrt(s), 1e-12)
	for i := range x {
		x[i] /= n
	}
}

// cosineSimilarity is F.cosine_similarity(a, b, dim=-1) with its default eps 1e-8: a.b / (max(||a||, eps) * max(||b||, eps)).
func cosineSimilarity(a, b []float32) float64 {
	var ab, aa, bb float64
	for i := range a {
		ab += float64(a[i]) * float64(b[i])
		aa += float64(a[i]) * float64(a[i])
		bb += float64(b[i]) * float64(b[i])
	}
	return ab / (math.Max(math.Sqrt(aa), 1e-8) * math.Max(math.Sqrt(bb), 1e-8))
}
