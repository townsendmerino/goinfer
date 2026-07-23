package decoder

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"

	"github.com/townsendmerino/aikit/embed"
	"github.com/townsendmerino/aikit/linalg"
)

// EagleHead is an imported EAGLE-3 draft head (05): a single transformer layer that
// fuses 3 of the target's hidden states (the ForwardCapture seam) with the previous
// token's embedding and predicts the next token over a reduced draft vocab, mapped
// back to the target vocab via d2t. It is a Drafter — the target verify keeps output
// lossless regardless of the head's quality. Weights/protocol confirmed against
// AngelSlim/Qwen3-1.7B_eagle3 (Eagle3LlamaForCausalLM); see docs/spec/05.
//
// Forward (per drafted token): feature = fc(concat(h_lo,h_mid,h_hi));
//
//	x = concat(inputNorm(targetEmbed(tok)), hiddenNorm(feature))  // [2*hidden]
//	x → midlayer (GQA+RoPE attn from 2*hidden, residual over the hidden feature,
//	    SwiGLU MLP) → norm → lmHead → draftLogits[draftVocab]
//	target_id = i + d2t[i]   for draft index i
type EagleHead struct {
	hidden, nHeads, nKV, headDim, inter int
	draftVocab, vocab                   int
	ropeTheta, normEps                  float64

	// projections (linalg.WeightMat, f32). q/k/v project from 2*hidden (embed‖feature).
	fc                      linalg.WeightMat // [hidden, 3*hidden] feature fusion
	q, k, v, o              linalg.WeightMat
	gate, up, down          linalg.WeightMat
	lmHead                  linalg.WeightMat // [draftVocab, hidden]
	hiddenNorm, inputNorm   []float32        // RMSNorm on feature / on embed
	postAttnNorm, finalNorm []float32

	d2t []int32 // draft index → target id offset: target = i + d2t[i]

	st *embed.SafetensorsFile // retained: the WeightMats alias its mmap (Close frees it)
}

// Close releases the head's mmap'd weights.
func (h *EagleHead) Close() error {
	if h.st != nil {
		return h.st.Close()
	}
	return nil
}

// eagleConfig is the subset of the head's config.json the loader needs.
type eagleConfig struct {
	HiddenSize        int     `json:"hidden_size"`
	NumAttentionHeads int     `json:"num_attention_heads"`
	NumKeyValueHeads  int     `json:"num_key_value_heads"`
	HeadDim           int     `json:"head_dim"`
	IntermediateSize  int     `json:"intermediate_size"`
	DraftVocabSize    int     `json:"draft_vocab_size"`
	VocabSize         int     `json:"vocab_size"`
	RopeTheta         float64 `json:"rope_theta"`
	RMSNormEps        float64 `json:"rms_norm_eps"`
}

// LoadEagleHead loads an EAGLE-3 head from dir (config.json + model.safetensors, the
// AngelSlim layout). The head's hidden size must match the target model's hidden dim
// (it reuses the target's token embedding and fuses the target's hidden states).
func LoadEagleHead(dir string) (*EagleHead, error) {
	cfgBytes, err := os.ReadFile(filepath.Join(dir, "config.json"))
	if err != nil {
		return nil, fmt.Errorf("eagle: read config: %w", err)
	}
	var c eagleConfig
	if err := json.Unmarshal(cfgBytes, &c); err != nil {
		return nil, fmt.Errorf("eagle: parse config: %w", err)
	}
	if c.HeadDim == 0 {
		c.HeadDim = c.HiddenSize / c.NumAttentionHeads
	}
	st, err := embed.OpenSafetensorsMmap(filepath.Join(dir, "model.safetensors"))
	if err != nil {
		return nil, fmt.Errorf("eagle: open safetensors: %w", err)
	}
	// st is retained on the head (the WeightMats alias its mmap); freed by Close. Every
	// error path below returned without closing it — leak one mmap per bad head. Close
	// it unless we reach the successful return.
	loaded := false
	defer func() {
		if !loaded {
			st.Close()
		}
	}()

	h := c.HiddenSize
	qDim := c.NumAttentionHeads * c.HeadDim
	kvDim := c.NumKeyValueHeads * c.HeadDim
	in2 := 2 * h // attn projects from concat(embed, feature)

	head := &EagleHead{
		hidden: h, nHeads: c.NumAttentionHeads, nKV: c.NumKeyValueHeads, headDim: c.HeadDim,
		inter: c.IntermediateSize, draftVocab: c.DraftVocabSize, vocab: c.VocabSize,
		ropeTheta: c.RopeTheta, normEps: c.RMSNormEps,
	}
	mats := []struct {
		dst        *linalg.WeightMat
		name       string
		rows, cols int
	}{
		{&head.fc, "fc.weight", h, 3 * h},
		{&head.q, "midlayer.self_attn.q_proj.weight", qDim, in2},
		{&head.k, "midlayer.self_attn.k_proj.weight", kvDim, in2},
		{&head.v, "midlayer.self_attn.v_proj.weight", kvDim, in2},
		{&head.o, "midlayer.self_attn.o_proj.weight", h, qDim},
		{&head.gate, "midlayer.mlp.gate_proj.weight", c.IntermediateSize, h},
		{&head.up, "midlayer.mlp.up_proj.weight", c.IntermediateSize, h},
		{&head.down, "midlayer.mlp.down_proj.weight", h, c.IntermediateSize},
		{&head.lmHead, "lm_head.weight", c.DraftVocabSize, h},
	}
	for _, m := range mats {
		w, err := loadMat(st, m.name, m.rows, m.cols)
		if err != nil {
			return nil, fmt.Errorf("eagle: load %s: %w", m.name, err)
		}
		*m.dst = w
	}
	norms := []struct {
		dst  *[]float32
		name string
	}{
		{&head.hiddenNorm, "midlayer.hidden_norm.weight"},
		{&head.inputNorm, "midlayer.input_layernorm.weight"},
		{&head.postAttnNorm, "midlayer.post_attention_layernorm.weight"},
		{&head.finalNorm, "norm.weight"},
	}
	for _, n := range norms {
		t, err := st.Tensor(n.name)
		if err != nil {
			return nil, fmt.Errorf("eagle: tensor %s: %w", n.name, err)
		}
		f, err := t.Float32s()
		if err != nil {
			return nil, fmt.Errorf("eagle: %s as f32: %w", n.name, err)
		}
		if len(f) != h {
			return nil, fmt.Errorf("eagle: %s len %d, want hidden %d", n.name, len(f), h)
		}
		*n.dst = append([]float32(nil), f...) // copy off the mmap before Close
	}
	d2tT, err := st.Tensor("d2t")
	if err != nil {
		return nil, fmt.Errorf("eagle: tensor d2t: %w", err)
	}
	d2t64, err := d2tT.Int64s()
	if err != nil {
		return nil, fmt.Errorf("eagle: d2t as i64: %w", err)
	}
	if len(d2t64) != c.DraftVocabSize {
		return nil, fmt.Errorf("eagle: d2t len %d, want draftVocab %d", len(d2t64), c.DraftVocabSize)
	}
	head.d2t = make([]int32, len(d2t64))
	for i, v := range d2t64 {
		head.d2t[i] = int32(v)
	}
	head.st = st
	loaded = true
	return head, nil
}

// eagleRecurFeature toggles the multi-step draft recurrence (inc 4a, set empirically
// by accepted-length): true feeds the head's own hidden output as the next step's
// feature (EAGLE-1/2 style); false keeps the step-0 target feature for all K steps
// (the chain then comes only from the drafted-token embeddings + the head's KV).
var eagleRecurFeature = true

// eagleState is the head's own KV cache across the K autoregressive draft steps,
// plus the RoPE inverse-frequency table (built once from the head's rope_theta).
type eagleState struct {
	k, v    [][]float32 // per step: [nKV*headDim]
	invFreq []float64   // [headDim/2]
}

// NewState returns a fresh per-draft-block head state.
func (h *EagleHead) NewState() *eagleState {
	half := h.headDim / 2
	inv := make([]float64, half)
	for d := range inv {
		inv[d] = math.Pow(h.ropeTheta, -float64(2*d)/float64(h.headDim))
	}
	return &eagleState{invFreq: inv}
}

// clone returns a copy of the head state's KV (sharing the immutable invFreq) so a
// drafted tree can branch: each branch continues from the shared prefix independently.
func (st *eagleState) clone() *eagleState {
	c := &eagleState{invFreq: st.invFreq, k: make([][]float32, len(st.k)), v: make([][]float32, len(st.v))}
	copy(c.k, st.k)
	copy(c.v, st.v)
	return c
}

// TreeDraft is a root-branched draft tree (05 EAGLE tree drafting): the head's top-b
// candidates for the first draft position each seed a linear continuation of total
// depth d. Verified in ONE batched target pass with tree attention, it recovers the
// cases where the correct next token is in the head's top-b but not top-1 (the head's
// top-1 is only ~0.56), and then the matching branch continues from the TRUE token.
type TreeDraft struct {
	Tokens []int    // node tokens in batch order
	RowPos []int    // absolute RoPE position per node (startPos + depth)
	Parent []int    // parent node index per node (-1 = child of the root/last-confirmed)
	Depth  []int    // tree depth per node (1 = first draft position)
	Mask   [][]bool // ancestor mask per node (its path to the root, plus itself)
	B, D   int      // branch factor, max depth
}

// Children returns the node indices whose parent is p (-1 for the depth-1 roots).
func (td TreeDraft) Children(p int) []int {
	var c []int
	for i, par := range td.Parent {
		if par == p {
			c = append(c, i)
		}
	}
	return c
}

// topKDraftIdx returns the indices of the k largest logits (descending).
func topKDraftIdx(logits []float32, k int) []int {
	idx := make([]int, len(logits))
	for i := range idx {
		idx[i] = i
	}
	// partial selection sort for the top k (k is small)
	for i := 0; i < k && i < len(idx); i++ {
		best := i
		for j := i + 1; j < len(idx); j++ {
			if logits[idx[j]] > logits[idx[best]] {
				best = j
			}
		}
		idx[i], idx[best] = idx[best], idx[i]
	}
	if k > len(idx) {
		k = len(idx)
	}
	return idx[:k]
}

// eagleTreeNodes returns the node count of a full b-ary draft tree of depth d
// (depths 1..d; the root/confirmed token is not a node): Σ_{i=1}^{d} b^i. This is
// exactly what DraftTree emits and what one verify round transiently writes into the
// cache before rolling back the rejected path — so it, not b*d, sizes the cache (M15).
func eagleTreeNodes(b, d int) int {
	n, pow := 0, 1
	for range d {
		pow *= b
		n += pow
	}
	return n
}

// DraftTree builds a full b-ary tree from a pre-built head state (KV over the
// context): every frontier node expands its top-b children at each of the d depths,
// so the tree has eagleTreeNodes(b, d) = Σ_{i=1}^{d} b^i nodes (NOT b chains — b is the
// branch factor at EVERY depth). firstTok/seedFeature/startPos are the root (last
// confirmed token). The first draft node is at absolute position startPos+1.
func (h *EagleHead) DraftTree(be Backend, st *eagleState, embedOf func(tok int, dst []float32), firstTok int, seedFeature []float32, startPos, b, d int) TreeDraft {
	emb := make([]float32, h.hidden)
	// expandable carries the per-node head state + feature needed to expand its children.
	type expandable struct {
		st      *eagleState
		feature []float32 // the feature this node's own Step consumes (its parent's hidden)
		idx     int       // node index, -1 for the synthetic root
	}
	td := TreeDraft{B: b, D: d}
	// The root is the last confirmed token; its Step (at startPos) yields the depth-1
	// candidates and the hidden feature the depth-1 nodes consume.
	embedOf(firstTok, emb)
	rootLogits, rootHidden := h.Step(be, emb, seedFeature, startPos, st)
	frontier := []expandable{{st: st, feature: rootHidden, idx: -1}}
	parentLogits := [][]float32{rootLogits}
	for depth := 1; depth <= d; depth++ {
		var next []expandable
		var parentLogitsNext [][]float32
		for fi, node := range frontier {
			cands := topKDraftIdx(parentLogits[fi], b)
			for _, di := range cands {
				tok := h.TargetID(di)
				idx := len(td.Tokens)
				td.Tokens = append(td.Tokens, tok)
				td.RowPos = append(td.RowPos, startPos+depth)
				td.Parent = append(td.Parent, node.idx)
				td.Depth = append(td.Depth, depth)
				if depth < d { // expand: this node's Step produces its children's logits
					stc := node.st.clone()
					embedOf(tok, emb)
					logits, hidden := h.Step(be, emb, node.feature, startPos+depth, stc)
					next = append(next, expandable{st: stc, feature: hidden, idx: idx})
					parentLogitsNext = append(parentLogitsNext, logits)
				}
			}
		}
		frontier = next
		parentLogits = parentLogitsNext
	}
	// ancestor masks: walk parent pointers from each node up to the root.
	n := len(td.Tokens)
	td.Mask = make([][]bool, n)
	for i := range td.Mask {
		td.Mask[i] = make([]bool, n)
		for j := i; j >= 0; j = td.Parent[j] {
			td.Mask[i][j] = true
		}
	}
	return td
}

// Fuse computes the fused feature [hidden] from the 3 concatenated target hidden
// states h3 [3*hidden] (the ForwardCapture seam output): feature = fc · h3.
func (h *EagleHead) Fuse(be Backend, h3 []float32) []float32 {
	feat := make([]float32, h.hidden)
	matmul(be, &h.fc, h3, feat, 1)
	return feat
}

// Step runs one head forward at position pos: embedRow is the TARGET embedding of the
// input token, feature is the fused target hidden (step 0) or the head's own previous
// hidden output (autoregressive steps 1+). It appends this step's K/V to st and
// returns the draft-vocab logits AND the head's hidden output (the pre-final-norm
// residual) — which is the feature the NEXT autoregressive step consumes. The decoder
// layer attends over concat(embed, feature) (2*hidden in), residual over the feature.
func (h *EagleHead) Step(be Backend, embedRow, feature []float32, pos int, st *eagleState) (logits, hiddenOut []float32) {
	hid := h.hidden
	e := append([]float32(nil), embedRow...)
	rmsNorm(e, h.inputNorm, 1, hid, h.normEps, false)
	f := append([]float32(nil), feature...)
	rmsNorm(f, h.hiddenNorm, 1, hid, h.normEps, false)
	x := make([]float32, 2*hid)
	copy(x[:hid], e)
	copy(x[hid:], f)

	qDim, kvDim := h.nHeads*h.headDim, h.nKV*h.headDim
	q := make([]float32, qDim)
	k := make([]float32, kvDim)
	v := make([]float32, kvDim)
	matmul(be, &h.q, x, q, 1)
	matmul(be, &h.k, x, k, 1)
	matmul(be, &h.v, x, v, 1)
	applyRoPE(q, h.nHeads, h.headDim, pos, st.invFreq, 1)
	applyRoPE(k, h.nKV, h.headDim, pos, st.invFreq, 1)
	st.k = append(st.k, k)
	st.v = append(st.v, v)

	ctx := make([]float32, qDim)
	h.attend(q, st, ctx)
	attnOut := make([]float32, hid)
	matmul(be, &h.o, ctx, attnOut, 1)
	resid := make([]float32, hid)
	for i := range resid {
		resid[i] = feature[i] + attnOut[i] // residual over the feature
	}

	x2 := append([]float32(nil), resid...)
	rmsNorm(x2, h.postAttnNorm, 1, hid, h.normEps, false)
	gate := make([]float32, h.inter)
	up := make([]float32, h.inter)
	matmul(be, &h.gate, x2, gate, 1)
	matmul(be, &h.up, x2, up, 1)
	mid := make([]float32, h.inter)
	for i := range mid {
		mid[i] = silu(gate[i]) * up[i]
	}
	down := make([]float32, hid)
	matmul(be, &h.down, mid, down, 1)
	for i := range resid {
		resid[i] += down[i]
	}

	hiddenOut = append([]float32(nil), resid...) // pre-final-norm: the next step's feature
	rmsNorm(resid, h.finalNorm, 1, hid, h.normEps, false)
	logits = make([]float32, h.draftVocab)
	matmul(be, &h.lmHead, resid, logits, 1)
	return logits, hiddenOut
}

// Draft autoregressively drafts up to k tokens. firstTok is the last confirmed token
// (its embedding is step 0's input); seedFeature = fc(3 target hidden) at firstTok's
// position; startPos = that position. Step 0 uses seedFeature; each later step feeds
// the head's own hidden output as the feature (the EAGLE recurrence, no new target
// forward). embedOf writes a token's TARGET embedding into dst.
func (h *EagleHead) Draft(be Backend, embedOf func(tok int, dst []float32), firstTok int, seedFeature []float32, startPos, k int) []int {
	return h.DraftFrom(be, h.NewState(), embedOf, firstTok, seedFeature, startPos, k)
}

// Prefill builds the head's KV over a context window so its attention has the prompt
// context before drafting (without this, multi-step drafts collapse — the head only
// sees the local draft chain). For each i it runs one head step over (token toks[i],
// fused target feature feats[i]) at position startPos+i, keeping the KV and discarding
// logits. Returns the populated state to DraftFrom.
func (h *EagleHead) Prefill(be Backend, embedOf func(tok int, dst []float32), toks []int, feats [][]float32, startPos int) *eagleState {
	st := h.NewState()
	h.Extend(be, st, embedOf, toks, feats, startPos)
	return st
}

// Extend appends context steps (toks[i] at position startPos+i, with feature feats[i])
// onto an existing head state, discarding logits — so a speculative loop can grow the
// head's KV incrementally across rounds instead of rebuilding it (O(C) → O(1)/round).
func (h *EagleHead) Extend(be Backend, st *eagleState, embedOf func(tok int, dst []float32), toks []int, feats [][]float32, startPos int) {
	emb := make([]float32, h.hidden)
	for i, tok := range toks {
		embedOf(tok, emb)
		h.Step(be, emb, feats[i], startPos+i, st)
	}
}

// DraftFrom is Draft continuing from a pre-built state (its KV already populated over
// the context by Prefill). startPos is the absolute position of firstTok.
func (h *EagleHead) DraftFrom(be Backend, st *eagleState, embedOf func(tok int, dst []float32), firstTok int, seedFeature []float32, startPos, k int) []int {
	feature := seedFeature
	tok := firstTok
	emb := make([]float32, h.hidden)
	out := make([]int, 0, k)
	for j := range k {
		embedOf(tok, emb)
		logits, hiddenOut := h.Step(be, emb, feature, startPos+j, st)
		d := h.TargetID(argmax(logits))
		out = append(out, d)
		tok = d
		if eagleRecurFeature {
			feature = hiddenOut // EAGLE recurrence (hypothesis A)
		}
		// else: keep the step-0 target feature for all steps (hypothesis B)
	}
	return out
}

// attend is the head's single-query GQA attention over its stored K/V (causal by
// construction — the head only attends positions it has drafted so far).
func (h *EagleHead) attend(q []float32, st *eagleState, ctx []float32) {
	n := len(st.k)
	group := h.nHeads / h.nKV
	scale := 1.0 / math.Sqrt(float64(h.headDim))
	scores := make([]float64, n)
	for qh := 0; qh < h.nHeads; qh++ {
		kvh := qh / group
		qHead := q[qh*h.headDim : (qh+1)*h.headDim]
		maxS := math.Inf(-1)
		for s := range n {
			kHead := st.k[s][kvh*h.headDim : (kvh+1)*h.headDim]
			var dot float64
			for d := range qHead {
				dot += float64(qHead[d]) * float64(kHead[d])
			}
			dot *= scale
			scores[s] = dot
			if dot > maxS {
				maxS = dot
			}
		}
		var sum float64
		for s := range n {
			scores[s] = math.Exp(scores[s] - maxS)
			sum += scores[s]
		}
		out := ctx[qh*h.headDim : (qh+1)*h.headDim]
		for d := range out {
			out[d] = 0
		}
		for s := range n {
			w := float32(scores[s] / sum)
			vHead := st.v[s][kvh*h.headDim : (kvh+1)*h.headDim]
			for d := range out {
				out[d] += w * vHead[d]
			}
		}
	}
}

// TargetID maps a draft-vocab index to the target token id: target = i + d2t[i].
func (h *EagleHead) TargetID(draftIdx int) int { return draftIdx + int(h.d2t[draftIdx]) }

// Hidden reports the head's hidden size (must equal the target model's HiddenDim).
func (h *EagleHead) Hidden() int { return h.hidden }
