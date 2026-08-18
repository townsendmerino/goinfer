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

// DFlashDrafter is an imported z-lab DFlash block drafter (P10 / docs/spec/08): a small
// non-causal transformer trunk that reads the target's hidden states (the ForwardCapture
// seam) and proposes a whole BLOCK of tokens in one pass, instead of one token per head
// forward the way 05's EAGLE head does. That is the draft-side economics the spec
// program's scorecard said was the lever.
//
// It is deliberately smaller than it looks. The checkpoint ships ONLY the trunk —
// 5 decoder layers + `fc` + two norms, 58 tensors — with **no embedding, no LM head, no
// Markov head and no confidence head** (verified against the published file byte-for-byte,
// see docs/spec/08). Both ends are the TARGET's: the block is embedded with the target's
// embed_tokens and the draft logits come out of the target's lm_head. So this type holds
// no vocab-sized weight at all, and the caller supplies both ends.
//
// One round, given the target's captured hidden states for the committed context:
//
//	fused    = hiddenNorm(fc · concat(h[l] for l in targetLayerIDs))   per context position
//	blockIn  = targetEmbed([anchor, MASK, MASK, ...])                  [blockSize, hidden]
//	trunk    = layers(blockIn, attending over concat(fused, block))    bidirectional
//	logits   = targetLMHead(norm(trunk)[1:])                           blockSize-1 drafts
//
// The attention is CROSS-attention and NON-causal: queries come only from the block,
// keys/values from the fused context concatenated with the block, and every block
// position sees every other. That is what makes the block one pass — and it is also why
// the block width is not truncatable (each position's hidden depends on all of them, so
// a narrower block is off the trained distribution).
type DFlashDrafter struct {
	blockTrunk
	blockSize, maskTokenID int
	targetLayerIDs         []int

	// OWN head + embedding, present only on drafters that ship them (poolside's
	// Laguna speculators do; z-lab's do not and borrow the target's — see
	// DrafterHeadLogits). When lmHead is set the drafter emits ids in its OWN
	// REDUCED vocabulary, which d2t maps back to target ids:
	//
	//	target_id = i + d2t[i]   for draft index i
	//
	// That is the same scheme EagleHead already implements; the mapping table is
	// identical in meaning, so the arithmetic is kept identical too rather than
	// re-derived. embed is the drafter's own token embedding over the TARGET vocab
	// (it is fed target ids and produces the trunk's block input).
	lmHead     linalg.WeightMat // [draftVocab, hidden]; zero-valued when the drafter borrows the target's
	embed      linalg.WeightMat // [vocab, hidden]; zero-valued when it borrows the target's
	d2t        []int32          // draft index → target id OFFSET
	draftVocab int

	st *embed.SafetensorsFile // retained: the WeightMats alias its mmap
}

// HasOwnHead reports whether this drafter emits ids from its own reduced vocabulary
// (and therefore must map them through d2t) rather than borrowing the target's LM
// head. Callers use it to pick between DraftTokenID and DrafterHeadLogits.
func (d *DFlashDrafter) HasOwnHead() bool { return d.lmHead.Rows() > 0 }

// DraftTokenID turns one already-normed trunk hidden row into a TARGET token id
// using the drafter's own reduced-vocab head, mapping the argmax back through d2t.
//
// The reduced vocab is the whole point of the design — a 32000-row head over a
// 100352-token target vocab is ~3x less work per drafted position — but it means an
// unmapped argmax is a VALID-LOOKING id in the wrong space. Drafting is lossless by
// construction (the target re-verifies every token), so a missed mapping would not
// corrupt output; it would silently destroy acceptance and read as "this pairing is
// just bad", which is exactly the failure P10 spent a day chasing on a wrong mask
// token. Hence the mapping lives here, next to the argmax, rather than in the caller.
func (d *DFlashDrafter) DraftTokenID(be Backend, h []float32) int {
	logits := make([]float32, d.draftVocab)
	matmul(be, &d.lmHead, h, logits, 1)
	i := argmax(logits)
	return i + int(d.d2t[i])
}

// EmbedDraft writes the drafter's OWN embedding row for a target token id. Drafters
// without their own embedding use the target's (Model.embedResident).
func (d *DFlashDrafter) EmbedDraft(id int, dst []float32) { d.embed.Row(id, dst) }

// EmbedBlock builds the trunk's block input for ids, using the DRAFTER's own
// embedding when it has one and falling back to the target's otherwise.
//
// The fallback is not cosmetic. A drafter that ships embed_tokens was trained
// against THOSE vectors; feeding it the target's instead is a silent
// off-distribution input — the trunk still runs, still emits ids, and the pairing
// still produces lossless output, so the only symptom is acceptance quietly landing
// below what the pairing is worth. Same failure shape as the mask token.
func (d *DFlashDrafter) EmbedBlock(m *Model, ids []int) [][]float32 {
	if d.embed.Rows() == 0 {
		return m.DrafterEmbedBlock(ids)
	}
	rows := make([][]float32, len(ids))
	for i, id := range ids {
		rows[i] = make([]float32, d.hidden)
		d.EmbedDraft(id, rows[i])
	}
	return rows
}

// DraftIDs turns the trunk's block output rows into TARGET token ids, using the
// drafter's own reduced-vocab head when present and the target's LM head otherwise.
// One call site for both drafter shapes, so a pairing cannot silently take the wrong
// output path.
func (d *DFlashDrafter) DraftIDs(m *Model, rows [][]float32) []int {
	out := make([]int, len(rows))
	for i, h := range rows {
		if d.HasOwnHead() {
			out[i] = d.DraftTokenID(m.be, h)
		} else {
			out[i] = argmax(m.DrafterHeadLogits(h))
		}
	}
	return out
}

// blockTrunk is the non-causal block trunk shared by DFlash and DSpark.
//
// Shared because they compute it IDENTICALLY, which was established from first-party source
// rather than assumed: DeepSpec's `_forward_backbone` is `hidden_norm(fc(ctx))` → rotary →
// N layers → `norm`, its attention sets `is_causal=False` and takes K/V from
// concat(raw fused context, block), and its `apply_rotary_pos_emb` — q taking
// `cos[..., -q_len:, :]` while k takes the full `cos` — is byte-identical to z-lab's. Two
// separately-developed drafters converged on the same trunk; carrying two copies of it here
// would be the third copy this repo's own rule warns about.
//
// What differs between them lives in the enclosing type: DFlash borrows the target's embedding
// and LM head and predicts from slot 1; DSpark ships its own, predicts from slot 0, and adds a
// rank-256 Markov chain plus a confidence head.
type blockTrunk struct {
	hidden, nHeads, nKV, headDim, inter int
	normEps                             float64
	invFreq                             []float64

	fc         linalg.WeightMat // [hidden, nTaps*hidden]
	hiddenNorm []float32        // RMSNorm on the fused context
	finalNorm  []float32        // RMSNorm at the end of the trunk
	layers     []dflashLayer
}

type dflashLayer struct {
	q, k, v, o     linalg.WeightMat
	gate, up, down linalg.WeightMat
	qNorm, kNorm   []float32 // [headDim] — Qwen3 per-head RMSNorm
	inputNorm      []float32 // normalizes the BLOCK only (never the context — see attend)
	postAttnNorm   []float32
}

// BlockSize is the trained block width. The drafter proposes BlockSize-1 tokens: slot 0
// is the anchor (a token already committed), so it is embedded but never predicted.
func (d *DFlashDrafter) BlockSize() int { return d.blockSize }

// MaskTokenID is the id the untrained block slots are filled with.
func (d *DFlashDrafter) MaskTokenID() int { return d.maskTokenID }

// TargetLayerIDs are the target layers whose OUTPUTS feed `fc`. HF's reference indexes
// hidden_states[id+1] because hidden_states[0] is the embedding output, so these name
// layer outputs — pass them straight to Model.ForwardCapture, which uses the same
// after-layer-l convention.
func (d *DFlashDrafter) TargetLayerIDs() []int { return append([]int(nil), d.targetLayerIDs...) }

// Close releases the drafter's mmap'd weights.
func (d *DFlashDrafter) Close() error {
	if d.st != nil {
		return d.st.Close()
	}
	return nil
}

// dflashConfig is the subset of the checkpoint's config.json the loader needs.
type dflashConfig struct {
	HiddenSize        int     `json:"hidden_size"`
	NumHiddenLayers   int     `json:"num_hidden_layers"`
	NumAttentionHeads int     `json:"num_attention_heads"`
	NumKeyValueHeads  int     `json:"num_key_value_heads"`
	HeadDim           int     `json:"head_dim"`
	IntermediateSize  int     `json:"intermediate_size"`
	RMSNormEps        float64 `json:"rms_norm_eps"`
	// RoPE in both spellings too — flat on Qwen3-4B-DFlash-b16, nested rope_parameters on
	// Qwen3.6-35B-A3B-DFlash (and on every DSpark checkpoint). Three checkpoints, three
	// config dialects; reading only the first one's is how a supported pairing looks broken.
	RopeTheta      float64         `json:"rope_theta"`
	RopeParameters json.RawMessage `json:"rope_parameters"`
	// block_size appears in BOTH places across z-lab's own checkpoints: top-level on
	// Qwen3-4B-DFlash-b16, nested in dflash_config on Qwen3.6-35B-A3B-DFlash. One publisher,
	// two spellings — so read either rather than assume the one the first checkpoint used.
	BlockSize int `json:"block_size"`
	DFlash    struct {
		BlockSize      int   `json:"block_size"`
		MaskTokenID    int   `json:"mask_token_id"`
		TargetLayerIDs []int `json:"target_layer_ids"`
	} `json:"dflash_config"`

	// vLLM "speculators" dialect (v0.5), which poolside's Laguna drafters ship —
	// a FOURTH config spelling for the same model. It differs from z-lab's in three
	// ways, all handled in LoadDFlashDrafter:
	//
	//   * the layer geometry is nested under transformer_layer_config (whose
	//     model_type says "llama" even though the layers carry Qwen3-style per-head
	//     q_norm/k_norm),
	//   * the taps are aux_hidden_state_layer_ids rather than target_layer_ids,
	//   * mask_token_id is top-level rather than inside dflash_config.
	//
	// DraftVocabSize marks the OTHER structural difference: this drafter ships its
	// own embed_tokens and a REDUCED-vocab lm_head plus d2t/t2d, where z-lab's
	// borrow the target's. See the drafter head fields on DFlashDrafter.
	MaskTokenIDTop         *int            `json:"mask_token_id"`
	AuxHiddenStateLayerIDs []int           `json:"aux_hidden_state_layer_ids"`
	DraftVocabSize         int             `json:"draft_vocab_size"`
	VocabSize              int             `json:"vocab_size"`
	TransformerLayerConfig json.RawMessage `json:"transformer_layer_config"`
	SpeculatorsModelType   string          `json:"speculators_model_type"`
}

// LoadDFlashDrafter loads a DFlash drafter from dir (config.json + model.safetensors, the
// z-lab layout). The drafter's hidden size must equal the target's — it consumes the
// target's hidden states and embeddings directly, with no projection between them.
func LoadDFlashDrafter(dir string) (*DFlashDrafter, error) {
	cfgBytes, err := os.ReadFile(filepath.Join(dir, "config.json"))
	if err != nil {
		return nil, fmt.Errorf("dflash: read config: %w", err)
	}
	var c dflashConfig
	if err := json.Unmarshal(cfgBytes, &c); err != nil {
		return nil, fmt.Errorf("dflash: parse config: %w", err)
	}
	// speculators dialect: the layer geometry lives one level down. Decode it into
	// the SAME struct so every field below is read once, from one place, whichever
	// dialect the checkpoint used. Top-level keys already parsed (block_size,
	// mask_token_id, the tap list) are not repeated there, so this cannot clobber them.
	if len(c.TransformerLayerConfig) > 0 && c.HiddenSize == 0 {
		if err := json.Unmarshal(c.TransformerLayerConfig, &c); err != nil {
			return nil, fmt.Errorf("dflash: parse transformer_layer_config: %w", err)
		}
	}
	if c.HeadDim == 0 {
		c.HeadDim = c.HiddenSize / c.NumAttentionHeads
	}
	if c.BlockSize == 0 {
		c.BlockSize = c.DFlash.BlockSize
	}
	// Taps and mask token in whichever spelling this checkpoint uses.
	if len(c.DFlash.TargetLayerIDs) == 0 {
		c.DFlash.TargetLayerIDs = c.AuxHiddenStateLayerIDs
	}
	if c.MaskTokenIDTop != nil && c.DFlash.MaskTokenID == 0 {
		c.DFlash.MaskTokenID = *c.MaskTokenIDTop
	}
	switch {
	case c.HiddenSize <= 0 || c.NumHiddenLayers <= 0 || c.NumAttentionHeads <= 0:
		return nil, fmt.Errorf("dflash: bad dims (hidden=%d layers=%d heads=%d)", c.HiddenSize, c.NumHiddenLayers, c.NumAttentionHeads)
	case c.BlockSize < 2:
		return nil, fmt.Errorf("dflash: block_size must be >= 2, got %d", c.BlockSize)
	case len(c.DFlash.TargetLayerIDs) == 0:
		return nil, fmt.Errorf("dflash: no taps — need dflash_config.target_layer_ids or aux_hidden_state_layer_ids")
	case c.RopeTheta <= 0 && len(c.RopeParameters) == 0:
		return nil, fmt.Errorf("dflash: no rope_theta and no rope_parameters")
	}

	theta := c.RopeTheta
	if len(c.RopeParameters) > 0 {
		spec, _, perr := parseRopeFlat(c.RopeParameters)
		if perr != nil {
			return nil, fmt.Errorf("dflash: rope_parameters: %w", perr)
		}
		theta = spec.base
	}
	if theta <= 0 {
		return nil, fmt.Errorf("dflash: rope_theta must be >0, got %v", theta)
	}

	st, err := embed.OpenSafetensorsMmap(filepath.Join(dir, "model.safetensors"))
	if err != nil {
		return nil, fmt.Errorf("dflash: open safetensors: %w", err)
	}
	loaded := false
	defer func() {
		if !loaded {
			st.Close()
		}
	}()

	h, hd := c.HiddenSize, c.HeadDim
	qDim, kvDim := c.NumAttentionHeads*hd, c.NumKeyValueHeads*hd
	d := &DFlashDrafter{
		blockTrunk: blockTrunk{
			hidden: h, nHeads: c.NumAttentionHeads, nKV: c.NumKeyValueHeads, headDim: hd,
			inter: c.IntermediateSize, normEps: c.RMSNormEps,
			invFreq: computeInvFreq(theta, hd, nil),
			layers:  make([]dflashLayer, c.NumHiddenLayers),
		},
		blockSize: c.BlockSize, maskTokenID: c.DFlash.MaskTokenID,
		targetLayerIDs: c.DFlash.TargetLayerIDs,
		st:             st,
	}

	mat := func(dst *linalg.WeightMat, name string, rows, cols int) error {
		m, err := loadMat(st, name, rows, cols)
		if err != nil {
			return fmt.Errorf("dflash: %w", err)
		}
		*dst = m
		return nil
	}
	vec := func(dst *[]float32, name string, n int) error {
		v, err := st.TensorF32(name, n)
		if err != nil {
			return fmt.Errorf("dflash: %w", err)
		}
		*dst = v
		return nil
	}

	if err := mat(&d.fc, "fc.weight", h, len(d.targetLayerIDs)*h); err != nil {
		return nil, err
	}
	if err := vec(&d.hiddenNorm, "hidden_norm.weight", h); err != nil {
		return nil, err
	}
	if err := vec(&d.finalNorm, "norm.weight", h); err != nil {
		return nil, err
	}
	// OWN head + embedding + d2t, when this drafter ships them (draft_vocab_size).
	// Loaded together and validated against each other: a reduced-vocab head with a
	// mismatched d2t maps argmaxes into the wrong ids, which is invisible in output
	// (drafting is lossless — the target re-verifies) and shows up only as acceptance
	// collapsing to noise.
	if c.DraftVocabSize > 0 {
		d.draftVocab = c.DraftVocabSize
		if err := mat(&d.lmHead, "lm_head.weight", c.DraftVocabSize, h); err != nil {
			return nil, err
		}
		if c.VocabSize > 0 {
			if err := mat(&d.embed, "embed_tokens.weight", c.VocabSize, h); err != nil {
				return nil, err
			}
		}
		t, terr := st.Tensor("d2t")
		if terr != nil {
			return nil, fmt.Errorf("dflash: draft_vocab_size=%d but no d2t: %w", c.DraftVocabSize, terr)
		}
		i64, ierr := t.Int64s()
		if ierr != nil {
			return nil, fmt.Errorf("dflash: d2t as i64: %w", ierr)
		}
		if len(i64) != c.DraftVocabSize {
			return nil, fmt.Errorf("dflash: d2t len %d, want draft_vocab_size %d", len(i64), c.DraftVocabSize)
		}
		d.d2t = make([]int32, len(i64))
		for i, v := range i64 {
			d.d2t[i] = int32(v)
		}
		// Every mapped id must land inside the TARGET vocab; an out-of-range one would
		// index the target's embedding out of bounds at verify time.
		if c.VocabSize > 0 {
			for i, off := range d.d2t {
				if tid := i + int(off); tid < 0 || tid >= c.VocabSize {
					return nil, fmt.Errorf("dflash: d2t[%d] maps to target id %d, outside vocab %d", i, tid, c.VocabSize)
				}
			}
		}
	}
	for i := range d.layers {
		l := &d.layers[i]
		p := fmt.Sprintf("layers.%d.", i)
		for _, m := range []struct {
			dst        *linalg.WeightMat
			name       string
			rows, cols int
		}{
			{&l.q, p + "self_attn.q_proj.weight", qDim, h},
			{&l.k, p + "self_attn.k_proj.weight", kvDim, h},
			{&l.v, p + "self_attn.v_proj.weight", kvDim, h},
			{&l.o, p + "self_attn.o_proj.weight", h, qDim},
			{&l.gate, p + "mlp.gate_proj.weight", c.IntermediateSize, h},
			{&l.up, p + "mlp.up_proj.weight", c.IntermediateSize, h},
			{&l.down, p + "mlp.down_proj.weight", h, c.IntermediateSize},
		} {
			if err := mat(m.dst, m.name, m.rows, m.cols); err != nil {
				return nil, err
			}
		}
		for _, v := range []struct {
			dst  *[]float32
			name string
			n    int
		}{
			{&l.qNorm, p + "self_attn.q_norm.weight", hd},
			{&l.kNorm, p + "self_attn.k_norm.weight", hd},
			{&l.inputNorm, p + "input_layernorm.weight", h},
			{&l.postAttnNorm, p + "post_attention_layernorm.weight", h},
		} {
			if err := vec(v.dst, v.name, v.n); err != nil {
				return nil, err
			}
		}
	}
	loaded = true
	return d, nil
}

// FuseContext turns the target's captured hidden states into the drafter's context
// representation: fused[i] = hiddenNorm(fc · concat(h[l][i] for l in targetLayerIDs)).
// ctxCat is [ctxLen][len(targetLayerIDs)*hidden] — the ForwardCapture seam's output for
// each committed position, already concatenated in targetLayerIDs order.
//
// This is the ONLY place the target's hidden states enter, and the result is what every
// layer's K/V read. It is computed once per round, not per layer.
func (d *blockTrunk) FuseContext(be Backend, ctxCat [][]float32) ([][]float32, error) {
	// fc's input width IS the tap count x hidden, so it validates the caller's concatenation
	// without the trunk needing to know which layers were tapped — that belongs to the drafter.
	want := d.fc.Cols()
	fused := make([][]float32, len(ctxCat))
	for i, row := range ctxCat {
		if len(row) != want {
			return nil, fmt.Errorf("block trunk: context row %d is %d wide, want %d (%d taps x %d)",
				i, len(row), want, want/d.hidden, d.hidden)
		}
		f := make([]float32, d.hidden)
		matmul(be, &d.fc, row, f, 1)
		rmsNorm(f, d.hiddenNorm, 1, d.hidden, d.normEps, false)
		fused[i] = f
	}
	return fused, nil
}

// DFlashContext caches the committed context's PROJECTED K/V per layer — K already
// RoPE'd at its absolute position, V neither roped nor normed.
//
// It exists because the context is projected once per position and then read by every
// subsequent round: without it, each round re-projects the whole context in every layer,
// which is O(ctx x layers) of pure repeat work per round and made the CPU trunk scale
// 1.6 s/block at ctx 64 to 12.9 s at ctx 2048 (measured, BenchmarkDFlashTrunk). Both
// reference implementations cache it — mlx-dspark's CtxCache and dflash.py's DynamicCache
// — so this matches them rather than inventing a shortcut.
//
// Append-only, plus TruncateTo for the speculative rollback: the drafter's context only
// ever grows with COMMITTED tokens, and a rejected draft's positions must come back off.
type DFlashContext struct {
	k, v [][][]float32 // [layer][pos][nKV*headDim]
}

// NewContext returns an empty per-layer context cache for this drafter.
func (d *blockTrunk) NewContext() *DFlashContext {
	return &DFlashContext{k: make([][][]float32, len(d.layers)), v: make([][][]float32, len(d.layers))}
}

// Len is the number of committed context positions cached.
func (c *DFlashContext) Len() int {
	if len(c.k) == 0 {
		return 0
	}
	return len(c.k[0])
}

// TruncateTo drops cached positions at index >= n — the rollback after a partial accept.
// The retained K was roped at its absolute position, so it stays valid after the trim.
func (c *DFlashContext) TruncateTo(n int) {
	for l := range c.k {
		if n < len(c.k[l]) {
			c.k[l] = c.k[l][:n]
			c.v[l] = c.v[l][:n]
		}
	}
}

// ExtendContext projects newly committed positions into the cache. fusedNew is
// FuseContext's output for those positions ONLY; they are assumed to sit immediately
// after the positions already cached, which is what fixes their RoPE offsets.
func (d *blockTrunk) ExtendContext(be Backend, ctx *DFlashContext, fusedNew [][]float32) {
	start := ctx.Len()
	kvDim := d.nKV * d.headDim
	for li := range d.layers {
		l := &d.layers[li]
		for i, row := range fusedNew {
			k := make([]float32, kvDim)
			v := make([]float32, kvDim)
			matmul(be, &l.k, row, k, 1)
			matmul(be, &l.v, row, v, 1)
			rmsNorm(k, l.kNorm, d.nKV, d.headDim, d.normEps, false)
			applyRoPE(k, d.nKV, d.headDim, start+i, d.invFreq, 1)
			ctx.k[li] = append(ctx.k[li], k)
			ctx.v[li] = append(ctx.v[li], v)
		}
	}
}

// DraftBlockCtx runs the trunk over one block against a cached context and returns the
// final-normed hidden states, [blockSize][hidden]. This is the form production should
// use; DraftBlock is the uncached convenience wrapper.
func (d *blockTrunk) DraftBlockCtx(be Backend, ctx *DFlashContext, blockIn [][]float32) ([][]float32, error) {
	// The block's WIDTH is whatever the caller passes — the trained width is the drafter's
	// property, not the trunk's, and the enclosing type checks it.
	if len(blockIn) == 0 {
		return nil, fmt.Errorf("block trunk: empty block")
	}
	h := make([][]float32, len(blockIn))
	for i, row := range blockIn {
		if len(row) != d.hidden {
			return nil, fmt.Errorf("block trunk: block row %d is %d wide, want %d", i, len(row), d.hidden)
		}
		h[i] = append([]float32(nil), row...)
	}
	for i := range d.layers {
		d.layer(be, &d.layers[i], ctx.k[i], ctx.v[i], h)
	}
	out := make([][]float32, len(h))
	for i, row := range h {
		r := append([]float32(nil), row...)
		rmsNorm(r, d.finalNorm, 1, d.hidden, d.normEps, false)
		out[i] = r
	}
	return out, nil
}

// DraftBlock runs the trunk over one block and returns its final-normed hidden states,
// [blockSize][hidden]. blockIn is the TARGET's embedding of the block's token ids (slot 0
// the anchor, the rest MASK); fused is FuseContext's output for the committed context.
// The caller applies the target's LM head to rows 1.. to get the drafted logits.
//
// Uncached: it projects the whole context every call. Convenient for a one-shot round (and
// for the parity gate, which is handed the reference's fused context directly), but a
// generation loop should hold a DFlashContext and call DraftBlockCtx.
func (d *blockTrunk) DraftBlock(be Backend, fused, blockIn [][]float32) ([][]float32, error) {
	ctx := d.NewContext()
	d.ExtendContext(be, ctx, fused)
	return d.DraftBlockCtx(be, ctx, blockIn)
}

// DrafterHeadLogits applies the TARGET's LM head to one already-normed hidden row and
// returns next-token logits over the target vocabulary. It is the output end that a
// drafter shipping no `lm_head` of its own (DFlash) borrows from the target.
//
// It deliberately does NOT apply the target's final norm — the drafter has already
// applied its OWN `norm`, and running both would be a second normalization the reference
// does not do. That is the only difference from logitsFromHidden, and it is why this
// cannot just call it. The post-head transforms (Gemma softcap, Granite logit scale) ARE
// applied, mirroring the reference's compute_logits.
//
// Declared here rather than in model.go on purpose: model.go is in the parity manifest's
// `core` set, and adding a drafter-only accessor there would re-stale all 23 families'
// deps_hash for a function no existing forward calls.
func (m *Model) DrafterHeadLogits(h []float32) []float32 {
	arch := m.w.arch
	logits := make([]float32, arch.VocabSize)
	if arch.TiedLMHead {
		matmul(m.be, &m.w.Embed, h, logits, 1)
	} else {
		matmul(m.be, &m.w.LMHead, h, logits, 1)
	}
	if arch.FinalLogitSoftcap > 0 {
		softcap := float32(arch.FinalLogitSoftcap)
		for i, v := range logits {
			logits[i] = softcap * float32(math.Tanh(float64(v/softcap)))
		}
	}
	if arch.LogitScale != 0 && arch.LogitScale != 1 {
		inv := float32(1 / arch.LogitScale)
		for i := range logits {
			logits[i] *= inv
		}
	}
	return logits
}

// DrafterEmbedBlock fills the drafter's block input with the TARGET's token embeddings —
// the input end DFlash borrows. ids must be blockSize long (slot 0 the anchor, the rest
// the mask token).
func (m *Model) DrafterEmbedBlock(ids []int) [][]float32 {
	rows := make([][]float32, len(ids))
	for i, id := range ids {
		rows[i] = make([]float32, m.w.arch.HiddenDim)
		m.embedToken(id, rows[i])
	}
	return rows
}

// layer runs one trunk layer in place over the block rows.
//
// The asymmetry worth naming: `input_layernorm` normalizes the BLOCK only. The context's
// K/V are projected from the fused context RAW (in ExtendContext) — the reference passes
// `target_hidden` straight into k_proj/v_proj while only `hidden_states` goes through the
// norm. Norming both would be the natural-looking port and would be wrong.
func (d *blockTrunk) layer(be Backend, l *dflashLayer, ctxK, ctxV [][]float32, h [][]float32) {
	hid, hd := d.hidden, d.headDim
	qDim, kvDim := d.nHeads*hd, d.nKV*hd
	nBlk, nCtx := len(h), len(ctxK)
	nKeys := nCtx + nBlk

	// Block rows, normed, are the attention input for Q and for the block's own K/V.
	xb := make([][]float32, nBlk)
	for i, row := range h {
		x := append([]float32(nil), row...)
		rmsNorm(x, l.inputNorm, 1, hid, d.normEps, false)
		xb[i] = x
	}

	// Keys/values: the context's come straight from the cache (projected and roped once,
	// when those positions were committed); only the BLOCK's are computed here. RoPE
	// positions are ABSOLUTE over [context ‖ block], which is what makes the block's start
	// at nCtx and q's likewise.
	keys := make([][]float32, nKeys)
	vals := make([][]float32, nKeys)
	copy(keys, ctxK)
	copy(vals, ctxV)
	for j := nCtx; j < nKeys; j++ {
		k := make([]float32, kvDim)
		v := make([]float32, kvDim)
		matmul(be, &l.k, xb[j-nCtx], k, 1)
		matmul(be, &l.v, xb[j-nCtx], v, 1)
		rmsNorm(k, l.kNorm, d.nKV, hd, d.normEps, false) // per-head, Qwen3 style
		applyRoPE(k, d.nKV, hd, j, d.invFreq, 1)         // V is NOT roped and NOT normed
		keys[j], vals[j] = k, v
	}

	scale := 1 / math.Sqrt(float64(hd))
	scores := make([]float64, nKeys)
	ctxRow := make([]float32, qDim)
	attnOut := make([]float32, hid)
	for i := range nBlk {
		q := make([]float32, qDim)
		matmul(be, &l.q, xb[i], q, 1)
		rmsNorm(q, l.qNorm, d.nHeads, hd, d.normEps, false)
		applyRoPE(q, d.nHeads, hd, nCtx+i, d.invFreq, 1)

		// Non-causal: every block query attends every context key AND every block key,
		// including positions after itself. No mask, by construction.
		for hI := range d.nHeads {
			kvH := hI / (d.nHeads / d.nKV) // GQA broadcast
			qo, ko := hI*hd, kvH*hd
			maxS := math.Inf(-1)
			for j := range nKeys {
				var dot float64
				kr := keys[j][ko : ko+hd]
				for t, qv := range q[qo : qo+hd] {
					dot += float64(qv) * float64(kr[t])
				}
				s := dot * scale
				scores[j] = s
				if s > maxS {
					maxS = s
				}
			}
			var sum float64
			for j := range nKeys {
				e := math.Exp(scores[j] - maxS)
				scores[j] = e
				sum += e
			}
			for t := range hd {
				var acc float64
				for j := range nKeys {
					acc += scores[j] * float64(vals[j][ko+t])
				}
				ctxRow[qo+t] = float32(acc / sum)
			}
		}
		matmul(be, &l.o, ctxRow, attnOut, 1)
		for t := range hid {
			h[i][t] += attnOut[t]
		}
	}

	// SwiGLU MLP over the post-attention residual.
	gate := make([]float32, d.inter)
	up := make([]float32, d.inter)
	down := make([]float32, hid)
	for i := range nBlk {
		x := append([]float32(nil), h[i]...)
		rmsNorm(x, l.postAttnNorm, 1, hid, d.normEps, false)
		matmul(be, &l.gate, x, gate, 1)
		matmul(be, &l.up, x, up, 1)
		for t := range gate {
			gate[t] = silu(gate[t]) * up[t]
		}
		matmul(be, &l.down, gate, down, 1)
		for t := range hid {
			h[i][t] += down[t]
		}
	}
}
