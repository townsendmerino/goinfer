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
	hidden, nHeads, nKV, headDim, inter int
	blockSize, maskTokenID              int
	normEps                             float64
	targetLayerIDs                      []int
	invFreq                             []float64

	fc         linalg.WeightMat // [hidden, len(targetLayerIDs)*hidden]
	hiddenNorm []float32        // RMSNorm on the fused context
	finalNorm  []float32        // RMSNorm before the target's LM head
	layers     []dflashLayer

	st *embed.SafetensorsFile // retained: the WeightMats alias its mmap
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
	RopeTheta         float64 `json:"rope_theta"`
	BlockSize         int     `json:"block_size"`
	DFlash            struct {
		MaskTokenID    int   `json:"mask_token_id"`
		TargetLayerIDs []int `json:"target_layer_ids"`
	} `json:"dflash_config"`
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
	if c.HeadDim == 0 {
		c.HeadDim = c.HiddenSize / c.NumAttentionHeads
	}
	switch {
	case c.HiddenSize <= 0 || c.NumHiddenLayers <= 0 || c.NumAttentionHeads <= 0:
		return nil, fmt.Errorf("dflash: bad dims (hidden=%d layers=%d heads=%d)", c.HiddenSize, c.NumHiddenLayers, c.NumAttentionHeads)
	case c.BlockSize < 2:
		return nil, fmt.Errorf("dflash: block_size must be >= 2, got %d", c.BlockSize)
	case len(c.DFlash.TargetLayerIDs) == 0:
		return nil, fmt.Errorf("dflash: dflash_config.target_layer_ids is required")
	case c.RopeTheta <= 0:
		return nil, fmt.Errorf("dflash: rope_theta must be >0, got %v", c.RopeTheta)
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
		hidden: h, nHeads: c.NumAttentionHeads, nKV: c.NumKeyValueHeads, headDim: hd,
		inter: c.IntermediateSize, blockSize: c.BlockSize, maskTokenID: c.DFlash.MaskTokenID,
		normEps: c.RMSNormEps, targetLayerIDs: c.DFlash.TargetLayerIDs,
		invFreq: computeInvFreq(c.RopeTheta, hd, nil),
		layers:  make([]dflashLayer, c.NumHiddenLayers),
		st:      st,
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
func (d *DFlashDrafter) FuseContext(be Backend, ctxCat [][]float32) ([][]float32, error) {
	want := len(d.targetLayerIDs) * d.hidden
	fused := make([][]float32, len(ctxCat))
	for i, row := range ctxCat {
		if len(row) != want {
			return nil, fmt.Errorf("dflash: context row %d is %d wide, want %d (%d layers x %d)",
				i, len(row), want, len(d.targetLayerIDs), d.hidden)
		}
		f := make([]float32, d.hidden)
		matmul(be, &d.fc, row, f, 1)
		rmsNorm(f, d.hiddenNorm, 1, d.hidden, d.normEps, false)
		fused[i] = f
	}
	return fused, nil
}

// DraftBlock runs the trunk over one block and returns its final-normed hidden states,
// [blockSize][hidden]. blockIn is the TARGET's embedding of the block's token ids (slot 0
// the anchor, the rest MASK); fused is FuseContext's output for the committed context.
// The caller applies the target's LM head to rows 1.. to get the drafted logits.
func (d *DFlashDrafter) DraftBlock(be Backend, fused, blockIn [][]float32) ([][]float32, error) {
	if len(blockIn) != d.blockSize {
		return nil, fmt.Errorf("dflash: block is %d rows, want block_size %d", len(blockIn), d.blockSize)
	}
	h := make([][]float32, len(blockIn))
	for i, row := range blockIn {
		if len(row) != d.hidden {
			return nil, fmt.Errorf("dflash: block row %d is %d wide, want %d", i, len(row), d.hidden)
		}
		h[i] = append([]float32(nil), row...)
	}
	for i := range d.layers {
		d.layer(be, &d.layers[i], fused, h)
	}
	out := make([][]float32, len(h))
	for i, row := range h {
		r := append([]float32(nil), row...)
		rmsNorm(r, d.finalNorm, 1, d.hidden, d.normEps, false)
		out[i] = r
	}
	return out, nil
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
// K/V are projected from the fused context RAW — the reference passes `target_hidden`
// straight into k_proj/v_proj while only `hidden_states` goes through the norm. Norming
// both would be the natural-looking port and would be wrong.
func (d *DFlashDrafter) layer(be Backend, l *dflashLayer, fused, h [][]float32) {
	hid, hd := d.hidden, d.headDim
	qDim, kvDim := d.nHeads*hd, d.nKV*hd
	nBlk, nCtx := len(h), len(fused)
	nKeys := nCtx + nBlk

	// Block rows, normed, are the attention input for Q and for the block's own K/V.
	xb := make([][]float32, nBlk)
	for i, row := range h {
		x := append([]float32(nil), row...)
		rmsNorm(x, l.inputNorm, 1, hid, d.normEps, false)
		xb[i] = x
	}

	// Keys/values: context first (from the RAW fused rows), then the block. RoPE positions
	// are ABSOLUTE over [context ‖ block], which is what makes q's positions start at nCtx.
	keys := make([][]float32, nKeys)
	vals := make([][]float32, nKeys)
	for j := range nKeys {
		var src []float32
		if j < nCtx {
			src = fused[j]
		} else {
			src = xb[j-nCtx]
		}
		k := make([]float32, kvDim)
		v := make([]float32, kvDim)
		matmul(be, &l.k, src, k, 1)
		matmul(be, &l.v, src, v, 1)
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
