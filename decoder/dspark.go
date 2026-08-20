package decoder

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"

	"github.com/townsendmerino/aikit/embed"
	"github.com/townsendmerino/aikit/linalg"
)

// DSparkDrafter is an imported DeepSeek DSpark block drafter (P10 / docs/spec/08): the same
// non-causal block trunk DFlash uses, plus the three things DFlash does not have — its own
// embedding and LM head, a rank-256 Markov chain, and a confidence head.
//
// It reuses blockTrunk rather than reimplementing the forward, and that is a measured claim,
// not a convenience: DeepSpec's `_forward_backbone` and z-lab's `DFlashDraftModel.forward`
// compute the same thing, down to the split RoPE application. See blockTrunk's doc.
//
// The differences that DO matter, all of which a port gets wrong silently:
//
//   - **logits_start = 0.** All blockSize positions are draft predictions; slot 0 both embeds
//     the anchor AND predicts the first token. DFlash reserves slot 0 and predicts from 1.
//     Slicing the wrong one makes every draft land one position late, which halves acceptance
//     while the text stays correct — so nothing crashes and no gate but this one notices.
//   - **Its own embed/head.** DSpark ships frozen COPIES of the target's (778 M of its 1.39 B).
//     They are loaded here; reusing the resident target's instead is a later optimization that
//     must be proven equal first, not assumed.
//   - **The Markov chain is SEQUENTIAL.** logits[i] += w2(w1[prev]) where prev is the token
//     just sampled at i-1, so the block is parallel in the trunk and serial in a blockSize-step
//     scalar chain. Each step is a [vocab, 256] matvec — small against a layer, but latency-serial
//     and therefore inside the draft term that gate 3 is most sensitive to.
//   - **The confidence head is adaptive block LENGTH, not a fire/don't-fire router.** Measured:
//     gating trims the proposal (chat 6.96 -> 4.87 positions) and barely moves acceptance
//     (3.04 -> 2.96), so what it buys is a cheaper verify. See docs/spec/08.
type DSparkDrafter struct {
	blockTrunk
	blockSize, maskTokenID int
	vocab, markovRank      int
	targetLayerIDs         []int
	confWithMarkov         bool

	embed    linalg.WeightMat // [vocab, hidden] — its own, not the target's
	lmHead   linalg.WeightMat // [vocab, hidden]
	markovW1 linalg.WeightMat // [vocab, rank] — indexed by the PREVIOUS token id
	markovW2 linalg.WeightMat // [vocab, rank] — projects the rank latent to a logit bias
	confW    linalg.WeightMat // [1, hidden(+rank)]
	confB    float32

	st *embed.SafetensorsFile
}

// BlockSize is the trained block width; DSpark proposes all BlockSize positions.
func (d *DSparkDrafter) BlockSize() int { return d.blockSize }

// MaskTokenID is the id the un-anchored block slots carry.
func (d *DSparkDrafter) MaskTokenID() int { return d.maskTokenID }

// TargetLayerIDs are the target layer OUTPUTS feeding `fc` — the ForwardCapture convention.
func (d *DSparkDrafter) TargetLayerIDs() []int { return append([]int(nil), d.targetLayerIDs...) }

// Close releases the drafter's mmap'd weights.
func (d *DSparkDrafter) Close() error {
	if d.st != nil {
		return d.st.Close()
	}
	return nil
}

type dsparkConfig struct {
	HiddenSize        int     `json:"hidden_size"`
	NumHiddenLayers   int     `json:"num_hidden_layers"`
	NumAttentionHeads int     `json:"num_attention_heads"`
	NumKeyValueHeads  int     `json:"num_key_value_heads"`
	HeadDim           int     `json:"head_dim"`
	IntermediateSize  int     `json:"intermediate_size"`
	VocabSize         int     `json:"vocab_size"`
	RMSNormEps        float64 `json:"rms_norm_eps"`
	// RoPE in BOTH spellings. DSpark's config is a deepcopy of the target's, saved by a
	// transformers >=5.10, so it carries the nested rope_parameters and NO top-level
	// rope_theta — the mirror of the granite case, where the RELEASED checkpoint had only the
	// flat field. Accept either rather than assume the one this checkpoint happens to use.
	RopeTheta             float64         `json:"rope_theta"`
	RopeParameters        json.RawMessage `json:"rope_parameters"`
	BlockSize             int             `json:"block_size"`
	TargetLayerIDs        []int           `json:"target_layer_ids"`
	MaskTokenID           int             `json:"mask_token_id"`
	MarkovRank            int             `json:"markov_rank"`
	MarkovHeadType        string          `json:"markov_head_type"`
	EnableConfidenceHead  bool            `json:"enable_confidence_head"`
	ConfidenceHeadWithMkv bool            `json:"confidence_head_with_markov"`
}

// LoadDSparkDrafter loads a DSpark drafter from dir (config.json + model.safetensors, the
// deepseek-ai layout). Its hidden size must equal the target's — it consumes the target's
// hidden states directly.
func LoadDSparkDrafter(dir string) (*DSparkDrafter, error) {
	cfgBytes, err := os.ReadFile(filepath.Join(dir, "config.json"))
	if err != nil {
		return nil, fmt.Errorf("dspark: read config: %w", err)
	}
	var c dsparkConfig
	if err := json.Unmarshal(cfgBytes, &c); err != nil {
		return nil, fmt.Errorf("dspark: parse config: %w", err)
	}
	if c.HeadDim == 0 {
		c.HeadDim = c.HiddenSize / c.NumAttentionHeads
	}
	switch {
	case c.HiddenSize <= 0 || c.NumHiddenLayers <= 0 || c.NumAttentionHeads <= 0:
		return nil, fmt.Errorf("dspark: bad dims (hidden=%d layers=%d heads=%d)", c.HiddenSize, c.NumHiddenLayers, c.NumAttentionHeads)
	case c.BlockSize < 1:
		return nil, fmt.Errorf("dspark: block_size must be >= 1, got %d", c.BlockSize)
	case len(c.TargetLayerIDs) == 0:
		return nil, fmt.Errorf("dspark: target_layer_ids is required")
	case c.RopeTheta <= 0 && len(c.RopeParameters) == 0:
		return nil, fmt.Errorf("dspark: no rope_theta and no rope_parameters")
	case c.MarkovRank > 0 && c.MarkovHeadType != "vanilla":
		// GatedMarkovHead carries an RNN state this forward does not implement; refuse rather
		// than silently running the vanilla chain against gated weights.
		return nil, fmt.Errorf("dspark: markov_head_type %q unsupported (only \"vanilla\")", c.MarkovHeadType)
	}

	st, err := embed.OpenSafetensorsMmap(filepath.Join(dir, "model.safetensors"))
	if err != nil {
		return nil, fmt.Errorf("dspark: open safetensors: %w", err)
	}
	loaded := false
	defer func() {
		if !loaded {
			st.Close()
		}
	}()

	theta := c.RopeTheta
	if len(c.RopeParameters) > 0 {
		spec, _, perr := parseRopeFlat(c.RopeParameters)
		if perr != nil {
			return nil, fmt.Errorf("dspark: rope_parameters: %w", perr)
		}
		theta = spec.base
	}
	if theta <= 0 {
		return nil, fmt.Errorf("dspark: rope_theta must be >0, got %v", theta)
	}

	h, hd := c.HiddenSize, c.HeadDim
	qDim, kvDim := c.NumAttentionHeads*hd, c.NumKeyValueHeads*hd
	d := &DSparkDrafter{
		hidden: h, nHeads: c.NumAttentionHeads, nKV: c.NumKeyValueHeads, headDim: hd,
		inter: c.IntermediateSize, normEps: c.RMSNormEps,
		invFreq:   computeInvFreq(theta, hd, nil),
		layers:    make([]dflashLayer, c.NumHiddenLayers),
		blockSize: c.BlockSize, maskTokenID: c.MaskTokenID,
		vocab: c.VocabSize, markovRank: c.MarkovRank,
		targetLayerIDs: c.TargetLayerIDs, confWithMarkov: c.ConfidenceHeadWithMkv,
		st: st,
	}

	mat := func(dst *linalg.WeightMat, name string, rows, cols int) error {
		m, err := loadMat(st, name, rows, cols)
		if err != nil {
			return fmt.Errorf("dspark: %w", err)
		}
		*dst = m
		return nil
	}
	vec := func(dst *[]float32, name string, n int) error {
		v, err := st.TensorF32(name, n)
		if err != nil {
			return fmt.Errorf("dspark: %w", err)
		}
		*dst = v
		return nil
	}

	for _, m := range []struct {
		dst        *linalg.WeightMat
		name       string
		rows, cols int
	}{
		{&d.fc, "fc.weight", h, len(c.TargetLayerIDs) * h},
		{&d.embed, "embed_tokens.weight", c.VocabSize, h},
		{&d.lmHead, "lm_head.weight", c.VocabSize, h},
	} {
		if err := mat(m.dst, m.name, m.rows, m.cols); err != nil {
			return nil, err
		}
	}
	if err := vec(&d.hiddenNorm, "hidden_norm.weight", h); err != nil {
		return nil, err
	}
	if err := vec(&d.finalNorm, "norm.weight", h); err != nil {
		return nil, err
	}
	if c.MarkovRank > 0 {
		if err := mat(&d.markovW1, "markov_head.markov_w1.weight", c.VocabSize, c.MarkovRank); err != nil {
			return nil, err
		}
		if err := mat(&d.markovW2, "markov_head.markov_w2.weight", c.VocabSize, c.MarkovRank); err != nil {
			return nil, err
		}
	}
	if c.EnableConfidenceHead {
		in := h
		if c.ConfidenceHeadWithMkv {
			in += c.MarkovRank
		}
		if err := mat(&d.confW, "confidence_head.proj.weight", 1, in); err != nil {
			return nil, err
		}
		b, berr := st.TensorF32("confidence_head.proj.bias", 1)
		if berr != nil {
			return nil, fmt.Errorf("dspark: %w", berr)
		}
		d.confB = b[0]
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

// EmbedBlock embeds the block's token ids with DSpark's OWN embedding table.
func (d *DSparkDrafter) EmbedBlock(ids []int) [][]float32 {
	rows := make([][]float32, len(ids))
	for i, id := range ids {
		rows[i] = make([]float32, d.hidden)
		d.embed.Row(id, rows[i])
	}
	return rows
}

// BaseLogits applies DSpark's own LM head to one trunk row — the pre-Markov draft logits.
func (d *DSparkDrafter) BaseLogits(be Backend, h []float32) []float32 {
	out := make([]float32, d.vocab)
	matmul(be, &d.lmHead, h, out, 1)
	return out
}

// SampleBlock greedily drafts the whole block, applying the Markov correction sequentially:
// each position's logits are biased by the token sampled at the position before it, starting
// from firstPrev (the anchor). Returns the drafted ids and the corrected logits per position.
//
// Sequential by construction — this is the chain that makes DSpark's draft latency-serial, and
// it is why the block is only partly a "one pass" proposal.
func (d *DSparkDrafter) SampleBlock(be Backend, trunk [][]float32, firstPrev int) (ids []int, corrected [][]float32) {
	ids = make([]int, 0, len(trunk))
	corrected = make([][]float32, 0, len(trunk))
	prev := firstPrev
	lat := make([]float32, d.markovRank)
	for _, row := range trunk {
		lg := d.BaseLogits(be, row)
		if d.markovRank > 0 {
			d.markovW1.Row(prev, lat)        // rank-256 latent for the previous token
			bias := make([]float32, d.vocab) // w2 · latent
			matmul(be, &d.markovW2, lat, bias, 1)
			for i := range lg {
				lg[i] += bias[i]
			}
		}
		next := argmax(lg)
		ids = append(ids, next)
		corrected = append(corrected, lg)
		prev = next
	}
	return ids, corrected
}

// Confidence returns the per-position accept-rate LOGIT (apply a sigmoid for a probability).
// prevIDs[i] is the token preceding block position i — the anchor for i=0, then the drafted
// tokens. Returns nil when the checkpoint ships no confidence head.
//
// This is DSpark's adaptive block length: the proposal is truncated at the first position whose
// sigmoid falls below the caller's threshold.
func (d *DSparkDrafter) Confidence(be Backend, trunk [][]float32, prevIDs []int) []float32 {
	if d.confW.Rows() == 0 {
		return nil
	}
	out := make([]float32, len(trunk))
	feat := make([]float32, d.hidden+d.markovRank)
	for i, row := range trunk {
		copy(feat, row)
		n := d.hidden
		if d.confWithMarkov {
			d.markovW1.Row(prevIDs[i], feat[d.hidden:])
			n += d.markovRank
		}
		acc := make([]float32, 1)
		matmul(be, &d.confW, feat[:n], acc, 1)
		out[i] = acc[0] + d.confB
	}
	return out
}
