package decoder

import (
	"encoding/json"
	"fmt"
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
	defer st.Close()

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
	return head, nil
}
