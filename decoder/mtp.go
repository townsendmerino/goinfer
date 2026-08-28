package decoder

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/townsendmerino/aikit/embed"
	"github.com/townsendmerino/aikit/linalg"
)

// MTPHead is a multi-token-prediction / NextN self-draft head, shipped INSIDE its target's
// checkpoint and skipped by every existing load path (docs/spec/09-mtp-heads.md).
//
// Shape, per the DeepSeek-V3 MTP layout the Qwen3.5/3.6/3.8 checkpoints use:
//
//	x       = fc( concat( preFCNormEmbed(embed(next)), preFCNormHidden(targetHidden) ) )
//	x       = one standard transformer block over x   (QK-normed GQA + SwiGLU)
//	logits  = targetLMHead( finalNorm(x) )
//
// It differs from the EAGLE-3 head in [eagle.go] by exactly one structural thing: MTP projects the
// concat down to hidden with `fc` and then runs a NORMAL block, where EAGLE folds the concat into
// the attention projections (its q/k/v read 2*hidden). Everything else — RoPE, the slice-based
// per-draft KV state, the residual shape — is the same, which is why this file mirrors that one
// rather than inventing a second way to do it.
//
// SCOPE: this is the Gate 1 measurement adapter. It loads the head and runs it. Nothing here is
// wired into serving, the router, or any generation path, and specRollbackSafe is not consulted —
// see 09, "the seam blocks deployment, not measurement".
type MTPHead struct {
	hidden, nHeads, nKV, headDim, inter int
	normEps                             float64

	preFCNormEmbed, preFCNormHidden []float32
	fc                              linalg.WeightMat // [hidden, 2*hidden]
	finalNorm                       []float32

	// lw is a synthetic LayerWeights holding the head's single block, so the head runs through the
	// TARGET's own forward — m.qwen35Attention and gatedMLP — rather than a second implementation
	// of them. That matters more than it looks: this family's attention emits [query ‖ gate]
	// interleaved per head, QK-norms the query half only, applies partial RoPE, and multiplies the
	// context by sigmoid(gate) before o_proj. Reimplementing that for a measurement is how a
	// plausible-but-wrong acceptance number gets produced.
	lw *LayerWeights

	// attnLayer is an index into the target's arch chosen so isLinearLayer(i) is FALSE — the MTP
	// block is a softmax-attention block, and the index selects the RoPE config the forward reads.
	attnLayer int

	st *embed.SafetensorsFile
}

// MTPState is one draft block's KV for the head's single layer: a real KVCache with NumLayers=1,
// thrown away per round. It never touches the target's cache and so never needs the rollback
// machinery that refuses these families (09, "the seam blocks deployment, not measurement").
type MTPState struct{ cache *KVCache }

// HasMTPHead reports whether dir carries a head, and by which route it was detected.
//
// TWO DETECTION PATHS, because the formats disagree and a single one reports "no head" on a
// checkpoint that has one (measured, 09 Gate 0): GGUF declares the count in arch-prefixed metadata
// (`…nextn_predict_layers`), while the safetensors Qwen checkpoints declare NOTHING in config.json
// and are discoverable only by tensor presence in the index.
func HasMTPHead(dir string) (bool, string) {
	idx := filepath.Join(dir, shardIndexFile)
	if b, err := os.ReadFile(idx); err == nil {
		var m struct {
			WeightMap map[string]string `json:"weight_map"`
		}
		if json.Unmarshal(b, &m) == nil {
			for k := range m.WeightMap {
				if strings.HasPrefix(k, "mtp.") {
					return true, "safetensors: mtp.* tensors present (config.json declares nothing)"
				}
			}
		}
	}
	if b, err := os.ReadFile(filepath.Join(dir, "config.json")); err == nil {
		var c map[string]any
		if json.Unmarshal(b, &c) == nil {
			t := c
			if tc, ok := c["text_config"].(map[string]any); ok {
				t = tc
			}
			if n, ok := t["num_nextn_predict_layers"].(float64); ok && n > 0 {
				return true, "safetensors: config.json declares num_nextn_predict_layers"
			}
		}
	}
	return false, "no mtp.* tensors and no declared nextn layers"
}

// mtpConfig is the subset of the TARGET's config the head needs. The head has no config of its own
// — it ships inside the target's — so its geometry is the target's geometry.
type mtpConfig struct {
	HiddenSize        int     `json:"hidden_size"`
	NumAttentionHeads int     `json:"num_attention_heads"`
	NumKeyValueHeads  int     `json:"num_key_value_heads"`
	HeadDim           int     `json:"head_dim"`
	IntermediateSize  int     `json:"intermediate_size"`
	RopeTheta         float64 `json:"rope_theta"`
	RMSNormEps        float64 `json:"rms_norm_eps"`
}

// LoadMTPHead reads the head the existing loaders skip. safetensors only for now: that is the
// format the Gate 1 target ships in, and a GGUF head lives in a trailing blk.<N> whose tensors the
// GGUF reader would need to be asked for separately (09, "what it requires").
func LoadMTPHead(dir string) (*MTPHead, error) {
	raw, err := os.ReadFile(filepath.Join(dir, "config.json"))
	if err != nil {
		return nil, fmt.Errorf("mtp: read config: %w", err)
	}
	var outer map[string]json.RawMessage
	if err := json.Unmarshal(raw, &outer); err != nil {
		return nil, fmt.Errorf("mtp: parse config: %w", err)
	}
	body := raw
	if tc, ok := outer["text_config"]; ok {
		body = tc
	}
	var c mtpConfig
	if err := json.Unmarshal(body, &c); err != nil {
		return nil, fmt.Errorf("mtp: parse text config: %w", err)
	}
	if c.HeadDim == 0 && c.NumAttentionHeads > 0 {
		c.HeadDim = c.HiddenSize / c.NumAttentionHeads
	}
	if c.RopeTheta == 0 {
		c.RopeTheta = 10000
	}
	if c.RMSNormEps == 0 {
		c.RMSNormEps = 1e-6
	}

	st, err := openCheckpointMmap(dir)
	if err != nil {
		return nil, fmt.Errorf("mtp: open checkpoint: %w", err)
	}
	loaded := false
	defer func() {
		if !loaded {
			st.Close()
		}
	}()

	hid := c.HiddenSize
	qDim := c.NumAttentionHeads * c.HeadDim
	kvDim := c.NumKeyValueHeads * c.HeadDim

	h := &MTPHead{
		hidden: hid, nHeads: c.NumAttentionHeads, nKV: c.NumKeyValueHeads,
		headDim: c.HeadDim, inter: c.IntermediateSize, normEps: c.RMSNormEps,
		st: st,
	}

	qa := &qwenAttnWeights{}
	lw := &LayerWeights{qattn: qa}
	mats := []struct {
		dst        *linalg.WeightMat
		name       string
		rows, cols int
	}{
		{&h.fc, "mtp.fc.weight", hid, 2 * hid},
		// q_proj emits [query ‖ gate] per head, so it is 2*qDim rows — the shape the target's own
		// attention expects, and the one a naive qDim guess gets wrong (caught by loadMat).
		{&qa.qProj, "mtp.layers.0.self_attn.q_proj.weight", 2 * qDim, hid},
		{&qa.kProj, "mtp.layers.0.self_attn.k_proj.weight", kvDim, hid},
		{&qa.vProj, "mtp.layers.0.self_attn.v_proj.weight", kvDim, hid},
		{&qa.oProj, "mtp.layers.0.self_attn.o_proj.weight", hid, qDim},
		{&lw.GateProj, "mtp.layers.0.mlp.gate_proj.weight", c.IntermediateSize, hid},
		{&lw.UpProj, "mtp.layers.0.mlp.up_proj.weight", c.IntermediateSize, hid},
		{&lw.DownProj, "mtp.layers.0.mlp.down_proj.weight", hid, c.IntermediateSize},
	}
	for _, m := range mats {
		w, err := loadMat(st, m.name, m.rows, m.cols)
		if err != nil {
			return nil, fmt.Errorf("mtp: load %s: %w", m.name, err)
		}
		*m.dst = w
	}

	norms := []struct {
		dst  *[]float32
		name string
		n    int
	}{
		{&h.preFCNormEmbed, "mtp.pre_fc_norm_embedding.weight", hid},
		{&h.preFCNormHidden, "mtp.pre_fc_norm_hidden.weight", hid},
		{&lw.PreAttnNorm, "mtp.layers.0.input_layernorm.weight", hid},
		{&lw.PreMLPNorm, "mtp.layers.0.post_attention_layernorm.weight", hid},
		{&qa.qNorm, "mtp.layers.0.self_attn.q_norm.weight", c.HeadDim},
		{&qa.kNorm, "mtp.layers.0.self_attn.k_norm.weight", c.HeadDim},
		{&h.finalNorm, "mtp.norm.weight", hid},
	}
	for _, n := range norms {
		// TensorF32, not Tensor().Float32s(): these norms ship BF16 in this checkpoint and the raw
		// accessor refuses to reinterpret. TensorF32 is what the target's own loader uses for the
		// same tensors, and it converts.
		f, err := st.TensorF32(n.name, n.n)
		if err != nil {
			return nil, fmt.Errorf("mtp: %s as f32: %w", n.name, err)
		}
		*n.dst = append([]float32(nil), f...) // copy off the mmap before Close
	}
	h.lw = lw

	loaded = true
	return h, nil
}

// NewMTPState builds the head's one-layer KV. capHint bounds a single draft block, not a
// generation: the state is discarded per round.
func (m *Model) NewMTPState(h *MTPHead, capHint int) *MTPState {
	c := NewKVCache(1, h.nKV, h.headDim, 0, capHint)
	c.scr = newDecodeScratch(m.w.arch)
	return &MTPState{cache: c}
}

// MTPStep runs the head for one drafted position and returns its final hidden.
//
//	x = fc( concat( preFCNormEmbed(embed(tok)), preFCNormHidden(targetHidden) ) )
//	x = one block  — the TARGET's own qwen35Attention + gatedMLP, not a second copy of them
//	x = finalNorm(x)
//
// embedRow is the embedding of the token whose successor is being predicted; feature is the hidden
// state at that position (the target's, for the first draft step; the head's own thereafter).
// The caller projects the result with the target's LM head — these heads carry none (09, Gate 0).
func (m *Model) MTPStep(h *MTPHead, embedRow, feature []float32, pos int, st *MTPState) ([]float32, error) {
	arch := m.w.arch
	hid := h.hidden
	eps := arch.NormEps

	e := append([]float32(nil), embedRow...)
	rmsNorm(e, h.preFCNormEmbed, 1, hid, eps, arch.RMSAddOne)
	f := append([]float32(nil), feature...)
	rmsNorm(f, h.preFCNormHidden, 1, hid, eps, arch.RMSAddOne)
	cat := make([]float32, 2*hid)
	copy(cat[:hid], e)
	copy(cat[hid:], f)

	x := make([]float32, hid)
	matmul(m.be, &h.fc, cat, x, 1) // the one structural difference from an EAGLE head

	// Attention sub-block (Pre2), through the target's forward.
	n := append([]float32(nil), x...)
	rmsNorm(n, h.lw.PreAttnNorm, 1, hid, eps, arch.RMSAddOne)
	attn := m.qwen35Attention(n, h.lw, arch, st.cache, 0, pos)
	for i := range x {
		x[i] += attn[i]
	}

	// FFN sub-block (Pre2), SwiGLU, also through the target's forward.
	n2 := append([]float32(nil), x...)
	rmsNorm(n2, h.lw.PreMLPNorm, 1, hid, eps, arch.RMSAddOne)
	ffn := make([]float32, hid)
	if err := gatedMLP(n2, ffn, h.lw, arch, m.be, st.cache.scr, nil); err != nil {
		return nil, fmt.Errorf("mtp: gatedMLP: %w", err)
	}
	for i := range x {
		x[i] += ffn[i]
	}

	st.cache.Advance()
	rmsNorm(x, h.finalNorm, 1, hid, eps, arch.RMSAddOne)
	return x, nil
}

// MTPPrefill builds the head's KV over an already-realized prefix, running it at each position with
// the target's own hidden there, so the head's attention sees the history the target produced.
func (m *Model) MTPPrefill(h *MTPHead, toks []int, feats [][]float32, capHint int) (*MTPState, error) {
	st := m.NewMTPState(h, capHint)
	emb := make([]float32, h.hidden)
	for j := range toks {
		m.embedToken(toks[j], emb)
		if _, err := m.MTPStep(h, emb, feats[j], j, st); err != nil {
			return nil, err
		}
	}
	return st, nil
}

// MTPDraftFrom drafts k tokens autoregressively. Step 0 uses the target's feature; later steps use
// the head's OWN hidden — the self-recurrence 05 measured as the multi-step weak link for an
// imported head, and the thing joint training is supposed to improve.
func (m *Model) MTPDraftFrom(h *MTPHead, st *MTPState, firstTok int, seedFeature []float32, pos, k int, cache *KVCache) ([]int, error) {
	out := make([]int, 0, k)
	emb := make([]float32, h.hidden)
	tok, feat := firstTok, seedFeature
	for i := 0; i < k; i++ {
		m.embedToken(tok, emb)
		hid, err := m.MTPStep(h, emb, feat, pos+i, st)
		if err != nil {
			return nil, err
		}
		next := argmax(m.mtpProject(hid, cache))
		out = append(out, next)
		tok, feat = next, hid
	}
	return out, nil
}

// mtpProject applies the TARGET's LM head to a head hidden ALREADY normed by MTPStep.
//
// Deliberately not logitsFromHidden: that applies the target's own FinalNorm first, and the head
// carries its own mtp.norm which MTPStep has applied. Routing through it would norm twice —
// silently, since the result is still a plausible logit vector.
// mtpProject applies the target's output projection to a head hidden state. The MTP head ships no
// LM head of its own on this family, so it borrows the trunk's — which is why a *KVCache is needed
// at all: only for its scratch.
//
// IT TOUCHES NO KEYS OR VALUES. The target's KV is unmodified, which is what lets the Gate 1 probe
// draft repeatedly against a cache the target already filled.
//
// BUT IT RETURNS THE TARGET'S SHARED LOGITS SCRATCH, and that is a trap for anything past the
// probe: a caller that interleaves drafting with target forwards will have its target logits
// silently overwritten by the next draft step. The probe is safe because it captures every target
// feature up front and only then drafts. An integration that alternates must copy out of, or not
// share, this buffer.
func (m *Model) mtpProject(x []float32, cache *KVCache) []float32 {
	logits := cache.scr.logits
	if m.w.arch.TiedLMHead {
		matmulInto(cache.scr.ws, m.be, &m.w.Embed, x, logits, 1)
	} else {
		matmulInto(cache.scr.ws, m.be, &m.w.LMHead, x, logits, 1)
	}
	return logits
}

// Close releases the head's mmap. The WeightMats alias it, so nothing may use the head after this.
func (h *MTPHead) Close() error {
	if h.st != nil {
		return h.st.Close()
	}
	return nil
}
