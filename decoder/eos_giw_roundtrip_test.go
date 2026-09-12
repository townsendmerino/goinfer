package decoder

import (
	"io"
	"os"
	"path/filepath"
	"testing"

	"github.com/townsendmerino/goinfer/internal/giw"
)

// buildEOSFixture writes a tiny synthetic llama-shaped safetensors checkpoint (same shape as
// resident_adapter_seam_test.go's buildLoRAFixture) whose config.json carries ONE eos id and
// whose generation_config.json adds a SECOND — the Qwen3 shape M-04 (docs/audit-2026-09-10.md)
// names: config.json's <|im_end|> (151645) beside generation_config.json's extra
// <|endoftext|> (151643).
func buildEOSFixture(t *testing.T) (dir string, wantMerged []int) {
	t.Helper()
	const hidden, heads, headDim, inter, vocab, layers = 8, 2, 4, 16, 16, 2
	qDim := heads * headDim
	dir = t.TempDir()
	cfg := `{"model_type":"llama","vocab_size":16,"hidden_size":8,"num_hidden_layers":2,
		"num_attention_heads":2,"num_key_value_heads":2,"head_dim":4,"intermediate_size":16,
		"max_position_embeddings":128,"rms_norm_eps":1e-6,"rope_theta":10000,"eos_token_id":151645}`
	if err := os.WriteFile(filepath.Join(dir, "config.json"), []byte(cfg), 0o644); err != nil {
		t.Fatal(err)
	}
	genCfg := `{"eos_token_id":[151645,151643]}`
	if err := os.WriteFile(filepath.Join(dir, "generation_config.json"), []byte(genCfg), 0o644); err != nil {
		t.Fatal(err)
	}
	fill := func(n, seed int) []float32 {
		d := make([]float32, n)
		for i := range d {
			d[i] = float32((i*5+seed)%11)*0.07 - 0.35
		}
		return d
	}
	ts := map[string]stTensor{
		"model.embed_tokens.weight": {[]int{vocab, hidden}, fill(vocab*hidden, 1)},
		"model.norm.weight":         {[]int{hidden}, fill(hidden, 2)},
		"lm_head.weight":            {[]int{vocab, hidden}, fill(vocab*hidden, 3)},
	}
	for l := range layers {
		p := func(s string) string { return "model.layers." + itoa(l) + s }
		ts[p(".self_attn.q_proj.weight")] = stTensor{[]int{qDim, hidden}, fill(qDim*hidden, 10+l)}
		ts[p(".self_attn.k_proj.weight")] = stTensor{[]int{qDim, hidden}, fill(qDim*hidden, 20+l)}
		ts[p(".self_attn.v_proj.weight")] = stTensor{[]int{qDim, hidden}, fill(qDim*hidden, 30+l)}
		ts[p(".self_attn.o_proj.weight")] = stTensor{[]int{hidden, qDim}, fill(hidden*qDim, 40+l)}
		ts[p(".mlp.gate_proj.weight")] = stTensor{[]int{inter, hidden}, fill(inter*hidden, 50+l)}
		ts[p(".mlp.up_proj.weight")] = stTensor{[]int{inter, hidden}, fill(inter*hidden, 60+l)}
		ts[p(".mlp.down_proj.weight")] = stTensor{[]int{hidden, inter}, fill(hidden*inter, 70+l)}
		ts[p(".input_layernorm.weight")] = stTensor{[]int{hidden}, fill(hidden, 80+l)}
		ts[p(".post_attention_layernorm.weight")] = stTensor{[]int{hidden}, fill(hidden, 90+l)}
	}
	writeSafetensors(t, filepath.Join(dir, "model.safetensors"), ts)
	return dir, []int{151645, 151643}
}

// TestLoad_staged_resolvesEOSIntoConfig is the direct M-04 gate: config.json alone would give
// {151645}; the merge with generation_config.json must land IN w.Cfg.EOSTokenID, not just on the
// Model's own eosIDs field, because Cfg is what a .giw bundle serializes.
func TestLoad_staged_resolvesEOSIntoConfig(t *testing.T) {
	dir, want := buildEOSFixture(t)
	m, err := Load(dir, Options{Quant: "f32"})
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	defer m.Close()

	if !eqInts(m.eosIDs, want) {
		t.Errorf("m.eosIDs = %v, want %v", m.eosIDs, want)
	}
	if got := m.Weights().Cfg.EOSIDs(); !eqInts(got, want) {
		t.Fatalf("Weights().Cfg.EOSIDs() = %v, want %v — the merge never reached Cfg, so a .giw "+
			"bundle serializing this Weights would only carry config.json's %v", got, want, m.Weights().Cfg.EOSIDs())
	}
}

// TestGIWRoundTrip_preservesMergedEOSIDs is the end-to-end M-04 gate: build the same fixture,
// load it staged (which now bakes the merged EOS set into Cfg), serialize it exactly the way
// cmd/prequant does (SerializeWeightsToForTarget -> giw.WriteStream), reload the resulting .giw
// file, and confirm the SECOND id (only ever named in generation_config.json, which the .giw
// loader has no directory to re-read) survives the round trip.
func TestGIWRoundTrip_preservesMergedEOSIDs(t *testing.T) {
	dir, want := buildEOSFixture(t)
	m, err := Load(dir, Options{Quant: "f32"})
	if err != nil {
		t.Fatalf("Load (staged): %v", err)
	}
	tok := []byte{} // fixture carries no tokenizer.json; a weights-only bundle is fine here

	out := filepath.Join(t.TempDir(), "model.giw")
	f, err := os.Create(out)
	if err != nil {
		t.Fatalf("create bundle: %v", err)
	}
	werr := giw.WriteStream(f, tok, func(w io.Writer) (int64, error) {
		return SerializeWeightsToForTarget(w, m.Weights(), "eos-fixture", GIWTargetNone)
	})
	if cerr := f.Close(); werr == nil {
		werr = cerr
	}
	m.Close()
	if werr != nil {
		t.Fatalf("write bundle: %v", werr)
	}

	m2, err := Load(out, Options{})
	if err != nil {
		t.Fatalf("Load (.giw): %v", err)
	}
	defer m2.Close()

	if !eqInts(m2.eosIDs, want) {
		t.Fatalf(".giw round-trip lost the merge: eosIDs = %v, want %v (M-04)", m2.eosIDs, want)
	}
}
