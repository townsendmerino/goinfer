//go:build darwin

package metal

import (
	"fmt"
	"math"
	"math/rand"
	"path/filepath"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestMoEKSlots_bitIdentical (D-P03): resident MoE decode with the k selected experts in four dispatches (k-slot
// gate|up, k-row SwiGLU, k-slot down, combine) writes the same logits, bit for bit, as the per-slot loop's 3k. The
// model is a tiny qwen2_moe whose experts all differ and whose shared expert is live, so a slot that read another
// slot's expert, activation or weight, or a combine in another order, changes the logits (writeMoEIdentical's
// identical experts could not see the first two).
func TestMoEKSlots_bitIdentical(t *testing.T) {
	if _, err := CreateSystemDefaultDevice(); err != nil {
		t.Skipf("no metal device: %v", err)
	}
	const nE, k = 8, 3
	dir := t.TempDir()
	writeMoEDistinct(t, dir, genTinyWeights(rand.New(rand.NewSource(77))), nE, k, rand.New(rand.NewSource(78)))
	m, err := decoder.Load(dir, decoder.Options{Quant: "int4"})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	defer m.Close()
	r, err := buildResident(m)
	if err != nil {
		t.Fatalf("resident: %v", err)
	}
	defer r.Close()
	if r.moe == nil || !r.moe.kSlots || r.moe.k != k { // kSlots: the shape admits it; moeKSlotsOn picks the arm
		t.Fatalf("moe %v: the k-slot path is not selected, this tests nothing", r.moe != nil)
	}
	prev := moeKSlotsOn
	defer func() { moeKSlotsOn = prev; r.stopExec() }()
	ids := []int{3, 11, 19, 27, 35, 43, 51, 59, 2, 10, 18, 26, 34, 42, 50, 58}
	run := func(on bool) [][]float32 {
		moeKSlotsOn = on
		r.stopExec()
		var out [][]float32
		for pos, tok := range ids {
			out = append(out, append([]float32(nil), r.Forward(tok, pos)...))
		}
		return out
	}
	loop, kslots := run(false), run(true)
	for i := range ids {
		for j, x := range loop[i] {
			if math.Float32bits(x) != math.Float32bits(kslots[i][j]) {
				t.Fatalf("position %d logit %d: k-slot %v, per-slot loop %v", i, j, kslots[i][j], x)
			}
		}
	}
	// the routing must actually vary, or a slot mix-up could hide: count distinct expert sets across positions
	sets := map[string]bool{}
	moeKSlotsOn = true
	r.stopExec()
	for pos, tok := range ids {
		r.Forward(tok, pos)
		sets[fmt.Sprint(r.moe.rIdx.U32s()[:k])] = true
	}
	if len(sets) < 4 {
		t.Fatalf("only %d distinct routed sets over %d positions: the fixture does not exercise the slots", len(sets), len(ids))
	}
	t.Logf("%d positions, %d distinct routed sets (last layer): logits bit-identical", len(ids), len(sets))
}

// writeMoEDistinct writes a qwen2_moe with nE different experts, top-k routing (renormalised) and a live gated shared
// expert, from w's attention weights and router.
func writeMoEDistinct(t *testing.T, dir string, w *tinyWeights, nE, k int, rng *rand.Rand) {
	t.Helper()
	cfg := fmt.Sprintf(`{"model_type":"qwen2_moe","vocab_size":%d,"hidden_size":%d,
		"num_hidden_layers":%d,"num_attention_heads":%d,"num_key_value_heads":%d,"head_dim":%d,
		"intermediate_size":%d,"moe_intermediate_size":%d,"shared_expert_intermediate_size":%d,
		"num_experts":%d,"num_experts_per_tok":%d,"norm_topk_prob":true,
		"max_position_embeddings":256,"rms_norm_eps":1e-6,"rope_theta":10000}`,
		tmVocab, tmHidden, tmLayers, tmHeads, tmKVHeads, tmHeadDim, tmInter, tmInter, tmInter, nE, k)
	writeConfig(t, dir, cfg)
	rnd := func(n int, s float32) []float32 {
		d := make([]float32, n)
		for i := range d {
			d[i] = (rng.Float32()*2 - 1) * s
		}
		return d
	}
	ts := map[string]stf32{
		"model.embed_tokens.weight": {[]int{tmVocab, tmHidden}, w.embed},
		"model.norm.weight":         {[]int{tmHidden}, w.norm},
		"lm_head.weight":            {[]int{tmVocab, tmHidden}, w.lmHead},
	}
	for l := range tmLayers {
		attnTensors(ts, w, l)
		p := fmt.Sprintf("model.layers.%d.", l)
		ts[p+"mlp.gate.weight"] = stf32{[]int{nE, tmHidden}, rnd(nE*tmHidden, 0.8)}
		for e := range nE {
			ep := fmt.Sprintf("%smlp.experts.%d.", p, e)
			ts[ep+"gate_proj.weight"] = stf32{[]int{tmInter, tmHidden}, rnd(tmInter*tmHidden, 0.3)}
			ts[ep+"up_proj.weight"] = stf32{[]int{tmInter, tmHidden}, rnd(tmInter*tmHidden, 0.3)}
			ts[ep+"down_proj.weight"] = stf32{[]int{tmHidden, tmInter}, rnd(tmHidden*tmInter, 0.3)}
		}
		ts[p+"mlp.shared_expert.gate_proj.weight"] = stf32{[]int{tmInter, tmHidden}, rnd(tmInter*tmHidden, 0.3)}
		ts[p+"mlp.shared_expert.up_proj.weight"] = stf32{[]int{tmInter, tmHidden}, rnd(tmInter*tmHidden, 0.3)}
		ts[p+"mlp.shared_expert.down_proj.weight"] = stf32{[]int{tmHidden, tmInter}, rnd(tmHidden*tmInter, 0.3)}
		ts[p+"mlp.shared_expert_gate.weight"] = stf32{[]int{1, tmHidden}, rnd(tmHidden, 0.3)}
	}
	writeSTF32(t, filepath.Join(dir, "model.safetensors"), ts)
}
