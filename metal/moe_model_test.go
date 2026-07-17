//go:build darwin

package metal

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"sort"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// Tiny synthetic dims. Every GEMV contraction (hidden, ffnInter, qDim) is a multiple of 32
// so the int4 (group=32) packing is exact.
const (
	tmHidden  = 64
	tmHeads   = 4
	tmHeadDim = 16 // qDim = 64
	tmKVHeads = 2  // kvDim = 32
	tmLayers  = 2
	tmVocab   = 64
	tmInter   = 64
	tmExperts = 8
	tmTopK    = 2
)

// TestMoE_assemblyVsDense is the decisive MoE assembly gate. It builds two models that share
// every weight: a qwen2_moe whose 8 experts are IDENTICAL (and shared expert zeroed), and a
// plain dense qwen2 whose FFN equals that one expert. With identical experts + norm_topk_prob,
// the routed top-k always sums to weight 1 over copies of the same expert, so the MoE FFN is
// mathematically equal to the dense FFN — and both run int4 on the GPU (SAME quant), so the
// only thing under test is the MoE WIRING (router → indexed stacked-expert GEMVs → weighted
// combine → residual). A correct assembly gives cosine ≈ 1; any offset/order/combine/hazard
// bug breaks it. Real MoE checkpoints don't fit 16 GB, so this is the e2e gate for FeatMoE.
func TestMoE_assemblyVsDense(t *testing.T) {
	if _, err := CreateSystemDefaultDevice(); err != nil {
		t.Skipf("no metal device: %v", err)
	}
	w := genTinyWeights(rand.New(rand.NewSource(1234)))
	moeDir, denseDir := t.TempDir(), t.TempDir()
	writeMoEIdentical(t, moeDir, w)
	writeDense(t, denseDir, w)

	moeM, err := decoder.Load(moeDir, decoder.Options{Quant: "int8int8"})
	if err != nil {
		t.Fatalf("load moe: %v", err)
	}
	moeR, err := BuildResident(moeM)
	if err != nil {
		t.Fatalf("build moe resident: %v", err)
	}
	if moeR.moe == nil {
		t.Fatal("resident has no MoE state")
	}
	denseM, err := decoder.Load(denseDir, decoder.Options{Quant: "int8int8"})
	if err != nil {
		t.Fatalf("load dense: %v", err)
	}
	denseR, err := BuildResident(denseM)
	if err != nil {
		t.Fatalf("build dense resident: %v", err)
	}

	ids := []int{1, 5, 9, 13, 17, 21, 25, 29, 33, 37}
	const steps = 16
	tok, pos, mism := ids[0], 0, 0
	var cosMin float64 = 1
	for i := 0; i < steps; i++ {
		gMoE := append([]float32(nil), moeR.Forward(tok, pos)...)
		gDense := denseR.Forward(tok, pos)
		c := cosF(gMoE, gDense)
		if c < cosMin {
			cosMin = c
		}
		if argmaxF(gMoE) != argmaxF(gDense) {
			mism++
		}
		if i+1 < len(ids) {
			tok = ids[i+1]
		} else {
			tok = argmaxF(gDense)
		}
		pos++
	}
	t.Logf("MoE-vs-dense (identical experts, same GPU int4 quant): cosine min=%.6f, argmax %d/%d match", cosMin, steps-mism, steps)
	if cosMin < 0.9999 {
		t.Fatalf("MoE assembly FAIL: min cosine %.6f < 0.9999 vs the equivalent dense FFN — wiring bug", cosMin)
	}
	if mism != 0 {
		t.Fatalf("MoE assembly FAIL: %d/%d argmax mismatches vs equivalent dense", mism, steps)
	}
}

// tinyWeights holds one shared weight set: attention/embed/norms per layer + one FFN
// (gate/up/down) that is used as the dense FFN and replicated across the MoE experts.
type tinyWeights struct {
	embed, norm, lmHead     []float32
	q, qb, k, kb, v, vb, o  [][]float32 // per layer
	inNorm, postNorm        [][]float32
	ffnGate, ffnUp, ffnDown [][]float32 // per layer, one expert's worth
	router                  [][]float32 // per layer [nExpert, hidden]
}

func genTinyWeights(rng *rand.Rand) *tinyWeights {
	qDim, kvDim := tmHeads*tmHeadDim, tmKVHeads*tmHeadDim
	rnd := func(n int, s float32) []float32 {
		d := make([]float32, n)
		for i := range d {
			d[i] = (rng.Float32()*2 - 1) * s
		}
		return d
	}
	ones := func(n int) []float32 {
		d := make([]float32, n)
		for i := range d {
			d[i] = 1 + (rng.Float32()*2-1)*0.05
		}
		return d
	}
	w := &tinyWeights{
		embed:  rnd(tmVocab*tmHidden, 0.4),
		norm:   ones(tmHidden),
		lmHead: rnd(tmVocab*tmHidden, 0.4),
	}
	for l := 0; l < tmLayers; l++ {
		w.q = append(w.q, rnd(qDim*tmHidden, 0.3))
		w.qb = append(w.qb, rnd(qDim, 0.1))
		w.k = append(w.k, rnd(kvDim*tmHidden, 0.3))
		w.kb = append(w.kb, rnd(kvDim, 0.1))
		w.v = append(w.v, rnd(kvDim*tmHidden, 0.3))
		w.vb = append(w.vb, rnd(kvDim, 0.1))
		w.o = append(w.o, rnd(tmHidden*qDim, 0.3))
		w.inNorm = append(w.inNorm, ones(tmHidden))
		w.postNorm = append(w.postNorm, ones(tmHidden))
		w.ffnGate = append(w.ffnGate, rnd(tmInter*tmHidden, 0.3))
		w.ffnUp = append(w.ffnUp, rnd(tmInter*tmHidden, 0.3))
		w.ffnDown = append(w.ffnDown, rnd(tmHidden*tmInter, 0.3))
		w.router = append(w.router, rnd(tmExperts*tmHidden, 0.8))
	}
	return w
}

func attnTensors(ts map[string]stf32, w *tinyWeights, l int) {
	qDim, kvDim := tmHeads*tmHeadDim, tmKVHeads*tmHeadDim
	p := fmt.Sprintf("model.layers.%d.", l)
	ts[p+"self_attn.q_proj.weight"] = stf32{[]int{qDim, tmHidden}, w.q[l]}
	ts[p+"self_attn.q_proj.bias"] = stf32{[]int{qDim}, w.qb[l]}
	ts[p+"self_attn.k_proj.weight"] = stf32{[]int{kvDim, tmHidden}, w.k[l]}
	ts[p+"self_attn.k_proj.bias"] = stf32{[]int{kvDim}, w.kb[l]}
	ts[p+"self_attn.v_proj.weight"] = stf32{[]int{kvDim, tmHidden}, w.v[l]}
	ts[p+"self_attn.v_proj.bias"] = stf32{[]int{kvDim}, w.vb[l]}
	ts[p+"self_attn.o_proj.weight"] = stf32{[]int{tmHidden, qDim}, w.o[l]}
	ts[p+"input_layernorm.weight"] = stf32{[]int{tmHidden}, w.inNorm[l]}
	ts[p+"post_attention_layernorm.weight"] = stf32{[]int{tmHidden}, w.postNorm[l]}
}

// writeMoEIdentical writes a qwen2_moe whose experts are all identical (= w.ffn*) and whose
// shared expert is zeroed — so the FFN reduces to that one expert (routed weights sum to 1).
func writeMoEIdentical(t *testing.T, dir string, w *tinyWeights) {
	cfg := fmt.Sprintf(`{"model_type":"qwen2_moe","vocab_size":%d,"hidden_size":%d,
		"num_hidden_layers":%d,"num_attention_heads":%d,"num_key_value_heads":%d,"head_dim":%d,
		"intermediate_size":%d,"moe_intermediate_size":%d,"shared_expert_intermediate_size":%d,
		"num_experts":%d,"num_experts_per_tok":%d,"norm_topk_prob":true,
		"max_position_embeddings":256,"rms_norm_eps":1e-6,"rope_theta":10000}`,
		tmVocab, tmHidden, tmLayers, tmHeads, tmKVHeads, tmHeadDim, tmInter, tmInter, tmInter, tmExperts, tmTopK)
	writeConfig(t, dir, cfg)
	zeros := func(n int) []float32 { return make([]float32, n) }
	ts := map[string]stf32{
		"model.embed_tokens.weight": {[]int{tmVocab, tmHidden}, w.embed},
		"model.norm.weight":         {[]int{tmHidden}, w.norm},
		"lm_head.weight":            {[]int{tmVocab, tmHidden}, w.lmHead},
	}
	for l := 0; l < tmLayers; l++ {
		attnTensors(ts, w, l)
		p := fmt.Sprintf("model.layers.%d.", l)
		ts[p+"mlp.gate.weight"] = stf32{[]int{tmExperts, tmHidden}, w.router[l]}
		for e := 0; e < tmExperts; e++ { // identical experts
			ep := fmt.Sprintf("%smlp.experts.%d.", p, e)
			ts[ep+"gate_proj.weight"] = stf32{[]int{tmInter, tmHidden}, w.ffnGate[l]}
			ts[ep+"up_proj.weight"] = stf32{[]int{tmInter, tmHidden}, w.ffnUp[l]}
			ts[ep+"down_proj.weight"] = stf32{[]int{tmHidden, tmInter}, w.ffnDown[l]}
		}
		// zeroed shared expert → contributes nothing (also exercises the shared path safely).
		ts[p+"mlp.shared_expert.gate_proj.weight"] = stf32{[]int{tmInter, tmHidden}, zeros(tmInter * tmHidden)}
		ts[p+"mlp.shared_expert.up_proj.weight"] = stf32{[]int{tmInter, tmHidden}, zeros(tmInter * tmHidden)}
		ts[p+"mlp.shared_expert.down_proj.weight"] = stf32{[]int{tmHidden, tmInter}, zeros(tmHidden * tmInter)}
		ts[p+"mlp.shared_expert_gate.weight"] = stf32{[]int{1, tmHidden}, zeros(tmHidden)}
	}
	writeSTF32(t, filepath.Join(dir, "model.safetensors"), ts)
}

// writeDense writes a plain qwen2 (dense) whose FFN equals w.ffn* — the reference the MoE
// with identical experts must match.
func writeDense(t *testing.T, dir string, w *tinyWeights) {
	cfg := fmt.Sprintf(`{"model_type":"qwen2","vocab_size":%d,"hidden_size":%d,
		"num_hidden_layers":%d,"num_attention_heads":%d,"num_key_value_heads":%d,"head_dim":%d,
		"intermediate_size":%d,"max_position_embeddings":256,"rms_norm_eps":1e-6,"rope_theta":10000}`,
		tmVocab, tmHidden, tmLayers, tmHeads, tmKVHeads, tmHeadDim, tmInter)
	writeConfig(t, dir, cfg)
	ts := map[string]stf32{
		"model.embed_tokens.weight": {[]int{tmVocab, tmHidden}, w.embed},
		"model.norm.weight":         {[]int{tmHidden}, w.norm},
		"lm_head.weight":            {[]int{tmVocab, tmHidden}, w.lmHead},
	}
	for l := 0; l < tmLayers; l++ {
		attnTensors(ts, w, l)
		p := fmt.Sprintf("model.layers.%d.", l)
		ts[p+"mlp.gate_proj.weight"] = stf32{[]int{tmInter, tmHidden}, w.ffnGate[l]}
		ts[p+"mlp.up_proj.weight"] = stf32{[]int{tmInter, tmHidden}, w.ffnUp[l]}
		ts[p+"mlp.down_proj.weight"] = stf32{[]int{tmHidden, tmInter}, w.ffnDown[l]}
	}
	writeSTF32(t, filepath.Join(dir, "model.safetensors"), ts)
}

func writeConfig(t *testing.T, dir, cfg string) {
	t.Helper()
	if err := os.WriteFile(filepath.Join(dir, "config.json"), []byte(cfg), 0o644); err != nil {
		t.Fatal(err)
	}
}

type stf32 struct {
	shape []int
	data  []float32
}

// writeSTF32 writes a minimal F32 .safetensors file (8-byte header length + JSON header +
// concatenated little-endian f32 blob) — the same format decoder.Load reads.
func writeSTF32(t *testing.T, path string, tensors map[string]stf32) {
	t.Helper()
	type meta struct {
		DType   string `json:"dtype"`
		Shape   []int  `json:"shape"`
		Offsets [2]int `json:"data_offsets"`
	}
	names := make([]string, 0, len(tensors))
	for n := range tensors {
		names = append(names, n)
	}
	sort.Strings(names)
	header := map[string]meta{}
	var blob []byte
	off := 0
	for _, n := range names {
		d := tensors[n]
		b := make([]byte, len(d.data)*4)
		for i, f := range d.data {
			binary.LittleEndian.PutUint32(b[i*4:], math.Float32bits(f))
		}
		header[n] = meta{"F32", d.shape, [2]int{off, off + len(b)}}
		blob = append(blob, b...)
		off += len(b)
	}
	hjson, _ := json.Marshal(header)
	var buf []byte
	lenHdr := make([]byte, 8)
	binary.LittleEndian.PutUint64(lenHdr, uint64(len(hjson)))
	buf = append(buf, lenHdr...)
	buf = append(buf, hjson...)
	buf = append(buf, blob...)
	if err := os.WriteFile(path, buf, 0o644); err != nil {
		t.Fatal(err)
	}
}
