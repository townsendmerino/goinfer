//go:build darwin

package metal

import (
	"fmt"
	"math/rand"
	"os"
	"path/filepath"
	"runtime/debug"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// The MC3 identity fixture (E-G01 and F-G02, docs/audit-metal-2026-09-30.md): a generated qwen2 with Qwen2.5-1.5B's
// attention geometry (12 query heads of 128 over 2 KV heads, so attention_fa's G=6 block kernel and the steel prefill
// kernel engage) and small everything else, written as f32 safetensors and loaded at int4. It lets the batched-step,
// chunked-prefill and spec-verify identity checks run on any Mac with no checkpoint; with GOINFER_METAL_MC3=1 they
// run on the real model as before. The vocabulary covers the token ids those tests draw (below 20000).
const (
	mfHidden, mfHeads, mfKVHeads, mfHeadDim = 256, 12, 2, 128
	mfLayers, mfInter, mfVocab              = 2, 512, 20480
)

// writeMC3Fixture writes the fixture into a fresh directory with a maxPositions-token window and returns it.
func writeMC3Fixture(t *testing.T, maxPositions int) string {
	t.Helper()
	dir := t.TempDir()
	rng := rand.New(rand.NewSource(20261001))
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
	qDim, kvDim := mfHeads*mfHeadDim, mfKVHeads*mfHeadDim
	writeConfig(t, dir, fmt.Sprintf(`{"model_type":"qwen2","vocab_size":%d,"hidden_size":%d,
		"num_hidden_layers":%d,"num_attention_heads":%d,"num_key_value_heads":%d,"head_dim":%d,
		"intermediate_size":%d,"max_position_embeddings":%d,"rms_norm_eps":1e-6,"rope_theta":1000000}`,
		mfVocab, mfHidden, mfLayers, mfHeads, mfKVHeads, mfHeadDim, mfInter, maxPositions))
	ts := map[string]stf32{
		"model.embed_tokens.weight": {[]int{mfVocab, mfHidden}, rnd(mfVocab*mfHidden, 0.4)},
		"model.norm.weight":         {[]int{mfHidden}, ones(mfHidden)},
		"lm_head.weight":            {[]int{mfVocab, mfHidden}, rnd(mfVocab*mfHidden, 0.4)},
	}
	for l := range mfLayers {
		p := fmt.Sprintf("model.layers.%d.", l)
		ts[p+"self_attn.q_proj.weight"] = stf32{[]int{qDim, mfHidden}, rnd(qDim*mfHidden, 0.3)}
		ts[p+"self_attn.q_proj.bias"] = stf32{[]int{qDim}, rnd(qDim, 0.1)}
		ts[p+"self_attn.k_proj.weight"] = stf32{[]int{kvDim, mfHidden}, rnd(kvDim*mfHidden, 0.3)}
		ts[p+"self_attn.k_proj.bias"] = stf32{[]int{kvDim}, rnd(kvDim, 0.1)}
		ts[p+"self_attn.v_proj.weight"] = stf32{[]int{kvDim, mfHidden}, rnd(kvDim*mfHidden, 0.3)}
		ts[p+"self_attn.v_proj.bias"] = stf32{[]int{kvDim}, rnd(kvDim, 0.1)}
		ts[p+"self_attn.o_proj.weight"] = stf32{[]int{mfHidden, qDim}, rnd(mfHidden*qDim, 0.1)}
		ts[p+"input_layernorm.weight"] = stf32{[]int{mfHidden}, ones(mfHidden)}
		ts[p+"post_attention_layernorm.weight"] = stf32{[]int{mfHidden}, ones(mfHidden)}
		ts[p+"mlp.gate_proj.weight"] = stf32{[]int{mfInter, mfHidden}, rnd(mfInter*mfHidden, 0.3)}
		ts[p+"mlp.up_proj.weight"] = stf32{[]int{mfInter, mfHidden}, rnd(mfInter*mfHidden, 0.3)}
		ts[p+"mlp.down_proj.weight"] = stf32{[]int{mfHidden, mfInter}, rnd(mfHidden*mfInter, 0.3)}
	}
	writeSTF32(t, filepath.Join(dir, "model.safetensors"), ts)
	return dir
}

// mc3FixtureResident loads the fixture at int4 with `slots` resident KV slots and a ctx-token resident context, and
// builds its Metal resident as the backend does. Both are closed when the test ends.
func mc3FixtureResident(t *testing.T, slots, ctx int) (*decoder.Model, *resident) {
	t.Helper()
	if _, err := CreateSystemDefaultDevice(); err != nil {
		t.Skipf("no metal device: %v", err)
	}
	m, err := decoder.Load(writeMC3Fixture(t, max(ctx, 4096)), decoder.Options{Quant: "int4", ResidentContext: ctx, ResidentKVSlots: slots})
	if err != nil {
		t.Fatalf("load the MC3 fixture: %v", err)
	}
	r, err := buildResident(m)
	if err != nil {
		m.Close()
		t.Fatalf("build the MC3 fixture's resident: %v", err)
	}
	t.Cleanup(func() {
		r.Close()
		m.Close()
		debug.FreeOSMemory()
	})
	return m, r
}

// TestMC3Fixture_isTheBatchedPath: the fixture builds the resident the identity checks need. A fixture that silently
// stopped qualifying would turn every one of them into a skip or a check of the wrong path.
func TestMC3Fixture_isTheBatchedPath(t *testing.T) {
	_, r := mc3FixtureResident(t, 2, 4096)
	if why := r.batchIneligible(); why != "" || r.batch == nil {
		t.Fatalf("the fixture builds no batched step: %q", why)
	}
	g := r.layers[0].geom
	if g.hd != 128 || r.nH/g.nKV != 6 || len(r.kvSlotBufs) < 2 {
		t.Fatalf("fixture geometry: hd %d, G %d, %d KV slots; want 128, 6, 2", g.hd, r.nH/g.nKV, len(r.kvSlotBufs))
	}
	if !r.prefillOK {
		t.Fatal("the fixture's resident does not admit batched prefill")
	}
	if _, steel := r.prefillAttnKernels(); !steel {
		t.Fatal("the fixture's prefill does not run attention_prefill_steel")
	}
	if r.attnFANKV == 0 || r.attnFABlkSplit == 0 {
		t.Fatalf("attention_fa's block kernel is not selected (attnFANKV %d, attnFABlkSplit %d)", r.attnFANKV, r.attnFABlkSplit)
	}
}

// mc3PrefillResident is the resident the chunked-prefill and spec-verify identity checks run on: the GOINFER_METAL_MC3
// checkpoint when that is set, otherwise the fixture; either with `slots` KV slots at a ctx-token context, wrapped as
// the backend wraps it. It skips a resident without 2 contiguous f16 KV slots, which both checks compare across.
func mc3PrefillResident(t *testing.T, slots, ctx int) *metalResident {
	t.Helper()
	var r *resident
	if os.Getenv("GOINFER_METAL_MC3") == "1" {
		_, r = mc3LoadCheckpoint(t, slots, ctx)
	} else {
		_, r = mc3FixtureResident(t, slots, ctx)
	}
	if len(r.kvSlotBufs) < 2 || !r.kvContig || r.kvF32 {
		t.Skip("needs a Metal resident with 2 contiguous f16 KV slots")
	}
	return &metalResident{r: r, hidden: r.H}
}
