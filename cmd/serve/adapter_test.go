package main

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

func TestAdapterFlag_parse(t *testing.T) {
	var f adapterFlag
	if err := f.Set("chat=base=/tmp/a"); err != nil {
		t.Fatalf("valid spec rejected: %v", err)
	}
	if err := f.Set("code=base=/tmp/b"); err != nil {
		t.Fatal(err)
	}
	if len(f) != 2 || f[0] != (adapterSpec{"chat", "base", "/tmp/a"}) || f[1].name != "code" {
		t.Fatalf("parsed = %+v", f)
	}
	for _, bad := range []string{"noequals", "name=baseonly", "=base=dir", "name==dir", "name=base="} {
		var g adapterFlag
		if err := g.Set(bad); err == nil {
			t.Errorf("expected error for %q", bad)
		}
	}
}

// TestLoadAdapters_errors covers the wiring guards that fire before any real model
// load — base missing, name collision, stream-weights, attaching to an adapter.
func TestLoadAdapters_errors(t *testing.T) {
	mk := func(adapters adapterFlag, models map[string]*loadedModel, stream bool) error {
		s := &server{models: models, cfg: config{}}
		return s.loadAdapters(config{adapters: adapters, streamWeights: stream})
	}
	if err := mk(adapterFlag{{"a", "missing", "/d"}}, map[string]*loadedModel{}, false); err == nil {
		t.Error("expected base-not-found error")
	}
	if err := mk(adapterFlag{{"a", "b", "/d"}}, map[string]*loadedModel{"b": {name: "b"}}, true); err == nil {
		t.Error("expected stream-weights incompatibility error")
	}
	if err := mk(adapterFlag{{"b", "b", "/d"}}, map[string]*loadedModel{"b": {name: "b"}}, false); err == nil {
		t.Error("expected name-collision error")
	}
	if err := mk(adapterFlag{{"a", "x", "/d"}}, map[string]*loadedModel{"x": {name: "x", adapter: "x"}}, false); err == nil {
		t.Error("expected attach-to-adapter error")
	}
}

// TestServe_multiAdapter is the end-to-end #7 proof: one synthetic safetensors base
// serves two compute-time LoRA adapters. It asserts the adapters SHARE the base's
// resident decoder.Model (the RAM win), route by served name, and produce distinct
// greedy outputs (base vs each adapter, and the two adapters vs each other) — i.e.
// the per-session adapter binding is live and isolated.
func TestServe_multiAdapter(t *testing.T) {
	base := buildSyntheticBase(t)
	a1 := buildSyntheticAdapter(t, +1)
	a2 := buildSyntheticAdapter(t, -1)

	// Construct the base loadedModel directly (the synthetic base has no
	// tokenizer.json, and generating from token IDs doesn't need one) so the test
	// exercises loadAdapters + session binding + routing without that fixture.
	m, err := decoder.Load(base, decoder.Options{Backend: "cpu"})
	if err != nil {
		t.Fatalf("decoder.Load base: %v", err)
	}
	defer m.Close()
	baseLM := &loadedModel{model: m, name: "base", fp: "base", sessions: newSessionLRU(m, 4, 0, "base")}
	srv := &server{models: map[string]*loadedModel{"base": baseLM}, cfg: config{}}
	if err := srv.loadAdapters(config{adapters: adapterFlag{{"a1", "base", a1}, {"a2", "base", a2}}, kvSessions: 4}); err != nil {
		t.Fatalf("loadAdapters: %v", err)
	}

	// All three route; the adapters share the base's resident weights.
	for _, n := range []string{"base", "a1", "a2"} {
		if srv.pick(n) == nil {
			t.Fatalf("model %q not routable", n)
		}
	}
	if srv.models["a1"].model != srv.models["base"].model || srv.models["a2"].model != srv.models["base"].model {
		t.Fatal("adapters must share the base decoder.Model (the RAM-density win)")
	}
	if srv.models["a1"].sessions == srv.models["a2"].sessions {
		t.Fatal("adapters must have separate session LRUs (no KV cross-contamination)")
	}

	prompt := []int{1, 4, 2, 7, 3}
	tb := genGreedy(t, srv.models["base"], prompt, 8)
	t1 := genGreedy(t, srv.models["a1"], prompt, 8)
	t2 := genGreedy(t, srv.models["a2"], prompt, 8)
	t.Logf("base=%v a1=%v a2=%v; adapters: a1=%v a2=%v", tb, t1, t2, m.HasAdapter("a1"), m.HasAdapter("a2"))

	if eqTokens(tb, t1) {
		t.Error("adapter a1 output == base: adapter not applied")
	}
	if eqTokens(tb, t2) {
		t.Error("adapter a2 output == base: adapter not applied")
	}
	if eqTokens(t1, t2) {
		t.Error("a1 == a2: the two adapters are not distinguished (binding/isolation broken)")
	}
}

func genGreedy(t *testing.T, lm *loadedModel, ids []int, n int) []int {
	t.Helper()
	sess := lm.sessions.acquire(ids)
	ch, gen := sess.Generate(context.Background(), ids, n, decoder.SamplingParams{})
	var out []int
	for tok := range ch {
		out = append(out, tok)
	}
	if gen.Err() != nil {
		t.Fatalf("generate (%s): %v", lm.name, gen.Err())
	}
	return out
}

func eqTokens(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// buildSyntheticBase writes a tiny 2-layer llama (safetensors + config.json) and
// returns its dir.
func buildSyntheticBase(t *testing.T) string {
	t.Helper()
	const hidden, qDim, inter, vocab, layers = 8, 8, 16, 16, 2
	dir := t.TempDir()
	cfg := `{"model_type":"llama","vocab_size":16,"hidden_size":8,"num_hidden_layers":2,
		"num_attention_heads":2,"num_key_value_heads":2,"head_dim":4,"intermediate_size":16,
		"max_position_embeddings":128,"rms_norm_eps":1e-6,"rope_theta":10000}`
	if err := os.WriteFile(filepath.Join(dir, "config.json"), []byte(cfg), 0o644); err != nil {
		t.Fatal(err)
	}
	fill := func(n, seed int) []float32 {
		d := make([]float32, n)
		for i := range d {
			// multiplicative hash → well-spread signed weights (avoids a degenerate
			// always-same-argmax model where an adapter shift can't flip anything).
			h := uint32((i+1)*2654435761 + seed*40503)
			d[i] = float32(int32(h)) / float32(1<<31) * 0.6
		}
		return d
	}
	ts := map[string][]float32{
		"model.embed_tokens.weight": fill(vocab*hidden, 1),
		"model.norm.weight":         fill(hidden, 2),
		"lm_head.weight":            fill(vocab*hidden, 3),
	}
	shapes := map[string][]int{
		"model.embed_tokens.weight": {vocab, hidden},
		"model.norm.weight":         {hidden},
		"lm_head.weight":            {vocab, hidden},
	}
	for l := range layers {
		p := "model.layers." + strconv.Itoa(l)
		add := func(name string, shape []int, seed int) {
			ts[p+name] = fill(prod(shape), seed)
			shapes[p+name] = shape
		}
		add(".self_attn.q_proj.weight", []int{qDim, hidden}, 10+l)
		add(".self_attn.k_proj.weight", []int{qDim, hidden}, 20+l)
		add(".self_attn.v_proj.weight", []int{qDim, hidden}, 30+l)
		add(".self_attn.o_proj.weight", []int{hidden, qDim}, 40+l)
		add(".mlp.gate_proj.weight", []int{inter, hidden}, 50+l)
		add(".mlp.up_proj.weight", []int{inter, hidden}, 60+l)
		add(".mlp.down_proj.weight", []int{hidden, inter}, 70+l)
		add(".input_layernorm.weight", []int{hidden}, 80+l)
		add(".post_attention_layernorm.weight", []int{hidden}, 90+l)
	}
	writeST(t, filepath.Join(dir, "model.safetensors"), ts, shapes)
	return dir
}

// buildSyntheticAdapter writes a PEFT adapter on q/v/gate/down for both layers,
// with a large alpha and sign-controlled B so two adapters (sign ±1) diverge.
func buildSyntheticAdapter(t *testing.T, sign int) string {
	t.Helper()
	const hidden, qDim, inter, layers, r = 8, 8, 16, 2, 2
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "adapter_config.json"),
		[]byte(`{"r":2,"lora_alpha":24}`), 0o644); err != nil {
		t.Fatal(err)
	}
	ts := map[string][]float32{}
	shapes := map[string][]int{}
	// fill A plainly; fill B scaled by sign — so the delta s·B·(A·x) flips sign
	// between the two adapters (scaling BOTH A and B would cancel: sign² = 1).
	fill := func(n, seed int, s float32) []float32 {
		d := make([]float32, n)
		for i := range d {
			d[i] = s * (float32((i*3+seed)%7)*0.15 - 0.3)
		}
		return d
	}
	pfx := "base_model.model.model.layers."
	for l := range layers {
		add := func(mod string, inDim, outDim, seed int) {
			an := pfx + strconv.Itoa(l) + mod + ".lora_A.weight"
			bn := pfx + strconv.Itoa(l) + mod + ".lora_B.weight"
			ts[an], shapes[an] = fill(r*inDim, seed, 1), []int{r, inDim}
			ts[bn], shapes[bn] = fill(outDim*r, seed+1, float32(sign)), []int{outDim, r}
		}
		add(".self_attn.q_proj", hidden, qDim, 100+l)
		add(".self_attn.v_proj", hidden, qDim, 120+l)
		add(".mlp.gate_proj", hidden, inter, 140+l)
		add(".mlp.down_proj", inter, hidden, 160+l)
	}
	writeST(t, filepath.Join(dir, "adapter_model.safetensors"), ts, shapes)
	return dir
}

func prod(shape []int) int {
	n := 1
	for _, d := range shape {
		n *= d
	}
	return n
}

// writeST writes an F32 safetensors file (header JSON + raw little-endian data).
func writeST(t *testing.T, path string, data map[string][]float32, shapes map[string][]int) {
	t.Helper()
	type meta struct {
		DType   string `json:"dtype"`
		Shape   []int  `json:"shape"`
		Offsets [2]int `json:"data_offsets"`
	}
	names := make([]string, 0, len(data))
	for n := range data {
		names = append(names, n)
	}
	sort.Strings(names)
	header := map[string]meta{}
	var blob []byte
	off := 0
	for _, n := range names {
		d := data[n]
		b := make([]byte, len(d)*4)
		for i, f := range d {
			binary.LittleEndian.PutUint32(b[i*4:], math.Float32bits(f))
		}
		header[n] = meta{"F32", shapes[n], [2]int{off, off + len(b)}}
		blob = append(blob, b...)
		off += len(b)
	}
	hjson, _ := json.Marshal(header)
	var buf []byte
	lenHdr := make([]byte, 8)
	binary.LittleEndian.PutUint64(lenHdr, uint64(len(hjson)))
	buf = append(append(append(buf, lenHdr...), hjson...), blob...)
	if err := os.WriteFile(path, buf, 0o644); err != nil {
		t.Fatal(err)
	}
}
