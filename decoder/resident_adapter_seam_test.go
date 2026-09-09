package decoder

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"sync/atomic"
	"testing"
)

// G3 seam gate (docs/task-gpu-paths-2026-09.md).
//
// WHY THIS EXISTS. Same shape as resident_seam_test.go's TestSeam_GenerateRunsOnTheResident and
// resident_embed_seam_test.go's G4 tests: a resident capability nothing ever calls is
// indistinguishable from one that was never wired. Model.generateInto's useGPU gate was widened
// to admit an adapter session (cache.lora != nil) onto the resident path ONLY when the resident
// backend implements ResidentAdapter — these tests pin both halves of that with a fake backend
// and an in-memory synthetic base+adapter, so no GPU and no downloaded model are needed:
//
//   - a ResidentAdapter-capable backend must actually be bound (SetAdapter with the adapter's
//     per-layer deltas) before decode and cleared (SetAdapter(nil)) after — never silently run
//     the adapter session's tokens through the base model's unmodified resident weights, which
//     would return correct-looking but WRONG output (audit R-01's failure mode).
//   - a resident backend that does NOT implement ResidentAdapter must decline the WHOLE request
//     to the CPU/staged path (which applies the adapter correctly via applyLoRA) rather than
//     running it resident with the adapter dropped.

// fakeResidentAdapter extends fakeResident with ResidentAdapter, recording every bind/clear.
// lastBind is NOT cleared by SetAdapter(nil) — a clear call is recorded via clears/boundNow so
// tests can inspect what was bound even after generation has cleared it.
type fakeResidentAdapter struct {
	*fakeResident
	fail bool

	binds    int32
	clears   int32
	lastBind []ResidentAdapterLayer
	boundNow bool
}

func (f *fakeResidentAdapter) SetAdapter(layers []ResidentAdapterLayer) error {
	if layers == nil {
		atomic.AddInt32(&f.clears, 1)
		f.boundNow = false
		return nil
	}
	if f.fail {
		return fmt.Errorf("fakeResidentAdapter: forced bind failure")
	}
	atomic.AddInt32(&f.binds, 1)
	f.lastBind = layers
	f.boundNow = true
	return nil
}

type fakeAdapterBackend struct {
	Backend
	rf *fakeResidentAdapter
}

func (b *fakeAdapterBackend) BuildResident(m *Model) (ResidentForward, bool, error) {
	_, _, _, _, _, _, vocab := m.Dims()
	b.rf = &fakeResidentAdapter{fakeResident: &fakeResident{vocab: vocab}}
	return b.rf, true, nil
}

func (b *fakeAdapterBackend) Close() error { return nil }

// buildLoRAFixture writes a tiny synthetic llama base (safetensors) plus a PEFT adapter
// directory targeting q/v/gate/down across two layers — the same shape
// TestLoRACompute_forwardParity (decoder/lora_compute_test.go) uses, factored out here as raw
// directories since these tests need to go through decoder.Load + Model.LoadAdapter themselves,
// not a pre-built *Model.
func buildLoRAFixture(t *testing.T) (baseDir, adapterDir string) {
	t.Helper()
	const hidden, heads, headDim, inter, vocab, layers = 8, 2, 4, 16, 16, 2
	qDim := heads * headDim
	base := t.TempDir()
	cfg := `{"model_type":"llama","vocab_size":16,"hidden_size":8,"num_hidden_layers":2,
		"num_attention_heads":2,"num_key_value_heads":2,"head_dim":4,"intermediate_size":16,
		"max_position_embeddings":128,"rms_norm_eps":1e-6,"rope_theta":10000}`
	if err := os.WriteFile(filepath.Join(base, "config.json"), []byte(cfg), 0o644); err != nil {
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
	writeSafetensors(t, filepath.Join(base, "model.safetensors"), ts)

	const r = 2
	adapter := t.TempDir()
	if err := os.WriteFile(filepath.Join(adapter, "adapter_config.json"),
		[]byte(`{"r":2,"lora_alpha":4}`), 0o644); err != nil {
		t.Fatal(err)
	}
	at := map[string]stTensor{}
	pfx := "base_model.model.model.layers."
	for l := range layers {
		add := func(mod string, inDim, outDim int) {
			at[pfx+itoa(l)+mod+".lora_A.weight"] = stTensor{[]int{r, inDim}, fill(r*inDim, 100+l)}
			at[pfx+itoa(l)+mod+".lora_B.weight"] = stTensor{[]int{outDim, r}, fill(outDim*r, 200+l)}
		}
		add(".self_attn.q_proj", hidden, qDim)
		add(".self_attn.v_proj", hidden, qDim)
		add(".mlp.gate_proj", hidden, inter)
		add(".mlp.down_proj", inter, hidden)
	}
	writeSafetensors(t, filepath.Join(adapter, "adapter_model.safetensors"), at)
	return base, adapter
}

func loadWithFakeAdapterResident(t *testing.T, base string) (*Model, *fakeAdapterBackend) {
	t.Helper()
	be := &fakeAdapterBackend{}
	name := fmt.Sprintf("fake-adapter-resident-%s", t.Name())
	RegisterBackend(name, func() (Backend, error) {
		cpu, err := NewBackend("cpu")
		if err != nil {
			return nil, err
		}
		be.Backend = cpu
		return be, nil
	})
	m, err := Load(base, Options{Backend: name})
	if err != nil {
		t.Fatalf("Load with fake ResidentAdapter backend: %v", err)
	}
	t.Cleanup(func() { _ = m.Close() })
	return m, be
}

// TestSeam_AdapterSessionBindsAndClearsResidentAdapter is the main G3 gate: a session with an
// adapter bound (cache.lora != nil) must, on a resident backend that implements
// ResidentAdapter, drive SetAdapter(layers) before decode and SetAdapter(nil) after.
func TestSeam_AdapterSessionBindsAndClearsResidentAdapter(t *testing.T) {
	base, adapter := buildLoRAFixture(t)
	m, be := loadWithFakeAdapterResident(t, base)
	if !m.ResidentActive() {
		t.Skip("fixture is not resident-eligible; the other seam tests still gate the wiring")
	}
	if err := m.LoadAdapter("a", adapter); err != nil {
		t.Fatalf("LoadAdapter: %v", err)
	}
	s := m.NewSession(0)
	if err := s.UseAdapter("a"); err != nil {
		t.Fatalf("UseAdapter: %v", err)
	}

	out, g := s.Generate(context.Background(), []int{1, 2, 3}, 2, SamplingParams{})
	for range out {
	}
	if err := g.Err(); err != nil {
		t.Fatalf("Generate: %v", err)
	}

	if be.rf.binds == 0 {
		t.Fatal("adapter session completed without ever calling the resident SetAdapter — the G3 " +
			"defect: an adapter session's tokens ran on the base model's unmodified resident weights")
	}
	if be.rf.clears == 0 {
		t.Error("SetAdapter was never called with nil to clear the bound adapter after generation — " +
			"the next (possibly non-adapter) generation on this shared resident would inherit this one's delta")
	}
	if be.rf.binds != be.rf.clears {
		t.Errorf("binds=%d clears=%d, want equal — every bind must be paired with a clear even on error paths", be.rf.binds, be.rf.clears)
	}
	if be.rf.boundNow {
		t.Error("resident adapter left bound (boundNow) after Generate returned")
	}
	if be.rf.forwards == 0 {
		t.Error("SetAdapter was called but the resident Forward never ran — decode did not actually use the resident path")
	}

	// The bound layers must match the adapter's actual targets (q,v,gate,down; not k,o,up).
	if len(be.rf.lastBind) != 2 {
		t.Fatalf("SetAdapter called with %d layers, want 2", len(be.rf.lastBind))
	}
	for i, l := range be.rf.lastBind {
		if l.Q == nil || l.V == nil || l.Gate == nil || l.Down == nil {
			t.Errorf("layer %d: targeted projection missing from the bound delta (Q=%v V=%v Gate=%v Down=%v)",
				i, l.Q != nil, l.V != nil, l.Gate != nil, l.Down != nil)
		}
		if l.K != nil || l.O != nil || l.Up != nil {
			t.Errorf("layer %d: untargeted projection present in the bound delta (K=%v O=%v Up=%v) — "+
				"the fixture's adapter never targets these", i, l.K != nil, l.O != nil, l.Up != nil)
		}
	}
}

// TestSeam_AdapterSessionDeclinesToCPUWithoutResidentAdapter pins the other half: a resident
// backend that does NOT implement ResidentAdapter must never run an adapter session's tokens
// through it — generateInto must decline the whole request to the CPU/staged path, which
// applies the adapter correctly via applyLoRA, exactly like any other missing resident
// capability (compare ResidentHiddenLast's own fallback).
func TestSeam_AdapterSessionDeclinesToCPUWithoutResidentAdapter(t *testing.T) {
	base, adapter := buildLoRAFixture(t)
	be := &fakeResidencyBackend{}
	name := fmt.Sprintf("fake-plain-resident-%s", t.Name())
	RegisterBackend(name, func() (Backend, error) {
		cpu, err := NewBackend("cpu")
		if err != nil {
			return nil, err
		}
		be.Backend = cpu
		return be, nil
	})
	m, err := Load(base, Options{Backend: name})
	if err != nil {
		t.Fatalf("Load with fake plain resident backend: %v", err)
	}
	defer m.Close()
	if !m.ResidentActive() {
		t.Skip("fixture is not resident-eligible")
	}
	if err := m.LoadAdapter("a", adapter); err != nil {
		t.Fatalf("LoadAdapter: %v", err)
	}

	prompt := []int{1, 2, 3}
	// CPU reference: same adapter, computed directly via prefillLogits (no resident/session).
	cRef := m.NewCache(len(prompt))
	cRef.lora = m.adapter("a")
	want, err := m.prefillLogits(context.Background(), prompt, cRef)
	if err != nil {
		t.Fatalf("CPU reference prefill: %v", err)
	}

	s := m.NewSession(0)
	if err := s.UseAdapter("a"); err != nil {
		t.Fatalf("UseAdapter: %v", err)
	}
	out, g := s.Generate(context.Background(), prompt, 1, SamplingParams{})
	var got int
	for tok := range out {
		got = tok
	}
	if err := g.Err(); err != nil {
		t.Fatalf("Generate: %v", err)
	}
	if be.rf.forwards != 0 {
		t.Fatalf("resident Forward was called %d times for an adapter session on a backend without "+
			"ResidentAdapter — the G3 defect in reverse: base weights ran resident while the adapter's "+
			"delta was silently dropped", be.rf.forwards)
	}
	if got != argmax(want) {
		t.Errorf("declined-to-CPU first token %d != adapter CPU reference argmax %d", got, argmax(want))
	}
}
