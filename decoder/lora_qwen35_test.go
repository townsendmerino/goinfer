package decoder

import (
	"context"
	"encoding/json"
	"errors"
	"io/fs"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// TestLoRA_mergeAtLoad_qwen35MatchesPEFT is D3's gate (docs/tasks/task-constrained-confidence.md): a PEFT adapter on
// Qwen3.5, merged at load, must give the hidden state PEFT's own merge_and_unload() gives. The adapter targets every module
// autotrust's JEV recipe does (the DeltaNet in_proj_qkv/in_proj_z/out_proj, the full-attention q/k/v/o_proj, and the MLP)
// on all four layers of the tiny checkpoint, three linear-attention and one full-attention. It was written by PEFT
// (scripts/pin_qwen35_lora.py), so the tensor names are PEFT's, and its B is non-zero so the adapter changes the output.
// The observable is PromptHidden, the vector Route B's head reads. Each prompt must match the merged reference to relative
// L2 1e-5, and the base must sit at least 100x that far from it, or the gate could not tell a merge from none.
func TestLoRA_mergeAtLoad_qwen35MatchesPEFT(t *testing.T) {
	raw, err := os.ReadFile(filepath.Join("testdata", "qwen35_lora_golden.json"))
	if errors.Is(err, fs.ErrNotExist) {
		t.Skip("no golden: run scripts/pin_qwen35_lora.py")
	}
	if err != nil {
		t.Fatal(err)
	}
	var g struct {
		Base, Adapter  string
		WrappedModules int `json:"wrapped_modules"`
		Prompts        []struct {
			IDs        []int
			Hidden     []float64
			BaseHidden []float64 `json:"base_hidden"`
		}
	}
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatal(err)
	}
	if len(g.Prompts) == 0 {
		t.Fatal("golden has no prompts")
	}
	base, adapter := filepath.Join("testdata", g.Base), filepath.Join("testdata", g.Adapter)
	if _, err := os.Stat(filepath.Join(base, "model.safetensors")); errors.Is(err, fs.ErrNotExist) {
		t.Skipf("no checkpoint at %s", base)
	}

	lo, err := loadLoRA(adapter)
	if err != nil {
		t.Fatal(err)
	}
	var dn, attn, mlp int
	for name := range lo.deltas {
		switch {
		case strings.Contains(name, ".linear_attn."):
			dn++
		case strings.Contains(name, ".self_attn."):
			attn++
		case strings.Contains(name, ".mlp."):
			mlp++
		}
	}
	lo.close()
	if len(lo.deltas) != g.WrappedModules || dn != 9 || attn != 4 || mlp != 12 {
		t.Fatalf("adapter has %d deltas (DeltaNet %d, attention %d, MLP %d), want %d (9, 4, 12)", len(lo.deltas), dn, attn, mlp, g.WrappedModules)
	}

	m, err := Load(base, Options{LoRA: adapter})
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	rel := func(h []float32, ref []float64) float64 {
		var ne, nb float64
		for j := range ref {
			d := float64(h[j]) - ref[j]
			ne, nb = ne+d*d, nb+ref[j]*ref[j]
		}
		return math.Sqrt(ne / nb)
	}
	for i, p := range g.Prompts {
		h, err := m.PromptHidden(context.Background(), p.IDs)
		if err != nil {
			t.Fatalf("prompt %d: %v", i, err)
		}
		if len(h) != len(p.Hidden) {
			t.Fatalf("prompt %d: hidden size %d, want %d", i, len(h), len(p.Hidden))
		}
		base64 := make([]float32, len(p.BaseHidden))
		for j, x := range p.BaseHidden {
			base64[j] = float32(x)
		}
		got, sep := rel(h, p.Hidden), rel(base64, p.Hidden)
		t.Logf("prompt %d (%d tokens): relative L2 vs PEFT merged %.3g; the base is %.3g away", i, len(p.IDs), got, sep)
		if sep < 1e-3 {
			t.Fatalf("prompt %d: the base is only %.3g from the merged reference; the adapter is too weak for this gate to see a missing merge", i, sep)
		}
		if got > 1e-5 {
			t.Errorf("prompt %d (%d tokens): relative L2 %.3g > 1e-5 against PEFT's merged model", i, len(p.IDs), got)
		}
	}
}

// TestLoRA_mergeAtLoad_refusesUnmergedDelta pins checkAllMerged: a delta on a tensor the load never merges fails the load
// instead of being ignored. The tensor chosen, a self_attn q_proj on a linear-attention layer, is a real projection name
// that validateTargets' list would accept, so only the post-load check can refuse it.
func TestLoRA_mergeAtLoad_refusesUnmergedDelta(t *testing.T) {
	base := filepath.Join("testdata", "qwen3_5-tiny")
	if _, err := os.Stat(filepath.Join(base, "model.safetensors")); errors.Is(err, fs.ErrNotExist) {
		t.Skipf("no checkpoint at %s", base)
	}
	const r, hidden = 2, 64
	adapter := t.TempDir()
	if err := os.WriteFile(filepath.Join(adapter, "adapter_config.json"), []byte(`{"r":2,"lora_alpha":4}`), 0o644); err != nil {
		t.Fatal(err)
	}
	fill := func(n int) []float32 {
		d := make([]float32, n)
		for i := range d {
			d[i] = 0.01
		}
		return d
	}
	writeSafetensors(t, filepath.Join(adapter, "adapter_model.safetensors"), map[string]stTensor{
		"base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": {[]int{r, hidden}, fill(r * hidden)},
		"base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight": {[]int{256, r}, fill(256 * r)},
	})
	m, err := Load(base, Options{LoRA: adapter})
	if err == nil {
		m.Close()
		t.Fatal("loaded an adapter whose only delta targets a tensor layer 0 (linear attention) does not have")
	}
	if !strings.Contains(err.Error(), "did not merge 1 of the adapter's 1 deltas") || !strings.Contains(err.Error(), "layers.0.self_attn.q_proj") {
		t.Fatalf("wrong refusal: %v", err)
	}
}

// TestLoRA_mergeAtLoad_refusesFamilyThatTakesNoAdapter is the same check on a loader that never sees the adapter.
// buildInternLM2Weights (like buildGptOssWeights) takes no lora argument, and its branch returned before any LoRA check, so
// before checkAllMerged an adapter on InternLM2 loaded clean and changed nothing.
func TestLoRA_mergeAtLoad_refusesFamilyThatTakesNoAdapter(t *testing.T) {
	base := filepath.Join("testdata", "internlm2-tiny")
	if _, err := os.Stat(filepath.Join(base, "model.safetensors")); errors.Is(err, fs.ErrNotExist) {
		t.Skipf("no checkpoint at %s", base)
	}
	const r, hidden = 2, 64
	adapter := t.TempDir()
	if err := os.WriteFile(filepath.Join(adapter, "adapter_config.json"), []byte(`{"r":2,"lora_alpha":4}`), 0o644); err != nil {
		t.Fatal(err)
	}
	one := make([]float32, r*hidden)
	for i := range one {
		one[i] = 0.01
	}
	writeSafetensors(t, filepath.Join(adapter, "adapter_model.safetensors"), map[string]stTensor{
		"base_model.model.model.layers.0.attention.wo.lora_A.weight": {[]int{r, hidden}, one},
		"base_model.model.model.layers.0.attention.wo.lora_B.weight": {[]int{hidden, r}, one},
	})
	m, err := Load(base, Options{LoRA: adapter})
	if err == nil {
		m.Close()
		t.Fatal("InternLM2 loaded a LoRA adapter its loader never merges")
	}
	if !strings.Contains(err.Error(), "did not merge 1 of the adapter's 1 deltas") {
		t.Fatalf("wrong refusal: %v", err)
	}
}
