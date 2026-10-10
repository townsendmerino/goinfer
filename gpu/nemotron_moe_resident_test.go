//go:build gpu && goinfer_testhooks

package gpu

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// nemotronMoEParityPrompt is an arbitrary fixed token sequence, not tied to the fixture's
// tokenizer — same convention as gemma3ParityPrompt/gptOssParityPrompt.
var nemotronMoEParityPrompt = []int{1, 7, 42, 20, 5, 30, 13, 40}

// TestNemotronMoEResidentParityWebGPU is the gate for Nemotron-H's fourth block kind
// (docs/tasks/task-gpu-paths-2026-09.md, G7 part 2): the MoE FFN (routed NON-GATED relu² experts
// plus an always-on ungated shared expert of the same shape, NOT moeMLP's gated SwiGLU) against
// testdata/nemotron3nano-tiny, whose 6-layer block pattern (linear_attention, moe,
// linear_attention, full_attention, moe, linear_attention) exercises mamba, attention AND moe
// block kinds in one fixture.
//
// It reuses moeRouteWGSL/moeExpertWGSL/relu2QuantWGSL: the router is DeepSeek/GLM's
// sigmoid+bias+group-limited-top-k shape (Nemotron's n_group=1 degenerates it to plain top-k, the
// kernel's nGroup==1 path), and moeExpert is generic per single projection (called here for
// up/down only, with no expGate).
func TestNemotronMoEResidentParityWebGPU(t *testing.T) {
	const dir = "../testdata/nemotron3nano-tiny"
	// Stat the WEIGHTS, not the directory: the dir and its config.json are tracked while the weights
	// are gitignored, so a dir stat passes on every clone and the test would then skip saying "no
	// webgpu device" when Load failed on the missing weights.
	if _, err := os.Stat(dir + "/model.safetensors"); err != nil {
		t.Skipf("no fixture weights (%s/model.safetensors; config.json alone is tracked)", dir)
	}
	if c, err := New(); err != nil {
		t.Skipf("no webgpu device: %v", err)
	} else {
		c.Close()
	}

	mg, err := decoder.Load(dir, decoder.Options{Backend: "webgpu", Quant: "int4"})
	if err != nil {
		t.Fatalf("load %s: %v", dir, err)
	}
	defer mg.Close()
	rf := mg.ResidentForwardForTest()
	if rf == nil {
		t.Fatalf("%s did not go resident (BuildResident refused) — decode path %q; decline: %s",
			dir, mg.DecodePath(), mg.ResidentDecline())
	}
	t.Logf("resident decode path: %s", mg.DecodePath())

	mcpu, err := decoder.Load(dir, decoder.Options{Quant: "int4"})
	if err != nil {
		t.Fatalf("load cpu: %v", err)
	}
	defer mcpu.Close()

	cache := mcpu.NewCache(len(nemotronMoEParityPrompt))
	minCos := 1.0
	for i, tok := range nemotronMoEParityPrompt {
		cpuL, err := mcpu.ForwardForTest(tok, cache)
		if err != nil {
			t.Fatalf("cpu pos %d: %v", i, err)
		}
		gpuL, err := rf.Forward(mg.EmbedResidentForTest(tok), i)
		if err != nil {
			t.Fatalf("resident forward pos %d: %v", i, err)
		}
		var dot, na, nb float64
		for j := range cpuL {
			v, w := float64(cpuL[j]), float64(gpuL[j])
			if v != v || w != w {
				t.Fatalf("pos %d: NaN in logits (cpu=%v gpu=%v at %d)", i, v, w, j)
			}
			dot += v * w
			na += v * v
			nb += w * w
		}
		cos := dot / (math.Sqrt(na)*math.Sqrt(nb) + 1e-30)
		if cos < minCos {
			minCos = cos
		}
		t.Logf("  pos %2d cosine %.6f", i, cos)
	}
	t.Logf("nemotron3nano-tiny, resident vs CPU (int4 both sides): minCosine=%.6f", minCos)
	// 0.95 is the established resident-vs-CPU floor (gpt2_resident_parity_test.go's precedent), not
	// a bar invented for this test.
	if minCos < 0.95 {
		t.Errorf("minCosine %.6f < 0.95 — resident diverges from CPU", minCos)
	}
}

// TestNemotronMoEResident_noSharedExpertBuilds pins that a Nemotron-H MoE block with
// n_shared_experts=0 builds resident. That is a real, validator-accepted config shape
// (decoder/config.go's validateNemotron only errors on the opposite combination,
// n_shared_experts>0 with the intermediate size unset), yet the case-3 branch of the per-layer
// switch once projected SharedExpert.Up/.Down unconditionally, so a zero-value SharedExpert hit
// proj() with nothing to wrap and BuildResident failed with "unsupported projection precision".
// Neither real Nemotron-H checkpoint has this shape (both ship a shared expert), so the parity
// test above never exercised it.
//
// The fixture derives from testdata/nemotron3nano-tiny: config.json copied with n_shared_experts
// and moe_shared_expert_intermediate_size zeroed (both, because decoder/registry.go's nemotron
// branch reads the intermediate size directly rather than deriving it from n_shared_experts), and
// the same real model.safetensors symlinked; its now-unreferenced mixer.shared_experts.* tensors
// are never looked up once SharedIntermediateDim is 0.
func TestNemotronMoEResident_noSharedExpertBuilds(t *testing.T) {
	const srcDir = "../testdata/nemotron3nano-tiny"
	srcWeights := filepath.Join(srcDir, "model.safetensors")
	if _, err := os.Stat(srcWeights); err != nil {
		t.Skipf("no fixture weights (%s; config.json alone is tracked)", srcWeights)
	}
	if c, err := New(); err != nil {
		t.Skipf("no webgpu device: %v", err)
	} else {
		c.Close()
	}

	raw, err := os.ReadFile(filepath.Join(srcDir, "config.json"))
	if err != nil {
		t.Fatalf("read config.json: %v", err)
	}
	var cfg map[string]any
	if err := json.Unmarshal(raw, &cfg); err != nil {
		t.Fatalf("parse config.json: %v", err)
	}
	if cfg["n_shared_experts"] == float64(0) {
		t.Fatal("test bug: source fixture already has n_shared_experts=0 — this test no longer exercises the gap")
	}
	cfg["n_shared_experts"] = 0
	cfg["moe_shared_expert_intermediate_size"] = 0
	edited, err := json.Marshal(cfg)
	if err != nil {
		t.Fatalf("marshal edited config: %v", err)
	}

	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "config.json"), edited, 0o644); err != nil {
		t.Fatalf("write config.json: %v", err)
	}
	if genCfg, err := os.ReadFile(filepath.Join(srcDir, "generation_config.json")); err == nil {
		if err := os.WriteFile(filepath.Join(dir, "generation_config.json"), genCfg, 0o644); err != nil {
			t.Fatalf("write generation_config.json: %v", err)
		}
	}
	absWeights, err := filepath.Abs(srcWeights)
	if err != nil {
		t.Fatalf("abs path: %v", err)
	}
	if err := os.Symlink(absWeights, filepath.Join(dir, "model.safetensors")); err != nil {
		t.Fatalf("symlink weights: %v", err)
	}

	mg, err := decoder.Load(dir, decoder.Options{Backend: "webgpu", Quant: "int4"})
	if err != nil {
		t.Fatalf("BuildResident refused an n_shared_experts=0 Nemotron-H MoE config: %v", err)
	}
	defer mg.Close()
	rf := mg.ResidentForwardForTest()
	if rf == nil {
		t.Fatalf("did not go resident — decode path %q; decline: %s", mg.DecodePath(), mg.ResidentDecline())
	}
	t.Logf("resident decode path: %s (n_shared_experts=0 config)", mg.DecodePath())

	// A forward pass must actually run without panicking on the nil shUp/shDown fields residency.go
	// leaves unbuilt: the decode-time `if lw.shUp != nil` guard (gpu/decoderunner.go) is what this
	// exercises end to end, not just that Load succeeds.
	if _, err := rf.Forward(mg.EmbedResidentForTest(1), 0); err != nil {
		t.Fatalf("resident forward with no shared expert: %v", err)
	}
}
