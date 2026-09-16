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

// TestNemotronMoEResidentParityWebGPU is the G7 part 2 gate (docs/tasks/task-gpu-paths-2026-09.md):
// Nemotron-H's fourth block kind (MoE FFN — routed NON-GATED relu² experts plus an always-on
// ungated shared expert of the same shape, NOT moeMLP's gated SwiGLU) against
// testdata/nemotron3nano-tiny, whose 6-layer block pattern (linear_attention, moe,
// linear_attention, full_attention, moe, linear_attention) exercises mamba, attention, AND moe
// block kinds in one fixture.
//
// Reuses moeRouteWGSL/moeExpertWGSL/relu2QuantWGSL wholesale — no new kernels were needed. The
// router is DeepSeek/GLM's own sigmoid+bias+group-limited-top-k shape (moeRouteWGSL already
// implements this; Nemotron's n_group=1 degenerates it to plain top-k, the kernel's own nGroup==1
// path), and moeExpert is already generic per single projection (called here for up/down only,
// with no expGate at all) — the gap was purely gpu/residency.go's per-layer switch never building
// case 3's weights and gpu/decoderunner.go never dispatching nemoKMoE, not a missing primitive.
func TestNemotronMoEResidentParityWebGPU(t *testing.T) {
	const dir = "../testdata/nemotron3nano-tiny"
	// Stat the WEIGHTS, not the directory (audit-2026-09-10 G-13(h)). The dir and its config.json
	// are tracked while the weights are gitignored, so a dir stat passes on every clone. Then Load
	// failed on the missing weights, and the test skipped saying "no webgpu device".
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
	// 0.95 is the established resident-vs-CPU floor this session's other new-family gates use
	// (gpt2_resident_parity_test.go's own precedent), not a bar invented for this test.
	if minCos < 0.95 {
		t.Errorf("minCosine %.6f < 0.95 — resident diverges from CPU", minCos)
	}
}

// TestNemotronMoEResident_noSharedExpertBuilds is M-35's gate (audit-2026-09-10): a Nemotron-H
// MoE block with n_shared_experts=0 (a real, validator-accepted config shape — decoder/config.go's
// validateNemotron only errors on the opposite combination, n_shared_experts>0 with the
// intermediate size unset) used to fail BuildResident with "unsupported projection precision
// \"\"" — the case-3 branch of the per-layer switch projected SharedExpert.Up/.Down
// unconditionally, so a zero-value SharedExpert (no tensors for it exist when
// moe_shared_expert_intermediate_size is also 0) hit proj() with nothing to wrap. Neither real
// Nemotron-H checkpoint has this shape (both ship a shared expert), so this is a config-space
// hole, not something the existing parity test above ever exercised.
//
// Derives its fixture from testdata/nemotron3nano-tiny by copying config.json with
// n_shared_experts and moe_shared_expert_intermediate_size zeroed (which makes
// arch.MoE.SharedIntermediateDim resolve to 0 — decoder/registry.go's nemotron branch reads it
// directly from moe_shared_expert_intermediate_size, NOT derived from n_shared_experts, so both
// must be zeroed) and symlinking the same real model.safetensors — its now-unreferenced
// mixer.shared_experts.* tensors are simply never looked up by either the CPU or GPU loader once
// SharedIntermediateDim is 0, so no fresh weights need to be generated.
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

	// A forward pass must actually run without panicking on the nil shUp/shDown fields the
	// fixed residency.go now leaves unbuilt — the decode-time `if lw.shUp != nil` guard
	// (gpu/decoderunner.go) is what this exercises end to end, not just that Load succeeds.
	if _, err := rf.Forward(mg.EmbedResidentForTest(1), 0); err != nil {
		t.Fatalf("resident forward with no shared expert: %v", err)
	}
}
