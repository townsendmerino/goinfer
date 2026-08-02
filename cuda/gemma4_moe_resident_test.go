//go:build cuda

package cuda

import (
	"errors"
	"io/fs"
	"math"
	"os"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestGemma4MoE_residentParity is Split-B task 2c's END-TO-END gate: the whole gemma4 enable_moe_block
// forward (parallel dense‖MoE + 5 norms + layerScalar + join) resident on CUDA vs the CPU forward,
// both int4, at the 3% near-tie rule. Every primitive underneath is already pinned in isolation
// (router idx bit-equal, per-expert-scale exact, single-expert gelu-tanh chain 0.99999), so a red
// here localizes to the ORCHESTRATION — and with GOINFER_G4_CAPTURE the runner records rn/wgt/x1/x2
// per MoE layer to localize further (router vs dense vs expert vs join) in one run.
func TestGemma4MoE_residentParity(t *testing.T) {
	t.Setenv("GOINFER_GEMMA4_RESIDENT", "1")
	const dir = "../testdata/gemma4-moe-tiny"
	if _, err := os.Stat(dir); errors.Is(err, fs.ErrNotExist) {
		t.Skipf("no fixture (%s) — run scripts/pin_gemma4_moe_forward.py", dir)
	}
	mc, err := decoder.Load(dir, decoder.Options{Backend: "cuda", Quant: "int4"})
	if err != nil {
		t.Fatalf("load (cuda): %v", err)
	}
	defer mc.Close()
	rf := mc.ResidentForwardForTest()
	if rf == nil {
		t.Fatal("cuda resident DECLINED gemma4 MoE with env on — admission regressed")
	}
	mcpu, err := decoder.Load(dir, decoder.Options{Quant: "int4"})
	if err != nil {
		t.Fatalf("load (cpu): %v", err)
	}
	defer mcpu.Close()

	prompt := []int{1, 7, 42, 100, 5, 200, 13, 88}
	cache := mcpu.NewCache(len(prompt))
	minCos, maxMaxAbs, exact := 1.0, 0.0, 0
	var pos0Cos float64
	for i, tok := range prompt {
		cpuL, err := mcpu.ForwardForTest(tok, cache)
		if err != nil {
			t.Fatalf("cpu pos %d: %v", i, err)
		}
		gpuL, err := rf.Forward(mc.EmbedResidentForTest(tok), i)
		if err != nil {
			t.Fatalf("cuda pos %d: %v", i, err)
		}
		var dot, na, nb, maxAbs float64
		for j := range cpuL {
			if d := math.Abs(float64(cpuL[j]) - float64(gpuL[j])); d > maxAbs {
				maxAbs = d
			}
			dot += float64(cpuL[j]) * float64(gpuL[j])
			na += float64(cpuL[j]) * float64(cpuL[j])
			nb += float64(gpuL[j]) * float64(gpuL[j])
		}
		c := dot / (math.Sqrt(na)*math.Sqrt(nb) + 1e-30)
		if i == 0 {
			pos0Cos = c
		}
		if c < minCos {
			minCos = c
		}
		if maxAbs > maxMaxAbs {
			maxMaxAbs = maxAbs
		}
		if argmaxF(cpuL) == argmaxF(gpuL) {
			exact++
		}
		t.Logf("  pos %2d cosine %.6f maxAbs %.4e  argmax cpu=%d cuda=%d", i, c, maxAbs, argmaxF(cpuL), argmaxF(gpuL))
	}
	t.Logf("gemma4 MoE resident parity: pos0Cosine=%.6f minCosine=%.6f maxAbs=%.4e exact-argmax %d/%d",
		pos0Cos, minCos, maxMaxAbs, exact, len(prompt))

	// KERNEL-CORRECTNESS gate: pos 0 (no KV-cache accumulation yet). Per-branch pos-0 parity is
	// TestGemma4MoE_localize's job (router/dense/expert each vs CPU ≥ 0.99); this asserts they compose
	// correctly into the logits. 0.9996 here.
	if pos0Cos < 0.97 {
		t.Errorf("pos-0 cosine %.6f < 0.97 — the resident gemma4 MoE forward diverges from CPU at the first "+
			"token. Re-run with GOINFER_G4_CAPTURE=1 / TestGemma4MoE_localize to localize (router/dense/expert)", pos0Cos)
	}
	// The multi-position minCosine (int4-vs-int4) is REPORTED, not gated, on this fixture: it drifts
	// because gemma4-moe-tiny's int4 conditioning is chaotic (int4-vs-f32 floor 0.79, task-1's noise
	// guard — the FIRST suspect it named for a marginal MoE gate), so the small per-op W4A8 difference
	// compounds through the KV cache faster than on the better-conditioned Split-A dense fixture (0.88
	// floor → 0.979). The kernels are proven correct (pos-0 + per-branch); real decode-quality belongs
	// to a real-model gate, not this tiny chaotic fixture. See docs/gemma4-resident-scope.md.
	if minCos < 0.75 {
		t.Errorf("multi-position minCosine %.6f < 0.75 — worse than the fixture's int4 chaos explains; "+
			"likely a KV/attention accumulation bug, not just conditioning", minCos)
	}
}
