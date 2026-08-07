//go:build cuda && goinfer_testhooks

package cuda

import (
	"math"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestMoEResidentParity is the ADMISSION gate for MoE on CUDA: the resident MoE forward must
// agree with the CPU forward on a real (if tiny) Mixtral checkpoint, under the repo's 3%
// near-tie rule (gpu/kv_i8_parity_test.go).
//
// testdata/mixtral-tiny is a genuine MoE: 8 experts, top-2, softmax routing with
// norm_topk_prob, 4 layers. Its required feature set is EXACTLY [moe] — nothing else CUDA
// lacks — which is what makes it a clean gate: anything that breaks here is the MoE block,
// not some other unimplemented axis leaking in.
//
// What this covers that the kernel-level tests cannot: they proved moe_route matches a CPU
// reference and that the indexed GEMV reads the expert the packing wrote, each in isolation
// with hand-built inputs. Neither can catch a DISPATCH bug — the router fed the wrong
// activation, gate and up swapped, the down-proj striding by the dense intermediate width
// instead of the MoE one, the combine landing on the wrong residual. Those are only visible
// end to end against a reference forward.
//
// On ids rather than text: this fixture has random weights and NO tokenizer, so there is no
// text to encode and no round-trip to assert — vocab is 512 and every id below it is equally
// meaningless. That is the opposite of the Gemma gate's bug, where invented ids were fed to a
// real tokenizer's model and silently decoded to garbage. Here the ids ARE the input; the only
// thing to check is that they are in range, which is asserted below.
func TestMoEResidentParity(t *testing.T) {
	const path = "../testdata/mixtral-tiny"
	requireDeviceAndFixture(t, path)

	mc, err := decoder.Load(path, decoder.Options{Backend: "cuda", Quant: "int4"})
	if err != nil {
		t.Fatalf("load (cuda): %v", err)
	}
	defer mc.Close()

	if got := mc.RequiredResidentFeatures(); len(got) != 1 || got[0] != decoder.FeatMoE {
		t.Fatalf("fixture requires %v, want exactly [moe] — this gate only isolates the MoE block "+
			"if MoE is the ONLY feature in play", got)
	}
	rf := mc.ResidentForwardForTest()
	if rf == nil {
		t.Fatal("cuda resident DECLINED a routed MoE model — FeatMoE is declared, so admission " +
			"granted and then BuildResident bailed: the two disagree")
	}

	mcpu, err := decoder.Load(path, decoder.Options{Quant: "int4"})
	if err != nil {
		t.Fatalf("load (cpu): %v", err)
	}
	defer mcpu.Close()

	_, _, _, _, _, _, vocab := mc.Dims()
	// Spread across the vocab so different tokens route to different experts — a prompt that
	// happened to pick one expert every step would pass while 7/8 of the stack went unread.
	prompt := []int{1, 37, 101, 200, 313, 404, 55, 289, 150, 466, 12, 377}
	for _, id := range prompt {
		if id < 0 || id >= vocab {
			t.Fatalf("prompt id %d out of range for vocab %d", id, vocab)
		}
	}

	cache := mcpu.NewCache(len(prompt) + 2)
	exact, hard := 0, 0
	worst, minCos := 0.0, 1.0
	for i, tok := range prompt {
		cpuL, err := mcpu.ForwardForTest(tok, cache)
		if err != nil {
			t.Fatalf("cpu pos %d: %v", i, err)
		}
		gpuL, err := rf.Forward(mc.EmbedResidentForTest(tok), i)
		if err != nil {
			t.Fatalf("cuda pos %d: %v", i, err)
		}
		var dot, na, nb float64
		for j := range cpuL {
			dot += float64(cpuL[j]) * float64(gpuL[j])
			na += float64(cpuL[j]) * float64(cpuL[j])
			nb += float64(gpuL[j]) * float64(gpuL[j])
		}
		c := dot / (math.Sqrt(na) * math.Sqrt(nb))
		// NaN must be caught explicitly: `NaN < minCos` is FALSE, so a NaN cosine would leave
		// minCos untouched and the floor below would pass on degenerate logits. Measured, not
		// hypothetical — the mis-strided-stack control (E) reads out of its row range and
		// reported "min cosine 1.000000" next to a 79% argmax gap.
		if math.IsNaN(c) {
			t.Fatalf("pos %d: logit cosine is NaN — the resident forward produced degenerate output", i)
		}
		if c < minCos {
			minCos = c
		}
		ca, ga := argmaxF(cpuL), argmaxF(gpuL)
		if ca == ga {
			exact++
			continue
		}
		lo, hi := cpuL[0], cpuL[0]
		for _, v := range cpuL {
			if v < lo {
				lo = v
			}
			if v > hi {
				hi = v
			}
		}
		gap := float64(cpuL[ca]-cpuL[ga]) / (float64(hi-lo) + 1e-9)
		if gap > worst {
			worst = gap
		}
		if gap > 0.03 {
			hard++
			t.Errorf("pos %d: CPU=%d CUDA=%d gap=%.3f%% > 3%% — a routed expert disagrees, which is "+
				"not int4 noise: a wrong expert is a different computation", i, ca, ga, gap*100)
		}
	}
	t.Logf("MoE resident vs CPU: %d/%d exact | worst near-tie %.3f%% | hard fails %d | min cosine %.6f",
		exact, len(prompt), worst*100, hard, minCos)

	// The 3% near-tie rule above is NECESSARY BUT NOT SUFFICIENT on this fixture, and the floor
	// below is what closes the gap. Both are calibrated by breaking the dispatch on purpose and
	// measuring — a green parity test proves nothing until it has been seen to go red.
	//
	// RE-MEASURED 2026-08-06 (RTX 2070 SUPER) after audit C-15 made cuda's f32tof16 match the
	// canonical cross-backend f16 scale representation (decoder.f32ToF16bits). The OLD table below
	// had correct=0.999906 and bug B=0.997687 — but that separation was ENTIRELY an artifact of the
	// old TRUNCATING f32tof16, which happened to resolve one int4 near-tie token the CPU's way. With
	// the correct scales each control applied alone measures:
	//
	//   dispatch state                            exact   worst gap   min cosine   3% rule   floor(0.995)
	//   ---------------------------------------  ------  ----------  -----------  --------  ------------
	//   correct                                   11/12      0.032%     0.997833   pass      pass
	//   A down-proj slot pinned to 0              11/12      0.959%     0.988509   PASSES ✗  FAILS ✓
	//   B glu_quant gOff/uOff swapped             11/12      0.032%     0.997846   PASSES ✗  PASSES ✗  ← see below
	//   C gate/up GEMV slot pinned to 0           10/12      6.124%     0.989101   fails     (fails)
	//   D router fed the raw residual              — (>3% argmax gap, structural — caught by the rule)
	//   E down-proj stack mis-strided              — (79% gap / NaN — caught by the rule + NaN guard)
	//
	// The correct run is no longer 12/12: one token is a genuine int4-KERNEL near-tie (device W8A8
	// GEMV vs CPU int4, 0.032% argmax gap, benign), so the correct cosine is 0.997833, not ~1.0. The
	// floor is therefore 0.995 — below correct (0.997833) and above bug A (0.988509), which is the
	// only STRUCTURAL bug that survives the 3% argmax rule (C/D/E have >3% gaps). ~2.5e-3 margin on
	// the correct side, ~6.5e-3 on the bug side.
	//
	// Bug B (gate/up swap) can NO LONGER be caught here: correct (0.997833) and bug B (0.997846) are
	// indistinguishable, because on RANDOM-weight experts silu(up)*gate ≈ silu(gate)*up in magnitude
	// and the difference washes out through down-proj + combine + argmax. It is instead caught by
	// TestMoeSwigluWiring_C15, which exercises launchGluSplit (the sole gate/up-split dispatch) with
	// crafted gate≠up and asserts the pre-quant SwiGLU output directly — scale- and fixture-independent.
	if minCos < 0.995 {
		t.Errorf("logit cosine %.6f < 0.995 — the correct dispatch measures 0.997833 here and the "+
			"tightest STRUCTURAL bug the 3%% rule misses (down-proj slot pinned) measures 0.988509, so "+
			"this is a routing or stacking bug, not int4 noise (gate/up swap → TestMoeSwigluWiring_C15)", minCos)
	}
}
