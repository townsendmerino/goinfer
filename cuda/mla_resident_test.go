//go:build cuda && goinfer_testhooks

package cuda

import (
	"math"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestMLAResidentParityCUDA verifies that deepseek-tiny (DeepSeek-V2 / V3 architecture with MLA
// compressed-KV latent attention + q-LoRA + ungated shared expert MoE) loads on CUDA, activates
// resident execution with isMLA=true, and produces forward logits matching the CPU reference
// with high cosine similarity across multiple decode positions, as well as greedy generation.
func TestMLAResidentParityCUDA(t *testing.T) {
	const ckpt = "../testdata/deepseek-tiny"
	requireDeviceAndFixture(t, ckpt)

	const quant = "int4"
	mRes, err := decoder.Load(ckpt, decoder.Options{Backend: "cuda", Quant: quant})
	if err != nil {
		t.Fatalf("load cuda (%s): %v", quant, err)
	}
	defer mRes.Close()

	if !mRes.ResidentActive() {
		t.Fatalf("deepseek-tiny did not go CUDA-resident (%s) — decline: %s", quant, mRes.ResidentDecline())
	}

	rf := mRes.ResidentForwardForTest()
	cr, ok := rf.(*cudaResident)
	if !ok {
		t.Fatalf("deepseek-tiny resident forward is %T, want *cudaResident", rf)
	}
	if !cr.isMLA {
		t.Fatalf("deepseek-tiny resident runner has isMLA=false, want true")
	}

	mCPU, err := decoder.Load(ckpt, decoder.Options{Backend: "cpu", Quant: quant})
	if err != nil {
		t.Fatalf("load cpu (%s): %v", quant, err)
	}
	defer mCPU.Close()

	prompt := []int{1, 7, 3, 42, 9, 5}
	const N = 8 // greedy tokens generated past the prompt

	// One continuous, teacher-forced walk (matched observations, not two independently-sampled
	// trajectories — the same principle CLAUDE.md's measurement discipline names for a peer
	// comparison applies here to a backend comparison): both arms are fed the SAME token at every
	// step. Past the prompt that token is CPU's own greedy pick, so a step's comparison is never
	// confounded by an earlier step's divergence, and CUDA's logits are always the model's
	// response to a context CPU actually visited — not to a context only CUDA chose to reach.
	rf.Reset()
	cache := mCPU.NewCache(len(prompt) + N)
	worstCos := 1.0
	var lgC, lgR []float32
	for i, tok := range prompt {
		var e error
		if lgC, e = mCPU.ForwardForTest(tok, cache); e != nil {
			t.Fatalf("[%s] cpu forward[%d]: %v", quant, i, e)
		}
		if lgR, e = rf.Forward(mRes.EmbedResidentForTest(tok), i); e != nil {
			t.Fatalf("[%s] resident forward[%d]: %v", quant, i, e)
		}
		cos, _ := cosF32(lgR, lgC)
		worstCos = math.Min(worstCos, cos)
		t.Logf("[%s] pos %d (tok %d): cosine vs cpu = %.8f", quant, i, tok, cos)
	}
	t.Logf("[%s] MLA resident decode, %d prompt positions: worst cosine %.6f", quant, len(prompt), worstCos)

	// Greedy generation, continuing the same walk past the prompt.
	cpuToks := make([]int, 0, N)
	gpuToks := make([]int, 0, N)
	base := len(prompt)
	fedTok := prompt[len(prompt)-1]
	for i := range N {
		cos, _ := cosF32(lgR, lgC)
		worstCos = math.Min(worstCos, cos)
		cTok, gTok := argmaxF(lgC), argmaxF(lgR)
		cpuToks = append(cpuToks, cTok)
		gpuToks = append(gpuToks, gTok)
		t.Logf("[%s] gen %d (fed tok %d): cpu argmax=%d cuda argmax=%d cosine=%.8f", quant, i, fedTok, cTok, gTok, cos)
		if cTok != gTok {
			// The FULL sequence, not just the first token: discrete expert selection can stay identical for
			// several tokens after a wrong-but-plausible routing decision and only visibly diverge once a
			// different expert combination is selected. A transposed nGroup/topkGroup in the router launch
			// (cuda/resident.go, the trap decoder/features.go's FeatMLA entry names) leaves the per-position
			// cosine check above and the first generated token untouched. So every divergence is a HARD stop,
			// UNLESS it passes the near-tie check immediately below.
			//
			// NEAR-TIE, NOT A DEFECT: on this synthetic random-weight fixture a greedy pick can be already
			// near-tied and decided by backend rounding noise (the MoE router-flip noise floor; the int4
			// group-scale precision change moved only CPU's rounding). A real routing defect (the transposition
			// class above) produces a CONFIDENT wrong pick, not a swap between each side's own top-2, so the
			// check requires MUTUAL containment: each side's pick is in the OTHER side's own top-2, which a
			// random unrelated token from a real bug would not satisfy.
			cRank2 := inTop2(lgC, gTok)
			rRank2 := inTop2(lgR, cTok)
			if !cRank2 || !rRank2 {
				// Errorf, not Fatalf: continuing (teacher-forced on CPU's own path either way)
				// keeps the rest of the run's diagnostic log, which is exactly what mattered the
				// first time this class of bug was found here.
				t.Errorf("[%s] MLA resident generation diverges at token %d: cpu=%d cuda=%d (full so far: cpu=%v cuda=%v) — "+
					"NOT a near-tie (cuda's pick %d in cpu's own top-2: %v; cpu's pick %d in cuda's own top-2: %v)",
					quant, i, cTok, gTok, cpuToks, gpuToks, gTok, cRank2, cTok, rRank2)
			} else {
				t.Logf("[%s] gen token %d: cpu=%d cuda=%d — mutual near-tie (each side's pick is the other's runner-up), tolerated", quant, i, cTok, gTok)
			}
		}
		fedTok = cTok // teacher-forced: both arms continue on CPU's own greedy pick
		var e error
		if lgC, e = mCPU.ForwardForTest(fedTok, cache); e != nil {
			t.Fatalf("[%s] gen cpu forward[%d]: %v", quant, base+i, e)
		}
		if lgR, e = rf.Forward(mRes.EmbedResidentForTest(fedTok), base+i); e != nil {
			t.Fatalf("[%s] gen resident forward[%d]: %v", quant, base+i, e)
		}
	}
	t.Logf("[%s] MLA generate: cpu=%v cuda=%v | worst cosine over %d positions (prompt+generation): %.6f",
		quant, cpuToks, gpuToks, len(prompt)+N, worstCos)
	if worstCos < 0.99 {
		t.Errorf("[%s] MLA resident forward diverges from CPU: worst cosine %.6f (want >= 0.99)", quant, worstCos)
	}
}

// inTop2 reports whether tok is one of l's two highest-valued indices.
func inTop2(l []float32, tok int) bool {
	b1, b2 := 0, -1
	for j := range l {
		if b2 == -1 || l[j] > l[b2] {
			if b1 == -1 || l[j] > l[b1] {
				b2, b1 = b1, j
			} else {
				b2 = j
			}
		}
	}
	return tok == b1 || tok == b2
}
