//go:build cuda && goinfer_testhooks

package cuda

import (
	"context"
	"math"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestCohereRealResidentParityCUDA is the REAL-CHECKPOINT gate for the pairwise rope on CUDA: Aya-expanse-8B
// (cohere) and Command-R7B (cohere2: sliding window + NoPE global layers) at int4 on the resident, against
// the CPU at the same quantization and against the HF f32 golden. Heavy (two 16 GB checkpoints, each
// loaded twice, ~2-4 min per family):
//
//	GOINFER_HEAVY_TESTS=1 go test -tags 'cuda goinfer_testhooks' ./cuda/ -run TestCohereRealResidentParityCUDA -v -timeout 40m
//
// WHY IT EXISTS. On 2026-10-01 these two checkpoints, on this resident with the NeoX half-split rope kernels,
// read per-position resident-vs-CPU worst cosine -0.075 (R7B) / -0.041 (Aya) on a 48-token prompt, exact at
// position 0 and diverging after, and Aya's greedy continuation matched the HF golden 1/8 against the CPU's
// 8/8. The committed flat-weight gate (TestCohereResidentParityCUDA, 0.02-std) read 0.9997 on the same wrong
// kernels. This gate reads the real weights.
//
// BARS, PRE-REGISTERED BEFORE the first run with the pairwise kernels (docs/measurements/cuda-pairwise-rope-2026-10-01.md):
//  1. every prompt position, resident vs CPU (same quant): cosine >= 0.995 AND relL2 <= 0.15, on the golden
//     prompt AND on a 48-token prompt, via sequential decode and via the batched prefill's last token;
//  2. last-token cosine against the HF golden within 0.01 of what the CPU int4 reads against the same golden
//     (resident >= CPU - 0.01);
//  3. the greedy continuation of the resident equals the CPU int4's token for token (an int4 near-tie flip is
//     reported with the cosines, not papered over).
//
// IT PROVES IT CAN FAIL: after the real measurement it rebinds the NeoX rope pipelines into the SAME resident
// and re-measures the 48-token prompt, which must read below bar 1 (the expected red is the 2026-10-01 numbers).
func TestCohereRealResidentParityCUDA(t *testing.T) {
	requireHeavyModel(t)
	for _, c := range []struct {
		name, env, golden string
	}{
		{"aya-expanse-8b", "GOINFER_COHERE_AYA", "../testdata/cohere_aya_golden.json.gz"},
		{"command-r7b", "GOINFER_COHERE2_R7B", "../testdata/cohere2_r7b_golden.json.gz"},
	} {
		t.Run(c.name, func(t *testing.T) { cohereRealResidentParity(t, decoder.AssetPathForTest(t, c.env), c.golden) })
	}
}

func cohereRealResidentParity(t *testing.T, ckpt, goldenPath string) {
	const (
		cosBar      = 0.995
		relBar      = 0.15
		goldenSlack = 0.01
		ctx         = 512 // the default context does not fit beside int4 weights on an 8 GB card
	)
	var g struct {
		PromptIDs       []int     `json:"prompt_ids"`
		Argmax          int       `json:"argmax"`
		LastLogits      []float32 `json:"last_logits"`
		NNew            int       `json:"n_new"`
		ContinuationIDs []int     `json:"continuation_ids"`
	}
	if err := decoder.ReadGoldenJSONForTest(goldenPath, &g); err != nil {
		t.Skipf("no golden (%v)", err)
	}
	t0 := time.Now()
	hb := func(s string) { t.Logf("[%6.1fs] %s", time.Since(t0).Seconds(), s) } // heartbeat: this test runs minutes
	hb("loading CUDA resident (int4, context 512)")
	mRes, err := decoder.Load(ckpt, decoder.Options{Backend: "cuda", Quant: "int4", ResidentContext: ctx})
	if err != nil {
		t.Fatalf("load cuda: %v", err)
	}
	defer mRes.Close()
	rf := mRes.ResidentForwardForTest()
	cr, ok := rf.(*cudaResident)
	if !ok {
		t.Fatalf("did not go CUDA-resident (%T) — decline: %s", rf, mRes.ResidentDecline())
	}
	if !cr.pairwiseRoPE {
		t.Fatal("resident.pairwiseRoPE=false: the pairwise kernels are not bound, this gate would measure the NeoX path")
	}
	hb("loading CPU model (int4)")
	mCPU, err := decoder.Load(ckpt, decoder.Options{Backend: "cpu", Quant: "int4"})
	if err != nil {
		t.Fatalf("load cpu: %v", err)
	}
	defer mCPU.Close()
	_, _, _, _, _, _, vocab := mCPU.Dims()

	// 48-token prompt: the golden prompt cycled, then a deterministic tail (the construction the 2026-10-01
	// measurement used, so the numbers are comparable).
	long := make([]int, 48)
	for i := range long {
		long[i] = g.PromptIDs[i%len(g.PromptIDs)]
		if i >= len(g.PromptIDs) {
			long[i] = (i*37 + 3) % vocab
		}
	}
	argmaxOf := func(x []float32) int {
		b := 0
		for i, v := range x {
			if v > x[b] {
				b = i
			}
		}
		return b
	}

	type perPos struct {
		cos, rel float64
		at       int
		last     []float32
	}
	// CPU logits per prompt position, computed once per prompt (the CPU int4 forward of an 8B model is the slow part).
	cpuMemo := map[string][][]float32{}
	cpuLogits := func(key string, prompt []int) [][]float32 {
		if v, ok := cpuMemo[key]; ok {
			return v
		}
		cache := mCPU.NewCache(len(prompt) + g.NNew)
		out := make([][]float32, len(prompt))
		for i, tok := range prompt {
			lc, err := mCPU.ForwardForTest(tok, cache)
			if err != nil {
				t.Fatalf("cpu forward[%d]: %v", i, err)
			}
			out[i] = append([]float32(nil), lc...)
		}
		cpuMemo[key] = out
		return out
	}
	// decodeVsCPU: sequential decode through prompt on the resident vs the CPU, every position.
	decodeVsCPU := func(key string, prompt []int) (perPos, []float32) {
		cpu := cpuLogits(key, prompt)
		rf.Reset()
		r := perPos{cos: 1}
		for i, tok := range prompt {
			lr, err := rf.Forward(mRes.EmbedResidentForTest(tok), i)
			if err != nil {
				t.Fatalf("resident forward[%d]: %v", i, err)
			}
			if cos, _ := cosF32(lr, cpu[i]); cos < r.cos {
				r.cos, r.at = cos, i
			}
			r.rel = math.Max(r.rel, relL2(lr, cpu[i]))
			r.last = lr
		}
		return r, cpu[len(prompt)-1]
	}
	prefillVsCPU := func(prompt []int, cpuLast []float32) (cos, rel float64) {
		rf.Reset()
		embs := make([][]float32, len(prompt))
		for i, tok := range prompt {
			embs[i] = mRes.EmbedResidentForTest(tok)
		}
		got, err := cr.PrefillLast(context.Background(), embs, 0)
		if err != nil {
			t.Fatalf("PrefillLast: %v (the batched path must not decline for this family)", err)
		}
		cos, _ = cosF32(got, cpuLast)
		return cos, relL2(got, cpuLast)
	}
	check := func(label string, cos, rel float64) {
		if cos < cosBar || rel > relBar {
			t.Errorf("%s: cosine %.6f (want >= %.3f) relL2 %.4f (want <= %.2f)", label, cos, cosBar, rel, relBar)
		}
	}

	// ---- bar 1 + 2, golden prompt ----
	hb("golden prompt: sequential decode, resident vs CPU every position")
	gp, cpuLastG := decodeVsCPU("golden", g.PromptIDs)
	cosRG, _ := cosF32(gp.last, g.LastLogits)
	cosCG, _ := cosF32(cpuLastG, g.LastLogits)
	t.Logf("GOLDEN PROMPT (%d tokens) decode, resident vs CPU: worst cosine %.6f (pos %d) worst relL2 %.4f", len(g.PromptIDs), gp.cos, gp.at, gp.rel)
	t.Logf("  last token vs HF f32 golden: resident cosine %.6f | CPU int4 cosine %.6f | argmax resident %d CPU %d golden %d",
		cosRG, cosCG, argmaxOf(gp.last), argmaxOf(cpuLastG), g.Argmax)
	check("golden prompt decode vs CPU", gp.cos, gp.rel)
	if cosRG < cosCG-goldenSlack {
		t.Errorf("last token vs HF golden: resident %.6f is more than %.2f below the CPU int4's %.6f", cosRG, goldenSlack, cosCG)
	}
	pc, pr := prefillVsCPU(g.PromptIDs, cpuLastG)
	t.Logf("GOLDEN PROMPT batched prefill last token, resident vs CPU: cosine %.6f relL2 %.4f", pc, pr)
	check("golden prompt batched prefill vs CPU", pc, pr)

	// ---- bar 1, 48-token prompt ----
	hb("48-token prompt: sequential decode, resident vs CPU every position")
	lp, cpuLastL := decodeVsCPU("long", long)
	t.Logf("48-TOKEN PROMPT decode, resident vs CPU: worst cosine %.6f (pos %d) worst relL2 %.4f", lp.cos, lp.at, lp.rel)
	check("48-token decode vs CPU", lp.cos, lp.rel)
	pc, pr = prefillVsCPU(long, cpuLastL)
	t.Logf("48-TOKEN PROMPT batched prefill last token, resident vs CPU: cosine %.6f relL2 %.4f", pc, pr)
	check("48-token batched prefill vs CPU", pc, pr)

	// ---- bar 3: greedy continuation ----
	hb("greedy continuation, resident vs CPU")
	rf.Reset()
	cache := mCPU.NewCache(len(g.PromptIDs) + g.NNew)
	var lc, lr []float32
	for _, tok := range g.PromptIDs {
		lc, _ = mCPU.ForwardForTest(tok, cache)
	}
	for i, tok := range g.PromptIDs {
		lr, _ = rf.Forward(mRes.EmbedResidentForTest(tok), i)
	}
	var resIDs, cpuIDs []int
	for k := 0; k < g.NNew; k++ {
		ri, ci := argmaxOf(lr), argmaxOf(lc)
		resIDs, cpuIDs = append(resIDs, ri), append(cpuIDs, ci)
		var err error
		if lr, err = rf.Forward(mRes.EmbedResidentForTest(ri), len(g.PromptIDs)+k); err != nil {
			t.Fatalf("resident continuation[%d]: %v", k, err)
		}
		if lc, err = mCPU.ForwardForTest(ci, cache); err != nil {
			t.Fatalf("cpu continuation[%d]: %v", k, err)
		}
	}
	mRG, mCG, mRC := 0, 0, 0
	for k := range g.NNew {
		if k < len(g.ContinuationIDs) {
			if resIDs[k] == g.ContinuationIDs[k] {
				mRG++
			}
			if cpuIDs[k] == g.ContinuationIDs[k] {
				mCG++
			}
		}
		if resIDs[k] == cpuIDs[k] {
			mRC++
		}
	}
	t.Logf("GREEDY CONTINUATION (%d tokens): resident == CPU %d/%d | resident == golden %d/%d | CPU == golden %d/%d", g.NNew, mRC, g.NNew, mRG, g.NNew, mCG, g.NNew)
	t.Logf("  golden   %v\n  CPU      %v\n  resident %v", g.ContinuationIDs, cpuIDs, resIDs)
	if mRC != g.NNew {
		t.Errorf("resident continuation differs from the CPU int4's (%d/%d match): resident %v vs CPU %v", mRC, g.NNew, resIDs, cpuIDs)
	}

	// ---- discrimination control: the SAME resident with the NeoX kernels ----
	hb("control: NeoX rope kernels forced into the same resident, 48-token prompt")
	restore := forceNeoXRope(t, cr)
	np, _ := decodeVsCPU("long", long)
	restore()
	t.Logf("NEOX CONTROL, 48-token decode, resident vs CPU: worst cosine %.6f (pos %d) worst relL2 %.4f (2026-10-01 reference: -0.075 R7B / -0.041 Aya)", np.cos, np.at, np.rel)
	if np.cos >= cosBar {
		t.Errorf("the gate is BLIND: with the NeoX kernels forced it still reads cosine %.6f >= %.3f", np.cos, cosBar)
	}
	hb("done")
}
