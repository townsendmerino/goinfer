//go:build cuda && goinfer_testhooks

package cuda

import (
	"fmt"
	"math"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/decoder"
)

// G2-R on CUDA, as registered in docs/tasks/task-forward-architecture-2026-10.md §4.6a: a model-level sanity gate for real Gemma 2 2B, and
// not a softcap test. On Metal the cap barely acted on the gate's prompts (a plant that dropped it passed the rule), so the CUDA
// declaration rests on G2-K, G2-D and G2-L; this test records the same arms beside them.
//
// One process, one session, in this order, each arm the harness of Metal's gate (greedy lockstep off the CPU, 24 steps, the min logit
// cosine and the argmax-equal count over positions from 2) at int8int8 and at int4:
//
//	B  the softcap removed on BOTH sides (decoder.SetAttnSoftcapOffForTest): the baseline, recorded.
//	control (dense Qwen2.5-Coder 1.5B, bar 0.95, a failure voids the session) and Gemma 3 4B at the same quant, the band's reference.
//	G  the softcap on both sides: must clear 0.88 with argmax equal at >= 50% of positions and no NaN, and not sit more than 0.03 under
//	   Gemma 3's min cosine from this session (else PARKED).
//	P  the softcap dropped from the CUDA build only (attnSoftcapDropForTest), the CPU keeping it: recorded, with how much the cap acts on
//	   the tokens P feeds (read on the CPU). It does not gate.
//
//	GOINFER_HEAVY_TESTS=1 go test -count=1 -timeout 40m -tags 'cuda goinfer_testhooks' -run '^TestG2R_cuda$' -v ./cuda/
func TestG2R_cuda(t *testing.T) {
	requireHeavyModel(t)
	const bar, band = 0.88, 0.03
	home := os.ExpandEnv("$HOME/models")
	g2 := home + "/gemma-2-2b-it-Q8_0.gguf"
	g3path := home + "/gemma-3-4b-it"
	dense := home + "/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"
	for _, p := range []string{g2, g3path, dense} {
		if _, err := os.Stat(p); err != nil {
			t.Skipf("no checkpoint at %s", p)
		}
	}
	var report []string
	say := func(format string, a ...any) {
		line := fmt.Sprintf(format, a...)
		report = append(report, line)
		t.Log(line)
	}
	defer func() { t.Log("== G2-R (CUDA) verdict block ==\n" + strings.Join(report, "\n")) }()
	seed2 := g2rSeed(t, g2)

	// the hook check: removing the softcap must change the CPU's logits
	changed := g2rHookChanged(t, g2, seed2)
	say("hook check: removing the softcap changes the CPU's logits: %v", changed)
	if !changed {
		t.Fatal("the hook is broken; the run stops")
	}

	quants := []string{"int8int8", "int4"}
	for _, q := range quants {
		was := decoder.SetAttnSoftcapOffForTest(true)
		st := g2rLockstep(t, g2, q, seed2, 24, 0)
		decoder.SetAttnSoftcapOffForTest(was)
		say("B  %-8s softcap removed on both sides: min cosine %.6f, argmax %d/%d, NaN %d", q, st.minCos, st.exact, st.steps, st.nan)
	}

	ctrl := g2rLockstep(t, dense, "int8int8", g2rSeed(t, dense), 24, -1)
	say("control (dense 1.5B, int8int8): min cosine %.6f, argmax %d/%d", ctrl.minCos, ctrl.exact, ctrl.steps)
	if ctrl.nan > 0 || ctrl.minCos < 0.95 || ctrl.exact*2 < ctrl.steps {
		say("VOID: the control failed its 0.95 bar")
		t.FailNow()
	}
	g3ref := map[string]g2rStats{}
	for _, q := range quants {
		g3ref[q] = g2rLockstep(t, g3path, q, g2rSeed(t, g3path), 24, -1)
		say("Gemma 3 4B (%s): min cosine %.6f, argmax %d/%d", q, g3ref[q].minCos, g3ref[q].exact, g3ref[q].steps)
	}

	gPass, parked := true, false
	for _, q := range quants {
		st := g2rLockstep(t, g2, q, seed2, 24, 50)
		pass := st.nan == 0 && st.minCos >= bar && st.exact*2 >= st.steps
		say("G  %-8s softcap on both sides: min cosine %.6f, argmax %d/%d, NaN %d, worst near-tie %.3f%% -> %s", q, st.minCos, st.exact, st.steps, st.nan, st.worstTie*100, map[bool]string{true: "pass", false: "FAIL"}[pass])
		gPass = gPass && pass
		if pass && st.minCos < g3ref[q].minCos-band {
			parked = true
			say("   PARKED: more than %.2f under Gemma 3's %.6f from this session", band, g3ref[q].minCos)
		}
	}

	for _, q := range quants {
		attnSoftcapDropForTest = true
		stop := decoder.StartAttnScoreStatsForTest()
		st := g2rLockstep(t, g2, q, seed2, 24, 0)
		n, over, maxAbs := stop()
		attnSoftcapDropForTest = false
		fails := st.nan > 0 || st.minCos < bar || st.exact*2 < st.steps
		say("P  %-8s softcap dropped from the CUDA build only: min cosine %.6f, argmax %d/%d -> %s (recorded, does not gate)", q, st.minCos, st.exact, st.steps, map[bool]string{true: "fails the rule", false: "passes the rule"}[fails])
		share := 0.0
		if n > 0 {
			share = float64(over) / float64(n)
		}
		say("   the cap on the CPU at %s over P's run: %d capped scores, %.4f%% with |score| > 15, largest |score| %.2f", q, n, 100*share, maxAbs)
	}
	switch {
	case !gPass:
		say("VERDICT: G2-R (model-level) FAILS at arm G. Stop and report; no retuning.")
		t.Fail()
	case parked:
		say("VERDICT: G2-R PARKED (band). The feature is not declared while parked.")
		t.Fail()
	default:
		say("VERDICT: G2-R (model-level) PASSES: G clears the bar at both quants, inside Gemma 3's band. The softcap declaration rests on G2-K, G2-D and G2-L.")
	}
}

type g2rStats struct {
	steps, exact, nan int
	minCos, worstTie  float64
}

func g2rSeed(t *testing.T, path string) []int {
	t.Helper()
	tk, err := loadTok(path)
	if err != nil {
		t.Skipf("tokenizer for %s: %v", path, err)
	}
	ids, err := tk.Encode("The capital of France is", true)
	if err != nil || len(ids) < 2 {
		t.Fatalf("encode: %v (%d ids)", err, len(ids))
	}
	return ids
}

// g2rLockstep drives the CUDA resident and the CPU over the same tokens: the seed, then the CPU's greedy continuation, steps in all. The
// CUDA side must be resident (a silent CPU fallback would pass every bar). wantCap is the cap the resident must carry for the arm, or
// negative for no check.
func g2rLockstep(t *testing.T, path, quant string, seed []int, steps int, wantCap float32) g2rStats {
	t.Helper()
	mg, err := decoder.Load(path, decoder.Options{Backend: "cuda", Quant: quant})
	if err != nil {
		t.Fatalf("load (cuda) %s at %s: %v", path, quant, err)
	}
	defer mg.Close()
	rf := mg.ResidentForwardForTest()
	if rf == nil {
		t.Fatalf("cuda resident DECLINED %s at %s (%s): %s", path, quant, mg.DecodePath(), mg.ResidentDecline())
	}
	if wantCap >= 0 { // a Gemma 2 arm: the resident's own cap must be the arm's (0 in B and P, the model's 50 in G)
		if got := rf.(*cudaResident).attnCap; got != wantCap {
			t.Fatalf("VOID: the CUDA resident carries an attention cap of %v, this arm's is %v", got, wantCap)
		}
	}
	mc, err := decoder.Load(path, decoder.Options{Quant: quant})
	if err != nil {
		t.Fatalf("load (cpu): %v", err)
	}
	defer mc.Close()
	cache := mc.NewCache(steps + 2)
	rf.Reset()
	st := g2rStats{steps: steps, minCos: 1}
	tok := seed[0]
	for i := range steps {
		cl, err := mc.ForwardForTest(tok, cache)
		if err != nil {
			t.Fatalf("cpu forward: %v", err)
		}
		gl, err := rf.Forward(mg.EmbedResidentForTest(tok), i)
		if err != nil {
			t.Fatalf("cuda forward: %v", err)
		}
		for _, v := range gl {
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				st.nan++
				break
			}
		}
		if i >= 2 { // the <bos> sink positions are not a meaningful cosine (Metal's gate skips them for the same reason)
			var dot, na, nb float64
			for j := range cl {
				dot += float64(cl[j]) * float64(gl[j])
				na += float64(cl[j]) * float64(cl[j])
				nb += float64(gl[j]) * float64(gl[j])
			}
			st.minCos = math.Min(st.minCos, dot/(math.Sqrt(na)*math.Sqrt(nb)))
		}
		ca, ga := argmaxF(cl), argmaxF(gl)
		if ca == ga {
			st.exact++
		} else {
			lo, hi := cl[0], cl[0]
			for _, v := range cl {
				lo, hi = min(lo, v), max(hi, v)
			}
			st.worstTie = math.Max(st.worstTie, float64(cl[ca]-cl[ga])/(float64(hi-lo)+1e-9))
		}
		if i+1 < len(seed) {
			tok = seed[i+1]
		} else {
			tok = ca
		}
	}
	t.Logf("%s at %s (%s): %d/%d argmax-exact, worst near-tie %.3f%%, min cosine %.6f", path, quant, mg.DecodePath(), st.exact, st.steps, st.worstTie*100, st.minCos)
	return st
}

// g2rHookChanged runs the CPU model over the seed with and without the softcap and reports whether any logit differs.
func g2rHookChanged(t *testing.T, path string, seed []int) bool {
	t.Helper()
	run := func(off bool) []float32 {
		was := decoder.SetAttnSoftcapOffForTest(off)
		defer decoder.SetAttnSoftcapOffForTest(was)
		mc, err := decoder.Load(path, decoder.Options{Quant: "int4"})
		if err != nil {
			t.Fatalf("load (cpu): %v", err)
		}
		defer mc.Close()
		cache := mc.NewCache(len(seed) + 2)
		var last []float32
		for _, tok := range seed {
			l, err := mc.ForwardForTest(tok, cache)
			if err != nil {
				t.Fatal(err)
			}
			last = append([]float32(nil), l...)
		}
		return last
	}
	a, b := run(false), run(true)
	for i := range a {
		if a[i] != b[i] {
			return true
		}
	}
	return false
}

// TestG2R_fitCheck loads, once and alone, every CUDA model TestG2R_cuda loads, with the CPU side not loaded, and reports whether the card
// admits it and how long the load took: the by-day fit check that precedes queueing the night job (CLAUDE.md, "Testing tiers").
//
//	GOINFER_HEAVY_TESTS=1 go test -count=1 -timeout 20m -tags 'cuda goinfer_testhooks' -run '^TestG2R_fitCheck$' -v ./cuda/
func TestG2R_fitCheck(t *testing.T) {
	requireHeavyModel(t)
	home := os.ExpandEnv("$HOME/models")
	for _, c := range []struct{ path, quant string }{
		{home + "/gemma-2-2b-it-Q8_0.gguf", "int8int8"}, {home + "/gemma-2-2b-it-Q8_0.gguf", "int4"},
		{home + "/gemma-3-4b-it", "int8int8"}, {home + "/gemma-3-4b-it", "int4"},
		{home + "/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf", "int8int8"},
	} {
		if _, err := os.Stat(c.path); err != nil {
			t.Skipf("no checkpoint at %s", c.path)
		}
		t0 := time.Now()
		m, err := decoder.Load(c.path, decoder.Options{Backend: "cuda", Quant: c.quant})
		if err != nil {
			t.Fatalf("load %s at %s: %v", c.path, c.quant, err)
		}
		took := time.Since(t0)
		resident := m.ResidentForwardForTest() != nil
		t.Logf("fit: %s at %s: resident %v (%s), load %.1fs, decline %q", c.path, c.quant, resident, m.DecodePath(), took.Seconds(), m.ResidentDecline())
		m.Close()
		if !resident {
			t.Errorf("%s at %s is not admitted by the card", c.path, c.quant)
		}
	}
}
