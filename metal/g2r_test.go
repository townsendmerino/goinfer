//go:build darwin && goinfer_testhooks

package metal

import (
	"fmt"
	"os"
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestG2R_registered is OP1's real Gemma 2 gate, G2-R, run as registered in docs/tasks/task-forward-architecture-2026-10.md
// §4.6 ("G2-R's baseline step"): one session, in this order.
//
//  0. The hook check: on the CPU, removing the softcap (decoder.SetAttnSoftcapOffForTest) must change the 2B's logits.
//  1. Arm B: the gate's harness with the softcap removed on both sides, at int8int8 and int4. If 0.88 is within 0.001 of
//     B's min cosine at either quant, the run stops there.
//  2. The control (TestDenseResidentParity's) and Gemma 3 (TestGemma3ResidentParity's), for G2-R's band.
//  3. Arm G: TestGemma2ResidentParity's measurement, the softcap on both sides, at both quants.
//  4. Arm P: the softcap dropped from the Metal build only (attnSoftcapDropForTest), at both quants; it must fail G2-R's
//     rule. P also records, on the CPU and on the tokens P feeds, how much the softcap acts: the share of scores with
//     |score| > 15 before the cap, the largest |score|, and the worst per-position logit cosine between the capped and the
//     uncapped CPU runs.
//
// It prints a verdict block and fails on anything but a pass. Run it alone, under the timing lock:
//
//	GOINFER_HEAVY_TESTS=1 go test -tags goinfer_testhooks ./metal -run '^TestG2R_registered$' -count=1 -v -timeout 60m
func TestG2R_registered(t *testing.T) {
	requireHeavyModel(t)
	const bar, stopMargin, band = 0.88, 0.001, 0.03
	g2 := os.ExpandEnv("$HOME/models/gemma-2-2b-it-Q8_0.gguf")
	g3 := os.ExpandEnv("$HOME/models/gemma-3-4b-it-Q4_K_M.gguf")
	dense := os.ExpandEnv("$HOME/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf")
	for _, p := range []string{g2, g3, dense} {
		if _, err := os.Stat(p); err != nil {
			t.Fatalf("VOID: a checkpoint the session needs is missing: %v", err)
		}
	}
	quants := []string{"int8int8", "int4"}
	seed := seedPrompt(t, g2, probeText)
	var report []string
	say := func(format string, a ...any) {
		line := fmt.Sprintf(format, a...)
		report = append(report, line)
		t.Log(line)
	}
	defer func() { t.Log("== G2-R verdict block ==\n" + strings.Join(report, "\n")) }()

	// 0. The hook check.
	tokens := g2rTokens(t, g2, "int8int8", seed, 24, false)
	with := g2rCPULogits(t, g2, "int8int8", tokens, true)
	without := g2rCPULogits(t, g2, "int8int8", tokens, false)
	changed := false
	for i := range with {
		for j := range with[i] {
			changed = changed || with[i][j] != without[i][j]
		}
	}
	if !changed {
		say("STOP: the hook is broken: the CPU's logits with the softcap removed equal its logits with it")
		t.FailNow()
	}
	say("hook check: removing the softcap changes the CPU's logits (ok)")

	// 1. Arm B.
	stop := false
	for _, q := range quants {
		was := decoder.SetAttnSoftcapOffForTest(true)
		g2rCheckCap(t, g2, q, 0)
		st := residentParityAt(t, g2, q, seed, 24)
		decoder.SetAttnSoftcapOffForTest(was)
		say("B  %-8s softcap removed on both sides: min cosine %.6f, argmax %d/%d, NaN %d", q, st.minCos, st.exact, st.steps, st.nan)
		if st.minCos < bar+stopMargin {
			stop = true
			say("STOP (registered): 0.88 is within %.3f of B's min cosine at %s; the graded arm does not run and the bar is not changed", stopMargin, q)
		}
	}
	if stop {
		t.FailNow()
	}

	// 2. The control and Gemma 3, same session.
	ctrl := residentParity(t, dense, seedPrompt(t, dense, probeText), 24)
	say("control (dense 1.5B, int8int8): min cosine %.6f, argmax %d/%d", ctrl.minCos, ctrl.exact, ctrl.steps)
	if ctrl.nan > 0 || ctrl.minCos < 0.95 || ctrl.exact*2 < ctrl.steps {
		say("VOID: the control failed its 0.95 bar")
		t.FailNow()
	}
	g3st := residentParity(t, g3, seedPrompt(t, g3, probeText), 24)
	say("Gemma 3 4B (int8int8): min cosine %.6f, argmax %d/%d", g3st.minCos, g3st.exact, g3st.steps)

	// 3. Arm G.
	gPass, parked := true, false
	for _, q := range quants {
		g2rCheckCap(t, g2, q, 50)
		st := residentParityAt(t, g2, q, seed, 24)
		pass := st.nan == 0 && st.minCos >= bar && st.exact*2 >= st.steps
		say("G  %-8s softcap on both sides: min cosine %.6f, argmax %d/%d, NaN %d, worst near-tie %.3f%% -> %s", q, st.minCos, st.exact, st.steps, st.nan, st.worstTie*100, map[bool]string{true: "pass", false: "FAIL"}[pass])
		gPass = gPass && pass
		if pass && st.minCos < g3st.minCos-band {
			parked = true
			say("   PARKED: more than %.2f under Gemma 3's %.6f from this session", band, g3st.minCos)
		}
	}

	// 4. Arm P, and how much the softcap acts.
	pFails := true
	for _, q := range quants {
		attnSoftcapDropForTest = true
		g2rCheckCap(t, g2, q, 0)
		st := residentParityAt(t, g2, q, seed, 24)
		attnSoftcapDropForTest = false
		fails := st.nan > 0 || st.minCos < bar || st.exact*2 < st.steps
		say("P  %-8s softcap dropped from the Metal build only: min cosine %.6f, argmax %d/%d -> %s", q, st.minCos, st.exact, st.steps, map[bool]string{true: "fails the rule, as it must", false: "PASSES the rule"}[fails])
		pFails = pFails && fails

		toks := g2rTokens(t, g2, q, seed, 24, false)
		stopStats := decoder.StartAttnScoreStatsForTest()
		capped := g2rCPULogits(t, g2, q, toks, false)
		n, over, maxAbs := stopStats()
		uncapped := g2rCPULogits(t, g2, q, toks, true)
		worst := 1.0
		for i := 2; i < len(capped); i++ {
			worst = min(worst, cosF(capped[i], uncapped[i]))
		}
		share := 0.0
		if n > 0 {
			share = float64(over) / float64(n)
		}
		say("   P's tokens on the CPU at %s: %d capped scores, %.4f%% with |score| > 15, largest |score| %.2f; worst capped-vs-uncapped logit cosine %.6f", q, n, 100*share, maxAbs, worst)
	}

	switch {
	case !gPass:
		say("VERDICT: G2-R FAILS (arm G). Stop and report; no retuning.")
		t.Fail()
	case parked:
		say("VERDICT: G2-R PARKED (band). The feature is not declared while parked.")
		t.Fail()
	case !pFails:
		say("VERDICT: VOID as a softcap test at the quant where P passed. Do not declare.")
		t.Fail()
	default:
		say("VERDICT: G2-R PASSES: G passes at both quants and P fails at both.")
	}
}

// g2rTokens is the token sequence G2-R's harness feeds: the seed, then the CPU's greedy continuation, for steps tokens.
func g2rTokens(t *testing.T, path, quant string, seed []int, steps int, softcapOff bool) []int {
	t.Helper()
	logits := g2rRun(t, path, quant, seed, steps, softcapOff, nil)
	toks := make([]int, steps)
	for i := range steps {
		if i < len(seed) {
			toks[i] = seed[i]
		} else {
			toks[i] = argmaxF(logits[i-1])
		}
	}
	return toks
}

// g2rCPULogits is the CPU's logits for each of toks, with or without the softcap.
func g2rCPULogits(t *testing.T, path, quant string, toks []int, softcapOff bool) [][]float32 {
	t.Helper()
	return g2rRun(t, path, quant, nil, len(toks), softcapOff, toks)
}

// g2rRun runs the CPU model: on toks when given, else the seed and its greedy continuation.
func g2rRun(t *testing.T, path, quant string, seed []int, steps int, softcapOff bool, toks []int) [][]float32 {
	t.Helper()
	was := decoder.SetAttnSoftcapOffForTest(softcapOff)
	m, err := decoder.Load(path, decoder.Options{Quant: quant})
	decoder.SetAttnSoftcapOffForTest(was)
	if err != nil {
		t.Fatalf("load (cpu): %v", err)
	}
	defer m.Close()
	if got := m.AttnLogitSoftcapResident(); (got == 0) != softcapOff {
		t.Fatalf("VOID: the CPU model's cap is %v with the softcap removed = %v", got, softcapOff)
	}
	_, nL, _, nKV, hd, _, _ := m.Dims()
	cache := decoder.NewKVCache(nL, nKV, hd, 0, 1024, nil)
	out := make([][]float32, 0, steps)
	tok := 0
	if toks != nil {
		tok = toks[0]
	} else {
		tok = seed[0]
	}
	for i := range steps {
		l, err := m.ForwardForTest(tok, cache)
		if err != nil {
			t.Fatalf("cpu forward: %v", err)
		}
		out = append(out, append([]float32(nil), l...))
		switch {
		case i+1 >= steps:
		case toks != nil:
			tok = toks[i+1]
		case i+1 < len(seed):
			tok = seed[i+1]
		default:
			tok = argmaxF(l)
		}
	}
	return out
}

// g2rCheckCap loads the 2B on Metal under the current hooks and checks the resident compiled the cap the arm expects:
// one of the void conditions.
func g2rCheckCap(t *testing.T, path, quant string, want float32) {
	t.Helper()
	m, err := decoder.Load(path, decoder.Options{Backend: "metal", Quant: quant})
	if err != nil {
		t.Fatalf("load (metal): %v", err)
	}
	defer m.Close()
	rf, ok := m.ResidentForwardForTest().(*metalResident)
	if !ok || rf == nil {
		skipIfMemoryDeclined(t, m)
		t.Fatalf("VOID: the 2B did not go resident on Metal at %s: %s", quant, m.ResidentDecline())
	}
	if rf.r.attnSoftcap != want {
		t.Fatalf("VOID: the resident's compiled cap is %v, the arm wants %v", rf.r.attnSoftcap, want)
	}
}
