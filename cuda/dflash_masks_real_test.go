//go:build cuda && goinfer_testhooks

package cuda

import (
	"fmt"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestDFlashMasks_speed is G3 of docs/measurements/dflash-window-2026-10/README.md, on the SP2 rig (Qwen3.5-9B with its DFlash
// drafter): BlockSpec at w = 4 with the drafter's per-layer masks on against off (drafterMasksOffForTest), interleaved A B B A in
// one process, on SP0's 12 code and math prompts as they are and behind the 3,800-token preamble, 10 pairs each. Each pass
// records its decode rate and its committed tokens a round; the arms' tokens must be equal (verify is lossless). The reading is
// on over off; it decides nothing but the registered stop. SP2_SMOKE=1 runs one pair over two prompts per context.
func TestDFlashMasks_speed(t *testing.T) {
	g := newSP2Rig(t)
	const maxNew, width = 256, 4
	pairs, perCtx := 10, 0
	if os.Getenv("SP2_SMOKE") == "1" {
		pairs, perCtx = 1, 2
	}
	preamble := sp2LongText(t, g.tk, filepath.Join("..", "docs", "benchmarks.md"), 3800)
	type ctxSet struct {
		name    string
		prompts [][]int
	}
	var sets []ctxSet
	for _, c := range []struct{ name, pre string }{{"short", ""}, {"long", preamble + "\n\n"}} {
		cs := ctxSet{name: c.name}
		for _, s := range []string{"code", "math"} {
			for _, p := range widthSweepSuites[s] {
				ids, err := sp2Encode(g.tk, c.pre+p, "off")
				if err != nil {
					t.Fatal(err)
				}
				cs.prompts = append(cs.prompts, ids)
			}
		}
		if perCtx > 0 {
			cs.prompts = cs.prompts[:perCtx]
		}
		sets = append(sets, cs)
	}
	// One arm's pass: decode tokens and seconds (first token to the end, so prefill is outside), rounds, and each prompt's tokens.
	pass := func(prompts [][]int, masksOff bool) (toks int, secs float64, rounds int, outs [][]int) {
		drafterMasksOffForTest = masksOff
		defer func() { drafterMasksOffForTest = false }()
		for _, ids := range prompts {
			g.mc.ForgetResidentForTest() // every generation prefills cold, in both arms
			out, r, first, err := g.spec.GenerateFirstTokenForTest(ids, decoder.BlockSpecOptions{VerifyWidth: width, MaxTokens: maxNew})
			if err != nil {
				t.Fatal(err)
			}
			toks += len(out) - 1
			secs += time.Since(first).Seconds()
			rounds += r
			outs = append(outs, out)
		}
		return toks, secs, rounds, outs
	}
	for _, cs := range sets {
		pass(cs.prompts[:2], false) // warm-up, discarded
		pass(cs.prompts[:2], true)
		var ratios []float64
		for i := range pairs {
			cell := g.sp2ValidCell(t, fmt.Sprintf("G3 %s pair %d", cs.name, i))
			var onT, offT, onR, offR int
			var onS, offS float64
			var onO, offO [][]int
			if i%2 == 0 {
				onT, onS, onR, onO = pass(cs.prompts, false)
				offT, offS, offR, offO = pass(cs.prompts, true)
			} else {
				offT, offS, offR, offO = pass(cs.prompts, true)
				onT, onS, onR, onO = pass(cs.prompts, false)
			}
			for k := range onO {
				if at := sp2Diverge(onO[k], offO[k]); at >= 0 {
					t.Errorf("%s pair %d prompt %d: the arms' tokens differ at %d: verify is not lossless here", cs.name, i, k, at)
				}
			}
			ratio := (float64(onT) / onS) / (float64(offT) / offS)
			ratios = append(ratios, ratio)
			g.record(t, map[string]any{"kind": "g3", "context": cs.name, "pair": i, "cell": cell,
				"order": map[bool]string{true: "on-off", false: "off-on"}[i%2 == 0], "on_tokens": onT, "on_s": onS, "on_rounds": onR,
				"off_tokens": offT, "off_s": offS, "off_rounds": offR, "speedup": ratio})
			t.Logf("%s pair %d: masks on %.1f tok/s, %.2f a round; off %.1f tok/s, %.2f a round; %.3fx", cs.name, i,
				float64(onT)/onS, float64(onT+len(onO))/float64(onR), float64(offT)/offS, float64(offT+len(offO))/float64(offR), ratio)
		}
		above := 0
		for _, r := range ratios {
			if r > 1 {
				above++
			}
		}
		med := sp2Median(ratios)
		g.record(t, map[string]any{"kind": "g3-summary", "context": cs.name, "median": med, "pairs_above_1": above})
		t.Logf("G3 %s: masks on over off, median %.3fx, %d of %d pairs above 1", cs.name, med, above, pairs)
		if pairs == 10 && cs.name == "short" && med < 0.98 {
			t.Logf("G3 STOP (registered): the short context's median %.3fx is under 0.98x; to the owner before merging", med)
		}
	}
}
