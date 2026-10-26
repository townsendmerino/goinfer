//go:build cuda && goinfer_testhooks

package cuda

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/chat"
	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/internal/mem/probe"
	"github.com/townsendmerino/goinfer/internal/modelload"
	"github.com/townsendmerino/goinfer/tokenizer"
)

// SP2's real-pairing gates on Qwen3.5-9B with z-lab's DFlash (docs/tasks/task-forward-architecture-2026-10.md §6a, "SP2 on
// nobara"): G1c identity with the plants P1 and P2 and G5 on the 9B (TestSP2_identity), G2 the fold's cost (TestSP2_foldCost) and
// G4 speed (TestSP2_speed). Each writes JSON lines to SP2_OUT for docs/measurements/sp2-2026-10/.
//
//	GOINFER_HEAVY_TESTS=1 SP2_OUT=<out.jsonl> [SP2_TARGET=<giw>] [SP2_DRAFTER=<dir>] [SP2_WIDTH=<w>] [SP2_MOE_STREAM=1] \
//	  go test -tags 'cuda goinfer_testhooks' -run '^TestSP2_identity$' -v -timeout 60m ./cuda/
//
// SP2_WIDTH is the verify width G1c and G4 run at (default 4); SP2_MOE_STREAM=1 streams a MoE target's experts
// (Options.MoECacheExperts), as the 35B needs on this card.

func sp2Target() string {
	if p := os.Getenv("SP2_TARGET"); p != "" {
		return p
	}
	return os.ExpandEnv("$HOME/models/Qwen3.5-9B-Q4_K_M.int4.e4h.cuda.v17.giw")
}

// sp2Width is the verify width G1c (beside 16) and G4 run at: SP2_WIDTH, else 4.
func sp2Width() int {
	if w, err := strconv.Atoi(os.Getenv("SP2_WIDTH")); err == nil && w >= 2 {
		return w
	}
	return 4
}

func sp2Drafter() string {
	if p := os.Getenv("SP2_DRAFTER"); p != "" {
		return p
	}
	return os.ExpandEnv("$HOME/models/qwen35-9b-dflash")
}

// sp2Rig is the 9B resident with its drafter attached through the production attach, so the arena is the one serve allocates.
type sp2Rig struct {
	mc   *decoder.Model
	r    *cudaResident
	dr   *decoder.DFlashDrafter
	spec *decoder.BlockSpec
	tk   *tokenizer.Tokenizer
	out  *json.Encoder
}

func newSP2Rig(t *testing.T) *sp2Rig {
	t.Helper()
	requireHeavyModel(t)
	if os.Getenv("SP2_OUT") == "" {
		t.Skip("set SP2_OUT")
	}
	f, err := os.OpenFile(os.Getenv("SP2_OUT"), os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o644)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { f.Close() })
	dr, err := decoder.OpenDFlashDrafter(sp2Drafter()) // as serve opens it: the attach streams its matrices
	if err != nil {
		t.Fatalf("load drafter: %v", err)
	}
	t.Cleanup(func() { dr.Close() })
	opts := decoder.Options{Backend: "cuda", Quant: "int4", ResidentContext: 4608, KVPrecision: "f32", Drafter: true,
		ExtraResidentBytes: decoder.DrafterResidentBytesEstimate(dr), ExtraResidentKVPerPosition: decoder.DrafterKVBytesPerPosition(dr),
		MoECacheExperts: os.Getenv("SP2_MOE_STREAM") == "1"}
	mc, err := decoder.Load(sp2Target(), opts)
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	t.Cleanup(func() { mc.Close() })
	r := mustResident(t, mc)
	spec, err := mc.NewBlockSpec(dr, dr.TargetLayerIDs())
	if err != nil {
		t.Fatalf("NewBlockSpec: %v", err)
	}
	tk, err := modelload.Tokenizer(sp2Target())
	if err != nil {
		t.Fatalf("tokenizer: %v", err)
	}
	rig := &sp2Rig{mc: mc, r: r, dr: dr, spec: spec, tk: tk, out: json.NewEncoder(f)}
	free, total, _ := r.dev.Context().MemInfo()
	rig.record(t, map[string]any{"kind": "load", "target": sp2Target(), "drafter": sp2Drafter(), "moe_stream": opts.MoECacheExperts, "arena_bytes": r.RecurrentSpecBytes(),
		"arena_rows": r.RecurrentSpecRows(), "reserve_bytes": mc.ExtraResidentBytes(), "ctx": mc.ResidentContextCap(),
		"vram_free": free, "vram_total": total})
	return rig
}

func (g *sp2Rig) record(t *testing.T, rec map[string]any) {
	t.Helper()
	rec["time"] = time.Now().Format(time.RFC3339)
	if err := g.out.Encode(rec); err != nil {
		t.Fatal(err)
	}
}

// sp2Prompt is one G1c prompt: its class and its ids.
type sp2Prompt struct {
	class string
	ids   []int
}

// sp2Prompts is G1c's set: SP0's code and math prompts with thinking off, its math prompts with thinking on, and two long prompts
// of about 3,000 tokens built from this repo's own sources.
func (g *sp2Rig) sp2Prompts(t *testing.T) []sp2Prompt {
	t.Helper()
	var ps []sp2Prompt
	add := func(class, text, think string) {
		ids, err := sp2Encode(g.tk, text, think)
		if err != nil {
			t.Fatalf("encode: %v", err)
		}
		ps = append(ps, sp2Prompt{class, ids})
	}
	for _, s := range []string{"code", "math"} {
		for _, p := range widthSweepSuites[s] {
			add(s, p, "off")
		}
	}
	for _, p := range widthSweepSuites["math"] {
		add("math-think", p, "on")
	}
	for _, lp := range []struct{ file, ask string }{
		{"deltanet.cu", "Explain what delta_rule_rows_128 computes, step by step, and write a plain Go reference for one row."},
		{filepath.Join("..", "decoder", "blockspec.go"), "List every exit of BlockSpec.generate and what it commits."},
	} {
		add("long", sp2LongText(t, g.tk, lp.file, 2900)+"\n\n"+lp.ask, "off")
	}
	return ps
}

// sp2LongText is the leading lines of file, as many as fit in about tokens tokens.
func sp2LongText(t *testing.T, tk *tokenizer.Tokenizer, file string, tokens int) string {
	t.Helper()
	b, err := os.ReadFile(file)
	if err != nil {
		t.Fatal(err)
	}
	lines := strings.Split(string(b), "\n")
	lo, hi := 1, len(lines)
	for lo < hi { // the most lines whose text stays within tokens
		mid := (lo + hi + 1) / 2
		ids, err := tk.Encode(strings.Join(lines[:mid], "\n"), false)
		if err != nil {
			t.Fatal(err)
		}
		if len(ids) <= tokens {
			lo = mid
		} else {
			hi = mid - 1
		}
	}
	return "```\n" + strings.Join(lines[:lo], "\n") + "\n```"
}

// sp2Diverge is the first index where a and b differ, or -1 when they are equal (a length difference counts at the shorter end).
func sp2Diverge(a, b []int) int {
	for i := range min(len(a), len(b)) {
		if a[i] != b[i] {
			return i
		}
	}
	if len(a) != len(b) {
		return min(len(a), len(b))
	}
	return -1
}

// TestSP2_identity is G1c on the real pairing, then the plants P1 and P2, then G5 on the 9B.
func TestSP2_identity(t *testing.T) {
	g := newSP2Rig(t)
	const maxNew = 256
	prompts := g.sp2Prompts(t)
	type plainRun struct {
		toks  []int
		state [][]float32
	}
	plains := make([]plainRun, len(prompts))
	fails := 0
	for i, p := range prompts {
		want, _ := plainGreedy(t, g.mc, g.r, p.ids, maxNew)
		plains[i] = plainRun{want, dnSnapshot(t, g.r)}
		// The served default's plain decode, without the exact-attention scope: recorded, no bar.
		ch, gen := g.mc.Generate(context.Background(), p.ids, maxNew, decoder.SamplingParams{})
		var served []int
		for id := range ch {
			served = append(served, id)
		}
		if e := gen.Err(); e != nil {
			t.Fatalf("served plain: %v", e)
		}
		for _, w := range []int{sp2Width(), 16} {
			got, rounds, err := g.spec.Generate(p.ids, decoder.BlockSpecOptions{VerifyWidth: w, MaxTokens: maxNew})
			if err != nil {
				t.Fatalf("prompt %d w=%d: %v", i, w, err)
			}
			at := sp2Diverge(got, want)
			stateDiff := ""
			if at < 0 {
				stateDiff = snapDiff(plains[i].state, dnSnapshot(t, g.r))
			}
			pass := at < 0 && stateDiff == ""
			if !pass {
				fails++
				t.Errorf("prompt %d (%s) w=%d: first divergence at token %d; state: %q", i, p.class, w, at, stateDiff)
			}
			g.record(t, map[string]any{"kind": "g1c", "prompt": i, "class": p.class, "prompt_tokens": len(p.ids), "width": w,
				"tokens": len(got), "plain_tokens": len(want), "rounds": rounds, "diverge_at": at, "state_diff": stateDiff, "pass": pass,
				"served_plain_diverge_at": sp2Diverge(got, served)})
		}
	}
	// The guard's fallback: the turn ends on plain steps after folded rounds.
	withGuardTripped(func() {
		for i := range 4 {
			got, rounds, err := g.spec.Generate(prompts[i].ids, decoder.BlockSpecOptions{VerifyWidth: 4, MaxTokens: maxNew})
			if err != nil {
				t.Fatalf("guard prompt %d: %v", i, err)
			}
			at := sp2Diverge(got, plains[i].toks)
			stateDiff := ""
			if at < 0 {
				stateDiff = snapDiff(plains[i].state, dnSnapshot(t, g.r))
			}
			if at >= 0 || stateDiff != "" {
				fails++
				t.Errorf("guard tripped, prompt %d: divergence at %d; state: %q", i, at, stateDiff)
			}
			g.record(t, map[string]any{"kind": "g1c-guard", "prompt": i, "rounds": rounds, "diverge_at": at, "state_diff": stateDiff})
		}
	})
	t.Logf("G1c: %d failures over %d prompts x 2 widths plus 4 guard runs", fails, len(prompts))

	// The plants, at w = 4 on the 12 thinking-off prompts: each must diverge somewhere.
	for _, pl := range []struct {
		name  string
		plant int
	}{{"P1-fold-skips-conv", dnSpecPlantFoldSkipsConv}, {"P2-verify-writes-state", dnSpecPlantVerifyWritesState}} {
		caught := 0
		withSpecPlant(pl.plant, func() {
			for i := range 12 {
				got, _, err := g.spec.Generate(prompts[i].ids, decoder.BlockSpecOptions{VerifyWidth: 4, MaxTokens: maxNew})
				if err != nil {
					t.Fatalf("plant %s prompt %d: %v", pl.name, i, err)
				}
				at := sp2Diverge(got, plains[i].toks)
				if at >= 0 {
					caught++
				}
				g.record(t, map[string]any{"kind": "plant", "plant": pl.name, "prompt": i, "diverge_at": at})
			}
		})
		if caught == 0 {
			t.Errorf("plant %s diverged on none of 12 prompts: G1c cannot see it", pl.name)
		} else {
			t.Logf("plant %s caught [expected] on %d of 12 prompts", pl.name, caught)
		}
	}

	// G5 on the 9B: a speculative turn A, then B extending it, plainly on A's slot, against cold.
	g.mc.ForgetResidentForTest()
	promptA := prompts[0].ids
	ckptBefore := ckptSnapshot(t, g.r)
	outA, _, err := g.spec.Generate(promptA, decoder.BlockSpecOptions{VerifyWidth: 4, MaxTokens: 96})
	if err != nil {
		t.Fatalf("G5 A: %v", err)
	}
	afterA := dnSnapshot(t, g.r)
	ckptDiff := snapDiff(ckptBefore, ckptSnapshot(t, g.r))
	suffix, err := sp2Encode(g.tk, "Now add tests for it.", "off")
	if err != nil {
		t.Fatal(err)
	}
	promptB := append(append(append([]int(nil), promptA...), outA...), suffix...)
	warmB, gen := plainGreedy(t, g.mc, g.r, promptB, 96)
	reused := gen.PrefillReused
	if _, err := g.r.PrefillLastNArgmax(embsOf(g.mc, append(append([]int(nil), promptA...), outA...)), 0); err != nil {
		t.Fatalf("G5 cold prefill of A: %v", err)
	}
	stateDiff := snapDiff(dnSnapshot(t, g.r), afterA)
	g.mc.ForgetResidentForTest()
	coldB, _ := plainGreedy(t, g.mc, g.r, promptB, 96)
	ok := ckptDiff == "" && reused == len(promptA)+len(outA) && stateDiff == "" && sp2Diverge(warmB, coldB) < 0
	if !ok {
		t.Errorf("G5: checkpoint %q; reused %d of %d; state after A vs cold %q; warm B vs cold B diverge at %d",
			ckptDiff, reused, len(promptA)+len(outA), stateDiff, sp2Diverge(warmB, coldB))
	}
	g.record(t, map[string]any{"kind": "g5", "ckpt_diff": ckptDiff, "reused": reused, "want_reused": len(promptA) + len(outA),
		"state_diff": stateDiff, "warm_cold_diverge_at": sp2Diverge(warmB, coldB), "pass": ok})
}

// sp2Cell reads the box at a cell's start and returns the record and, when the registered rule voids the cell, why: the
// 1-minute load average above 8.0 (half nobara's 16 cores), or available memory under the fit need plus 2 GB. Every cell starts
// with the load already resident in this process, so what it still needs is 0 and the term is the 2 GB margin alone.
func sp2Cell() (map[string]any, string) {
	b, _ := os.ReadFile("/proc/loadavg")
	la, _ := strconv.ParseFloat(strings.Fields(string(b))[0], 64)
	avail := probe.Available()
	cell := map[string]any{"loadavg": la, "mem_available": avail}
	switch {
	case la > 8.0:
		return cell, fmt.Sprintf("load %.2f > 8.0", la)
	case avail < 2<<30:
		return cell, fmt.Sprintf("available %d B < 2 GB with the load resident", avail)
	}
	return cell, ""
}

// sp2ValidCell waits until a cell may start under the void rule, recording each void reading, and returns the valid one. A box
// that stays busy for 30 minutes fails the test rather than measuring under load.
func (g *sp2Rig) sp2ValidCell(t *testing.T, what string) map[string]any {
	t.Helper()
	for try := 0; ; try++ {
		cell, why := sp2Cell()
		if why == "" {
			return cell
		}
		g.record(t, map[string]any{"kind": "void", "cell": what, "why": why, "reading": cell})
		t.Logf("VOID at the start of %s (%s); re-run in order after 60 s", what, why)
		if try >= 30 {
			t.Fatalf("%s: the box stayed busy for 30 minutes (%s)", what, why)
		}
		time.Sleep(60 * time.Second)
	}
}

func sp2Median(xs []float64) float64 {
	s := append([]float64(nil), xs...)
	sort.Float64s(s)
	if len(s)%2 == 1 {
		return s[len(s)/2]
	}
	return (s[len(s)/2-1] + s[len(s)/2]) / 2
}

func sp2Quartiles(xs []float64) (q1, q3 float64) {
	s := append([]float64(nil), xs...)
	sort.Float64s(s)
	return s[len(s)/4], s[(3*len(s))/4]
}

// TestSP2_foldCost is G2: in one process, interleaved and rotated A B B A over 3 warm-ups and 20 repetitions, a plain step, a
// w = 4 verify on the fold path and on today's path, the fold of 1 to 4 rows on its own (FlushFold), and a w = 4 verify folding
// 1 to 4 rows the previous verify marked (the fold inside the next pass's scan, the registered response to a flagged G2), at
// L = 512 and 4096. The in-scan fold's cost is that verify less a verify with nothing to fold.
func TestSP2_foldCost(t *testing.T) {
	g := newSP2Rig(t)
	text := sp2LongText(t, g.tk, filepath.Join("..", "decoder", "model.go"), 4200)
	fill, err := g.tk.Encode(text, false)
	if err != nil {
		t.Fatal(err)
	}
	sync := func() {
		if err := g.r.do(func() error { return g.r.stream.Sync() }); err != nil {
			t.Fatal(err)
		}
	}
	rows := fill[100:104]
	for _, L := range []int{512, 4096} {
		if len(fill) < L+8 {
			t.Fatalf("fill has %d tokens, want %d", len(fill), L+8)
		}
		g.mc.ForgetResidentForTest()
		if _, err := g.r.PrefillLast(context.Background(), embsOf(g.mc, fill[:L]), 0); err != nil {
			t.Fatalf("fill to %d: %v", L, err)
		}
		cell := g.sp2ValidCell(t, fmt.Sprintf("G2 L=%d", L))
		embs := embsOf(g.mc, rows)
		arms := []string{"step", "verify_fold", "verify_today", "fold1", "fold2", "fold3", "fold4", "vpend1", "vpend2", "vpend3", "vpend4"}
		times := map[string][]float64{}
		run := func(arm string) float64 {
			var t0 time.Time
			switch arm {
			case "step":
				sync()
				t0 = time.Now()
				if _, err := g.r.ForwardArgmax(embs[0], L); err != nil {
					t.Fatal(err)
				}
			case "verify_fold":
				sync()
				t0 = time.Now()
				if _, err := g.r.VerifyLastNArgmax(embs, L); err != nil {
					t.Fatal(err)
				}
			case "verify_today":
				sync()
				t0 = time.Now()
				if _, err := g.r.PrefillLastNArgmax(embs, L); err != nil {
					t.Fatal(err)
				}
			default:
				n := int(arm[len(arm)-1] - '0')
				if _, err := g.r.VerifyLastNArgmax(embs, L); err != nil {
					t.Fatal(err)
				}
				if err := g.r.FoldVerified(n); err != nil {
					t.Fatal(err)
				}
				sync()
				t0 = time.Now()
				if strings.HasPrefix(arm, "fold") {
					if err := g.r.FlushFold(); err != nil {
						t.Fatal(err)
					}
				} else if _, err := g.r.VerifyLastNArgmax(embs, L); err != nil {
					t.Fatal(err)
				}
			}
			sync()
			return float64(time.Since(t0).Microseconds()) / 1000
		}
		for rep := range 23 {
			order := append([]string(nil), arms...)
			k := rep % len(order)
			order = append(order[k:], order[:k]...)
			if rep%2 == 1 {
				for i, j := 0, len(order)-1; i < j; i, j = i+1, j-1 {
					order[i], order[j] = order[j], order[i]
				}
			}
			for _, a := range order {
				ms := run(a)
				if rep >= 3 {
					times[a] = append(times[a], ms)
				}
			}
		}
		rec := map[string]any{"kind": "g2", "L": L, "cell": cell}
		med := map[string]float64{}
		for _, a := range arms {
			q1, q3 := sp2Quartiles(times[a])
			med[a] = sp2Median(times[a])
			rec[a] = map[string]float64{"median_ms": med[a], "q1_ms": q1, "q3_ms": q3}
		}
		fold330 := med["fold3"] + 0.30*(med["fold4"]-med["fold3"])
		inScan330 := med["vpend3"] + 0.30*(med["vpend4"]-med["vpend3"]) - med["verify_fold"]
		rec["fold_at_3.30_ms"] = fold330
		rec["fold_over_step"] = fold330 / med["step"]
		rec["inscan_fold_at_3.30_ms"] = inScan330
		rec["inscan_fold_over_step"] = inScan330 / med["step"]
		rec["verify_fold_over_today"] = med["verify_fold"] / med["verify_today"]
		g.record(t, rec)
		t.Logf("L=%d: step %.3f ms; the fold at 3.30 rows on its own %.3f ms = %.1f%% of a step, inside the next scan %.3f ms = %.1f%%; verify fold/today %.3f",
			L, med["step"], fold330, 100*fold330/med["step"], inScan330, 100*inScan330/med["step"], med["verify_fold"]/med["verify_today"])
		// The registered bands, on the fold as the loop runs it (inside the next pass's scan): under 5.0% passes; 5.0-11.0% passes
		// the comparison, flagged; 11.0% or more fails.
		switch r := inScan330 / med["step"]; {
		case r < 0.05:
			t.Logf("VERDICT G2 L=%d: PASS (%.1f%% of a step, under 5.0%%)", L, 100*r)
		case r < 0.11:
			t.Logf("VERDICT G2 L=%d: PASS, FLAGGED (%.1f%% of a step, inside 5.0-11.0%%)", L, 100*r)
		default:
			t.Errorf("VERDICT G2 L=%d: FAIL (%.1f%% of a step, at or past R-01 phase 1's 11.0%%)", L, 100*r)
		}
	}
}

// TestSP2_speed is G4: plain greedy against BlockSpec at w = 4, whole generations through the public API on one resident, 10 A B
// B A pairs per context over SP0's 12 code and math prompts, decode timed from the first token to the end. Both arms' windows
// hold the same work outside decode: each starts when the prompt's first token is out (BlockSpec has folded the prompt into its
// drafter's context by then) and ends after the last token is forwarded and the turn committed, which plain decode also does
// before its channel closes. The plain arm runs under the exact-attention scope BlockSpec holds, the plain greedy G1 compares
// with, and every prompt's tokens must be equal in the two arms of a pair, so the rates compare the same generations. A pair
// started under heavy load is void and re-run in order (sp2ValidCell).
//
// SP2_SMOKE=1 runs one pair over two prompts per context into SP2_OUT, to prove the harness before the registered run; its
// readings are not G4's.
func TestSP2_speed(t *testing.T) {
	g := newSP2Rig(t)
	const maxNew = 256
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
	// One arm's pass over every prompt: decode tokens and decode seconds, summed, and each prompt's tokens.
	pass := func(prompts [][]int, spec bool) (toks int, secs float64, outs [][]int) {
		for _, ids := range prompts {
			g.mc.ForgetResidentForTest() // every generation prefills cold, in both arms
			var n int
			var first, end time.Time
			if spec {
				out, _, f, err := g.spec.GenerateFirstTokenForTest(ids, decoder.BlockSpecOptions{VerifyWidth: sp2Width(), MaxTokens: maxNew})
				if err != nil {
					t.Fatal(err)
				}
				n, first, end = len(out), f, time.Now()
				outs = append(outs, out)
			} else {
				leave := g.r.EnterExactAttention()
				ch, gen := g.mc.Generate(context.Background(), ids, maxNew, decoder.SamplingParams{})
				var out []int
				for id := range ch {
					if n == 0 {
						first = time.Now()
					}
					n++
					out = append(out, id)
				}
				end = time.Now()
				leave()
				if e := gen.Err(); e != nil {
					t.Fatal(e)
				}
				outs = append(outs, out)
			}
			toks += n - 1
			secs += end.Sub(first).Seconds()
		}
		return toks, secs, outs
	}
	for _, cs := range sets {
		pass(cs.prompts[:2], false) // warm-up, discarded
		pass(cs.prompts[:2], true)
		g.record(t, map[string]any{"kind": "g4-mode", "context": cs.name, "fa_verify": g.r.faVerify,
			"note": "fa_verify true: the multi-row flash-decode lane serves verify rows and the exact-attention scope is a no-op"})
		var ratios []float64
		for i := range pairs {
			cell := g.sp2ValidCell(t, fmt.Sprintf("G4 %s pair %d", cs.name, i))
			var pt, st int
			var ps, ss float64
			var po, so [][]int
			if i%2 == 0 {
				pt, ps, po = pass(cs.prompts, false)
				st, ss, so = pass(cs.prompts, true)
			} else {
				st, ss, so = pass(cs.prompts, true)
				pt, ps, po = pass(cs.prompts, false)
			}
			for k := range po {
				if at := sp2Diverge(so[k], po[k]); at >= 0 {
					t.Errorf("%s pair %d prompt %d: the arms' tokens differ at %d (spec %d, plain %d tokens): G1's equality does not hold here",
						cs.name, i, k, at, len(so[k]), len(po[k]))
					g.record(t, map[string]any{"kind": "g4-diverge", "context": cs.name, "pair": i, "prompt": k, "at": at})
				}
			}
			ratio := (float64(st) / ss) / (float64(pt) / ps)
			ratios = append(ratios, ratio)
			g.record(t, map[string]any{"kind": "g4", "context": cs.name, "pair": i, "cell": cell, "order": map[bool]string{true: "AB", false: "BA"}[i%2 == 0],
				"plain_tokens": pt, "plain_s": ps, "spec_tokens": st, "spec_s": ss, "speedup": ratio})
			t.Logf("%s pair %d: plain %.1f tok/s, spec %.1f tok/s, %.3fx", cs.name, i, float64(pt)/ps, float64(st)/ss, ratio)
		}
		above := 0
		for _, r := range ratios {
			if r > 1 {
				above++
			}
		}
		g.record(t, map[string]any{"kind": "g4-summary", "context": cs.name, "width": sp2Width(), "median": sp2Median(ratios), "pairs_above_1": above})
		t.Logf("%s: median %.3fx, %d of %d pairs above 1", cs.name, sp2Median(ratios), above, pairs)
		if pairs < 10 {
			continue // a smoke run grades nothing
		}
		// The owner's merge rule for a lossless opt-in feature, per context: a median of 1.05x or more with at least 9 of 10 pairs
		// above 1 delivers. SP0's 1.30x is an investment bar, not a merge bar (docs/measurements/sp2-2026-10/README.md, "G4 graded
		// against the owner's merge rule"); a record may report against it.
		if med := sp2Median(ratios); med >= 1.05 && above >= 9 {
			t.Logf("VERDICT G4 %s: DELIVERS by the merge rule (median %.3fx, %d of 10 pairs above 1)", cs.name, med, above)
		} else {
			t.Logf("VERDICT G4 %s: DOES NOT DELIVER by the merge rule (median %.3fx, %d of 10 pairs above 1; the rule is >= 1.05x and >= 9 of 10)", cs.name, med, above)
		}
	}
}

// sp2Encode renders one user message through the checkpoint's template, thinking on or off, and encodes it as the chat routes do
// (the same rendering as SP0's harness).
func sp2Encode(tk *tokenizer.Tokenizer, text, think string) ([]int, error) {
	tm, err := chat.Detect(chat.Meta{ChatTemplate: tk.ChatTemplate(), HasToken: tk.Has})
	if err != nil {
		return nil, err
	}
	switch think {
	case "on":
		tm = tm.WithThinking(chat.ThinkOn)
	case "off":
		tm = tm.WithThinking(chat.ThinkOff)
	}
	return tk.EncodeSegments(tm.RenderSegments("", []chat.Turn{{Role: "user", Content: text}}), false)
}
