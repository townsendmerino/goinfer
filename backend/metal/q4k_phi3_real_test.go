//go:build darwin && goinfer_testhooks

package metal

import (
	"compress/gzip"
	"context"
	"crypto/sha256"
	"encoding/gob"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/chat"
	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/internal/fidelity"
	"github.com/townsendmerino/goinfer/tokenizer"
)

// TestQ4KLane_realPhi3NonInferiority is G-Q2: S1's
// G3 procedure (g3Run: the eight G3 prompts, 32 greedy tokens free-running on each side, then the CPU's sequence
// teacher-forced through Metal), on Phi-3 mini 4k at --quant q4k, Metal's q4k lane against the CPU at q4k on the same
// file, with the validated Qwen2.5-Coder-1.5B (its int4 Metal sidecar, as G3 runs it) as the reference in this process.
// PASS: agreement >= the reference's - 2.0 points and free-run passes >= the reference's - 1; 2.0-4.0 points below is
// ambiguous (parked); worse, or free-run passes 2+ short, fails. The decode path must read metal-resident (q4k).
//
//	GOINFER_HEAVY_TESTS=1 go test -count=1 -timeout 30m -tags goinfer_testhooks -run '^TestQ4KLane_realPhi3NonInferiority$' -v ./metal/
//
// The Phi-3 file is the registered asset GOINFER_PHI3_GGUF (testdata/assets.json).
func TestQ4KLane_realPhi3NonInferiority(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") != "1" {
		t.Skip("heavy-checkpoint test: set GOINFER_HEAVY_TESTS=1")
	}
	home, _ := os.UserHomeDir()
	t0 := time.Now()
	logf := func(format string, a ...any) {
		fmt.Fprintf(os.Stderr, "[G-Q2 %6.1fs] %s\n", time.Since(t0).Seconds(), fmt.Sprintf(format, a...))
	}
	phi3 := decoder.AssetPathForTest(t, "GOINFER_PHI3_GGUF") // the asset registry (testdata/assets.json) resolves it
	qb := filepath.Join(home, "models", "qwen2.5-coder-1.5b-instruct-q4_k_m")
	rp, ra, rn := g3Model(t, fmt.Sprintf("%s.int4.metal.v%d.giw", qb, decoder.GIWFormatVersion), qb+".gguf", "reference Qwen2.5-Coder-1.5B", logf)
	pp, pa, pn := g3ModelQ4K(t, phi3, "Phi-3 mini q4k", false, logf)
	refPct, pct := 100*float64(ra)/float64(rn), 100*float64(pa)/float64(pn)
	delta := pct - refPct
	logf("G-Q2 non-inferiority: Phi-3 %.2f%% (%d/%d prompts) vs reference %.2f%% (%d/%d prompts): delta %+.2f points", pct, pp, len(g3Prompts), refPct, rp, len(g3Prompts), delta)
	switch {
	case delta < -4.0 || pp <= rp-2:
		t.Errorf("G-Q2 FAIL: delta %+.2f points, free-run %d vs reference %d", delta, pp, rp)
	case delta < -2.0:
		t.Errorf("G-Q2 AMBIGUOUS (parked for the owner): delta %+.2f points", delta)
	default:
		t.Logf("G-Q2 PASS: delta %+.2f points (margin -2.0), free-run %d vs reference %d", delta, pp, rp)
	}
}

// TestQ4KLane_g3OneLoadMatchesTwo is G-Q2's procedure check, by day on Qwen2.5-Coder-0.5B q4_k_m: g3ModelQ4K with one
// load (the model that builds the resident is also the CPU reference, which is how Phi-3 fits the 16 GB Mac) must give
// exactly the counts of two separate loads, as g3Model does. If building the resident changed the CPU forward of the
// model it was built from, the two would differ.
//
//	GOINFER_HEAVY_TESTS=1 go test -count=1 -tags goinfer_testhooks -run '^TestQ4KLane_g3OneLoadMatchesTwo$' -v ./metal/
func TestQ4KLane_g3OneLoadMatchesTwo(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") != "1" {
		t.Skip("heavy-checkpoint test: set GOINFER_HEAVY_TESTS=1")
	}
	home, _ := os.UserHomeDir()
	t0 := time.Now()
	logf := func(format string, a ...any) {
		fmt.Fprintf(os.Stderr, "[G-Q2 check %6.1fs] %s\n", time.Since(t0).Seconds(), fmt.Sprintf(format, a...))
	}
	gguf := filepath.Join(home, "models", "qwen2.5-coder-0.5b-instruct-q4_k_m.gguf")
	p1, a1, n1 := g3ModelQ4K(t, gguf, "0.5B q4k, one load", false, logf)
	p2, a2, n2 := g3ModelQ4K(t, gguf, "0.5B q4k, two loads", true, logf)
	if p1 != p2 || a1 != a2 || n1 != n2 {
		t.Errorf("one load %d prompts, %d/%d; two loads %d prompts, %d/%d", p1, a1, n1, p2, a2, n2)
	}
}

// g3ModelQ4K is g3Model for a GGUF at --quant q4k, which has no Metal sidecar: it loads the file at G3's pinned context,
// builds Metal's resident from it (which must take the q4k lane), and runs g3Run with the CPU reference either on the same
// model (twoLoads false) or on a second load of the file.
func g3ModelQ4K(t *testing.T, gguf, label string, twoLoads bool, logf func(string, ...any)) (pass, agree, n int) {
	t.Helper()
	if strings.HasPrefix(gguf, "/Volumes/") || strings.HasPrefix(gguf, "/srv/models") {
		t.Fatalf("%s is on the archive, not the bench set (CLAUDE.md)", gguf)
	}
	if _, err := os.Stat(gguf); err != nil {
		t.Skipf("no model %s: %v", gguf, err)
	}
	tk, err := tokenizer.LoadGGUF(gguf)
	if err != nil {
		t.Fatal(err)
	}
	tmpl, err := chat.Detect(chat.Meta{ChatTemplate: tk.ChatTemplate(), HasToken: tk.Has})
	if err != nil {
		t.Fatal(err)
	}
	opts := decoder.Options{Quant: "q4k", ResidentContext: 512}
	mg, err := decoder.Load(gguf, opts)
	if err != nil {
		t.Fatal(err)
	}
	defer mg.Close()
	if why := residentMemoryDecline(mg, metalCtxPlan{}); why != "" {
		t.Fatalf("%s: Metal's fit guard declines it: %s", label, why)
	}
	r, err := buildResident(mg, metalCtxPlan{})
	if err != nil {
		t.Fatal(err)
	}
	defer r.Close()
	if !r.q4kLane {
		t.Fatalf("%s: the resident did not take the q4k lane", label)
	}
	tot, byKind := kso1LaneBytes(r)
	logf("%s: q4k lane weights %d bytes on the device, by kind %v", label, tot, byKind)
	mc := mg
	if twoLoads {
		if mc, err = decoder.Load(gguf, opts); err != nil {
			t.Fatal(err)
		}
		defer mc.Close()
	}
	logf("%s: loaded (template %s, the q4k lane)", label, tmpl.Name())
	pass, agree, n = g3Run(t, tk, tmpl, r, mg, mc, logf)
	logf("%s: %d/%d prompts pass the free-run rule; teacher-forced agreement %d/%d = %.2f%%", label, pass, len(g3Prompts), agree, n, 100*float64(agree)/float64(n))
	return pass, agree, n
}

// op6mpPhi3 loads the registered Phi-3 asset (GOINFER_PHI3_GGUF) at --quant q4k for the CPU and builds Metal's resident from it
// with ctx positions, as g3ModelQ4K does: one load is both the CPU reference and the q4k lane, which is how Phi-3 fits a 16 GB
// Mac. A load a guard declines is a SKIP naming the guard, never bypassed.
func op6mpPhi3(t *testing.T, ctx int) (*decoder.Model, *metalResident, string) {
	t.Helper()
	path := decoder.AssetPathForTest(t, "GOINFER_PHI3_GGUF")
	if strings.HasPrefix(path, "/Volumes/") || strings.HasPrefix(path, "/srv/models") {
		t.Fatalf("%s is on the archive, not the bench set (CLAUDE.md)", path)
	}
	m, err := decoder.Load(path, decoder.Options{Quant: "q4k", ResidentContext: ctx, ResidentKVSlots: 1})
	if err != nil && strings.Contains(err.Error(), "NOT loaded") {
		t.Skipf("the fit guard declines this load now (never bypassed here): %v", err)
	}
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	t.Cleanup(func() { m.Close() })
	if why := residentMemoryDecline(m, metalCtxPlan{}); why != "" {
		t.Skipf("Metal's memory guard declines this load now: %s", why)
	}
	r, err := buildResident(m, metalCtxPlan{})
	if err != nil {
		t.Fatalf("build resident: %v", err)
	}
	t.Cleanup(func() { r.Close() })
	a := &metalResident{r: r, hidden: r.H}
	if !r.q4kLane || !r.prefillOK || !a.stepPrefill() {
		t.Fatalf("not the q4k lane with the pass and the step (lane %v, pass %v, step %v)", r.q4kLane, r.prefillOK, a.stepPrefill())
	}
	return m, a, path
}

// TestOP6MP_q4kPassSpeed times the q4k lane's MMA pass against the route a Phi-3 prompt took before the lane was admitted to it:
// one load of Phi-3 mini 4k q4, the step (prefillByStep, decode rows batchMaxSeqs at a time) against PrefillLast (which must run
// the pass: its attention launches are counted once, untimed), at P = 512 and 3900 tokens from position 0. Per P, one untimed
// call of each, then 5 pairs in alternating order (A B, B A, ...). Each arm is bracketed by voidWindow; a void pair is re-run, at
// most 4 a P, and listed. It prints every pair and the median step ÷ pass per P; the registration grades them. Under the
// exclusive lock, about 25 minutes (the step's 3900-token call is about 80 s):
//
//	GOINFER_OP6MP_SPEED=1 go test -count=1 -tags goinfer_testhooks -timeout 60m -run '^TestOP6MP_q4kPassSpeed$' -v ./backend/metal/
func TestOP6MP_q4kPassSpeed(t *testing.T) {
	if os.Getenv("GOINFER_OP6MP_SPEED") == "" {
		t.Skip("set GOINFER_OP6MP_SPEED=1 (Phi-3 mini, about 25 minutes, under the exclusive timing lock)")
	}
	m, a, path := op6mpPhi3(t, 4096)
	ctx := context.Background()
	_, _, _, _, _, _, vocab := m.Dims()
	t0 := time.Now()
	for _, P := range []int{512, 3900} {
		embs := make([][]float32, P)
		for i := range embs {
			embs[i] = m.EmbedResidentForTest((i*2654435761 + 1) % (vocab - 1))
		}
		step := func() time.Duration {
			s := time.Now()
			if _, err := a.prefillByStep(ctx, embs, 0); err != nil {
				t.Fatalf("P %d step: %v", P, err)
			}
			return time.Since(s)
		}
		pass := func() time.Duration {
			s := time.Now()
			if _, err := a.PrefillLast(ctx, embs, 0); err != nil {
				t.Fatalf("P %d pass: %v", P, err)
			}
			return time.Since(s)
		}
		prefillAttnRanForTest = map[string]int{}
		pass()
		ran := prefillAttnRanForTest
		prefillAttnRanForTest = nil
		if len(ran) == 0 {
			t.Fatalf("P %d: PrefillLast did not run the pass", P)
		}
		step()
		var ratios []float64
		voids := 0
		for p := 0; len(ratios) < 5; p++ {
			var ds, dp time.Duration
			var oks, okp bool
			var ns, np string
			arm := func(f func() time.Duration) (time.Duration, bool, string) {
				end := voidWindow(t)
				d := f()
				ok, note := end()
				return d, ok, note
			}
			if len(ratios)%2 == 0 {
				ds, oks, ns = arm(step)
				dp, okp, np = arm(pass)
			} else {
				dp, okp, np = arm(pass)
				ds, oks, ns = arm(step)
			}
			if !oks || !okp {
				voids++
				fmt.Fprintf(os.Stderr, "[op6mp-speed %5.0fs] P %d pair %d VOID (step: %s; pass: %s)\n", time.Since(t0).Seconds(), P, p+1, ns, np)
				if voids > 4 {
					t.Fatalf("P %d: %d void pairs; the box is too busy for this cell", P, voids)
				}
				continue
			}
			r := ds.Seconds() / dp.Seconds()
			ratios = append(ratios, r)
			fmt.Fprintf(os.Stderr, "[op6mp-speed %5.0fs] P %d pair %d (%s first): step %.2f s, pass %.2f s, step/pass %.3f (step: %s; pass: %s)\n",
				time.Since(t0).Seconds(), P, len(ratios), map[bool]string{true: "step", false: "pass"}[len(ratios)%2 == 1], ds.Seconds(), dp.Seconds(), r, ns, np)
		}
		sorted := slices.Clone(ratios)
		slices.Sort(sorted)
		above := 0
		for _, r := range ratios {
			if r > 1 {
				above++
			}
		}
		fmt.Fprintf(os.Stderr, "[op6mp-speed] %s P %d: median step/pass %.3f over %d pairs, %d above 1, %d void; pairs %v\n",
			filepath.Base(path), P, sorted[len(sorted)/2], len(ratios), above, voids, ratios)
	}
}

// TestOP6MP_q4kPassFidelity is the q4k pass's fidelity gate on Phi-3 mini 4k q4: the pass (candidate, PrefillLast) and the step
// (exact, prefillByStep, bit-identical to the sequential path) are each scored against the CPU at q4k on the same load (the
// lane's own quant, so the arms differ from it by arithmetic only), over the prefill gate's prose prompts at K = 512 and 2048: the
// seed position from the prefill and 31 positions teacher-forced on the CPU's greedy tokens. fidelity.NonInferior with
// AgreeMarginPts 1, KLMargin 0.10, alpha 0.05. Any prompt the pass declines (its f16 overflow check) fails the gate. Not timed
// (the shared lock).
//
// The CPU reference is nearly all of the wall time (about 3.5 minutes a prompt at K = 512 and 16 at 2048 on an M1 Pro), so
// $OP6MP_REF_CACHE names a directory that keeps each prompt's reference, keyed by the checkpoint and the prompt's tokens, and a
// re-run reuses it. $OP6MP_FIDELITY_K (comma-separated) runs a subset of the cells and labels the verdict PARTIAL; each cell's
// own summary prints as it finishes, and $OP6MP_POSITIONS_LOG is rewritten after each cell. The deadline is the test's -timeout.
//
//	GOINFER_OP6MP_FIDELITY=1 go test -count=1 -tags goinfer_testhooks -timeout 240m -run '^TestOP6MP_q4kPassFidelity$' -v ./backend/metal/
func TestOP6MP_q4kPassFidelity(t *testing.T) {
	if os.Getenv("GOINFER_OP6MP_FIDELITY") == "" {
		t.Skip("set GOINFER_OP6MP_FIDELITY=1 (Phi-3 mini on Metal and the CPU, about 3 hours cold on an M1 Pro)")
	}
	const cont = 32
	t.Setenv("GOINFER_CPU_FAST_ATTENTION", "0") // the CPU reference's exact attention (PrefillLogitsForTest); knobs are read at Load
	m, a, path := op6mpPhi3(t, 2048+cont+8)
	tk, err := tokenizer.LoadGGUF(path)
	if err != nil {
		t.Fatal(err)
	}
	ctx := context.Background()
	if d, ok := t.Deadline(); ok {
		var cancel context.CancelFunc
		ctx, cancel = context.WithDeadline(ctx, d.Add(-2*time.Minute))
		defer cancel()
	}
	st, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	cacheDir := os.Getenv("OP6MP_REF_CACHE")
	type refEntry struct {
		Logits [][]float32
		Toks   []int
	}
	// ref is the CPU's seed logits and cont-1 greedy steps after it: logits for every scored position, and the tokens the arms
	// are teacher-forced on.
	ref := func(ids []int) ([][]float32, []int) {
		h := sha256.New()
		fmt.Fprintf(h, "%s %d %d %v", filepath.Base(path), st.Size(), cont, ids)
		file := filepath.Join(cacheDir, fmt.Sprintf("%x.gob.gz", h.Sum(nil)[:12]))
		if cacheDir != "" {
			if f, err := os.Open(file); err == nil {
				defer f.Close()
				zr, err := gzip.NewReader(f)
				if err != nil {
					t.Fatalf("reference cache %s: %v", file, err)
				}
				var e refEntry
				if err := gob.NewDecoder(zr).Decode(&e); err != nil {
					t.Fatalf("reference cache %s: %v", file, err)
				}
				return e.Logits, e.Toks
			}
		}
		c := m.NewCache(len(ids) + cont)
		lg, err := m.PrefillLogitsForTest(ctx, ids, c)
		if err != nil {
			t.Fatalf("CPU prefill: %v", err)
		}
		e := refEntry{Logits: [][]float32{slices.Clone(lg)}}
		for range cont - 1 {
			e.Toks = append(e.Toks, argmaxF(e.Logits[len(e.Logits)-1]))
			if lg, err = m.ForwardForTest(e.Toks[len(e.Toks)-1], c); err != nil {
				t.Fatal(err)
			}
			e.Logits = append(e.Logits, slices.Clone(lg))
		}
		if cacheDir != "" {
			if err := os.MkdirAll(cacheDir, 0o755); err != nil {
				t.Fatal(err)
			}
			f, err := os.Create(file + ".part")
			if err != nil {
				t.Fatal(err)
			}
			zw := gzip.NewWriter(f)
			if err := gob.NewEncoder(zw).Encode(e); err != nil {
				t.Fatal(err)
			}
			if err := zw.Close(); err != nil {
				t.Fatal(err)
			}
			if err := f.Close(); err != nil {
				t.Fatal(err)
			}
			if err := os.Rename(file+".part", file); err != nil {
				t.Fatal(err)
			}
		}
		return e.Logits, e.Toks
	}
	arm := func(ids, toks []int, step bool) [][]float32 {
		embs := make([][]float32, len(ids))
		for i, id := range ids {
			embs[i] = m.EmbedResidentForTest(id)
		}
		a.Reset()
		var seed []float32
		var err error
		if step {
			seed, err = a.prefillByStep(ctx, embs, 0)
		} else {
			seed, err = a.PrefillLast(ctx, embs, 0)
		}
		if err != nil {
			t.Fatalf("prefill (step %v): %v", step, err)
		}
		logits := [][]float32{slices.Clone(seed)}
		for j, tok := range toks {
			lg, err := a.Forward(m.EmbedResidentForTest(tok), len(ids)+j)
			if err != nil {
				t.Fatal(err)
			}
			logits = append(logits, slices.Clone(lg))
		}
		return logits
	}
	score := func(ref, got [][]float32) []fidelity.Position {
		var ps []fidelity.Position
		for i := range got {
			agree, _, hard := decoder.NearTieArgmaxForTest(ref[i], got[i])
			ps = append(ps, fidelity.Position{Agree: argmaxF(ref[i]) == argmaxF(got[i]), HardFlip: !agree && hard, KL: decoder.KLDivergenceForTest(ref[i], got[i])})
		}
		return ps
	}
	_, files := decoder.PrefillGatePromptSet()
	Ks, partial := []int{512, 2048}, false
	if v := os.Getenv("OP6MP_FIDELITY_K"); v != "" {
		Ks, partial = nil, true
		for f := range strings.SplitSeq(v, ",") {
			k, err := strconv.Atoi(strings.TrimSpace(f))
			if err != nil || k <= 0 || k > 2048 {
				t.Fatalf("OP6MP_FIDELITY_K %q: each K is an integer in 1..2048", v)
			}
			Ks = append(Ks, k)
		}
	}
	crit := fidelity.Criteria{AgreeMarginPts: 1, KLMargin: 0.10, Alpha: 0.05}
	writePositions := func(cells []fidelity.Cell) {
		if lp := os.Getenv("OP6MP_POSITIONS_LOG"); lp != "" {
			if f, err := os.Create(lp); err == nil {
				_ = fidelity.WritePositions(f, cells)
				f.Close()
			}
		}
	}
	var cells []fidelity.Cell
	var allIDs [][]int
	start := time.Now()
	for _, K := range Ks {
		cell := fidelity.Cell{Name: fmt.Sprintf("phi3-q4k/K=%d", K)}
		for fi, f := range files {
			ids := decoder.PrefillGateProseIDsForTest(t, tk, f, K)[:K]
			allIDs = append(allIDs, ids)
			rl, toks := ref(ids)
			p := fidelity.Prompt{ID: fmt.Sprintf("%s/%d", cell.Name, fi), Candidate: score(rl, arm(ids, toks, false)), Exact: score(rl, arm(ids, toks, true))}
			cell.Prompts = append(cell.Prompts, p)
			fmt.Fprintf(os.Stderr, "[op6mp-fidelity %5.0fs] K=%d prompt %d/%d done\n", time.Since(start).Seconds(), K, fi+1, len(files))
		}
		cells = append(cells, cell)
		writePositions(cells)
		if cv, err := fidelity.NonInferior([]fidelity.Cell{cell}, crit); err == nil {
			p := cv.Pooled
			fmt.Fprintf(os.Stderr, "[op6mp-fidelity %5.0fs] cell %s alone (reported, not the verdict): agree pass %.2f%% / step %.2f%%, diff %+.2f pts (LCB %+.2f), KL ratio %.3f (UCB %.3f), hard flips %d / %d\n",
				time.Since(start).Seconds(), cell.Name, p.AgreeCandPct, p.AgreeExactPct, p.AgreeDiffPts, p.AgreeDiffLCB, p.KLRatio, p.KLRatioUCB, p.FlipsCand, p.FlipsExact)
		}
	}
	v, err := fidelity.NonInferior(cells, crit)
	if err != nil {
		t.Fatal(err)
	}
	for _, c := range cells {
		s := v.PerCell[c.Name]
		fmt.Fprintf(os.Stderr, "[op6mp-fidelity] %s: agree pass %.2f%% / step %.2f%%, mean KL pass %.4g / step %.4g, hard flips %d / %d\n",
			c.Name, s.AgreeCandPct, s.AgreeExactPct, s.MeanKLCand, s.MeanKLExact, s.FlipsCand, s.FlipsExact)
	}
	p := v.Pooled
	label := "the registered cells"
	if partial {
		label = fmt.Sprintf("PARTIAL, cells K=%v only (OP6MP_FIDELITY_K)", Ks)
	}
	fmt.Fprintf(os.Stderr, "[op6mp-fidelity] %s: pooled over %d prompts, %d positions (prompt set %s): agreement diff %+.2f pts (LCB %+.2f), KL ratio %.3f (UCB %.3f), hard flips %d vs %d -> pass %v %v\n",
		label, p.Prompts, p.Positions, fidelity.PromptSetHash(allIDs), p.AgreeDiffPts, p.AgreeDiffLCB, p.KLRatio, p.KLRatioUCB, p.FlipsCand, p.FlipsExact, v.Pass, v.Reasons)
	if !v.Pass {
		t.Errorf("OP6-MP fidelity FAIL (%s): %v", label, v.Reasons)
	}
}
