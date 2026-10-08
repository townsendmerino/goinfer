//go:build darwin && goinfer_testhooks

package metal

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/chat"
	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/tokenizer"
)

// TestQ4KLane_realPhi3NonInferiority is G-Q2 of docs/tasks/task-metal-q4k-2026-10.md (registered before any code): S1's
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
	rp, ra, rn := g3Model(t, qb+".int4.metal.giw", qb+".gguf", "reference Qwen2.5-Coder-1.5B", logf)
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
	if why := residentMemoryDecline(mg); why != "" {
		t.Fatalf("%s: Metal's fit guard declines it: %s", label, why)
	}
	r, err := buildResident(mg)
	if err != nil {
		t.Fatal(err)
	}
	defer r.Close()
	if !r.q4kLane {
		t.Fatalf("%s: the resident did not take the q4k lane", label)
	}
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
