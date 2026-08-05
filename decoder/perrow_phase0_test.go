package decoder

import (
	"fmt"
	"math"
	"os"
	"sort"
	"testing"

	"github.com/townsendmerino/aikit/linalg"
)

// TestPerRowScalePhase0 is §7's cheap gate: measure how much PER-ROW int4 scales (one scale per output
// row — the granularity that makes IMMA int32 accumulation associative, so a tensor-core GEMM is
// bit-identical by construction) cost in weight-reconstruction quality versus the shipped PER-GROUP
// scales (one f32 scale per 32 input features, which forces a float accumulate every 8 values and puts
// tensor cores out of reach). No kernel, no forward: it fake-quantizes every real projection matrix both
// ways with the production "sym" scheme (aikit's runtime int4) and reports the relative Frobenius error
// ||W−Wq||/||W|| per tensor-type and overall, plus the per-row/per-group RATIO.
//
// Decision rule (the doc's framing): if per-row costs ~nothing over per-group, the §7 fork collapses to
// a cheap format change (no rotation machinery). If per-row is materially worse, the extra error is what
// rotation must buy back before the tensor-core 3× is reachable — i.e. the expensive path is real.
//
// A Q8_0 GGUF is the f32 proxy: Q8 dequant is ~lossless (0.99995), so its weights carry the real outlier
// distribution (the 5th representativeness axis) that uniform fixtures do not.
//
//	GOINFER_PERROW_PHASE0=1 go test -run TestPerRowScalePhase0 -v ./decoder/
func TestPerRowScalePhase0(t *testing.T) {
	if os.Getenv("GOINFER_PERROW_PHASE0") == "" {
		t.Skip("§7 Phase 0 (loads a real model at f32); set GOINFER_PERROW_PHASE0=1")
	}
	path := os.ExpandEnv("$HOME/models/qwen3-1.7b-q8_0.gguf")
	if _, err := os.Stat(path); err != nil {
		t.Skipf("no fixture at %s", path)
	}
	m, err := Load(path, Options{Backend: "cpu"}) // no Quant ⇒ f32 weights
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	w := m.Weights()

	// relErr = ||W − quantDequant(W)|| / ||W|| (Frobenius), for the given scheme at the given group.
	relErr := func(scheme string, f32 []float32, rows, cols, group int) float64 {
		recon := fakeQuantInt4(scheme, f32, rows, cols, group)
		var num, den float64
		for i, v := range f32 {
			d := float64(v - recon[i])
			num += d * d
			den += float64(v) * float64(v)
		}
		if den == 0 {
			return 0
		}
		return math.Sqrt(num / den)
	}

	type acc struct {
		grp, row, rowMSE float64
		n                int
	}
	byType := map[string]*acc{}
	add := func(name string, wm *linalg.WeightMat) {
		f32, ok := wm.F32()
		if !ok || len(f32) == 0 {
			return
		}
		rows, cols := wm.Rows(), wm.Cols()
		if rows == 0 || cols == 0 || cols < int4GroupSize {
			return
		}
		g := relErr("sym", f32, rows, cols, int4GroupSize) // per-group (32), sym — shipped
		r := relErr("sym", f32, rows, cols, cols)          // per-row, sym maxabs scale — the naive §7 fork
		rm := relErr("symmse", f32, rows, cols, cols)      // per-row, MSE-optimal scale — cheap mitigation (no rotation)
		a := byType[name]
		if a == nil {
			a = &acc{}
			byType[name] = a
		}
		a.grp += g
		a.row += r
		a.rowMSE += rm
		a.n++
	}

	for li := range w.Layers {
		L := &w.Layers[li]
		add("q", &L.QProj)
		add("k", &L.KProj)
		add("v", &L.VProj)
		add("o", &L.OProj)
		add("gate", &L.GateProj)
		add("up", &L.UpProj)
		add("down", &L.DownProj)
	}
	if len(byType) == 0 {
		t.Skip("no f32 projection weights exposed (this GGUF loaded pre-quantized); need an f32/Q8 source")
	}

	names := make([]string, 0, len(byType))
	for n := range byType {
		names = append(names, n)
	}
	sort.Strings(names)

	t.Logf("§7 Phase 0 — sym int4 weight reconstruction, per-row vs per-group (%s)", path)
	t.Logf("%-6s %-4s %-11s %-13s %-15s", "type", "n", "grp/sym", "row/sym (×)", "row/symmse (×)")
	var totG, totR, totRM float64
	var totN int
	for _, n := range names {
		a := byType[n]
		g, r, rm := a.grp/float64(a.n), a.row/float64(a.n), a.rowMSE/float64(a.n)
		t.Logf("%-6s %-4d %-11.5f %-13s %-15s", n, a.n, g,
			fmtRatio(r, g), fmtRatio(rm, g))
		totG += a.grp
		totR += a.row
		totRM += a.rowMSE
		totN += a.n
	}
	mg, mr, mrm := totG/float64(totN), totR/float64(totN), totRM/float64(totN)
	t.Logf("%-6s %-4d %-11.5f %-13s %-15s", "ALL", totN, mg, fmtRatio(mr, mg), fmtRatio(mrm, mg))
	t.Logf("VERDICT: per-row int4 costs %.2f× (sym maxabs) / %.2f× (symmse, MSE-optimal scale) the per-group error, no rotation.", mr/mg, mrm/mg)
	t.Logf("  ~1.1× ⇒ fork collapses to a cheap format change; ≫1 ⇒ rotation (or a forward-quality Phase 0b) must justify the gap.")
}

// fmtRatio renders an error value and its ×-ratio to the per-group baseline.
func fmtRatio(v, base float64) string {
	if base == 0 {
		return "n/a"
	}
	return fmt.Sprintf("%.5f (%.2f×)", v, v/base)
}
