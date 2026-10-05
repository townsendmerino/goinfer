//go:build darwin && goinfer_testhooks

package metal

import (
	"math"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestDB04R_expertRowsAB is D-B04's routed-expert half's speed read: the routed-expert GEMVs one row per simdgroup
// against R18's rows form (moeExpertRows), swapped on one resident, every token of a prompt decoded in each arm, arms
// alternated rep by rep, logits compared bit for bit first. A resident generic MoE (the Qwen1.5-MoE slice, the default)
// is timed by GPU time a token; a paged one (M26 with GOINFER_DB04R_SLOTS) by its GPU-busy time summed over the token's
// command buffers (the pager's profile), since its wall time is dominated by expert staging from disk. Exploratory, by day.
//
//	GOINFER_DB04R=1 [GOINFER_DB04R_MODEL=<path>] [GOINFER_DB04R_SLOTS=24] go test -tags goinfer_testhooks -count=1 \
//	  -run '^TestDB04R_expertRowsAB$' -v ./metal/
func TestDB04R_expertRowsAB(t *testing.T) {
	if os.Getenv("GOINFER_DB04R") != "1" {
		t.Skip("set GOINFER_DB04R=1 (loads a real MoE checkpoint)")
	}
	path := os.Getenv("GOINFER_DB04R_MODEL")
	if path == "" {
		home, _ := os.UserHomeDir()
		path = filepath.Join(home, "models", "qwen15-moe-a27b-l4slice")
	}
	if strings.HasPrefix(path, "/Volumes/") || strings.HasPrefix(path, "/srv/models") {
		t.Fatalf("%s is on the archive, not the bench set (CLAUDE.md)", path)
	}
	if _, err := os.Stat(path); err != nil {
		t.Skipf("no checkpoint at %s: %v", path, err)
	}
	const tokens = 32
	opts := decoder.Options{Quant: "int4", Backend: "metal", ResidentContext: tokens + 64}
	if s := os.Getenv("GOINFER_DB04R_SLOTS"); s != "" {
		opts.MoECacheExperts = true
		for _, c := range s {
			opts.MoECacheSlots = opts.MoECacheSlots*10 + int(c-'0')
		}
	}
	m, err := decoder.Load(path, opts)
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	defer m.Close()
	a, ok := m.ResidentForwardForTest().(*metalResident)
	if !ok {
		t.Fatalf("not metal-resident: %s", m.ResidentDecline())
	}
	r := a.r
	t0 := time.Now()
	one := func() (gu, down Pipeline) {
		runtime.LockOSThread()
		defer runtime.UnlockOSThread()
		pool := NewARPool()
		defer pool.Drain()
		lib, err := r.d.CompileLibrary(allKernels, MSL3_1)
		if err != nil {
			t.Fatalf("compile: %v", err)
		}
		gu, err1 := r.d.NewComputePipeline(lib, "gemv_w4a8_moe")
		down, err2 := r.d.NewComputePipeline(lib, "gemv_w4a8_moe_wacc")
		if err1 != nil || err2 != nil {
			t.Fatalf("pipelines: %v %v", err1, err2)
		}
		return gu, down
	}
	oneGU, oneDown := one()
	type arm struct {
		gu, down   Pipeline
		guR, downR int
	}
	var rows arm
	paged := false
	var set func(x arm)
	switch {
	case r.g4moe != nil:
		g := r.g4moe
		rows, paged = arm{g.pGU, g.pDownWacc, g.guR, g.downR}, g.paged
		set = func(x arm) { g.pGU, g.pDownWacc, g.guR, g.downR = x.gu, x.down, x.guR, x.downR }
	case r.moe != nil && !r.moe.isGptOss && !r.moe.w8:
		mo := r.moe
		rows, paged = arm{mo.pGU, mo.pDownWacc, mo.guR, mo.downR}, mo.paged
		set = func(x arm) { mo.pGU, mo.pDownWacc, mo.guR, mo.downR = x.gu, x.down, x.guR, x.downR }
	default:
		t.Fatalf("%s: no int4 routed-expert MoE resident", path)
	}
	if rows.guR == 0 || rows.downR == 0 {
		t.Fatalf("the rows form is not engaged on this model (gate|up R %d, down R %d)", rows.guR, rows.downR)
	}
	arms := map[string]arm{"one": {oneGU, oneDown, 0, 0}, "rows": rows}
	defer set(rows)
	embs := auditEmbs(r, tokens, 31)
	logits := map[string][][]float32{}
	for _, name := range []string{"one", "rows"} {
		set(arms[name])
		r.stopExec()
		a.Reset()
		for i, e := range embs {
			logits[name] = append(logits[name], append([]float32(nil), r.ForwardEmb(e, i)...))
		}
	}
	for i := range embs {
		for j, x := range logits["one"][i] {
			if math.Float32bits(x) != math.Float32bits(logits["rows"][i][j]) {
				t.Fatalf("token %d logit %d differs between the one-row and rows expert GEMVs", i, j)
			}
		}
	}
	auditHB("d-b04r", t0, "%s (paged %v, gate|up R %d, down R %d): %d tokens bit-identical between the arms", path, paged, rows.guR, rows.downR, tokens)
	reps := auditReps(7)
	ms := map[string][]float64{}
	for rep := range reps {
		order := []string{"one", "rows"}
		if rep%2 == 1 {
			order = []string{"rows", "one"}
		}
		for _, name := range order {
			set(arms[name])
			r.stopExec()
			a.Reset()
			var ks []float64
			busy := func() int64 { p := r.PagedProfile(); return p.p1GpuNanos + p.p2GpuNanos + p.denseGpuNanos }
			for i, e := range embs {
				b0 := busy()
				r.ForwardEmb(e, i)
				if paged { // the token's GPU-busy time over its command buffers: staging and round trips are not the kernels'
					ks = append(ks, float64(busy()-b0)/1e6)
				} else {
					ks = append(ks, (r.gpuEnd-r.gpuStart)*1e3)
				}
			}
			ms[name] = append(ms[name], auditMedian(ks))
		}
	}
	ratio := make([]float64, reps)
	above := 0
	for i := range reps {
		ratio[i] = ms["one"][i] / ms["rows"][i]
		if ratio[i] > 1 {
			above++
		}
	}
	auditHB("d-b04r", t0, "token ms (%s), per rep: one-row %s; rows %s", map[bool]string{false: "GPU", true: "GPU-busy"}[paged], auditFmt3(ms["one"]), auditFmt3(ms["rows"]))
	auditHB("d-b04r", t0, "RESULT %s: one-row / rows median %.3f, %d of %d reps above 1 (per rep %s)", path, auditMedian(ratio), above, reps, auditFmt3(ratio))
}
