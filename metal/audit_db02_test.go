//go:build darwin && goinfer_testhooks

package metal

import (
	"context"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/tokenizer"
)

// TestDB02_expertMajorProbe is T1.12, D-B02's probe (docs/audit-metal-2026-09-30.md, "D-B02"): what the host-grouped
// expert-major MoE prefill spends besides expert work, on a fully resident generic MoE (default the Qwen1.5-MoE 4-layer
// slice; GOINFER_DP04_MODEL picks another). For prose prompts of M tokens it reads, per pass, through db02Trace:
//   - the GPU-idle share: wall time minus the GPU time of the pass's command buffers, which is the per-layer routing
//     sync (commit, wait, host grouping, re-encode) that device-side scheduling would remove;
//   - expert dispatches per MoE layer (5 per active expert), the count device-side scheduling would collapse;
//   - row padding: rows the expert GEMMs run (padded to 8, then to each GEMM's row tile) over routed rows.
//
// Exploratory sizing, not a graded result.
//
//	GOINFER_DP04=1 go test -tags goinfer_testhooks -count=1 -run '^TestDB02_expertMajorProbe$' -v ./metal/
func TestDB02_expertMajorProbe(t *testing.T) {
	if os.Getenv("GOINFER_DP04") != "1" {
		t.Skip("set GOINFER_DP04=1 (loads a real MoE checkpoint)")
	}
	path := os.Getenv("GOINFER_DP04_MODEL")
	if path == "" {
		home, _ := os.UserHomeDir()
		path = filepath.Join(home, "models", "qwen15-moe-a27b-l4slice")
	}
	if strings.HasPrefix(path, "/Volumes/") || strings.HasPrefix(path, "/srv/models") {
		t.Fatalf("%s is on the archive, not the bench set (CLAUDE.md)", path)
	}
	if _, err := os.Stat(filepath.Join(path, "config.json")); err != nil {
		t.Skipf("no checkpoint at %s: %v", path, err)
	}
	Ms := []int{64, 512, 2048}
	m, err := decoder.Load(path, decoder.Options{Quant: "int4", ResidentContext: slices.Max(Ms) + 64})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	defer m.Close()
	r, err := buildResident(m)
	if err != nil {
		t.Fatalf("resident: %v", err)
	}
	defer r.Close()
	a := &metalResident{r: r}
	if r.moe == nil || !r.prefillOK {
		t.Fatalf("%s: moe %v, prefillOK %v", path, r.moe != nil, r.prefillOK)
	}
	nMoE := 0
	for l := range r.layers {
		if r.layers[l].moe != nil {
			nMoE++
			if r.layers[l].moe.pool != nil {
				t.Fatalf("layer %d is paged", l)
			}
		}
	}
	tk, err := tokenizer.Load(filepath.Join(path, "tokenizer.json"))
	if err != nil {
		t.Fatalf("tokenizer: %v", err)
	}
	_, files := decoder.PrefillGatePromptSet()
	t0 := time.Now()
	type pass struct {
		wall, busy                        float64 // ms
		group, encode                     float64 // ms: host grouping after each routing sync; host encoding not overlapped by GPU work
		pre, post                         float64 // ms: host time before the first buffer's GPU work, and after the last End
		active, rows, pad8, tileGu, tileD int
	}
	var cur pass
	// Host time, from the callback's own clock: an End returns (tEnd); the first expert event of that layer marks
	// grouping done (tFirst); the next End returns after encoding the experts and the next layer, committing, and
	// waiting for its GPU work, so encoding is that interval less the buffer's GPU time.
	var tEnd, tFirst, tStart time.Time
	db02Trace = func(ev db02Event) {
		now := time.Now()
		if ev.gpuEnd > 0 {
			gpu := (ev.gpuEnd - ev.gpuStart) * 1e3
			if tEnd.IsZero() { // the pass's first End: everything before its GPU work was host time
				cur.pre = now.Sub(tStart).Seconds()*1e3 - gpu
			}
			cur.busy += gpu
			if !tFirst.IsZero() {
				cur.encode += now.Sub(tFirst).Seconds()*1e3 - gpu
			}
			tEnd, tFirst = now, time.Time{}
			return
		}
		if tFirst.IsZero() {
			tFirst = now
			cur.group += now.Sub(tEnd).Seconds() * 1e3
		}
		cur.active++
		cur.rows += ev.rows
		cur.pad8 += ev.padRows
		cur.tileGu += ev.tileRowsGu
		cur.tileD += ev.tileRowsD
	}
	t.Cleanup(func() { db02Trace = nil })
	auditHB("d-b02", t0, "%s: %d MoE layers, %d experts, top %d, shared %d", filepath.Base(path), nMoE, r.moe.nE, r.moe.k, r.moe.sharedInter)
	for _, M := range Ms {
		embs := getEmbs(r, decoder.PrefillGateProseIDsForTest(t, tk, files[0], M)[:M])
		var ps []pass
		for rep := range 4 {
			cur = pass{}
			st := time.Now()
			tStart, tEnd, tFirst = st, time.Time{}, time.Time{}
			if _, err := a.PrefillLast(context.Background(), embs, 0); err != nil {
				t.Fatalf("M=%d: %v", M, err)
			}
			cur.wall = time.Since(st).Seconds() * 1e3
			cur.post = time.Since(tEnd).Seconds() * 1e3
			if rep > 0 { // the first pass warms pipelines and pages
				ps = append(ps, cur)
			}
		}
		n := len(ps)
		wall, busy, idle, group, enc := make([]float64, n), make([]float64, n), make([]float64, n), make([]float64, n), make([]float64, n)
		pre, post := make([]float64, n), make([]float64, n)
		for i, p := range ps {
			wall[i], busy[i], idle[i], group[i], enc[i] = p.wall, p.busy, (p.wall-p.busy)/p.wall, p.group, p.encode
			pre[i], post[i] = p.pre, p.post
		}
		p := ps[0]
		perLayer := float64(p.active) / float64(nMoE)
		auditHB("d-b02", t0, "M=%d: wall %.2f ms, GPU busy %.2f ms, GPU idle %.1f%% (medians of %d); per MoE layer %.1f active experts = %.0f expert dispatches",
			M, auditMedian(wall), auditMedian(busy), 100*auditMedian(idle), len(ps), perLayer, 5*perLayer)
		auditHB("d-b02", t0, "M=%d: of the idle, host grouping %.2f ms and unoverlapped encoding after the routing syncs %.2f ms; before the first buffer's GPU work %.2f ms, after the last End %.2f ms (medians)",
			M, auditMedian(group), auditMedian(enc), auditMedian(pre), auditMedian(post))
		auditHB("d-b02", t0, "M=%d: routed rows %d; GEMM rows padded to 8: %.3fx; to the row tile: gate|up %.3fx, down %.3fx",
			M, p.rows, float64(p.pad8)/float64(p.rows), float64(p.tileGu)/float64(p.rows), float64(p.tileD)/float64(p.rows))
	}
}
