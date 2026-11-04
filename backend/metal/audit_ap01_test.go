//go:build darwin

package metal

import (
	"context"
	"fmt"
	"math"
	"testing"
	"time"
)

// A-P01's night grade (docs/tasks/task-metal-audit-2026-10.md, "A-P01: built, pending its grade"): the GEMM tile
// selector's policies against each other, timed. Gated like the Batch A probes (GOINFER_METAL_AUDIT_A=1, a timing),
// GOINFER_AUDIT_MODEL picks the checkpoint and GOINFER_AUDIT_REPS the repetitions.

var ap01Policies = []string{"shipped", "bm32", "bn32", ""} // "" is the default selector, both rules

func ap01Name(p string) string {
	if p == "" {
		return "both"
	}
	return p
}

// TestAuditAP01_passCost: one PrefillLast pass's wall time at startPos 64 and 2048 for C = 16, 32, 48 and 64 tokens
// under each gemmTilePolicy, cells and policies interleaved rep by rep in a rotating order. The reading is shipped ÷
// policy per cell, paired by rep, median over reps. One KV slot, so E-P01's step route cannot take a pass.
func TestAuditAP01_passCost(t *testing.T) {
	auditA(t)
	defer func() { gemmTilePolicy = "" }()
	name, a := auditLoad(t, "qwen2.5-coder-1.5b-instruct-q4_k_m.gguf", 1, 2048+128, nil)
	t0 := time.Now()
	embs := auditEmbs(a.r, 2048+64, 3)
	if _, err := a.PrefillLast(context.Background(), embs[:2048], 0); err != nil {
		t.Fatalf("fill 2048 positions: %v", err)
	}
	type cell struct {
		at, C  int
		policy string
	}
	var cells []cell
	for _, at := range []int{64, 2048} {
		for _, C := range []int{16, 32, 48, 64} {
			for _, p := range ap01Policies {
				cells = append(cells, cell{at, C, p})
			}
		}
	}
	reps := auditReps(7)
	wall := map[cell][]float64{}
	for rep := range reps + 1 { // rep 0 warms every pipeline and is dropped
		for k := range cells {
			c := cells[(k+rep)%len(cells)]
			gemmTilePolicy = c.policy
			st := time.Now()
			if _, err := a.PrefillLast(context.Background(), embs[c.at:c.at+c.C], c.at); err != nil {
				t.Fatalf("pass at %d, C=%d, policy %q: %v", c.at, c.C, c.policy, err)
			}
			if rep > 0 {
				wall[c] = append(wall[c], time.Since(st).Seconds()*1e3)
			}
		}
		if rep > 0 {
			auditHB("a-p01", t0, "rep %d/%d done", rep, reps)
		}
	}
	gemmTilePolicy = ""
	ratio := func(at, C int, p string) (float64, string) {
		var rs []float64
		for i := range wall[cell{at, C, "shipped"}] {
			rs = append(rs, wall[cell{at, C, "shipped"}][i]/wall[cell{at, C, p}][i])
		}
		return auditMedian(rs), auditFmt(rs)
	}
	for _, at := range []int{64, 2048} {
		for _, C := range []int{16, 32, 48, 64} {
			line := fmt.Sprintf("startPos %d C=%d: shipped %.1f ms", at, C, auditMedian(wall[cell{at, C, "shipped"}]))
			for _, p := range ap01Policies[1:] {
				m, _ := ratio(at, C, p)
				line += fmt.Sprintf(" | %s %.1f ms (%.3fx)", ap01Name(p), auditMedian(wall[cell{at, C, p}]), m)
			}
			auditHB("a-p01", t0, "%s", line)
		}
	}
	for _, p := range ap01Policies[1:] {
		m, pairs := ratio(64, 32, p)
		auditHB("a-p01", t0, "RESULT %s: pass wall shipped / %s at startPos 64, C=32 = %.3fx (pairs %s)", name, ap01Name(p), m, pairs)
	}
}

// TestAuditAP01_gemmSmallM: each prefill GEMM of the model's shape alone (qkv, o, gate/up, down), at M = 16, 32, 48
// and 64 rows, gemm_w4f16_store against the tile gemmTile's default picks, GPU time per dispatch; arms interleaved
// rep by rep, weight copies rotated past the system-level cache. The audit's kill line reads gate/up at M = 32.
func TestAuditAP01_gemmSmallM(t *testing.T) {
	auditA(t)
	_, a := auditLoad(t, "qwen2.5-coder-1.5b-instruct-q4_k_m.gguf", 1, 512, nil)
	r, t0 := a.r, time.Now()
	r.ensurePrefill()
	pf := r.pf
	d := r.d
	g0 := r.layers[0].geom
	H, I := r.H, r.I
	shapes := []struct {
		what string
		N, K int
	}{{"qkv", r.nH*g0.hd + 2*g0.kvDim, H}, {"o", H, r.nH * g0.hd}, {"gate/up", 2 * I, H}, {"down", H, I}}
	seed := uint32(77)
	rnd := func() uint32 { seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5; return seed }
	cq := d.NewCommandQueue()
	reps := auditReps(7)
	for _, sh := range shapes {
		N, K := sh.N, sh.K
		bytes := N*K/2 + N*(K/32)*2
		copies := min(32, max(1, int(math.Ceil(float64(256<<20)/float64(bytes)))))
		var ws, ss []Buffer
		for range copies {
			w := d.NewBufferLen(N * K / 8)
			wv := w.U32s()[:N*K/8]
			for i := range wv {
				wv[i] = rnd()
			}
			sc := make([]uint16, N*K/32)
			for i := range sc {
				sc[i] = f32ToF16(float32(rnd()%1000+1) * 1e-5)
			}
			ws, ss = append(ws, w), append(ss, NewBufferU16s(d, sc))
		}
		uN, uK, uMode, bias := NewBufferU32(d, uint32(N)), NewBufferU32(d, uint32(K)), NewBufferU32(d, 0), NewBufferFloats(d, make([]float32, N))
		for _, M := range []int{16, 32, 48, 64} {
			av := make([]uint16, M*K)
			for i := range av {
				av[i] = f32ToF16(float32(int(rnd()%2001)-1000) * 1e-3)
			}
			A, C, uM := NewBufferU16s(d, av), d.NewBufferLen(M*N/2+1), NewBufferU32(d, uint32(M))
			pp, tm, tn := pf.gemmTile(M, N)
			arms := []struct {
				name   string
				p      Pipeline
				tm, tn int
			}{{"shipped", pf.pGemmStore, 64, 64}, {fmt.Sprintf("m%dn%d", tm, tn), pp, tm, tn}}
			ms := [2][]float64{}
			for rep := range reps + 1 {
				for k := range arms {
					ai := (k + rep) % 2
					arm := arms[ai]
					e := cq.Begin()
					for i := range 2 * copies {
						e.Dispatch2D(arm.p, (N+arm.tn-1)/arm.tn, (M+arm.tm-1)/arm.tm, 128, 1, A, ws[i%copies], ss[i%copies], C, uM, uN, uK, bias, uMode)
					}
					e.End()
					if err := e.Err(); err != nil {
						t.Fatalf("%s M=%d %s: %v", sh.what, M, arm.name, err)
					}
					if rep > 0 {
						ms[ai] = append(ms[ai], (e.GPUEnd()-e.GPUStart())*1e3/float64(2*copies))
					}
				}
			}
			var rs []float64
			for i := range ms[0] {
				rs = append(rs, ms[0][i]/ms[1][i])
			}
			line := fmt.Sprintf("%s (N %d, K %d) M=%d: shipped %.3f ms, %s %.3f ms: %.3fx (pairs %s)", sh.what, N, K, M,
				auditMedian(ms[0]), arms[1].name, auditMedian(ms[1]), auditMedian(rs), auditFmt(rs))
			if sh.what == "gate/up" && M == 32 {
				line = "RESULT " + line
			}
			auditHB("a-p01-gemm", t0, "%s", line)
		}
	}
}
