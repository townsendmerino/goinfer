//go:build darwin

package metal

import (
	"math"
	"os"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestGemma4MoE_noiseFloor is the pre-flight the track's rule requires BEFORE any resident MoE
// number is interpreted: measure the fixture's own CPU-int4-vs-CPU-f32 floor on THIS machine. It is
// CPU-only (no Metal), so it runs before the gemma4MoeMLP kernels exist. The CUDA MoE fixture had to
// be rebuilt once (9275f94) because the original couldn't hold a tolerance — degenerate routing at
// 68.8% agreement, 0.77 logit floor. A fixture whose int4-vs-f32 logit cosine or routing agreement
// sits below the near-tie bar cannot gate Metal regardless of how good the port is; better to learn
// that here than after the MoE forward is wired.
//
// Three signals per fixture: (1) logit cosine CPU-int4 vs CPU-f32 (the "as well as int4 can agree"
// bound); (2) argmax agreement; (3) routing agreement — do int4 and f32 select the SAME top-k
// experts — plus the top-k boundary margin (the quantity a quant flip must overcome). Diagnostic:
// logs everything, and fails only if a fixture is degenerate enough that it could not gate.
func TestGemma4MoE_noiseFloor(t *testing.T) {
	for _, fx := range []string{"gemma4-moe-tiny", "gemma4-moe-kv-tiny"} {
		fx := fx
		t.Run(fx, func(t *testing.T) {
			dir := "../testdata/" + fx
			if _, err := os.Stat(dir); err != nil {
				t.Skipf("no fixture (%s)", dir)
			}
			// Router idx/margin for the f32 run, then the int4 run — SetRouterCaptureForTest clears on
			// enable, so capture, copy, re-enable, capture again.
			m4, err := decoder.Load(dir, decoder.Options{Quant: "int4"})
			if err != nil {
				t.Fatalf("load int4: %v", err)
			}
			defer m4.Close()
			mF, err := decoder.Load(dir, decoder.Options{Quant: "f32"})
			if err != nil {
				t.Fatalf("load f32: %v", err)
			}
			defer mF.Close()

			decoder.SetRouterCaptureForTest(true)
			defer decoder.SetRouterCaptureForTest(false)

			// f32 reference forward.
			cacheF := mF.NewCache(len(twoGeomPrompt))
			var logF [][]float32
			for _, tok := range twoGeomPrompt {
				l, err := mF.ForwardForTest(tok, cacheF)
				if err != nil {
					t.Fatalf("f32 forward: %v", err)
				}
				logF = append(logF, append([]float32(nil), l...))
			}
			idxF, _ := decoder.RouterCaptureForTest()
			idxF = copyIdx(idxF)
			marginF := append([]float32(nil), decoder.RouterMarginForTest()...)

			// int4 forward.
			decoder.SetRouterCaptureForTest(true) // clears
			cache4 := m4.NewCache(len(twoGeomPrompt))
			var log4 [][]float32
			for _, tok := range twoGeomPrompt {
				l, err := m4.ForwardForTest(tok, cache4)
				if err != nil {
					t.Fatalf("int4 forward: %v", err)
				}
				log4 = append(log4, append([]float32(nil), l...))
			}
			idx4, _ := decoder.RouterCaptureForTest()

			// (1)/(2) logit cosine + argmax agreement, int4 vs f32.
			minCos, exact := 1.0, 0
			for i := range logF {
				c, _ := cosMaxAbs(logF[i], log4[i])
				if c < minCos {
					minCos = c
				}
				if argmaxF(logF[i]) == argmaxF(log4[i]) {
					exact++
				}
			}
			// (3) routing agreement: fraction of decisions whose top-k SET matches, + min margin.
			agree, total := 0, min(len(idxF), len(idx4))
			for i := 0; i < total; i++ {
				if sameSet(idxF[i], idx4[i]) {
					agree++
				}
			}
			minMargin := math.Inf(1)
			for _, mg := range marginF {
				if float64(mg) < minMargin {
					minMargin = float64(mg)
				}
			}
			agreePct := 100.0
			if total > 0 {
				agreePct = 100 * float64(agree) / float64(total)
			}
			t.Logf("%s noise floor: CPU-int4-vs-f32 minCosine=%.6f argmax %d/%d | routing agree %d/%d (%.1f%%) minMargin=%.4f",
				fx, minCos, exact, len(logF), agree, total, agreePct, minMargin)

			// The gateable property for a router-first MoE gate is that routing is dominated by the
			// TRAINED router, not by quant noise — i.e. a wide top-k margin and high int4-vs-f32
			// routing agreement. A degenerate fixture (margin ~0) flips top-k on quant noise, so a
			// resident idx==CPU-int4 idx match is CIRCULAR (both int4 impls agree on the same coin
			// flip); it validates nothing about the router. The logit cosine is LOGGED, not asserted:
			// a low int4-vs-f32 cosine (moe-tiny is 0.79) is the toy's quant hostility and only means
			// the whole-forward gate must be argmax-primary (which it is), NOT that the port is wrong.
			degenerate := (total > 0 && agreePct < 90) || minMargin < 0.02
			// gemma4-moe-tiny is THE gating fixture for Step 5 (as on CUDA, which gates every MoE test
			// on it and never references moe-kv-tiny). It must stay healthy; a regression here is a real
			// failure. moe-kv-tiny is informational — the MoE+K=V COMPOSITION is validated by the real
			// 26B (Step 6), not this toy, so its degeneracy is expected and skipped, not failed.
			gating := fx == "gemma4-moe-tiny"
			if degenerate {
				if gating {
					t.Errorf("%s is the Step-5 GATING fixture but routing is degenerate (agree %.1f%%, margin %.4f) — "+
						"a resident idx match would be circular; rebuild it on the box before trusting the router gate", fx, agreePct, minMargin)
				} else {
					t.Skipf("%s routing is degenerate (agree %.1f%%, margin %.4f) — NOT a gating fixture: CUDA gates the "+
						"MoE delta on moe-tiny and the MoE+K=V composition via the real 26B, not this toy. Informational only.", fx, agreePct, minMargin)
				}
			}
		})
	}
}

func copyIdx(in [][]int) [][]int {
	out := make([][]int, len(in))
	for i, s := range in {
		out[i] = append([]int(nil), s...)
	}
	return out
}

func sameSet(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	seen := map[int]int{}
	for _, x := range a {
		seen[x]++
	}
	for _, x := range b {
		seen[x]--
	}
	for _, v := range seen {
		if v != 0 {
			return false
		}
	}
	return true
}
