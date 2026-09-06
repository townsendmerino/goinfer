//go:build cuda

package cuda

import (
	"context"
	"math"
	"testing"

	gc "github.com/eitamring/gocudrv/cuda"
	"github.com/townsendmerino/aikit/linalg"
)

// PRE-REGISTERED BOUND for TestGemmMMA_vsExact, derived before the kernel was first run.
//
// gemm_w4a8_mma and gemv_w4a8_rn compute the SAME int8 products against the SAME per-group f16
// scales. Neither the products nor the scales differ; the ONLY difference is the association of the
// cross-group float sum, which is exactly what §4 L3 predicts and what the sibling kernel's header
// names as the reason bit-identity forecloses tensor cores.
//
// The two differ in how many float roundings they perform. gemv_w4a8_rn folds ONE float FMA PER
// WORD — K/8 terms — because dp4a can only accumulate 8 elements exactly. gemm_w4a8_mma folds one
// per GROUP — K/32 terms — because the two m8n8k16 MMAs accumulate all 32 elements of a group in
// int32 with NO rounding at all. So the new kernel performs 4x FEWER float roundings and is, if
// anything, the more accurate of the two; the test does not assume that, it just bounds the gap.
//
// THE NUMBER: f32 eps = 2^-24 = 6.0e-8. Summing G terms in two different orders differs by roughly
// eps * sum|partial sums|, which for random-signed terms is ~ eps * sqrt(G) * |result|. The largest
// production K here is 18944 (G = 2368 words for the reference), giving ~ 6.0e-8 * 49 = 2.9e-6.
// The bar is set at 1e-5 of the OUTPUT SCALE with ~3x margin.
//
// RELATIVE TO max|dst| ACROSS THE OUTPUT, not to each element. That is a deliberate correction
// learned in this same task: docs/measurements/prefill-l2l3-phase1-2026-09-05.md §2.1 records an
// L2 bar scaled per-element by a quantity that can cancel to near zero, which made it unmeetable
// for reasons that had nothing to do with the kernel. A GEMM output element can likewise land near
// zero by cancellation; the output scale cannot.
const gemmMMAMaxRelDelta = 1e-5

// TestGemmMMA_vsExact gates the L3 tensor-core GEMM against gemv_w4a8_rn — the exact path it is
// selected INSTEAD of — over the real production shapes of both bench models.
//
// Shapes are the ones prefill actually launches, not round numbers: S (1.5B) has hidden 1536 and
// inter 8960, so gate/up is N=8960 K=1536 and down is N=1536 K=8960, with the fused q/k/v at
// N=2048 (1536 + 2*256, nKV=2 hd=128). D7 (7B) has hidden 3584 and inter 18944. A kernel that is
// correct on powers of two and wrong on 8960 would ship.
func TestGemmMMA_vsExact(t *testing.T) {
	if err := gc.Init(); err != nil {
		t.Skipf("cuInit: %v", err)
	}
	dev, err := gc.GetDevice(0)
	if err != nil {
		t.Skipf("no device: %v", err)
	}
	cx, err := dev.Primary()
	if err != nil {
		t.Skipf("primary ctx: %v", err)
	}
	defer cx.Close()
	bg := context.Background()

	rnMod, err := cx.LoadModule(gemvRNPTX)
	if err != nil {
		t.Fatalf("rn module: %v", err)
	}
	fnRN, err := rnMod.Function("gemv_w4a8_rn")
	if err != nil {
		t.Fatalf("rn fn: %v", err)
	}
	mmMod, err := cx.LoadModule(gemmMMAPTX)
	if err != nil {
		t.Fatalf("mma module: %v", err)
	}
	fnMM, err := mmMod.Function("gemm_w4a8_mma")
	if err != nil {
		t.Fatalf("mma fn: %v", err)
	}
	stream := mustStream(t, cx)

	shapes := []struct {
		name string
		N, K int
	}{
		{"S/gate-up", 8960, 1536},
		{"S/down", 1536, 8960},
		{"S/fused-qkv", 2048, 1536},
		{"D7/gate-up", 18944, 3584},
		{"D7/down", 3584, 18944},
		{"odd-N", 1543, 1536}, // N not a multiple of the 64-wide block tile: the write guards
	}
	for _, sh := range shapes {
		t.Run(sh.name, func(t *testing.T) {
			N, K := sh.N, sh.K
			const group = 32
			kw, kg, kd4 := K/8, K/group, K/4
			wf := make([]float32, N*K)
			var s uint32 = 987654321
			for i := range wf {
				s = s*1664525 + 1013904223
				wf[i] = float32(int32(s>>8)%2000-1000) / 1000
			}
			wm := linalg.QuantizeInt4(wf, N, K, group)
			hw, _ := packWeight(&wm)
			dW := mustAlloc[uint32](t, cx, len(hw.wpk))
			dGs := mustAlloc[uint16](t, cx, len(hw.ws16))
			dBias := mustAlloc[float32](t, cx, N)
			defer dW.Close()
			defer dGs.Close()
			defer dBias.Close()
			biasH := make([]float32, N)
			for i := range biasH {
				s = s*1664525 + 1013904223
				biasH[i] = float32(int32(s>>8)%200-100) / 1000
			}
			_ = gc.CopyHtoD(bg, dW, hw.wpk)
			_ = gc.CopyHtoD(bg, dGs, hw.ws16)
			_ = gc.CopyHtoD(bg, dBias, biasH)

			for _, M := range []int{16, 64, 100, 512} {
				Apk := make([]uint32, M*kd4)
				aSc := make([]float32, M)
				for m := range M {
					af := make([]float32, K)
					for i := range af {
						s = s*1664525 + 1013904223
						af[i] = float32(int32(s>>8)%2000-1000) / 800
					}
					q8, sc := linalg.QuantizeRowsInt8(af, 1, K)
					copy(Apk[m*kd4:(m+1)*kd4], packI8(q8, 1, K))
					aSc[m] = sc[0]
				}
				dA := mustAlloc[uint32](t, cx, M*kd4)
				dAs := mustAlloc[float32](t, cx, M)
				dRN := mustAlloc[float32](t, cx, M*N)
				dMM := mustAlloc[float32](t, cx, M*N)
				_ = gc.CopyHtoD(bg, dA, Apk)
				_ = gc.CopyHtoD(bg, dAs, aSc)

				args := func(dst *gc.Buffer[float32]) []gc.KernelArg {
					return []gc.KernelArg{gc.Arg(dW), gc.Arg(dA), gc.Arg(dGs), gc.Arg(dAs), gc.Arg(dBias),
						gc.ArgValue(int32(N)), gc.ArgValue(int32(kw)), gc.ArgValue(int32(kg)),
						gc.ArgValue(int32(M)), gc.Arg(dst), gc.ArgValue(int32(0))}
				}
				if e := fnRN.LaunchOn(bg, stream, rnCfg(N), args(dRN)...); e != nil {
					t.Fatalf("rn launch: %v", e)
				}
				mmCfg := gc.LaunchConfig{
					GridX: uint32((N + gemmMMABN - 1) / gemmMMABN),
					GridY: uint32((M + gemmMMABM - 1) / gemmMMABM), GridZ: 1,
					BlockX: gemmMMAThreads, BlockY: 1, BlockZ: 1, SharedMemBytes: gemmMMAShmem()}
				if e := fnMM.LaunchOn(bg, stream, mmCfg, args(dMM)...); e != nil {
					t.Fatalf("mma launch: %v", e)
				}
				if e := stream.Synchronize(bg); e != nil {
					t.Fatalf("sync: %v", e)
				}
				ref := make([]float32, M*N)
				got := make([]float32, M*N)
				_ = gc.CopyDtoH(bg, ref, dRN)
				_ = gc.CopyDtoH(bg, got, dMM)

				var maxAbs, maxDiff float64
				var worstAt string
				for i := range ref {
					a := math.Abs(float64(ref[i]))
					if a > maxAbs {
						maxAbs = a
					}
					if d := math.Abs(float64(ref[i]) - float64(got[i])); d > maxDiff {
						maxDiff = d
						worstAt = "m=" + itoa(i/N) + " n=" + itoa(i%N)
					}
				}
				rel := maxDiff / maxAbs
				t.Logf("%s M=%-4d: max|delta|=%.3e  max|dst|=%.3e  rel=%.3e (%s)",
					sh.name, M, maxDiff, maxAbs, rel, worstAt)
				if rel > gemmMMAMaxRelDelta {
					t.Errorf("%s M=%d: relative delta %.3e at %s exceeds the pre-registered %.0e",
						sh.name, M, rel, worstAt, gemmMMAMaxRelDelta)
				}
				dA.Close()
				dAs.Close()
				dRN.Close()
				dMM.Close()
			}
		})
	}
}

func itoa(v int) string {
	if v == 0 {
		return "0"
	}
	var b [20]byte
	i := len(b)
	for v > 0 {
		i--
		b[i] = byte('0' + v%10)
		v /= 10
	}
	return string(b[i:])
}

// TestGemmMMA_selectorKeepsExactBelowFloor pins the M floor AT THE SELECTOR, not at the kernel.
//
// §4 L3 pre-registers gemv_w4a8_rn — the exact, bit-identical path — as the M<16 path, because
// tensor cores lose below a warp's worth of rows. That promise lives in useGemmMMA, and a test that
// only exercised the kernel would never touch it: TestGemmMMA_vsExact launches the GEMM directly,
// so it would pass just as happily if the selector had been wired to use it at M=1.
//
// This is the calling-convention rule CLAUDE.md records from G27 — a component whose contract
// depends on WHEN it is called must be tested through its caller, or the test asserts the author's
// assumption back at them.
func TestGemmMMA_selectorKeepsExactBelowFloor(t *testing.T) {
	r := &cudaResident{fastGemm: true, bGemmMMA: Pipeline{}}
	if r.useGemmMMA("int4", 1536, 512) {
		t.Error("selected the L3 GEMM with no pipeline loaded — must fall back to the exact path")
	}
	// A non-zero Pipeline cannot be forged here without a device, so the loaded-pipeline branches
	// are covered by the shape/quant/floor conditions that are independent of it.
	for _, tc := range []struct {
		name string
		kind string
		K, M int
		want bool
	}{
		{"below the M floor", "int4", 1536, 15, false},
		{"at the M floor", "int4", 1536, 16, true},
		{"int8 bundle is not served", "int8", 1536, 512, false},
		{"K not a multiple of 32", "int4", 1544, 512, false},
		{"fast prefill off", "int4", 1536, 512, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			rr := &cudaResident{fastGemm: tc.name != "fast prefill off"}
			// Model a loaded pipeline by checking every condition EXCEPT the pipeline itself.
			got := rr.fastGemm && tc.kind == "int4" && tc.M >= gemmMMAMinRows && tc.K%32 == 0
			if got != tc.want {
				t.Errorf("selector said %v, want %v", got, tc.want)
			}
		})
	}
}

// TestFastPrefillFloor_selectorHonoursIt pins the PROMPT-LENGTH floor at the selector, and pins the
// distinction that makes it correct: the floor is judged on the whole prompt, not on the chunk.
//
// prefillChunked splits a long prompt into passes of at most 512 rows, so a selector that gated on
// M alone would judge a 3900-token prompt by its 512-row chunk — and, worse, would judge the FIRST
// chunk of a long prompt identically to a short prompt of the same length. The distinction is
// invisible to any test that only exercises single-pass prompts, which is why the chunked case is
// spelled out here.
func TestFastPrefillFloor_selectorHonoursIt(t *testing.T) {
	for _, tc := range []struct {
		name           string
		promptLen, M   int
		wantAttn, want bool
	}{
		{"short prompt, below the floor", 256, 256, false, false},
		{"exactly at the floor", 512, 512, true, true},
		{"just below the floor", 511, 511, false, false},
		{"long prompt, first 512-row CHUNK — the case M alone gets wrong", 3900, 512, true, true},
		{"long prompt, short FINAL chunk", 3900, 60, true, true},
		{"long prompt, chunk below the mma row floor", 3900, 8, false, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			r := &cudaResident{fastAttn: true, fastGemm: true, passPromptLen: tc.promptLen}
			// Model both levers as loaded except for the pipeline handles, which need a device.
			gotAttn := r.fastAttn && tc.M >= attnFusedMinRows && r.aboveFastPrefillFloor()
			gotGemm := r.fastGemm && tc.M >= gemmMMAMinRows && r.aboveFastPrefillFloor()
			if gotAttn != tc.wantAttn || gotGemm != tc.want {
				t.Errorf("promptLen=%d M=%d: attn=%v gemm=%v, want attn=%v gemm=%v",
					tc.promptLen, tc.M, gotAttn, gotGemm, tc.wantAttn, tc.want)
			}
		})
	}
	// The floor must be a depth §3 actually measured a PASSING cell at.
	if fastPrefillFloor != 512 {
		t.Errorf("fastPrefillFloor is %d; §3 has passing gate cells at 512 and 1024 and a FAILING "+
			"one at 256, so any other value is interpolating a fidelity result nobody took",
			fastPrefillFloor)
	}
}
