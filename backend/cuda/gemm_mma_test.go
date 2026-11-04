//go:build cuda

package cuda

import "testing"

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
