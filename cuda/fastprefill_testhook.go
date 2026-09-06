//go:build cuda && goinfer_testhooks

package cuda

// SetFastPrefillForTest flips the L2/L3 lever selection on an ALREADY-LOADED resident, loading the
// fused kernels on first use if the env did not already ask for them.
//
// WHY THIS SEAM EXISTS. docs/task-prefill-gap.md Phase 3 requires the exact arm and the fast arm to
// run "in one process ... teacher-forced on the reference's tokens", and the env var is read once
// at model load (backend.go). Without this hook the gate would have to load the model twice, which
// is not merely wasteful: D7 at int4 is ~4 GB of an 8 GB card, so two residents do not fit at once,
// and two sequential processes cannot share the resident KV cache that the teacher-forced
// continuation walks position by position.
//
// It is a TEST HOOK, behind goinfer_testhooks, for the reason B-08 gives: it gates a measurement,
// not production inference, so it stays off the public API surface. Production selection remains
// exactly one thing — fastPrefillEnabled() reading the env at load.
//
// Returns an error if the PTX cannot be compiled; callers should skip rather than silently score a
// "fast" arm that is quietly running the exact kernels — a gate that cannot tell those apart would
// report the exact path's numbers twice and call the result a pass.
func (r *cudaResident) SetFastPrefillForTest(attn, gemm bool) error {
	if attn && (r.bAttnFused64 == (Pipeline{}) || r.bAttnFused128 == (Pipeline{})) {
		mod, err := r.dev.CompileLibrary(attnFusedPTX)
		if err != nil {
			return err
		}
		p64, err := r.dev.NewComputePipeline(mod, "attn_fused_hd64")
		if err != nil {
			return err
		}
		p128, err := r.dev.NewComputePipeline(mod, "attn_fused_hd128")
		if err != nil {
			return err
		}
		r.bAttnFused64, r.bAttnFused128 = p64, p128
	}
	if gemm && r.bGemmMMA == (Pipeline{}) {
		mod, err := r.dev.CompileLibrary(gemmMMAPTX)
		if err != nil {
			return err
		}
		p, err := r.dev.NewComputePipeline(mod, "gemm_w4a8_mma")
		if err != nil {
			return err
		}
		r.bGemmMMA = p
	}
	r.fastAttn, r.fastGemm = attn, gemm
	return nil
}

// FastPrefillActiveForTest reports which levers a launch would actually use at this M, so a gate
// can ASSERT that its "fast" arm is genuinely fast rather than assuming it. The M argument matters:
// both levers decline below their row floors, so at small M the fast arm legitimately runs the
// exact kernels and a gate that did not check would be comparing the exact path with itself.
func (r *cudaResident) FastPrefillActiveForTest(hd, M int) (attn, gemm bool) {
	_, _, a := r.useAttnFused(hd, M)
	return a, r.useGemmMMA("int4", 1536, M) && r.bGemmMMA != (Pipeline{})
}
