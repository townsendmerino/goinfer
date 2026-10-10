//go:build cuda && goinfer_testhooks

package cuda

// SetFastPrefillForTest flips the L2/L3 lever selection on an already-loaded resident, loading the fused kernels on
// first use if the env did not already ask for them. The seam exists because the fidelity gate
// (docs/completed/task-prefill-gap.md Phase 3) must run the exact arm and the fast arm in one process, teacher-forced on
// the reference's tokens, while the env var is read once at model load: two residents do not fit on the 8 GB card, and
// two processes cannot share the KV cache the continuation walks. It is a test hook, behind goinfer_testhooks, because
// it gates a measurement and not production inference; production selection stays fastPrefillEnabled() reading the env
// at load.
//
// It returns an error if the PTX cannot be compiled; callers should skip rather than score a "fast" arm that quietly
// runs the exact kernels, since a gate that cannot tell those apart would report the exact path's numbers twice and call
// it a pass.
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
	saved := r.passPromptLen
	r.passPromptLen = M // ask about a prompt of exactly this length
	defer func() { r.passPromptLen = saved }()
	_, a := r.useAttnFused(hd, M)
	return a, r.useGemmMMA("int4", 1536, M) && r.bGemmMMA != (Pipeline{})
}
