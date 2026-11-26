// gemv_w4a8_sa_bias_resid: FeatOutBias's kernel — the o-proj GEMV needs BOTH an additive
// per-row bias AND direct residual accumulation (gemv_w4a8_sa_resid has no bias epilogue,
// gemv_w4a8_sa_bias overwrites instead of accumulating; neither alone is o-proj's shape for a
// family with OutBias, e.g. GPT-2/gpt-oss). Not currently dispatched anywhere — no family
// resident on Metal declares FeatOutBias yet — gated standalone (gptoss_kernels_test.go-style)
// against a hand-computed reference until a real family wires it in.
kernel void gemv_w4a8_sa_bias_resid(device const uint4* wq[[buffer(0)]], device const half* sct[[buffer(1)]],
    device const char* aq[[buffer(2)]], device const float* asc[[buffer(3)]], device float* out[[buffer(4)]],
    device const float* bias[[buffer(5)]], constant uint& K[[buffer(6)]], threadgroup short* As [[threadgroup(0)]], uint tgid[[threadgroup_position_in_grid]],
    uint tid[[thread_index_in_threadgroup]], uint tgs[[threads_per_threadgroup]],
    uint sgid[[simdgroup_index_in_threadgroup]], uint lane[[thread_index_in_simdgroup]]) {
    SA_BODY
    if (lane==0) out[row] += acc*asc[0] + bias[row];
}

// Fused block-argmax lm-head (Fable): computes the SAME logits as gemv_w4a8_sa but never
// materializes them — each threadgroup emits (maxLogit, rowIndex) over its 8 rows; a tiny
// second pass (argmax_finish) reduces the tiles to one token. Kills the 608KB logit readback
// + CPU scan. Merge key (v, -idx) is a commutative monoid → order-independent, tie-broken
// identically to a CPU first-max-wins scan (strict >, lower index wins).
struct AmaxPart { float v; uint i; };
// NOTE (N-09): NOT currently dispatched — no pipeline is created for it (see model.go). Unlike the
// batch-K variant it takes no N and has no row>=N guard, so before wiring it, add N and mask any
// out-of-range row's logit to -INFINITY (it's a reduction: all simdgroups must reach the barrier, so
// do NOT early-return like the store variants). SA_BODY reads weight row=row, so guard the read too.
kernel void gemv_w4a8_sa_amax(device const uint4* wq[[buffer(0)]], device const half* sct[[buffer(1)]],
    device const char* aq[[buffer(2)]], device const float* asc[[buffer(3)]], device AmaxPart* part[[buffer(4)]],
    constant uint& K[[buffer(5)]], threadgroup short* As [[threadgroup(0)]], uint tgid[[threadgroup_position_in_grid]],
    uint tid[[thread_index_in_threadgroup]], uint tgs[[threads_per_threadgroup]],
    uint sgid[[simdgroup_index_in_threadgroup]], uint lane[[thread_index_in_simdgroup]]) {
    SA_BODY                                       // acc = this row's dot; row = output index
    threadgroup float tv[8]; threadgroup uint ti[8];
    if (lane==0) { tv[sgid] = acc*asc[0]; ti[sgid] = row; }   // the logit, exactly as the store variant
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid==0) {
        // NaN is skipped, as argmax_rows_part skips it: a NaN leading the tile would otherwise win it, hiding the tile's maximum.
        // A tile of NaN alone writes (-inf, 0xFFFFFFFF), and argmax_finish then writes 0xFFFFFFFF only for a row that is NaN
        // throughout, whose host argmax is 0 (headID).
        uint nsg = tgs>>5u; float bv = -INFINITY; uint bi = 0xFFFFFFFFu;
        for (uint s=0u; s<nsg; s++) if (!isnan(tv[s]) && (tv[s]>bv || (tv[s]==bv && ti[s]<bi))) { bv=tv[s]; bi=ti[s]; }
        part[tgid].v = bv; part[tgid].i = bi;
    }
}
