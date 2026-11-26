// lora_delta: compute-time LoRA, the two-GEMV low-rank
// delta y[o] += scale·Σ_r B[o,r]·(A·x)[r], fused into ONE dispatch (P-11, audit-2026-09-10 —
// mirrors CUDA's own P-11 finding, "14 extra launches per layer per token... before the
// serialized reduction": up to 7 targeted projections/layer each paid TWO dispatches — down then
// up — for pure per-dispatch launch overhead no different in kind from CUDA's). x is the SAME
// quantized activation (aq/asc) the base projection this delta rides alongside already consumed
// — matching applyLoRA's CPU reference, which takes the identical input the base matmul does
// (decoder's lora.go).
//
// M-08 (audit-metal-2026-09-12.md): P-11's fused-into-one-threadgroup shape traded a whole
// dispatch for making the up stage SERIAL over Out — every one of 256 threads striding across
// however many output rows there are, up to thousands. Now launched over ceil(Out/256)
// threadgroups (tgid selects a FIXED, disjoint 256-row block each owns outright — no stride, no
// race, still exactly one thread per row across the whole dispatch), each independently
// RECOMPUTING the down-stage's t[R] rather than reading it from a device-memory scratch buffer a
// separate kernel wrote (CUDA's own P-11 fix, cuda/lora.cu's lora_delta_down/_up split): still
// ONE dispatch (no second launch's overhead, the more expensive line item on Metal's launch/sync
// ceiling), and the redundant R reductions over K this costs are cheap relative to a second
// dispatch — A stays L2-resident across the ceil(Out/256) threadgroups reading it, and R·K MACs
// is small next to Out·R's up-stage work at any Out worth splitting over more than one
// threadgroup. A/B are read as half (PEFT ships them f32; SetAdapter converts once at bind time,
// metal/lora.go) — the delta feeds an int8-quantised activation, so f32 A/B precision was never
// load-bearing, and halving their bytes matters most here: A is re-read whole by every
// threadgroup in the group, not just once per dispatch.
kernel void lora_delta(device const char* aq[[buffer(0)]], device const float* asc[[buffer(1)]],
    device const half* A[[buffer(2)]], device const half* B[[buffer(3)]],
    device float* out[[buffer(4)]],
    constant uint& K[[buffer(5)]], constant uint& R[[buffer(6)]], constant uint& Out[[buffer(7)]],
    constant float& scale[[buffer(8)]],
    uint tgid[[threadgroup_position_in_grid]],
    uint tid[[thread_position_in_threadgroup]], uint tgs[[threads_per_threadgroup]]) {
    // Fixed-size STATIC threadgroup arrays (matches rmsnorm_quant's own reduction, kernels.go
    // top) — tgs is pinned to tgReduceNorm (256, model.go) by the dispatch site, same as every
    // other norm-class reduction kernel in this file. Deliberately NOT a [[threadgroup(0)]]
    // dynamic parameter: that form requires the caller to use DispatchTG (which sets the length
    // explicitly) rather than plain Dispatch — using a static array here avoids that footgun
    // entirely, the same way every other single-threadgroup reduction kernel in this file does.
    threadgroup float red[256];
    threadgroup float t[256]; // R <= loraRMax (256, model.go)
    float sc = asc[0];
    for (uint r = 0; r < R; r++) {
        device const half* Ar = A + r*K;
        float part = 0.0f;
        for (uint k = tid; k < K; k += tgs) part += float(Ar[k]) * (float(aq[k]) * sc);
        red[tid] = part;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint s = tgs/2; s > 0; s >>= 1) { if (tid < s) red[tid] += red[tid+s]; threadgroup_barrier(mem_flags::mem_threadgroup); }
        if (tid == 0) t[r] = red[0];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    uint row = tgid*tgs + tid;
    if (row < Out) {
        device const half* Br = B + row*R;
        float acc = 0.0f;
        for (uint r = 0; r < R; r++) acc += float(Br[r]) * t[r];
        out[row] += scale*acc;
    }
}

