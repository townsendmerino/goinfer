// argmax.cu — the greedy-decode argmax reduction, split out of glue.cu.
//
// A SEPARATE .cu from glue.cu on purpose (same reasoning as router_f32.cu). glue.ptx ships built at
// CUDA 12.6; this box's NVRTC is 12.9, so regenerating glue.ptx to change a kernel would rewrite EVERY
// shipped glue kernel's codegen (rmsnorm/rope/attention/glu — the audited numeric kernels), a toolchain
// bump masquerading as a one-kernel fix. A fresh file adds only argmax.ptx and leaves the audited 12.6
// glue.ptx untouched (it still carries a now-unused argmax_reduce copy, loaded from here instead).
//
// C-14: index tie-break. The CPU reference (decoder.argmax, strict >) returns the FIRST maximal index.
// This tree reduction pairs by THREAD index, and a left thread's strided max can sit at a HIGHER
// absolute index than the right thread's — so `>` alone returned that higher index on an exact tie
// (e.g. V=512, ties at 1 and 256: thread 0 holds 256, thread 1 holds 1, and 256 wrongly won). Taking
// the right half on strictly-greater OR equal-with-lower-index makes the on-device argmax return the
// same token the CPU does on ANY tie, so the greedy fast path stays bit-identical to CPU decode.

extern "C" {

__global__ void argmax_reduce(const float* __restrict__ logits, int V, int* __restrict__ outIdx, float* __restrict__ outVal) {
    extern __shared__ float sv[];
    int* si = (int*)(sv + blockDim.x);
    int t = threadIdx.x, nt = blockDim.x;
    float bv = -1e30f; int bi = -1;
    for (int k = t; k < V; k += nt) if (logits[k] > bv) { bv = logits[k]; bi = k; } // strict > ⇒ lowest index per thread
    sv[t] = bv; si[t] = bi; __syncthreads();
    for (int o = nt >> 1; o > 0; o >>= 1) {
        if (t < o && (sv[t + o] > sv[t] || (sv[t + o] == sv[t] && si[t + o] < si[t]))) { sv[t] = sv[t + o]; si[t] = si[t + o]; }
        __syncthreads();
    }
    if (t == 0) { *outIdx = si[0]; *outVal = sv[0]; }
}

} // extern "C"
