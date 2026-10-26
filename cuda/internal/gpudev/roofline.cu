// roofline.cu — the DENOMINATORS. Three probes that measure what this device can do
// at all, so a kernel's throughput can be reported as a fraction of something real
// rather than of a spec sheet or of another machine's constant.
//
// WHY THIS SHIPS AS A KERNEL RATHER THAN LIVING IN SOMEONE'S SCRATCH DIRECTORY. The
// amd64 GEMM benchmark reported "~50% of peak" for months against a hardcoded M1 Pro
// figure; the real number on that box was 38%. A wrong denominator does not announce
// itself — it produces a plausible number, and a plausible wrong number is harder to
// catch than an obviously wrong one. Every ratio in docs/internal/roofline-2026-08.md
// is against a ceiling one of these three produced on the machine that ran it, so the
// probes have to be re-runnable by whoever reads that doc on different hardware.
//
// Each probe seeds from a buffer and stores its result, so nothing folds away, and
// each varies ONE thing: bytes moved (stream_read), int32 multiply-add issue rate
// (imad_peak), or __dp4a issue rate (dp4a_peak). Reading an instruction mix and
// predicting the bottleneck was wrong three times in a row during that campaign;
// varying one thing in isolation was right every time.
//
// BIT-IDENTITY-EXEMPT: there is no float arithmetic here at all. Every accumulator is
// an int, and integer addition is associative, so nothing about these probes' results
// depends on evaluation order and there is no MAC for a compiler to contract. They
// also compute nothing any consumer reads — the stores exist only to defeat dead-code
// elimination. Adding a float accumulate would end both properties.

// stream_read: grid-stride int4 (16-byte) loads over a buffer far larger than L2.
// The DRAM roof. Report N*K/seconds against this to place a bandwidth-bound kernel.
extern "C" __global__ void stream_read(
    const int4* __restrict__ src,
    int* __restrict__ out,
    const unsigned int* __restrict__ n4,    // element count, in int4 units
    const unsigned int* __restrict__ total) // launch bound
{
    unsigned int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= *total) return;
    unsigned int n = *n4;
    unsigned int stride = gridDim.x * blockDim.x;
    int acc = 0;
    for (unsigned int i = g; i < n; i += stride) {
        int4 v = src[i];
        acc += v.x + v.y + v.z + v.w;
    }
    out[g] = acc;
}

// imad_peak: UNROLL independent `acc = acc * b + c` chains, no memory traffic in the
// loop. This is the instruction a byte-wise int8 tile GEMM's inner loop compiles to
// (`acc += (int)As[..] * (int)Bs[..]`), so it is the roof such a kernel works against
// when it is not DRAM-bound.
//
// The chains must be INDEPENDENT and there must be enough of them: a multiply-add has
// multi-cycle latency, so too few chains measure latency rather than throughput. 16 is
// comfortably past the knee on Turing.
#define ROOF_UNROLL 16

extern "C" __global__ void imad_peak(
    const int* __restrict__ seed,
    int* __restrict__ out,
    const unsigned int* __restrict__ iters,
    const unsigned int* __restrict__ total)
{
    unsigned int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= *total) return;
    int b = seed[g & 255u], c = seed[(g + 1u) & 255u];
    int a[ROOF_UNROLL];
    #pragma unroll
    for (int i = 0; i < ROOF_UNROLL; i++) a[i] = seed[(g + (unsigned)i) & 255u];
    unsigned int it = *iters;
    for (unsigned int t = 0; t < it; t++) {
        #pragma unroll
        for (int i = 0; i < ROOF_UNROLL; i++) a[i] = a[i] * b + c;
    }
    int s = 0;
    #pragma unroll
    for (int i = 0; i < ROOF_UNROLL; i++) s += a[i];
    out[g] = s;
}

// dp4a_peak: the same shape, but each instruction retires FOUR int8 MACs. Comparing
// the two is also the probes' self-check — see TestDeviceCeilings.
extern "C" __global__ void dp4a_peak(
    const int* __restrict__ seed,
    int* __restrict__ out,
    const unsigned int* __restrict__ iters,
    const unsigned int* __restrict__ total)
{
    unsigned int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= *total) return;
    int x = seed[g & 255u], y = seed[(g + 1u) & 255u];
    int a[ROOF_UNROLL];
    #pragma unroll
    for (int i = 0; i < ROOF_UNROLL; i++) a[i] = seed[(g + (unsigned)i) & 255u];
    unsigned int it = *iters;
    for (unsigned int t = 0; t < it; t++) {
        #pragma unroll
        for (int i = 0; i < ROOF_UNROLL; i++) a[i] = __dp4a(x, y, a[i]);
    }
    int s = 0;
    #pragma unroll
    for (int i = 0; i < ROOF_UNROLL; i++) s += a[i];
    out[g] = s;
}
