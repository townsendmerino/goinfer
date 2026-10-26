// smoke.cu — the device layer's own end-to-end proof (cuda_smoke_test.go): reach the
// GPU, JIT PTX, alloc buffers, upload, dispatch, read back. The CUDA twin of the MSL
// `vadd` in smoke_test.go.
//
// Note the bounds guard: unlike Metal's dispatchThreads, a CUDA launch rounds up to
// whole blocks, so the tail block runs past n. Every kernel on this device layer
// carries its own count and checks it (gpu/cuda.go, divergence 2).
extern "C" __global__ void vadd(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ out,
    const unsigned int* __restrict__ n)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= *n) return;
    out[i] = a[i] + b[i];
}

// ---- the tuned-kernel calling convention (gpu.KernelArg / LaunchConfig / Queue.Launch) ----
//
// The three kernels below exist to prove the surface a tuned decode kernel set needs,
// which `vadd` above does not exercise. They take their scalars BY VALUE rather than as
// 1-element buffers — the convention goinfer's kernels use, and the reason gpu.ArgValue
// exists alongside gpu.Arg.

// saxpy: y = a*x + y, with BOTH scalars (the coefficient and the count) passed by value
// and positionally mixed with the buffer arguments.
extern "C" __global__ void saxpy(
    float* __restrict__ y,
    const float* __restrict__ x,
    float a,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    y[i] = a * x[i] + y[i];
}

// blocksum: sum n elements in ONE cooperating block, staging partials in DYNAMIC shared
// memory — the shape gpu.GridOne(block, sharedBytes) exists for, and the one a whole-vector
// reduction (an RMS norm over the hidden state) uses. The caller sizes the shared memory,
// so the kernel declares it extern. blockDim must be a power of two for the tree reduction.
extern "C" __global__ void blocksum(
    const float* __restrict__ x,
    float* __restrict__ out,
    int n)
{
    extern __shared__ float sm[];
    int t = threadIdx.x, nt = blockDim.x;
    float acc = 0.f;
    for (int i = t; i < n; i += nt) acc += x[i];
    sm[t] = acc;
    __syncthreads();
    for (int s = nt / 2; s > 0; s >>= 1) {
        if (t < s) sm[t] += sm[t + s];
        __syncthreads();
    }
    if (t == 0) *out = sm[0];
}

// scale: x *= s in place. Enqueued repeatedly with no sync between launches, to prove that
// Queue.Launch batches asynchronously and that one Queue.Sync at the end is sufficient —
// the launch-many-then-sync-at-a-boundary model. Because the launches share a stream they
// run in issue order, so k launches compose to s^k exactly.
extern "C" __global__ void scale(
    float* __restrict__ x,
    float s,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    x[i] *= s;
}
