// kvtype.cuh — the element type of the KV cache, per kernel module (docs/tasks/task-cuda-f16-kv-2026-10.md).
//
// A kernel that reads or writes the resident K/V cache declares its cache pointers as `KV_T` and touches them only through the
// macros below. Compiled as it always was (no -DKV_F16), KV_T is `float` and every macro expands to the exact expression the kernel
// used to spell out, so the module's PTX regenerates byte for byte unchanged (that identity is what build_ptx.sh's check and
// TestKVTwins_f32ModulesUnchanged rest on). Compiled with -DKV_F16, KV_T is `__half`: a load converts to f32 (exact) and a store
// rounds to nearest even; every other operation in the kernel stays f32, in the same order, so a kernel run on a cache that already
// holds f16-representable values produces the same bits as its f32 original.
//
// Macros, not functions, on purpose: an inline function can lose the `__restrict__ const` provenance that selects the non-coherent
// load, and the f32 expansion must be textually the original code.
#ifndef GOINFER_KVTYPE_CUH
#define GOINFER_KVTYPE_CUH
#include <cuda_fp16.h>
#ifdef KV_F16
typedef __half KV_T;
// kv_ld4 / kv_ld2: 4 or 2 consecutive cache elements as one 8-byte or 4-byte load, widened to f32.
__device__ __forceinline__ float4 kv_ld4(const __half* p) {
    uint2 u = *reinterpret_cast<const uint2*>(p);
    float2 a = __half22float2(*reinterpret_cast<const __half2*>(&u.x));
    float2 b = __half22float2(*reinterpret_cast<const __half2*>(&u.y));
    return make_float4(a.x, a.y, b.x, b.y);
}
__device__ __forceinline__ float2 kv_ld2(const __half* p) {
    unsigned u = *reinterpret_cast<const unsigned*>(p);
    return __half22float2(*reinterpret_cast<const __half2*>(&u));
}
#define KV_LD(p) __half2float(*(p))
// KV_LD4V(p, i) / KV_LD2V(p, i): element i of the float4 / float2 view of p, i.e. the 4 (2) elements starting at p + 4*i (2*i).
#define KV_LD4V(p, i) kv_ld4((p) + 4 * (i))
#define KV_LD2V(p, i) kv_ld2((p) + 2 * (i))
#define KV_ST(p, v) (*(p) = __float2half_rn(v))
// KV_LDH(p): the element as an f16 value, for the tensor-core kernels that stage K and V as f16 anyway (f32 caches round here, f16 caches
// are already that value).
#define KV_LDH(p) (*(p))
#else
typedef float KV_T;
#define KV_LD(p) (*(p))
#define KV_LD4V(p, i) (((const float4*)(p))[i])
#define KV_LD2V(p, i) (((const float2*)(p))[i])
#define KV_ST(p, v) (*(p) = (v))
#define KV_LDH(p) __float2half_rn(*(p))
#endif
#endif
