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
// The i8 build (-DKV_I8, docs/measurements/cuda-kv-i8-2026-11/scope.md): KV_T is `signed char` and a row's values are int8 with ONE f32 scale per
// (position, KV head), the scheme the CPU, Metal and WebGPU caches use. A load needs that scale, so the readers of an i8-capable module read
// through KV_LDS(p, sc), whose `sc` is ignored (and not evaluated) in the f32 and f16 builds, so those modules' PTX is unchanged; and the
// module's kernels take the K and V scale pools as trailing parameters (KV_SCALE_PARAMS, empty in the f32 and f16 builds). A module is built
// with -DKV_I8 only once every KV access in it goes through KV_LDS / the quantizing store, so an access that forgot the scale does not compile.
#if defined(KV_I8) && defined(KV_F16)
#error "KV_I8 and KV_F16 are two builds of one cache"
#endif
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
#define KV_LDS(p, sc) KV_LD(p)
#define KV_LD4VS(p, i, sc) KV_LD4V(p, i)
#define KV_LD2VS(p, i, sc) KV_LD2V(p, i)
#define KV_SCALE_PARAMS
#define KV_SCALE_ARGS
#elif defined(KV_I8)
typedef signed char KV_T;
// KV_LDS(p, sc): the element widened to f32 and multiplied by its (position, head) scale. For a scale that is a power of two the product
// is exact, so a reader folding these in the f32 original's order gives the original's bits on a cache that holds int8 x 2^e values.
#define KV_LDS(p, sc) ((float)(*(p)) * (sc))
// KV_ST in an i8 build stores nothing: the rope/store kernels still write the rotated K back into their f32 K buffer in place and leave V in its own,
// and kv8_store.cu quantizes both into the int8 pools with their scales (one scale needs the absmax over a head's hd values, which the per-pair
// threads of the rope kernels do not see). A caller of an i8 build launches kv8_store after the rope kernel.
#define KV_ST(p, v) ((void)(p), (void)(v))
// KV_LD4VS(p, i, sc) / KV_LD2VS(p, i, sc): the 4 (2) int8 elements starting at p + 4*i (2*i), as one 32-bit (16-bit) load, each widened and multiplied by sc.
__device__ __forceinline__ float4 kv_ld4_s8(const signed char* p, float sc) {
    const char4 c = *reinterpret_cast<const char4*>(p);
    return make_float4((float)c.x * sc, (float)c.y * sc, (float)c.z * sc, (float)c.w * sc);
}
__device__ __forceinline__ float2 kv_ld2_s8(const signed char* p, float sc) {
    const char2 c = *reinterpret_cast<const char2*>(p);
    return make_float2((float)c.x * sc, (float)c.y * sc);
}
#define KV_LD4VS(p, i, sc) kv_ld4_s8((p) + 4 * (i), (sc))
#define KV_LD2VS(p, i, sc) kv_ld2_s8((p) + 2 * (i), (sc))
#define KV_SCALE_PARAMS , const float* __restrict__ kscl, const float* __restrict__ vscl
#define KV_SCALE_ARGS , kscl, vscl
#else
typedef float KV_T;
#define KV_LD(p) (*(p))
#define KV_LD4V(p, i) (((const float4*)(p))[i])
#define KV_LD2V(p, i) (((const float2*)(p))[i])
#define KV_ST(p, v) (*(p) = (v))
#define KV_LDH(p) __float2half_rn(*(p))
#define KV_LDS(p, sc) KV_LD(p)
#define KV_LD4VS(p, i, sc) KV_LD4V(p, i)
#define KV_LD2VS(p, i, sc) KV_LD2V(p, i)
#define KV_SCALE_PARAMS
#define KV_SCALE_ARGS
#endif
#endif
