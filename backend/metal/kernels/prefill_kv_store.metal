// kv_store_f16: scatter M rows' K,V (slices of the fused qkv[M×stride]) into the f16 KV cache
// at positions[m]. grid = M*kvDim.
kernel void kv_store_f16(device const half* qkv[[buffer(0)]], device half* kc[[buffer(1)]],
    device half* vc[[buffer(2)]], device const uint* positions[[buffer(3)]],
    constant uint& kvDim[[buffer(4)]], constant uint& stride[[buffer(5)]],
    constant uint& kOff[[buffer(6)]], constant uint& vOff[[buffer(7)]], device const uint* kt[[buffer(8)]],
    uint gid[[thread_position_in_grid]]) {
    uint m = gid / kvDim, i = gid % kvDim;
    uint row = kv_row(kt, positions[m]);
    kc[row*kvDim + i] = qkv[m*stride + kOff + i];
    vc[row*kvDim + i] = qkv[m*stride + vOff + i];
}

