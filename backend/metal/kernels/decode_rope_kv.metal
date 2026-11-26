// rope: NeoX half-split. Rotates pairs (d, half+d) for d in [0,half) within each head (stride
// hd), where half = rotaryDim/2 = len(invf). half<hd/2 is PARTIAL rotary (Phi): dims
// [2*half, hd) pass through unrotated. total = nHeads*half (the rotate-pair count).
// scale is YaRN's mscale (attention_factor), applied to cos/sin exactly like
// decoder's rope.go's applyRoPE (c := cos(theta)*scale; s := sin(theta)*scale) — NOT to the
// rotated output afterward, which is a different (and wrong) place to apply it. 1.0 for
// every family without YaRN, so this is a no-op multiply, not a new branch. FeatRopeMscale
// (decoder's features.go) is not yet declared for Metal — nothing resident here exercises
// scale != 1.0 end-to-end — but the kernel needs to accept it before wiring can proceed, and
// a no-op multiply on every existing dispatch is the lowest-risk way to add the parameter.
kernel void rope(device float* x[[buffer(0)]], device const float* invf[[buffer(1)]],
    constant uint& hd[[buffer(2)]], constant uint& pos[[buffer(3)]], constant uint& total[[buffer(4)]],
    constant uint& rhalf[[buffer(5)]], constant float& scale[[buffer(6)]], uint gid[[thread_position_in_grid]]) {
    if(gid>=total) return; uint head=gid/rhalf; uint dd=gid%rhalf; uint base=head*hd;
    float th=float(pos)*invf[dd]; float c=cos(th)*scale,s=sin(th)*scale;
    float x0=x[base+dd],x1=x[base+rhalf+dd]; x[base+dd]=x0*c-x1*s; x[base+rhalf+dd]=x0*s+x1*c;
}
// rope2: merges the Q and K rope dispatches for one layer into ONE dispatch — every production
// call site launches rope twice per layer back-to-back, once for Q (buffer offset 0 into the
// fused qkv buffer) and once for K (bound at a Metal buffer-offset kOff via aikit/gpu's
// Buffer.At). Bind-time buffer offsets can't express "two ranges of the SAME dispatch", so this
// takes the UNOFFSET base buffer and does the offset itself: gid in [0,qTotal) addresses Q at
// x[dd..], gid in [qTotal,qTotal+kTotal) addresses K at x[kOff+dd..]. kOff is in ELEMENTS
// (floats), matching how the kernel indexes x — reuse geom's existing uNHhd (= nH*hd) buffer
// unchanged rather than adding a new one, since it is already exactly that value. Same math as
// rope otherwise (verified bit-identical to running rope twice — TestRope2_matchesTwoRope,
// metal/rope2_test.go); rope itself is untouched and kept for its own standalone kernel tests.
// qTempScale (buffer 9, Ministral 3 FeatAttnTemp): a post-rotation multiplier on Q ONLY, applied
// after the pair rotation below — decoder's attention.go's sequential path does the SAME thing in
// the SAME order (rope first, attn-temp on top), never touching K. 1.0 (exact no-op) for every
// family without this feature — Model.AttnTempScale's own comment has the formula.
kernel void rope2(device float* x[[buffer(0)]], device const float* invf[[buffer(1)]],
    constant uint& hd[[buffer(2)]], constant uint& pos[[buffer(3)]], constant uint& qTotal[[buffer(4)]],
    constant uint& kTotal[[buffer(5)]], constant uint& rhalf[[buffer(6)]], constant float& scale[[buffer(7)]],
    constant uint& kOff[[buffer(8)]], constant float& qTempScale[[buffer(9)]], uint gid[[thread_position_in_grid]]) {
    uint total = qTotal + kTotal;
    if (gid >= total) return;
    bool isQ = gid < qTotal;
    uint g = gid; uint off = 0u;
    if (!isQ) { g = gid - qTotal; off = kOff; }
    uint head = g/rhalf; uint dd = g%rhalf; uint base = off + head*hd;
    float th=float(pos)*invf[dd]; float c=cos(th)*scale,s=sin(th)*scale;
    float x0=x[base+dd],x1=x[base+rhalf+dd];
    float r0=x0*c-x1*s, r1=x0*s+x1*c;
    if (isQ) { r0 *= qTempScale; r1 *= qTempScale; }
    x[base+dd]=r0; x[base+rhalf+dd]=r1;
}
// rope_pw / rope2_pw: rope / rope2 for GPT-J PAIRWISE rotation (Cohere,
// Command-R7B, Aya, GLM-OCR), pair dd = dims (2dd, 2dd+1) at invf[dd] instead of NeoX's (dd, rhalf+dd), as
// decoder's rope.go's applyRoPEInterleaved. Same arguments and grid, bound into the same pipeline fields when
// decoder.Model.BackendView().PairwiseRoPE(); every other family keeps rope / rope2. Partial rotary rotates dims [0, 2*rhalf)
// and leaves the tail as it is; scale and qTempScale are applied exactly where the NeoX kernels apply them.
kernel void rope_pw(device float* x[[buffer(0)]], device const float* invf[[buffer(1)]],
    constant uint& hd[[buffer(2)]], constant uint& pos[[buffer(3)]], constant uint& total[[buffer(4)]],
    constant uint& rhalf[[buffer(5)]], constant float& scale[[buffer(6)]], uint gid[[thread_position_in_grid]]) {
    if(gid>=total) return; uint head=gid/rhalf; uint dd=gid%rhalf; uint base=head*hd+2u*dd;
    float th=float(pos)*invf[dd]; float c=cos(th)*scale,s=sin(th)*scale;
    float x0=x[base],x1=x[base+1u]; x[base]=x0*c-x1*s; x[base+1u]=x0*s+x1*c;
}
kernel void rope2_pw(device float* x[[buffer(0)]], device const float* invf[[buffer(1)]],
    constant uint& hd[[buffer(2)]], constant uint& pos[[buffer(3)]], constant uint& qTotal[[buffer(4)]],
    constant uint& kTotal[[buffer(5)]], constant uint& rhalf[[buffer(6)]], constant float& scale[[buffer(7)]],
    constant uint& kOff[[buffer(8)]], constant float& qTempScale[[buffer(9)]], uint gid[[thread_position_in_grid]]) {
    uint total = qTotal + kTotal;
    if (gid >= total) return;
    bool isQ = gid < qTotal;
    uint g = gid; uint off = 0u;
    if (!isQ) { g = gid - qTotal; off = kOff; }
    uint head = g/rhalf; uint dd = g%rhalf; uint base = off + head*hd + 2u*dd;
    float th=float(pos)*invf[dd]; float c=cos(th)*scale,s=sin(th)*scale;
    float x0=x[base],x1=x[base+1u];
    float r0=x0*c-x1*s, r1=x0*s+x1*c;
    if (isQ) { r0 *= qTempScale; r1 *= qTempScale; }
    x[base]=r0; x[base+1u]=r1;
}
kernel void kv_store(device const float* k[[buffer(0)]], device const float* v[[buffer(1)]],
    device half* kc[[buffer(2)]], device half* vc[[buffer(3)]], constant uint& kvDim[[buffer(4)]],
    constant uint& pos[[buffer(5)]], device const uint* kt[[buffer(6)]], uint i[[thread_position_in_grid]]) {
    uint row = kv_row(kt, pos);
    kc[row*kvDim+i]=half(k[i]); vc[row*kvDim+i]=half(v[i]); // f16 KV: half the cache bytes + read BW
}
// kv_store_i8 — stores K and V as symmetric int8 per-head with f32 scales (ks, vs).
// Each thread handles one KV head: reduces absmax → scale (maxabs/127) → stores quantized int8.
kernel void kv_store_i8(device const float* k[[buffer(0)]], device const float* v[[buffer(1)]],
    device char* kc[[buffer(2)]], device char* vc[[buffer(3)]],
    device float* ks[[buffer(4)]], device float* vs[[buffer(5)]],
    constant uint& nKV[[buffer(6)]], constant uint& hd[[buffer(7)]],
    constant uint& pos[[buffer(8)]], device const uint* kt[[buffer(9)]], uint kvh[[thread_position_in_grid]]) {
    if (kvh >= nKV) return;
    uint kvDim = nKV * hd;
    uint base = kvh * hd;
    uint row = kv_row(kt, pos);
    float amax_k = 0.0f, amax_v = 0.0f;
    for (uint d = 0; d < hd; d++) {
        amax_k = max(amax_k, abs(k[base + d]));
        amax_v = max(amax_v, abs(v[base + d]));
    }
    float sc_k = amax_k / 127.0f; if (sc_k == 0.0f) sc_k = 1.0f;
    float sc_v = amax_v / 127.0f; if (sc_v == 0.0f) sc_v = 1.0f;
    ks[row * nKV + kvh] = sc_k;
    vs[row * nKV + kvh] = sc_v;
    float inv_k = 1.0f / sc_k;
    float inv_v = 1.0f / sc_v;
    for (uint d = 0; d < hd; d++) {
        kc[row * kvDim + base + d] = char(clamp(round(k[base + d] * inv_k), -127.0f, 127.0f));
        vc[row * kvDim + base + d] = char(clamp(round(v[base + d] * inv_v), -127.0f, 127.0f));
    }
}
// rope2_kv: fuses rope2 (merged Q+K RoPE) with kv_store (K/V cache write) into ONE dispatch --
// every production call site launches these back-to-back, RoPE-then-store, and K's cache write
// needs exactly the ROTATED value RoPE just computed, so storing it inline removes a full
// device round-trip (rope2 writes rotated K into the qkv buffer, kv_store immediately re-reads
// those same bytes). Grid: [0,qTotal) rotates Q (no cache write -- Q isn't cached, same as
// rope2); [qTotal,qTotal+kTotal) rotates K AND stores it into kc; [qTotal+kTotal,qTotal+2*kTotal)
// copies V (untouched by RoPE) into vc. V shares K's thread count and pairing since kvDim
// (=nKV*hd) is always exactly 2*kTotal (=2*nKV*half). Same math as running rope2 then kv_store --
// verified bit-identical on real weights (TestRope2Kv_matchesRope2ThenKv; TestMetalSnapshotGolden
// and TestGPT2ResidentParityMetal wired against this call site too, also byte-identical).
// MEASURED (TestQwen35ResidentDecodeRateMetal, 20 samples each way): eliminating one dispatch/
// layer is a null result at the whole-decode level (~0.6%, within noise) -- consistent with
// TestLayerA_bindingTax's own finding that per-dispatch marginal cost (~2.2us) is tiny against
// everything else in a token's critical path. NOT wired into any production dispatch site --
// correctness-proven, kept for the record, not a live kernel (same status as gemv_w4a8_sa_qv).
kernel void rope2_kv(device float* x[[buffer(0)]], device const float* invf[[buffer(1)]],
    constant uint& hd[[buffer(2)]], constant uint& pos[[buffer(3)]], constant uint& qTotal[[buffer(4)]],
    constant uint& kTotal[[buffer(5)]], constant uint& rhalf[[buffer(6)]], constant float& scale[[buffer(7)]],
    constant uint& kOff[[buffer(8)]], constant uint& vOff[[buffer(9)]], device half* kc[[buffer(10)]],
    device half* vc[[buffer(11)]], constant uint& kvDim[[buffer(12)]], uint gid[[thread_position_in_grid]]) {
    uint total = qTotal + 2u*kTotal;
    if (gid >= total) return;
    if (gid < qTotal) {
        uint head = gid/rhalf; uint dd = gid%rhalf; uint base = head*hd;
        float th=float(pos)*invf[dd]; float c=cos(th)*scale,s=sin(th)*scale;
        float x0=x[base+dd],x1=x[base+rhalf+dd];
        x[base+dd]=x0*c-x1*s; x[base+rhalf+dd]=x0*s+x1*c;
    } else if (gid < qTotal+kTotal) {
        uint g = gid - qTotal; uint head = g/rhalf; uint dd = g%rhalf; uint base = kOff + head*hd;
        float th=float(pos)*invf[dd]; float c=cos(th)*scale,s=sin(th)*scale;
        float x0=x[base+dd],x1=x[base+rhalf+dd];
        float k0=x0*c-x1*s, k1=x0*s+x1*c;
        x[base+dd]=k0; x[base+rhalf+dd]=k1;
        uint kvi = head*hd;
        kc[pos*kvDim+kvi+dd]=half(k0); kc[pos*kvDim+kvi+rhalf+dd]=half(k1);
    } else {
        uint g = gid - qTotal - kTotal; uint head = g/rhalf; uint dd = g%rhalf; uint base = vOff + head*hd;
        uint kvi = head*hd;
        vc[pos*kvDim+kvi+dd]=half(x[base+dd]); vc[pos*kvDim+kvi+rhalf+dd]=half(x[base+rhalf+dd]);
    }
}
// kv_store_f32 — the f32 KV cache's store (--kv f32); its attention, attention_f32, is derived from attention (metal/kv_f32.go)
// (resident.kvF32). Gemma's low-magnitude attention contexts amplify f16-KV rounding into a
// catastrophic per-layer context error (0.64 vs f32's 0.92 cosine; matched-input confirmer
// isolated it to the KV cache). Qwen is insensitive and keeps the f16 path (half the cache BW).
kernel void kv_store_f32(device const float* k[[buffer(0)]], device const float* v[[buffer(1)]],
    device float* kc[[buffer(2)]], device float* vc[[buffer(3)]], constant uint& kvDim[[buffer(4)]],
    constant uint& pos[[buffer(5)]], device const uint* kt[[buffer(6)]], uint i[[thread_position_in_grid]]) {
    uint row = kv_row(kt, pos);
    kc[row*kvDim+i]=k[i]; vc[row*kvDim+i]=v[i];
}
