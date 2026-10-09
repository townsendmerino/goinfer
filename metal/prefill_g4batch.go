//go:build darwin

package metal

// D-P01's batched expert GEMM (docs/audit-metal-2026-09-30.md; docs/tasks/task-m26-mac-2026-10.md, "D-P01"): a
// layer-major paged Gemma 4 prefill runs one slot group's routed experts expert by expert, every (row, expert) pair the
// group routes to that expert in one pass over its weights, instead of row by row. Bit-identical to the per-row phase 2
// (encodeG4Phase2Paged) by construction:
//   - gate|up and down run moe_batch_gemv, which computes sa_rows_acc's per-row sums for up to B activations at once:
//     each (activation, weight row) accumulator sees exactly sa_rows_acc's operations in its order, and the weights are
//     read once for all of them;
//   - SwiGLU and its int8 quantisation run mc3_swiglu_quant_rows, swiglu_quant's body per pair;
//   - the down projection stores each pair's raw sum, and moe_batch_combine adds a row's k experts in route order with
//     the per-row path's own epilogue, fma(wgt*acc, asc, out) from zero. The floating-point order is the per-row one.
//
// g4ExpertBatchOn picks it in g4LayerMajorRows. ON once its gate passes and its speed read clears the owner's bar.
var g4ExpertBatchOn = true

// moeBatchB is the most pairs one moe_batch_gemv threadgroup takes: B activations of K halves in threadgroup memory,
// 22.5 KB at M26's hidden size of 2816 for gate|up.
const moeBatchB = 4

const moeBatchKernels = `
// moe_batch_gemv<R,B>: for one routed expert and up to B pairs routed to it (entry = slot, count, B pair ids), the
// expert's rows [slot*rowsPerExpert, +rows) dotted with each pair's int8 activation (pair p reads activation p/actDiv),
// R rows a simdgroup as sa_rows_acc. Every (b, r) accumulator runs sa_rows_acc's operations in sa_rows_acc's order, so
// each output equals the single-activation kernel's bit for bit. scaleOut: out = acc*asc[act] (gate|up, as
// gemv_w4a8_moe_rows), else out = acc (the down projection's raw sum, combined by moe_batch_combine). Grid
// nEntries*tiles threadgroups of 256, tiles = rowsPerExpert/(8R) (full threadgroups only, as sa_rows_acc).
template <uint R, uint B>
kernel void moe_batch_gemv(device const uint4* wq[[buffer(0)]], device const half* sct[[buffer(1)]],
    device const char* aq_all[[buffer(2)]], device const float* asc_all[[buffer(3)]], device float* out_all[[buffer(4)]],
    constant uint& K[[buffer(5)]], device const uint* ent[[buffer(6)]], constant uint& rowsPerExpert[[buffer(7)]],
    constant uint& tiles[[buffer(8)]], constant uint& actDiv[[buffer(9)]], constant uint& scaleOut[[buffer(10)]],
    threadgroup half* Ah[[threadgroup(0)]], uint tg_b[[threadgroup_position_in_grid]],
    uint tid[[thread_index_in_threadgroup]], uint tgs[[threads_per_threadgroup]],
    uint sgid[[simdgroup_index_in_threadgroup]], uint lane[[thread_index_in_simdgroup]]) {
    constexpr float P4[4] = {1.0f, 0.0625f, 0.00390625f, 0.000244140625f};
    constexpr float4 U4 = float4(1.0f, 16.0f, 256.0f, 4096.0f);
    uint e = tg_b / tiles, tgid = tg_b % tiles;
    device const uint* en = ent + e*(2u + B);
    uint slot = en[0], cnt = en[1];
    uint G = K>>5u;
    device const uint4* w0 = wq + slot*rowsPerExpert*G;
    device const half* s0 = sct + slot*rowsPerExpert*G;
    for (uint b=0;b<cnt;b++) {
        device const char* aq = aq_all + (en[2u+b]/actDiv)*K;
        for (uint i=tid;i<K;i+=tgs) Ah[b*K+i] = half(float(aq[i]) * P4[i & 3u]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint row0 = (tgid*(tgs>>5u) + sgid)*R;
    float acc[B][R];
    SA_ROWS_UNROLL for (uint b=0;b<B;b++) SA_ROWS_UNROLL for (uint r=0;r<R;r++) acc[b][r]=0.0f;
    for (uint g=lane; g<G; g+=32u) {
        uint4 w[R];
        SA_ROWS_UNROLL for (uint r=0;r<R;r++) w[r] = w0[(row0+r)*G + g];
        SA_ROWS_UNROLL for (uint b=0;b<B;b++) {
            if (b < cnt) {
                threadgroup const half4* a4 = reinterpret_cast<threadgroup const half4*>(Ah + b*K + g*32u);
                float gf[R];
                SA_ROWS_UNROLL for (uint r=0;r<R;r++) gf[r] = 0.0f;
                float sa = 0.0f;
                SA_ROWS_UNROLL for (uint t=0;t<8u;t++) {
                    float4 x = float4(a4[t]);
                    sa += dot(x, U4);
                    SA_ROWS_UNROLL for (uint r=0;r<R;r++) {
                        uint word = (t < 2u) ? w[r].x : (t < 4u) ? w[r].y : (t < 6u) ? w[r].z : w[r].w;
                        uint u = (t & 1u) ? (word >> 16) : (word & 0xFFFFu);
                        gf[r] += dot(float4(float(u & 0xFu), float(u & 0xF0u), float(u & 0xF00u), float(u & 0xF000u)), x);
                    }
                }
                SA_ROWS_UNROLL for (uint r=0;r<R;r++) acc[b][r] += (gf[r] - 8.0f*sa) * float(s0[(row0+r)*G + g]);
            }
        }
    }
    SA_ROWS_UNROLL for (uint b=0;b<B;b++) SA_ROWS_UNROLL for (uint r=0;r<R;r++) acc[b][r] = simd_sum(acc[b][r]);
    if (lane==0) {
        SA_ROWS_UNROLL for (uint b=0;b<B;b++) {
            if (b < cnt) {
                uint p = en[2u+b];
                device float* out = out_all + p*rowsPerExpert;
                if (scaleOut != 0u) {
                    float sc = asc_all[p/actDiv];
                    SA_ROWS_UNROLL for (uint r=0;r<R;r++) out[row0+r] = acc[b][r]*sc;
                } else {
                    SA_ROWS_UNROLL for (uint r=0;r<R;r++) out[row0+r] = acc[b][r];
                }
            }
        }
    }
}
template [[host_name("moe_batch_gemv2")]] kernel decltype(moe_batch_gemv<2,4>) moe_batch_gemv<2,4>;
template [[host_name("moe_batch_gemv4")]] kernel decltype(moe_batch_gemv<4,4>) moe_batch_gemv<4,4>;

// moe_batch_combine: a row's routed experts into its own H floats, in route order j = 0..k-1, with gemv_w4a8_moe_wacc's
// epilogue fma(wgt*acc, asc, out) starting from zero, as encodeG4Phase2Paged's zero_vec and k wacc dispatches leave it.
// Pair p = row*k + j: wgt[p] its router weight, dacc[p*H + h] its raw down sum, dsc[p] its activation scale.
kernel void moe_batch_combine(device const float* dacc[[buffer(0)]], device const float* wgt[[buffer(1)]],
    device const float* dsc[[buffer(2)]], device float* out[[buffer(3)]], constant uint& H[[buffer(4)]],
    constant uint& k[[buffer(5)]], constant uint& n[[buffer(6)]], uint i[[thread_position_in_grid]]) {
    if (i >= n*H) return;
    uint row = i / H, h = i % H;
    float o = 0.0f;
    for (uint j=0;j<k;j++) {
        uint p = row*k + j;
        o = fma(wgt[p]*dacc[p*H + h], dsc[p], o);
    }
    out[i] = o;
}
`

// moeBatchR is the rows a simdgroup moe_batch_gemv takes for a projection of n output rows: 4 or 2 where full
// threadgroups divide n (sa_rows_acc's condition), else 0 (the batched path declines).
func moeBatchR(n int) int {
	for _, R := range []int{4, 2} {
		if n%(8*R) == 0 {
			return R
		}
	}
	return 0
}

// g4Batch is one layer-major call's scratch for the batched phase 2, grown to the largest group it meets: pair-major
// activations, scales, router weights, gate|up outputs, SwiGLU outputs, raw down sums and each row's combined experts.
type g4Batch struct {
	rows, pairs                               int
	aq, asc, wgt, gu, dq, dsc, dacc, x2       Buffer
	ent                                       Buffer
	entCap                                    int
	uK, uN, uTilesGU, uTilesDown, uOne, uZero Buffer
}

func (r *resident) g4BatchEnsure(b *g4Batch, n, entries int, made func(Buffer) Buffer) {
	g, d := r.g4moe, r.d
	P := n * g.topK
	if n > b.rows {
		b.rows = n
		b.aq, b.asc = made(byteBuf(d, n*r.H)), made(d.NewBufferLen(n))
		b.x2 = made(d.NewBufferLen(n * r.H))
	}
	if P > b.pairs {
		b.pairs = P
		b.wgt, b.dsc = made(d.NewBufferLen(P)), made(d.NewBufferLen(P))
		b.gu = made(d.NewBufferLen(P * 2 * g.moeInter))
		b.dq = made(byteBuf(d, P*g.moeInter))
		b.dacc = made(d.NewBufferLen(P * r.H))
	}
	if entries > b.entCap {
		b.entCap = entries
		b.ent = made(NewBufferUint32s(d, make([]uint32, entries*(2+moeBatchB))))
	}
	if b.uK == (Buffer{}) {
		b.uK, b.uOne = made(NewBufferU32(d, uint32(g.topK))), made(NewBufferU32(d, 1))
		b.uZero = made(NewBufferU32(d, 0))
		b.uN = made(NewBufferU32(d, 0))
		b.uTilesGU, b.uTilesDown = made(NewBufferU32(d, 0)), made(NewBufferU32(d, 0))
	}
}

// g4BatchEntries lays out a group's moe_batch_gemv entries: for each pool slot the group uses, in first-use order, its
// pairs (row*k + j, rows counted from the group's first) in order, at most moeBatchB to an entry.
func g4BatchEntries(slots [][]uint32, topK int) []uint32 {
	var order []uint32
	pairs := map[uint32][]uint32{}
	for i, si := range slots {
		for j := range topK {
			s := si[j]
			if _, ok := pairs[s]; !ok {
				order = append(order, s)
			}
			pairs[s] = append(pairs[s], uint32(i*topK+j))
		}
	}
	var out []uint32
	for _, s := range order {
		for ps := pairs[s]; len(ps) > 0; {
			c := min(len(ps), moeBatchB)
			e := make([]uint32, 2+moeBatchB)
			e[0], e[1] = s, uint32(c)
			copy(e[2:], ps[:c])
			out = append(out, e...)
			ps = ps[c:]
		}
	}
	return out
}

// encodeG4Phase2Batch encodes a slot group's phase 2 (rows grp, their slot tables already filled) through the batched
// kernels, then each row's join from its own combined experts. The rows' phase 1 is complete (the host has read their
// routes), so their activations, scales and router weights are packed here, pair-major.
func (r *resident) encodeG4Phase2Batch(e *Encoder, pool *expertPool, b *g4Batch, grp []*g4Row,
	join func(e *Encoder, rw *g4Row, x2 Buffer), made func(Buffer) Buffer) {
	g := r.g4moe
	n, k, H, I := len(grp), g.topK, r.H, g.moeInter
	slots := make([][]uint32, n)
	for i, rw := range grp {
		slots[i] = rw.slotIdx.U32s()[:k]
	}
	ent := g4BatchEntries(slots, k)
	nEnt := len(ent) / (2 + moeBatchB)
	r.g4BatchGroups++
	for x := range nEnt {
		r.g4BatchMaxCnt = max(r.g4BatchMaxCnt, int(ent[x*(2+moeBatchB)+1]))
	}
	r.g4BatchEnsure(b, n, nEnt, made)
	aq, asc, wgt := b.aq.Int8s(), b.asc.Floats(), b.wgt.Floats()
	for i, rw := range grp {
		copy(aq[i*H:(i+1)*H], rw.mq.Int8s()[:H])
		asc[i] = rw.mSc.Floats()[0]
		copy(wgt[i*k:(i+1)*k], rw.rWgt.Floats()[:k])
	}
	copy(b.ent.U32s(), ent)
	b.uN.SetU32(uint32(n))
	guR, downR := moeBatchR(2*I), moeBatchR(H)
	tilesGU, tilesDown := 2*I/(8*guR), H/(8*downR)
	b.uTilesGU.SetU32(uint32(tilesGU))
	b.uTilesDown.SetU32(uint32(tilesDown))
	e.DispatchTG(r.pMoeBatch[guR], nEnt*tilesGU*256, 256, moeBatchB*H*2,
		pool.guW, pool.guS, b.aq, b.asc, b.gu, r.uH, b.ent, g.uMoeGU, b.uTilesGU, b.uK, b.uOne)
	e.Dispatch(r.pSwRows, n*k*256, 256, b.gu, b.gu, b.dq, b.dsc, g.uMoeInter, r.uAct)
	e.DispatchTG(r.pMoeBatch[downR], nEnt*tilesDown*256, 256, moeBatchB*I*2,
		pool.dW, pool.dS, b.dq, b.dsc, b.dacc, g.uMoeInter, b.ent, r.uH, b.uTilesDown, b.uOne, b.uZero)
	e.Dispatch(r.pMoeCombine, n*H, 256, b.dacc, b.wgt, b.dsc, b.x2, r.uH, b.uK, b.uN)
	for i, rw := range grp {
		join(e, rw, b.x2.At(4*i*H))
	}
}

// g4ExpertBatchOK reports whether the batched phase 2 can take this resident's shapes: both projections in full
// threadgroups, and the gate|up activations of a full entry within threadgroup memory.
func (r *resident) g4ExpertBatchOK() bool {
	g := r.g4moe
	return g4ExpertBatchOn && g != nil && r.pMoeCombine != (Pipeline{}) && moeBatchR(2*g.moeInter) > 0 &&
		moeBatchR(r.H) > 0 && moeBatchB*max(r.H, g.moeInter)*2 <= 32*1024
}
