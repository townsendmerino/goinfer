//go:build cuda

package kernels

import (
	"fmt"
	"math"
	"testing"

	gc "github.com/eitamring/gocudrv/cuda"
)

// G2-K, OP1 on CUDA (docs/tasks/task-forward-architecture-2026-10.md §4.6a): each kernel that is built with the attention softcap computes
// cap·tanh(score/cap) on the scaled score, before the softmax, against a float64 reference. The three covered kernels are attention
// (glue), attn_batched (prefill_batched, which decode takes at M = 1) and the split-KV chain (splitkv_scores, then softmax and V-sum, which
// read the stored capped scores), each at f32 and f16 KV.
//
// The inputs are built so that the cap acts: every case asserts, from the reference, that scores lie on both sides of ±cap before it
// runs, because a tiny model's scores sit far inside the cap where cap·tanh(s/cap) is nearly the identity (§4.6, "G2-D is why the tiny
// fixture is not enough alone"). A peaked case puts one key at about three caps.
//
// Bar: cosine >= 0.9999 and max abs error <= 1e-2, TestAttention_ShippedKernelShapes' bar.
//
//	go test -count=1 -tags 'cuda goinfer_testhooks' -run '^TestAttnSoftcap_' -v ./cuda/

var (
	adGlueCap    = [2]adMod{{"glue_cap", glueCapPTX}, {"glue_kv16_cap", glueKV16CapPTX}}
	adPrefillCap = [2]adMod{{"prefill_batched_cap", prefillBatchedCapPTX}, {"prefill_batched_kv16_cap", prefillBatchedKV16CapPTX}}
	adSplitCap   = [2]adMod{{"decode_splitkv_cap", decodeSplitKVCapPTX}, {"decode_splitkv_kv16_cap", decodeSplitKVKV16CapPTX}}
)

// capInputs returns q rows, K and V (as the kernel will read them: f16-rounded for an f16 cache) for a case whose scores cross the cap.
// amp is chosen from the cap (a score's spread is about amp²/3). peaked makes key keyAt equal to q scaled to a score of three caps.
type capInputs struct {
	q, k, v []float32
}

func capMakeInputs(nRows, nTot, nH, nKV, hd int, cap float32, peaked bool, keyAt int, f16 bool, seed uint32) capInputs {
	amp := float32(math.Sqrt(3 * float64(cap) * 1.5))
	in := capInputs{q: adRand(nRows*nH*hd, seed, amp), k: adRand(nTot*nKV*hd, seed+1, amp), v: adRand(nTot*nKV*hd, seed+2, 1)}
	if peaked && keyAt < nTot {
		// key keyAt, kv head 0: parallel to head 0's last row's query, scaled so scale·q·k = 3·cap
		qrow := in.q[(nRows-1)*nH*hd : (nRows-1)*nH*hd+hd]
		var qq float64
		for _, x := range qrow {
			qq += float64(x) * float64(x)
		}
		f := 3 * float64(cap) / (float64(adScale(hd)) * qq)
		for d := range hd {
			in.k[keyAt*nKV*hd+d] = float32(f * float64(qrow[d]))
		}
	}
	if f16 {
		for _, s := range [][]float32{in.k, in.v} {
			for i, x := range s {
				s[i] = f16BitsToF32(f32ToF16RNE(x))
			}
		}
	}
	return in
}

// capReference is one query row's attention output in float64 with the softcap, and the share of its scores past ±cap and the largest |score|.
// sink is a learned per-head logit added to the softmax as the kernels add it (not capped), or nil.
func capReference(q []float32, in capInputs, nH, nKV, hd, nKeys, window int, cap float64, sink []float32) (out []float64, past, total int, maxAbs float64) {
	scale := float64(adScale(hd))
	winStart := 0
	if window > 0 && nKeys > window {
		winStart = nKeys - window
	}
	out = make([]float64, nH*hd)
	for h := range nH {
		kvh := h / (nH / nKV)
		sc := make([]float64, 0, nKeys-winStart)
		mx := math.Inf(-1)
		for s := winStart; s < nKeys; s++ {
			var d float64
			for j := range hd {
				d += float64(q[h*hd+j]) * float64(in.k[s*nKV*hd+kvh*hd+j])
			}
			raw := d * scale
			total++
			if math.Abs(raw) > cap {
				past++
			}
			maxAbs = math.Max(maxAbs, math.Abs(raw))
			c := cap * math.Tanh(raw/cap)
			sc = append(sc, c)
			mx = math.Max(mx, c)
		}
		if sink != nil {
			mx = math.Max(mx, float64(sink[h]))
		}
		var denom float64
		for i := range sc {
			sc[i] = math.Exp(sc[i] - mx)
			denom += sc[i]
		}
		if sink != nil {
			denom += math.Exp(float64(sink[h]) - mx)
		}
		for j := range hd {
			var a float64
			for i, p := range sc {
				a += p * float64(in.v[(winStart+i)*nKV*hd+kvh*hd+j])
			}
			out[h*hd+j] = a / denom
		}
	}
	return
}

func capCompare(t *testing.T, id string, got []float32, want []float64) {
	t.Helper()
	var dot, gg, ww, maxAbs float64
	for i := range got {
		g, w := float64(got[i]), want[i]
		dot += g * w
		gg += g * g
		ww += w * w
		maxAbs = math.Max(maxAbs, math.Abs(g-w))
	}
	cos := dot / (math.Sqrt(gg)*math.Sqrt(ww) + 1e-300)
	if !(cos >= 0.9999) || !(maxAbs <= 1e-2) {
		t.Errorf("%s: cosine %.8f, max abs error %.3e (bar: cosine >= 0.9999 and max abs <= 1e-2)", id, cos, maxAbs)
	}
}

func TestAttnSoftcap_kernelsAgainstFloat64(t *testing.T) {
	r, _ := newAdRig(t)
	cases, void := 0, 0
	for kvi := range 2 {
		f16 := kvi == 1
		for _, cap := range []float32{50, 2} {
			for _, hd := range []int{64, 256} {
				for _, peaked := range []bool{false, true} {
					for _, window := range []int{0, 100} {
						for _, sink := range []bool{false, true} {
							nH, nKV := 4, 2
							id := func(k string, s ...any) string {
								return fmt.Sprintf("%s/%s/cap%g/hd%d/peaked=%v/win%d/sink=%v/%v", k, adKVNames[kvi], cap, hd, peaked, window, sink, s)
							}
							var sinkVals []float32
							if sink {
								sinkVals = adRand(nH, 77, 2) // adSink's values
							}
							sinkArg := func() gc.KernelArg {
								if !sink {
									return gc.ArgDevicePtr(0)
								}
								return gc.Arg(r.f32(sinkVals))
							}

							// attention (glue) and the split-KV chain: one query, nKeys keys either side of the 128-thread tile
							for _, nKeys := range []int{130, 300} {
								in := capMakeInputs(1, nKeys, nH, nKV, hd, cap, peaked, 5, f16, 101)
								want, past, total, maxAbs := capReference(in.q, in, nH, nKV, hd, nKeys, window, float64(cap), sinkVals)
								if past*20 < total || maxAbs < 1.5*float64(cap) {
									t.Errorf("VOID %s: only %d of %d scores past ±cap, largest |score| %.2f: the cap does not act on this input", id("decode", nKeys), past, total, maxAbs)
									void++
									continue
								}
								q := r.f32(in.q)
								kc, vc := r.kv(in.k, f16), r.kv(in.v, f16)
								nWin := nKeys
								winStart := 0
								if window > 0 && nKeys > window {
									winStart, nWin = nKeys-window, window
								}
								if !sink { // attention carries no sink
									ctx := r.zeros(nH * hd)
									r.launch(adGlueCap[kvi], "attention", adCfg(nH, 1, 1, 128, (nWin+128)*4), gc.Arg(q), kc, vc, r.kvt(nKeys), adI(nH), adI(nKV), adI(hd), adI(nKeys),
										gc.ArgValue(adScale(hd)), adI(window), gc.Arg(ctx), gc.ArgValue(cap))
									capCompare(t, id("attention", nKeys), r.read(ctx, nH*hd), want)
									cases++
								}
								sc, inv, ctx := r.zeros(nH*nWin), r.zeros(nH), r.zeros(nH*hd)
								r.launch(adSplitCap[kvi], "splitkv_scores", adCfg(nH, (nWin+127)/128, 1, 128, hd*4), gc.Arg(q), kc, r.kvt(nKeys), adI(nH), adI(nKV), adI(hd),
									adI(winStart), adI(nKeys), gc.ArgValue(adScale(hd)), gc.Arg(sc), adI(nWin), gc.ArgValue(cap))
								r.launch(adSplitCap[kvi], "splitkv_softmax", adCfg(nH, 1, 1, 128, 128*4), gc.Arg(sc), adI(nH), adI(nWin), gc.Arg(inv), sinkArg())
								r.launch(adSplitCap[kvi], "splitkv_vsum", adCfg(nH, (hd+31)/32, 1, 32, 0), gc.Arg(sc), vc, r.kvt(nKeys), gc.Arg(inv), adI(nH), adI(nKV), adI(hd),
									adI(winStart), adI(nKeys), adI(nWin), gc.Arg(ctx))
								capCompare(t, id("splitkv", nKeys), r.read(ctx, nH*hd), want)
								cases++
								r.release()
							}

							// attn_batched: M rows at consecutive positions, each with its own causal key count
							for _, sm := range [][2]int{{0, 17}, {130, 5}} {
								startPos, M := sm[0], sm[1]
								nTot := startPos + M
								in := capMakeInputs(M, nTot, nH, nKV, hd, cap, peaked, 5, f16, 201)
								// the largest-|score| row is the last: check the cap acts there, compare every row
								wants := make([][]float64, M)
								past, total, maxAbs := 0, 0, 0.0
								for m := range M {
									w, p, tt, mx := capReference(in.q[m*nH*hd:(m+1)*nH*hd], in, nH, nKV, hd, startPos+m+1, window, float64(cap), sinkVals)
									wants[m], past, total, maxAbs = w, past+p, total+tt, math.Max(maxAbs, mx)
								}
								if past*20 < total || maxAbs < 1.5*float64(cap) {
									t.Errorf("VOID %s: only %d of %d scores past ±cap, largest |score| %.2f", id("batched", startPos, M), past, total, maxAbs)
									void++
									continue
								}
								q := r.f32(in.q)
								kc, vc := r.kv(in.k, f16), r.kv(in.v, f16)
								ctx := r.zeros(M * nH * hd)
								maxWin := nTot
								if window > 0 && window < maxWin {
									maxWin = window
								}
								r.launch(adPrefillCap[kvi], "attn_batched", adCfg(nH, M, 1, 128, (maxWin+128)*4), gc.Arg(q), kc, vc, r.kvt(nTot),
									adI(nH), adI(nKV), adI(hd), adI(startPos), gc.ArgValue(adScale(hd)), adI(window), adI(M), gc.Arg(ctx), sinkArg(), gc.ArgValue(cap))
								got := r.read(ctx, M*nH*hd)
								for m := range M {
									capCompare(t, id("attn_batched", startPos, M, "row", m), got[m*nH*hd:(m+1)*nH*hd], wants[m])
								}
								cases++
								r.release()
							}
							r.release()
						}
					}
				}
			}
		}
	}
	if cases < 200 {
		t.Fatalf("only %d cases ran (%d void): the matrix did not run", cases, void)
	}
	if void > 0 {
		t.Fatalf("%d case(s) void: the inputs did not make the cap act", void)
	}
	t.Logf("%d kernel cases inside the bar against float64", cases)
}
