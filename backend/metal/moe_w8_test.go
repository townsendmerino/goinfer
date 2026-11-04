//go:build darwin && goinfer_testhooks

package metal

import (
	"math"
	"math/rand"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestMoEW8_X1_kernelsMatchReference is MoE int8's gate X1 (docs/tasks/task-metal-int8-2026-10.md, "Slice 4: MoE
// int8"): gemv_w8a8_moe (overwrite) and gemv_w8a8_moe_wacc (weighted accumulate) against a Go reference, bit for bit,
// over a stacked buffer of experts, every slot of a routed set, at the Qwen1.5-MoE expert shapes (gate|up K = 2048,
// down K = 1408). Compiled precise, as production compiles a native int8 model (w8PreciseMath). The reference is
// float32(Σ int8·int8)·aScale·wScale, rounded after each product, then out + wgt·y.
func TestMoEW8_X1_kernelsMatchReference(t *testing.T) {
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Skipf("no metal device: %v", err)
	}
	defer d.ReleaseObjects()
	defer d.ReleaseAll()
	lib, err := d.CompileLibraryPrecise(allKernels, MSL3_1)
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	pGU, err := d.NewComputePipeline(lib, "gemv_w8a8_moe")
	if err != nil {
		t.Fatal(err)
	}
	pD, err := d.NewComputePipeline(lib, "gemv_w8a8_moe_wacc")
	if err != nil {
		t.Fatal(err)
	}
	q := d.NewCommandQueue()
	rng := rand.New(rand.NewSource(808))
	const nE = 6
	checked := 0
	for _, sh := range []struct{ rows, K int }{{2 * 1408, 2048}, {2048, 1408}, {128, 512}} {
		rows, K := sh.rows, sh.K
		codes := make([]int8, nE*rows*K)
		for i := range codes {
			codes[i] = int8(rng.Intn(255) - 127)
		}
		sct := make([]float32, nE*rows)
		for i := range sct {
			sct[i] = rng.Float32()*0.02 + 1e-4
		}
		a := make([]int8, K)
		for i := range a {
			a[i] = int8(rng.Intn(255) - 127)
		}
		asc := float32(0.013)
		idx := []uint32{4, 1, 5}
		wgt := []float32{0.5, 0.3, 0.2}
		wq, ws, aq, aS := NewBufferInt8(d, codes), NewBufferFloats(d, sct), NewBufferInt8(d, a), NewBufferFloats(d, []float32{asc})
		bIdx, bWgt, uK, uRows := NewBufferUint32s(d, idx), NewBufferFloats(d, wgt), NewBufferU32(d, uint32(K)), NewBufferU32(d, uint32(rows))
		ref := func(slot, row int) float32 {
			wr := int(idx[slot])*rows + row
			acc := int32(0)
			for k := range K {
				acc += int32(codes[wr*K+k]) * int32(a[k])
			}
			y := float32(acc) * asc
			return y * sct[wr]
		}
		resid := make([]float32, rows)
		for i := range resid {
			resid[i] = rng.Float32()*2 - 1
		}
		out := NewBufferFloats(d, resid)
		want := append([]float32(nil), resid...)
		for slot := range idx {
			// overwrite form, this slot alone
			gu := d.NewBufferLen(rows)
			e := q.Begin()
			e.DispatchTG(pGU, rows*32, 256, K*2, wq, ws, aq, aS, gu, uK, bIdx, NewBufferU32(d, uint32(slot)), uRows)
			e.End()
			got := gu.Floats()[:rows]
			for row := range rows {
				if r := ref(slot, row); math.Float32bits(got[row]) != math.Float32bits(r) {
					t.Fatalf("gemv_w8a8_moe rows %d K %d slot %d row %d: %v, reference %v", rows, K, slot, row, got[row], r)
				}
			}
			// weighted accumulate into the shared residual, slot after slot
			e = q.Begin()
			e.DispatchTG(pD, rows*32, 256, K*2, wq, ws, aq, aS, out, uK, bIdx, bWgt, NewBufferU32(d, uint32(slot)), uRows)
			e.End()
			for row := range rows {
				want[row] = want[row] + wgt[slot]*ref(slot, row)
			}
			checked += 2 * rows
		}
		got := out.Floats()[:rows]
		for row := range rows {
			if math.Float32bits(got[row]) != math.Float32bits(want[row]) {
				t.Fatalf("gemv_w8a8_moe_wacc rows %d K %d row %d: %v, reference %v", rows, K, row, got[row], want[row])
			}
		}
	}
	t.Logf("%d outputs bit-identical to the reference", checked)
}

// TestMoEW8_X2_assembly is gate X2: (a) a tiny qwen2_moe with identical experts and a zeroed shared expert against the
// dense FFN it equals, both on the native int8 path, minimum logit cosine >= 0.9999 and argmax 16/16; (b) a tiny
// qwen2_moe with distinct experts and a live shared expert, Metal native int8 against the CPU at int8int8, beside the
// int4 re-quant arm against the CPU: the native arm's minimum cosine >= the re-quant arm's, and its hard flips (gaps
// over 3%) <= the re-quant arm's.
func TestMoEW8_X2_assembly(t *testing.T) {
	if _, err := CreateSystemDefaultDevice(); err != nil {
		t.Skipf("no metal device: %v", err)
	}
	prev := nativeInt8MoE
	defer func() { nativeInt8MoE = prev }()
	nativeInt8MoE = true
	build := func(dir string) *resident {
		m, err := decoder.Load(dir, decoder.Options{Quant: "int8int8"})
		if err != nil {
			t.Fatalf("load %s: %v", dir, err)
		}
		t.Cleanup(func() { m.Close() })
		r, err := buildResident(m)
		if err != nil {
			t.Fatalf("resident %s: %v", dir, err)
		}
		t.Cleanup(func() { r.Close() })
		return r
	}
	ids := []int{1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49, 53, 57, 61}
	t.Run("identical-vs-dense", func(t *testing.T) {
		w := genTinyWeights(rand.New(rand.NewSource(1234)))
		moeDir, denseDir := t.TempDir(), t.TempDir()
		writeMoEIdentical(t, moeDir, w)
		writeDense(t, denseDir, w)
		moeR, denseR := build(moeDir), build(denseDir)
		if moeR.moe == nil || !moeR.w8 || !moeR.moe.w8 || !denseR.w8 {
			t.Fatalf("native int8: moe %v (experts %v), dense %v", moeR.w8, moeR.moe != nil && moeR.moe.w8, denseR.w8)
		}
		minCos, mism := 1.0, 0
		for pos, tok := range ids {
			a := append([]float32(nil), moeR.Forward(tok, pos)...)
			b := denseR.Forward(tok, pos)
			minCos = math.Min(minCos, cosF(a, b))
			if argmaxF(a) != argmaxF(b) {
				mism++
			}
		}
		t.Logf("X2a: MoE int8 (identical experts) against the dense int8 FFN: min cosine %.6f, argmax %d/%d", minCos, len(ids)-mism, len(ids))
		if minCos < 0.9999 || mism != 0 {
			t.Errorf("X2a fails: min cosine %.6f (bar 0.9999), %d argmax mismatches", minCos, mism)
		}
	})
	t.Run("distinct-vs-cpu", func(t *testing.T) {
		dir := t.TempDir()
		writeMoEDistinct(t, dir, genTinyWeights(rand.New(rand.NewSource(77))), 8, 3, rand.New(rand.NewSource(78)))
		cpuM, err := decoder.Load(dir, decoder.Options{Quant: "int8int8"})
		if err != nil {
			t.Fatal(err)
		}
		defer cpuM.Close()
		_, nL, _, nKV, hd, _, _ := cpuM.Dims()
		cache := decoder.NewKVCache(nL, nKV, hd, 0, 256, nil)
		var cpu [][]float32
		for _, tok := range ids {
			l, err := cpuM.ForwardForTest(tok, cache)
			if err != nil {
				t.Fatal(err)
			}
			cpu = append(cpu, append([]float32(nil), l...))
		}
		arm := func(native bool) (float64, int) {
			nativeInt8MoE = native
			defer func() { nativeInt8MoE = true }()
			r := build(dir)
			if r.w8 != native {
				t.Fatalf("native int8 = %v, want %v", r.w8, native)
			}
			minCos, hard := 1.0, 0
			for pos, tok := range ids {
				g := r.Forward(tok, pos)
				minCos = math.Min(minCos, cosF(g, cpu[pos]))
				if ga, ca := argmaxF(g), argmaxF(cpu[pos]); ga != ca {
					lo, hi := cpu[pos][0], cpu[pos][0]
					for _, v := range cpu[pos] {
						lo, hi = min(lo, v), max(hi, v)
					}
					if float64(cpu[pos][ca]-cpu[pos][ga])/(float64(hi-lo)+1e-9) > 0.03 {
						hard++
					}
				}
			}
			return minCos, hard
		}
		natCos, natHard := arm(true)
		reqCos, reqHard := arm(false)
		t.Logf("X2b: against the CPU int8int8: native min cosine %.6f, hard flips %d; re-quant min cosine %.6f, hard flips %d", natCos, natHard, reqCos, reqHard)
		if natCos < reqCos || natHard > reqHard {
			t.Errorf("X2b fails: native %.6f / %d against re-quant %.6f / %d", natCos, natHard, reqCos, reqHard)
		}
	})
}
