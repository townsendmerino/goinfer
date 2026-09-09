//go:build cuda && goinfer_testhooks

package cuda

import (
	"fmt"
	"math"
	"testing"

	gpu "github.com/townsendmerino/aikit/gpu"
	"github.com/townsendmerino/goinfer/decoder"
)

// requireCUDAResidentMRoPE is requireCUDAResident's m-RoPE twin: qwen3-4b (that helper's model)
// has no MRopeSection, so rope_kv_mrope_batched never loads on it. Needs a real Qwen2.5-VL
// checkpoint specifically.
func requireCUDAResidentMRoPE(t *testing.T) *cudaResident {
	t.Helper()
	requireHeavyModel(t)
	path := decoder.AssetPathForTest(t, "GOINFER_QWEN25VL_3B")
	mc, err := decoder.Load(path, decoder.Options{Backend: "cuda", Quant: "int4"})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	t.Cleanup(func() { mc.Close() })
	r, ok := mc.ResidentForwardForTest().(*cudaResident)
	if !ok {
		t.Fatal("resident did not engage")
	}
	if !r.mropePrefillReady {
		t.Skip("rope_kv_mrope_batched did not load on this build")
	}
	return r
}

// ropeMRopeArgs builds rope_kv_mrope_batched's argument list — the scalar rope_kv_batched prefix
// plus the three per-row position arrays and the cumulative section boundaries.
func ropeMRopeArgs(q, k, v, invF, kc, vc Buffer, nH, nKV, hd, startPos, rhalf, M int, mscale float32, posT, posH, posW Buffer, sec0, sec1 int32) []gpu.KernelArg {
	return []gpu.KernelArg{
		Arg(q), Arg(k), Arg(v), Arg(invF), Arg(kc), Arg(vc),
		gpu.ArgValue(int32(nH)), gpu.ArgValue(int32(nKV)), gpu.ArgValue(int32(hd)),
		gpu.ArgValue(int32(startPos)), gpu.ArgValue(int32(rhalf)), gpu.ArgValue(int32(M)),
		gpu.ArgValue(mscale),
		Arg(posT), Arg(posH), Arg(posW), gpu.ArgValue(sec0), gpu.ArgValue(sec1),
	}
}

// ropeScalarArgs builds rope_kv_batched's own (scalar-position) argument list — the same prefix,
// no trailing m-RoPE fields.
func ropeScalarArgs(q, k, v, invF, kc, vc Buffer, nH, nKV, hd, startPos, rhalf, M int, mscale float32) []gpu.KernelArg {
	return []gpu.KernelArg{
		Arg(q), Arg(k), Arg(v), Arg(invF), Arg(kc), Arg(vc),
		gpu.ArgValue(int32(nH)), gpu.ArgValue(int32(nKV)), gpu.ArgValue(int32(hd)),
		gpu.ArgValue(int32(startPos)), gpu.ArgValue(int32(rhalf)), gpu.ArgValue(int32(M)),
		gpu.ArgValue(mscale),
	}
}

// TestRopeKVMRoPEBatched_degenerateMatchesScalarKernel pins decoder/rope.go's own claim, at the
// kernel level: "for a TEXT token the three positions are equal, so this reduces EXACTLY to
// applyRoPE(pos[0])" — i.e. when every row's (t,h,w) triple collapses to (startPos+m,startPos+m,
// startPos+m), rope_kv_mrope_batched must be BIT-IDENTICAL to the plain scalar rope_kv_batched on
// the same q/k/v/invFreq, regardless of how sec0/sec1 partition the frequencies (every partition
// picks the same scalar when all three components agree).
func TestRopeKVMRoPEBatched_degenerateMatchesScalarKernel(t *testing.T) {
	r := requireCUDAResidentMRoPE(t)

	const (
		nH, nKV    = 4, 2
		hd         = 8
		rhalf      = 3 // tail = hd - 2*rhalf = 2, exercises the untouched-tail branch too
		startPos   = 5
		M          = 6
		mscale     = 1.0
		sec0, sec1 = 1, 2 // arbitrary nontrivial 3-way split of rhalf=3
	)
	qDim, kvDim := nH*hd, nKV*hd
	nKeys := startPos + M

	q := make([]float32, M*qDim)
	k := make([]float32, M*kvDim)
	v := make([]float32, M*kvDim)
	for i := range q {
		q[i] = float32(math.Sin(float64(i)*0.31)) * 0.6
	}
	for i := range k {
		k[i] = float32(math.Cos(float64(i)*0.19)) * 0.6
		v[i] = float32(math.Sin(float64(i)*0.11)) * 0.4
	}
	invF := make([]float32, rhalf)
	for i := range invF {
		invF[i] = float32(1.0 / math.Pow(10000, float64(i)/float64(rhalf)))
	}
	posT := make([]int32, M)
	for i := range posT {
		posT[i] = int32(startPos + i)
	}

	var qOutA, kOutA, kcA, vcA []float32
	var qOutB, kOutB, kcB, vcB []float32
	err := r.do(func() error {
		mk := func(host []float32) (Buffer, error) {
			b := r.af(len(host))
			return b, gpu.Upload(b, host)
		}
		mki := func(host []int32) (Buffer, error) {
			b := r.ai(len(host))
			return b, gpu.Upload(b, host)
		}
		qA, e := mk(q)
		if e != nil {
			return e
		}
		kA, e := mk(k)
		if e != nil {
			return e
		}
		vA, e := mk(v)
		if e != nil {
			return e
		}
		invFB, e := mk(invF)
		if e != nil {
			return e
		}
		kcAb, vcAb := r.af(nKeys*kvDim), r.af(nKeys*kvDim)

		qB, e := mk(q)
		if e != nil {
			return e
		}
		kB, e := mk(k)
		if e != nil {
			return e
		}
		vB, e := mk(v)
		if e != nil {
			return e
		}
		kcBb, vcBb := r.af(nKeys*kvDim), r.af(nKeys*kvDim)
		posTB, e := mki(posT)
		if e != nil {
			return e
		}
		posHB, e := mki(posT)
		if e != nil {
			return e
		}
		posWB, e := mki(posT)
		if e != nil {
			return e
		}

		ropeN := nH*rhalf + nKV*rhalf + nKV*(hd-2*rhalf)
		cfg := LaunchConfig{GridX: uint32((ropeN + 255) / 256), GridY: uint32(M), GridZ: 1, BlockX: 256, BlockY: 1, BlockZ: 1}

		scalarArgs := ropeScalarArgs(qA, kA, vA, invFB, kcAb, vcAb, nH, nKV, hd, startPos, rhalf, M, mscale)
		if e := r.launch(r.bRopeKV, cfg, scalarArgs...); e != nil {
			return fmt.Errorf("rope_kv_batched: %w", e)
		}
		mropeArgsList := ropeMRopeArgs(qB, kB, vB, invFB, kcBb, vcBb, nH, nKV, hd, startPos, rhalf, M, mscale, posTB, posHB, posWB, sec0, sec1)
		if e := r.launch(r.bRopeKVMRoPE, cfg, mropeArgsList...); e != nil {
			return fmt.Errorf("rope_kv_mrope_batched: %w", e)
		}
		if e := r.stream.Sync(); e != nil {
			return e
		}

		qOutA, kOutA, kcA, vcA = make([]float32, len(q)), make([]float32, len(k)), make([]float32, nKeys*kvDim), make([]float32, nKeys*kvDim)
		qOutB, kOutB, kcB, vcB = make([]float32, len(q)), make([]float32, len(k)), make([]float32, nKeys*kvDim), make([]float32, nKeys*kvDim)
		for _, d := range []struct {
			buf  Buffer
			host []float32
		}{{qA, qOutA}, {kA, kOutA}, {kcAb, kcA}, {vcAb, vcA}, {qB, qOutB}, {kB, kOutB}, {kcBb, kcB}, {vcBb, vcB}} {
			if e := gpu.Download(d.buf, d.host); e != nil {
				return e
			}
		}
		return nil
	})
	if err != nil {
		t.Fatalf("launch: %v", err)
	}

	assertEqual := func(name string, a, b []float32) {
		for i := range a {
			if math.Abs(float64(a[i]-b[i])) > 1e-5 {
				t.Fatalf("%s[%d]: scalar rope_kv_batched=%v, mrope (degenerate) rope_kv_mrope_batched=%v — must be bit-identical when every row's (t,h,w) collapses to the scalar position", name, i, a[i], b[i])
			}
		}
	}
	assertEqual("q", qOutA, qOutB)
	assertEqual("k", kOutA, kOutB)
	assertEqual("kc", kcA, kcB)
	assertEqual("vc", vcA, vcB)
}

// TestRopeKVMRoPEBatched_matchesCPUReference is the adversarial half: per-row (t,h,w) triples that
// genuinely diverge from EACH OTHER and from the naive pos=startPos+m row-sequential formula
// (mirroring what a post-image-block text row's compressed position actually looks like —
// decoder/rope.go's own header on this file), checked against decoder.ApplyMRoPEForTest (the exact
// CPU reference, never reimplemented here) applied per row.
func TestRopeKVMRoPEBatched_matchesCPUReference(t *testing.T) {
	r := requireCUDAResidentMRoPE(t)

	const (
		nH, nKV    = 4, 2
		hd         = 8
		rhalf      = 3
		startPos   = 0
		M          = 4
		mscale     = 1.0
		sec0, sec1 = 1, 2
	)
	qDim, kvDim := nH*hd, nKV*hd
	nKeys := startPos + M

	q := make([]float32, M*qDim)
	k := make([]float32, M*kvDim)
	v := make([]float32, M*kvDim)
	for i := range q {
		q[i] = float32(math.Sin(float64(i)*0.31)) * 0.6
	}
	for i := range k {
		k[i] = float32(math.Cos(float64(i)*0.19)) * 0.6
		v[i] = float32(math.Sin(float64(i)*0.11)) * 0.4
	}
	invF32 := make([]float32, rhalf)
	invF64 := make([]float64, rhalf)
	for i := range invF32 {
		f := 1.0 / math.Pow(10000, float64(i)/float64(rhalf))
		invF32[i], invF64[i] = float32(f), f
	}
	// Deliberately adversarial: every row's own triple, none equal to startPos+m, none equal to
	// each other across components — the exact shape a compressed post-image position takes.
	mropeRows := [][3]int{{2, 9, 40}, {2, 9, 41}, {17, 3, 7}, {30, 30, 30}}
	posT := make([]int32, M)
	posH := make([]int32, M)
	posW := make([]int32, M)
	for i, p := range mropeRows {
		posT[i], posH[i], posW[i] = int32(p[0]), int32(p[1]), int32(p[2])
	}

	var qOut, kcOut, vcOut []float32
	err := r.do(func() error {
		mk := func(host []float32) (Buffer, error) {
			b := r.af(len(host))
			return b, gpu.Upload(b, host)
		}
		mki := func(host []int32) (Buffer, error) {
			b := r.ai(len(host))
			return b, gpu.Upload(b, host)
		}
		qB, e := mk(q)
		if e != nil {
			return e
		}
		kB, e := mk(k)
		if e != nil {
			return e
		}
		vB, e := mk(v)
		if e != nil {
			return e
		}
		invFB, e := mk(invF32)
		if e != nil {
			return e
		}
		kcBb, vcBb := r.af(nKeys*kvDim), r.af(nKeys*kvDim)
		posTB, e := mki(posT)
		if e != nil {
			return e
		}
		posHB, e := mki(posH)
		if e != nil {
			return e
		}
		posWB, e := mki(posW)
		if e != nil {
			return e
		}

		ropeN := nH*rhalf + nKV*rhalf + nKV*(hd-2*rhalf)
		cfg := LaunchConfig{GridX: uint32((ropeN + 255) / 256), GridY: uint32(M), GridZ: 1, BlockX: 256, BlockY: 1, BlockZ: 1}
		args := ropeMRopeArgs(qB, kB, vB, invFB, kcBb, vcBb, nH, nKV, hd, startPos, rhalf, M, mscale, posTB, posHB, posWB, sec0, sec1)
		if e := r.launch(r.bRopeKVMRoPE, cfg, args...); e != nil {
			return fmt.Errorf("rope_kv_mrope_batched: %w", e)
		}
		if e := r.stream.Sync(); e != nil {
			return e
		}
		qOut, kcOut, vcOut = make([]float32, len(q)), make([]float32, nKeys*kvDim), make([]float32, nKeys*kvDim)
		if e := gpu.Download(qB, qOut); e != nil {
			return e
		}
		if e := gpu.Download(kcBb, kcOut); e != nil {
			return e
		}
		return gpu.Download(vcBb, vcOut)
	})
	if err != nil {
		t.Fatalf("launch: %v", err)
	}

	section := []int{sec0, sec1 - sec0, rhalf - sec1}

	// q: applyMRoPEForTest rotates each row in place, heads=nH.
	for m, p := range mropeRows {
		row := append([]float32(nil), q[m*qDim:(m+1)*qDim]...)
		decoder.ApplyMRoPEForTest(row, nH, hd, p, section, invF64, mscale, false)
		got := qOut[m*qDim : (m+1)*qDim]
		for d := range row {
			if math.Abs(float64(row[d]-got[d])) > 1e-4 {
				t.Fatalf("q row %d[%d]: CPU=%v GPU=%v (pos=%v) — resident m-RoPE rotation diverges from decoder.applyMRoPE", m, d, row[d], got[d], p)
			}
		}
	}
	// k: rotated in place then stored into kc at absolute position startPos+m; v copied unrotated.
	for m, p := range mropeRows {
		row := append([]float32(nil), k[m*kvDim:(m+1)*kvDim]...)
		decoder.ApplyMRoPEForTest(row, nKV, hd, p, section, invF64, mscale, false)
		pos := startPos + m
		gotK := kcOut[pos*kvDim : (pos+1)*kvDim]
		for d := range row {
			if math.Abs(float64(row[d]-gotK[d])) > 1e-4 {
				t.Fatalf("k(cache) row %d[%d]: CPU=%v GPU=%v (pos=%v) — resident m-RoPE k-rotation diverges from decoder.applyMRoPE", m, d, row[d], gotK[d], p)
			}
		}
		wantV := v[m*kvDim : (m+1)*kvDim]
		gotV := vcOut[pos*kvDim : (pos+1)*kvDim]
		for d := range wantV {
			if wantV[d] != gotV[d] {
				t.Fatalf("v(cache) row %d[%d]: want %v (unrotated), got %v", m, d, wantV[d], gotV[d])
			}
		}
	}

	// Adversarial pin: the naive row-sequential formula (pos=startPos+m for the rotation angle,
	// not just the store index) must NOT match — otherwise this test could pass on a kernel that
	// silently ignored posT/posH/posW and fell back to rope_kv_batched's own scalar formula.
	naiveDiffers := false
	for m := range mropeRows {
		if mropeRows[m][0] != startPos+m || mropeRows[m][1] != startPos+m || mropeRows[m][2] != startPos+m {
			naiveDiffers = true
			break
		}
	}
	if !naiveDiffers {
		t.Fatal("test setup: every row's mrope position equals the naive startPos+m formula — this test cannot distinguish the two kernels, fix the fixture")
	}
}
