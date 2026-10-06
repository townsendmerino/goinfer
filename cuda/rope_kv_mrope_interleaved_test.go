//go:build cuda && goinfer_testhooks

package cuda

import (
	"fmt"
	"math"
	"os"
	"path/filepath"
	"testing"

	gpu "github.com/townsendmerino/aikit/gpu"
	"github.com/townsendmerino/goinfer/decoder"
)

// requireCUDAResidentHybridVL loads the tiny Qwen3.5 vision-language fixture (a Gated-DeltaNet hybrid with an interleaved m-RoPE
// section) on CUDA. Unlike requireCUDAResidentMRoPE it needs no multi-GB checkpoint, so it runs wherever a GPU does.
func requireCUDAResidentHybridVL(t *testing.T) (*cudaResident, *decoder.Model) {
	t.Helper()
	requireCUDADevice(t)
	path := filepath.Join("..", "testdata", "qwen35vl-tiny")
	if _, err := os.Stat(filepath.Join(path, "model.safetensors")); err != nil {
		t.Skipf("no fixture at %s", path)
	}
	m, err := decoder.Load(path, decoder.Options{Backend: "cuda", Quant: "int4", ResidentContext: 4096})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	t.Cleanup(func() { m.Close() })
	r, ok := m.ResidentForwardForTest().(*cudaResident)
	if !ok {
		t.Skipf("not CUDA-resident (%T)", m.ResidentForwardForTest())
	}
	return r, m
}

// TestRopeKVMRoPEBatched_interleavedModeMatchesCPUReference is the interleaved twin of
// TestRopeKVMRoPEBatched_matchesCPUReference: rope_kv_mrope_batched in mode 1 (Qwen3-VL / Qwen3.5's per-frequency-index component
// layout, decoder.mropeComponentInterleaved) against decoder.ApplyMRoPEForTest(interleaved=true), on rows whose (t,h,w) triples all
// differ. Mode 0 is run beside it against the contiguous reference so the shared kernel is held to both rules, and each mode's
// result must DIFFER from the other's on these rows (otherwise the test could pass on a kernel that ignores its mode argument).
// It also asserts the resident chose mode 1, with the boundaries 3*section[1] and 3*section[2], for this model.
func TestRopeKVMRoPEBatched_interleavedModeMatchesCPUReference(t *testing.T) {
	r, m := requireCUDAResidentHybridVL(t)
	if !r.mropePrefillReady {
		t.Skip("rope_kv_mrope_batched did not load on this build")
	}
	if sec := m.MRopeSectionResident(); r.mropeMode != 1 || r.mropeSec0 != int32(3*sec[1]) || r.mropeSec1 != int32(3*sec[2]) {
		t.Fatalf("the resident for an interleaved model has mode %d sec0 %d sec1 %d, want mode 1 and 3*section[1], 3*section[2] of %v", r.mropeMode, r.mropeSec0, r.mropeSec1, sec)
	}

	const (
		nH, nKV  = 4, 2
		hd       = 20
		rhalf    = 8 // partial rotary: tail = 4 untouched dims, like Qwen3.5's 64 of 256
		startPos = 0
		M        = 4
		mscale   = 1.0
	)
	section := []int{3, 3, 2} // sums to rhalf; interleaved boundaries are 3*3 = 9 and 3*2 = 6
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
	rows := [][3]int{{2, 9, 40}, {2, 9, 41}, {17, 3, 7}, {30, 22, 11}}
	posT, posH, posW := make([]int32, M), make([]int32, M), make([]int32, M)
	for i, p := range rows {
		posT[i], posH[i], posW[i] = int32(p[0]), int32(p[1]), int32(p[2])
	}

	run := func(mode, sec0, sec1 int32) (qOut, kcOut []float32) {
		err := r.do(func() error {
			mk := func(host []float32) (Buffer, error) { b := r.af(len(host)); return b, gpu.Upload(b, host) }
			mki := func(host []int32) (Buffer, error) { b := r.ai(len(host)); return b, gpu.Upload(b, host) }
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
			kcB, vcB := r.af(nKeys*kvDim), r.af(nKeys*kvDim)
			pT, e := mki(posT)
			if e != nil {
				return e
			}
			pH, e := mki(posH)
			if e != nil {
				return e
			}
			pW, e := mki(posW)
			if e != nil {
				return e
			}
			ropeN := nH*rhalf + nKV*rhalf + nKV*(hd-2*rhalf)
			cfg := LaunchConfig{GridX: uint32((ropeN + 255) / 256), GridY: uint32(M), GridZ: 1, BlockX: 256, BlockY: 1, BlockZ: 1}
			args := ropeMRopeArgs(qB, kB, vB, invFB, kcB, vcB, nH, nKV, hd, startPos, rhalf, M, mscale, pT, pH, pW, sec0, sec1, mode)
			if e := r.launch(r.bRopeKVMRoPE, cfg, args...); e != nil {
				return fmt.Errorf("rope_kv_mrope_batched: %w", e)
			}
			if e := r.stream.Sync(); e != nil {
				return e
			}
			qOut, kcOut = make([]float32, len(q)), make([]float32, nKeys*kvDim)
			if e := gpu.Download(qB, qOut); e != nil {
				return e
			}
			return gpu.Download(kcB, kcOut)
		})
		if err != nil {
			t.Fatalf("launch (mode %d): %v", mode, err)
		}
		return qOut, kcOut
	}
	check := func(name string, interleaved bool, qOut, kcOut []float32) {
		for m, p := range rows {
			rq := append([]float32(nil), q[m*qDim:(m+1)*qDim]...)
			decoder.ApplyMRoPEForTest(rq, nH, hd, p, section, invF64, mscale, interleaved)
			for d := range rq {
				if g := qOut[m*qDim+d]; math.Abs(float64(rq[d]-g)) > 1e-4 {
					t.Fatalf("%s q row %d[%d]: CPU=%v GPU=%v (pos=%v)", name, m, d, rq[d], g, p)
				}
			}
			rk := append([]float32(nil), k[m*kvDim:(m+1)*kvDim]...)
			decoder.ApplyMRoPEForTest(rk, nKV, hd, p, section, invF64, mscale, interleaved)
			for d := range rk {
				if g := kcOut[(startPos+m)*kvDim+d]; math.Abs(float64(rk[d]-g)) > 1e-4 {
					t.Fatalf("%s k row %d[%d]: CPU=%v GPU=%v (pos=%v)", name, m, d, rk[d], g, p)
				}
			}
		}
	}
	qC, kcC := run(0, int32(section[0]), int32(section[0]+section[1]))
	check("mode 0 (contiguous)", false, qC, kcC)
	qI, kcI := run(1, int32(3*section[1]), int32(3*section[2]))
	check("mode 1 (interleaved)", true, qI, kcI)
	same := true
	for i := range qC {
		if math.Abs(float64(qC[i]-qI[i])) > 1e-4 {
			same = false
			break
		}
	}
	if same {
		t.Fatal("mode 0 and mode 1 gave the same rotation on rows whose components all differ: the test cannot tell the layouts apart, so it proves nothing about the mode argument")
	}
}
