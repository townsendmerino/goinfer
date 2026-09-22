//go:build gpu && goinfer_testhooks

package gpu

import (
	"fmt"
	"math"
	"math/rand"
	"os"
	"sort"
	"testing"
	"time"

	"github.com/oliverbestmann/webgpu/wgpu"
	"github.com/townsendmerino/goinfer/decoder"
)

// TestDecodeArgmaxHeadroom is R10's "measure before building" step for the greedy-decode item named
// in docs/tasks/red-october.md R10 ("on-device argmax for the greedy path with a K-entry MapAsync ...
// so the 608 KB readback goes"): ForwardSample's greedy branch (gpu/resident_sample.go) still calls
// Run() — full LM-head GEMV + a vocab*4-byte staging copy + MapAsync + a host linear scan — where a
// device-side argmax kernel would need only the GEMV (unavoidable either way) plus a tiny readback.
//
// TestDecodeTWE_split already found TSync at 91% of the token on the real 1.5B, but TSync there is
// the WHOLE GPU-blocked wait (real trunk compute + the LM-head GEMV + the copy/map), not isolated to
// the copy/map specifically -- this repo has a retired mistake built on exactly that kind of
// conflation (an analogy that transfers a diagnosis also transfers what it was measuring). This
// isolates the achievable UPPER BOUND: RunNoLogits skips the LM-head GEMV dispatch, the staging copy,
// AND the MapAsync/readback entirely, so Run-vs-RunNoLogits bounds what ANY logits-avoiding scheme
// (device argmax included) could recover -- a real device-argmax kernel still pays the GEMV, so it
// recovers LESS than this delta, never more.
//
//	GOINFER_DECODE_GGUF=~/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf \
//	  go test -tags 'gpu goinfer_testhooks' ./gpu/ -run TestDecodeArgmaxHeadroom -v
func TestDecodeArgmaxHeadroom(t *testing.T) {
	requireHeavyModel(t)
	path := os.Getenv("GOINFER_DECODE_GGUF")
	if path == "" {
		path = os.ExpandEnv("$HOME/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf")
	}
	if _, err := os.Stat(path); err != nil {
		t.Skipf("model not found: %s (set GOINFER_DECODE_GGUF)", path)
	}
	m, err := decoder.Load(path, decoder.Options{Backend: "webgpu", Quant: "int8int8"})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	defer m.Close()
	if !m.ResidentActive() {
		t.Skipf("model not GPU-resident")
	}
	rd, ok := m.ResidentForwardForTest().(*residentDecoder)
	if !ok {
		t.Skipf("resident forward is not the gpu DecodeRunner (%T)", m.ResidentForwardForTest())
	}
	_, _, _, _, _, _, vocab := m.Dims()
	hidden, _, _, _, _, _, _ := m.Dims()
	rng := rand.New(rand.NewSource(1))
	emb := make([]float32, hidden)
	for i := range emb {
		emb[i] = float32(rng.NormFloat64()) * 0.5
	}
	r := rd.runner
	c := rd.c

	median := func(xs []float64) float64 {
		s := append([]float64(nil), xs...)
		sort.Float64s(s)
		return s[len(s)/2]
	}
	timeIt := func(n int, f func() error) []float64 {
		out := make([]float64, n)
		for i := range out {
			t0 := time.Now()
			if err := f(); err != nil {
				t.Fatalf("call %d: %v", i, err)
			}
			out[i] = float64(time.Since(t0).Microseconds())
		}
		return out
	}

	rd.Reset()
	for i := range 8 {
		if _, err := rd.Forward(emb, i); err != nil {
			t.Fatalf("warmup: %v", err)
		}
	}
	const N = 200
	full := timeIt(N, func() error { _, e := r.Run(emb, 8+N, 8+N); return e })
	// same trunk state, continuing positions -- RunNoLogits still advances the KV cache correctly,
	// this only compares per-token WALL COST, not correctness (Run's own parity gates cover that).
	noLog := timeIt(N, func() error { return r.RunNoLogits(emb, 8+2*N, 8+2*N) })

	mFull, mNoLog := median(full), median(noLog)
	t.Logf("vocab=%d hidden=%d", vocab, hidden)
	t.Logf("Run()        median %8.1f us/token", mFull)
	t.Logf("RunNoLogits  median %8.1f us/token", mNoLog)
	t.Logf("delta (LM-head GEMV + %.0f KB copy + MapAsync + host scan, upper bound on the argmax-kernel win): %.1f us = %.1f%% of the token",
		float64(vocab*4)/1024, mFull-mNoLog, (mFull-mNoLog)/mFull*100)

	// Split the delta: re-copy+map+read r.lastLogits (whatever a prior Run() left there) with NO new
	// dispatch at all, so this isolates the copy(vocab*4)+MapAsync+host-scan cost alone, with the
	// LM-head GEMV's own compute cost excluded entirely -- the piece a device-argmax kernel actually
	// removes, vs the GEMV cost it does NOT (a real kernel still runs the GEMV; only the reduction and
	// the readback size change).
	copyMapOnly := timeIt(N, func() error {
		enc, e := c.device.TryCreateCommandEncoder(nil)
		if e != nil {
			return e
		}
		defer enc.Release()
		if e := enc.TryCopyBufferToBuffer(r.lastLogits, 0, r.stag, 0, uint64(r.vocab*4)); e != nil {
			return e
		}
		cmd, e := enc.TryFinish(nil)
		if e != nil {
			return e
		}
		defer cmd.Release()
		c.queue.Submit(cmd)
		var st wgpu.MapAsyncStatus
		if e := r.stag.TryMapAsync(wgpu.MapModeRead, 0, uint64(r.vocab*4), func(s wgpu.MapAsyncStatus) { st = s }); e != nil {
			return e
		}
		c.device.Poll(true, nil)
		if st != wgpu.MapAsyncStatusSuccess {
			return fmt.Errorf("map failed: %v", st)
		}
		best, bi := float32(math.Inf(-1)), 0
		for i, v := range wgpu.FromBytes[float32](r.stag.GetMappedRange(0, uint(r.vocab*4))) {
			if v > best {
				best, bi = v, i
			}
		}
		_ = bi
		r.stag.TryUnmap()
		return nil
	})
	mCopyMap := median(copyMapOnly)
	t.Logf("copy+MapAsync+host-scan ALONE (no new GEMV dispatch): %.1f us = %.1f%% of the delta above, %.1f%% of the token",
		mCopyMap, mCopyMap/(mFull-mNoLog)*100, mCopyMap/mFull*100)
	t.Logf("  -> implied LM-head GEMV's OWN cost (delta minus this): %.1f us = %.1f%% of the token",
		(mFull-mNoLog)-mCopyMap, ((mFull-mNoLog)-mCopyMap)/mFull*100)
}
