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
// Isolates the piece a device-argmax kernel would remove: RunNoLogits skips the LM-head GEMV
// dispatch, the staging copy, AND the MapAsync/readback entirely, so Run-vs-RunNoLogits bounds what
// ANY logits-avoiding scheme (device argmax included) could recover.
//
// RETRACTION, 2026-09-21 (docs/QUEUE.md G38-CORRECTED): the first version of this test read the
// delta at 81-85% of the token and reported an "LM-head GEMV 10-17x over roofline" finding — WRONG,
// and wrong for a reason worth stating plainly: RunNoLogits calls c.device.Poll(false, nil) —
// NON-BLOCKING — while Run() calls Poll(true, nil). Timing RunNoLogits back-to-back without an
// explicit blocking poll measures CPU-side submission only, not GPU completion, so the very first
// delta compared "wait for everything" against "don't wait at all" — an apples-to-oranges bug in
// THIS test, not in RunNoLogits itself (whose real callers do not need synchronous timing). Fixed
// by adding an explicit c.device.Poll(true, nil) after RunNoLogits and alternating small blocks of
// each arm (not two long back-to-back runs) so drift cannot bias one side. The corrected delta is
// 4.7-4.8% of the token, reproduced across two independent runs — the on-device-argmax item really
// is low-value, but not for the reason first reported, and the "LM-head GEMV" finding is retracted
// in full; it does not exist.
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
	const perRound = 40
	const rounds = 8 // 320 calls total per arm, alternating in small blocks so drift cannot bias one arm
	pos := 8
	var full, noLog []float64
	for round := 0; round < rounds; round++ {
		full = append(full, timeIt(perRound, func() error { _, e := r.Run(emb, pos, pos); pos++; return e })...)
		noLog = append(noLog, timeIt(perRound, func() error {
			if e := r.RunNoLogits(emb, pos, pos); e != nil {
				return e
			}
			pos++
			c.device.Poll(true, nil)
			return nil
		})...)
	}

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
	copyMapOnly := timeIt(perRound*rounds, func() error {
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
