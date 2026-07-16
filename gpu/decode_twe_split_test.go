//go:build gpu

package gpu

import (
	"math"
	"math/rand"
	"os"
	"sort"
	"testing"
	"time"

	"github.com/cogentcore/webgpu/wgpu"
	"github.com/townsendmerino/goinfer/decoder"
)

// TestDecodeTWE_split runs the DecodeRunner's TWrite/TEncode/TSync instrumentation on
// REAL resident decode (load → ResidentForwardForTest → drive Forward per token) and
// reports the per-token split with median + p10/p90 spread over ~400 steady-state
// tokens. It also times batched ForwardN (RunN) to size how much the batched verify
// already amortizes fences. This adjudicates the §3 (spine fusion) lever: if TSync
// dominates and ≈ the device-timestamp GPU portion, the resident path is GPU-bound and
// §3's only headroom is removing glue dispatches; material TEncode/TWrite means CPU
// headroom remains. Pure measurement — no production decode change.
//
//	GOINFER_DECODE_GGUF=~/models/qwen3-1.7b-q8_0.gguf go test -tags gpu -run TestDecodeTWE_split -v
func TestDecodeTWE_split(t *testing.T) {
	if testing.Short() {
		t.Skip("twe split")
	}
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
		t.Skipf("model not GPU-resident (ineligible arch) — this test measures the resident path")
	}
	rd, ok := m.ResidentForwardForTest().(*residentDecoder)
	if !ok {
		t.Skipf("resident forward is not the gpu DecodeRunner (%T)", m.ResidentForwardForTest())
	}
	hidden, _, _, _, _, _, _ := m.Dims()
	rng := rand.New(rand.NewSource(1))
	emb := make([]float32, hidden)
	for i := range emb {
		emb[i] = float32(rng.NormFloat64()) * 0.5
	}

	// warmup (compile pipelines, fill some KV), then steady-state.
	rd.Reset()
	for i := 0; i < 8; i++ {
		if _, err := rd.Forward(emb, i); err != nil {
			t.Fatalf("warmup Forward: %v", err)
		}
	}

	const N = 400
	tw := make([]float64, N)
	te := make([]float64, N)
	ts := make([]float64, N)
	tot := make([]float64, N)
	us := func(d time.Duration) float64 { return float64(d.Microseconds()) }
	for i := 0; i < N; i++ {
		if _, err := rd.Forward(emb, 8+i); err != nil {
			t.Fatalf("Forward[%d]: %v", i, err)
		}
		r := rd.runner
		tw[i], te[i], ts[i] = us(r.TWrite), us(r.TEncode), us(r.TSync)
		tot[i] = tw[i] + te[i] + ts[i]
	}
	pct := func(xs []float64, p float64) float64 {
		s := append([]float64(nil), xs...)
		sort.Float64s(s)
		return s[int(p*float64(len(s)-1))]
	}
	stat := func(name string, xs []float64) {
		t.Logf("  %-14s median %7.1f µs   (p10 %7.1f, p90 %7.1f)", name, pct(xs, 0.5), pct(xs, 0.1), pct(xs, 0.9))
	}

	// device-timestamp GPU portion (true on-GPU ns), if the resident device exposes
	// timestamp queries; else report TSync as the GPU-blocked wall and note poll cost.
	gpuMs := gpuTimePlanTWE(rd.c, rd.runner.steps)

	// batched ForwardN (RunN): K tokens in one Submit/Poll → per-token fence amortization.
	runNPerTok := func(K int) float64 {
		embs := make([][]float32, K)
		for i := range embs {
			embs[i] = emb
		}
		rd.Reset()
		if _, err := rd.ForwardN(embs, 0); err != nil { // warm the K-runner batch
			t.Fatalf("ForwardN warm: %v", err)
		}
		const reps = 20
		best := time.Hour
		for r := 0; r < reps; r++ {
			rd.Reset()
			t0 := time.Now()
			if _, err := rd.ForwardN(embs, 0); err != nil {
				t.Fatalf("ForwardN: %v", err)
			}
			if d := time.Since(t0); d < best {
				best = d
			}
		}
		return float64(best.Microseconds()) / float64(K)
	}

	t.Logf("=== %s  (resident int8, %d steady-state tokens) ===", path, N)
	stat("TWrite", tw)
	stat("TEncode", te)
	stat("TSync", ts)
	stat("total", tot)
	medTot := pct(tot, 0.5)
	bestTot := pct(tot, 0.0) // min over the run = throttle-free per-token (compare to ForwardN min)
	t.Logf("  per-token median total %.1f µs = %.1f tok/s  |  best %.1f µs = %.1f tok/s (throttle-free)",
		medTot, 1e6/medTot, bestTot, 1e6/bestTot)
	t.Logf("  split: TWrite %.0f%% | TEncode %.0f%% | TSync %.0f%%",
		pct(tw, 0.5)/medTot*100, pct(te, 0.5)/medTot*100, pct(ts, 0.5)/medTot*100)
	if gpuMs > 0 {
		t.Logf("  device-timestamp GPU portion: %.2f ms = %.0f µs (of TSync %.0f µs → CPU poll/submit %.0f µs)",
			gpuMs, gpuMs*1000, pct(ts, 0.5), pct(ts, 0.5)-gpuMs*1000)
	} else {
		t.Logf("  device-timestamp: unavailable on the resident device (feature not requested); TSync is GPU-blocked wall, poll overhead ~0.4 ms measured separately")
	}
	t.Logf("  Run() vs RunN (all best/min): single %.0f µs/tok | ForwardN K=8 %.0f µs/tok | K=16 %.0f µs/tok",
		bestTot, runNPerTok(8), runNPerTok(16))
}

// gpuTimePlanTWE returns the min/30 on-GPU ms for the runner's dispatch plan via a
// timestamp query, or -1 if the device lacks the timestamp feature (so the harness
// degrades to TSync wall-clock). Cogentcore lacks pass-descriptor TimestampWrites, so
// this brackets the pass with encoder WriteTimestamp.
func gpuTimePlanTWE(c *Context, steps []runStep) float64 {
	bestTicks := math.MaxFloat64
	for run := 0; run < 30; run++ {
		qset, e := c.device.CreateQuerySet(&wgpu.QuerySetDescriptor{Label: "ts", Type: wgpu.QueryTypeTimestamp, Count: 2})
		if e != nil || qset == nil {
			return -1
		}
		resolve, _ := c.device.CreateBuffer(&wgpu.BufferDescriptor{Label: "ts-resolve", Size: 16, Usage: wgpu.BufferUsageQueryResolve | wgpu.BufferUsageCopySrc})
		readback, _ := c.device.CreateBuffer(&wgpu.BufferDescriptor{Label: "ts-rb", Size: 16, Usage: wgpu.BufferUsageMapRead | wgpu.BufferUsageCopyDst})
		enc, _ := c.device.CreateCommandEncoder(nil)
		enc.WriteTimestamp(qset, 0)
		pass := enc.BeginComputePass(nil)
		for _, s := range steps {
			pass.SetPipeline(s.pl)
			pass.SetBindGroup(0, s.bg, nil)
			pass.DispatchWorkgroups(s.gx, s.gy, 1)
		}
		pass.End()
		pass.Release()
		enc.WriteTimestamp(qset, 1)
		enc.ResolveQuerySet(qset, 0, 2, resolve, 0)
		enc.CopyBufferToBuffer(resolve, 0, readback, 0, 16)
		cmd, _ := enc.Finish(nil)
		c.queue.Submit(cmd)
		cmd.Release()
		enc.Release()
		st := wgpu.BufferMapAsyncStatusUnknown
		readback.MapAsync(wgpu.MapModeRead, 0, 16, func(s wgpu.BufferMapAsyncStatus) { st = s })
		c.device.Poll(true, nil)
		if st == wgpu.BufferMapAsyncStatusSuccess {
			tsv := wgpu.FromBytes[uint64](readback.GetMappedRange(0, 16))
			if d := float64(tsv[1] - tsv[0]); d < bestTicks {
				bestTicks = d
			}
		}
		readback.Release()
		resolve.Release()
		qset.Release()
	}
	if bestTicks == math.MaxFloat64 {
		return -1
	}
	return bestTicks / 1e6 // period ~1.0 ns/tick on this RTX 2070 (measured)
}
