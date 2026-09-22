//go:build gpu && goinfer_testhooks

package gpu

import (
	"context"
	"math/rand"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestPrefill_dispatchProfile is R10's prefill investigation (docs/tasks/red-october.md): "extend
// the ablation harness to the batched path at P in {256, 1024}: per-class time with one class
// omitted, so the GEMM, the attention, the batched norms/rope and the KV write each carry a
// number. Then one arithmetic line: GFLOPS achieved by the GEMM class against the card's
// naive-f32 and its shared-memory-tiled expectations."
//
// TestDecode_dispatchProfile's own ABLATION method (re-record R times into one pass, omit one
// pipeline class, difference against the full plan) does not transfer: PrefillLastW8A8 is not a
// recorded/replayable step list (it builds fresh buffers and issues M-row-scaled dispatches
// inline, flushing every 32 passes for Metal's uncommitted-command-buffer cap) — re-running it
// with a class's dispatches skipped would cascade into shape/buffer errors for every downstream
// stage in the same layer, not a clean subtraction. Instead this uses PER-CATEGORY WALL-CLOCK
// ACCUMULATION (gpu/prefill_prof.go's profTic/profToc, wired into PrefillLastW8A8 itself, nil by
// default so it costs nothing when off) — the SAME method cuda/prefill_decomp_test.go already
// uses for the CUDA side, including its own accepted trade-off: category boundaries are syncs
// (c.device.Poll(true, nil)), so the category sum runs a bit over the pipelined wall time — "the
// price of per-kernel attribution", reported as such, not hidden.
//
//	GOINFER_HEAVY_TESTS=1 go test -tags gpu -run TestPrefill_dispatchProfile -v ./gpu/ -timeout 20m
func TestPrefill_dispatchProfile(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") == "" {
		t.Skip("heavy-checkpoint test: set GOINFER_HEAVY_TESTS=1")
	}
	home, _ := os.UserHomeDir()
	path := os.Getenv("GOINFER_RESIDENT_GGUF")
	if path == "" {
		path = filepath.Join(home, "models", "qwen2.5-coder-1.5b-instruct-q4_k_m.gguf")
	}
	if _, err := os.Stat(path); err != nil {
		t.Skipf("no resident model at %s: %v", path, err)
	}
	m, err := decoder.Load(path, decoder.Options{Backend: "webgpu", Quant: "int8int8"})
	if err != nil {
		t.Fatalf("load %s: %v", path, err)
	}
	defer m.Close()
	if !m.ResidentActive() {
		t.Skip("model not GPU-resident")
	}
	rf := m.ResidentForwardForTest()
	rd, isRD := rf.(*residentDecoder)
	if !isRD {
		t.Fatalf("ResidentForwardForTest returned %T, want *residentDecoder", rf)
	}
	pf, ok := rf.(decoder.Prefiller)
	if !ok {
		t.Skip("resident forward does not implement Prefiller")
	}
	c := rd.c

	hidden, nLayers, nH, nKV, hd, inter, vocab := m.Dims()
	t.Logf("geometry: hidden=%d layers=%d nH=%d nKV=%d headDim=%d inter=%d vocab=%d", hidden, nLayers, nH, nKV, hd, inter, vocab)

	rng := rand.New(rand.NewSource(3))
	emb := func() []float32 {
		e := make([]float32, hidden)
		for j := range e {
			e[j] = float32(rng.NormFloat64()) * 0.5
		}
		return e
	}

	// FLOP count for the GEMM class (int8 MAC = 2 FLOPs/MAC, the standard convention this repo's
	// own roofline notes use elsewhere): per token, per layer, the tiled GEMM projections are
	// Q/K/V/O (each hidden*hidden, or hidden*(nKV*hd) for K/V under GQA) + gate/up/down
	// (hidden*inter each, x2 for gate+up, x1 for down at inter*hidden). qDim = nH*hd, kvDim = nKV*hd.
	qDim := nH * hd
	kvDim := nKV * hd
	flopsPerTokenPerLayer := int64(2) * int64(hidden) * int64(qDim+2*kvDim+qDim /* Q+K+V+O, O is qDim->hidden = same as Q's hidden->qDim count */ +2*inter /* gate+up */ +inter /* down */)
	// The line above double-counts O (hidden*qDim, same shape as Q's qDim*hidden) correctly since
	// both directions cost the same MACs; written out for clarity rather than algebraic minimalism.

	// GOINFER_PREFILL_PROF_P overrides the depth list. The 1.5B int8int8 OOMs at P=1024 on this 8 GB
	// card (the WebGPU path does not chunk the way cuda's prefillChunked does; O(M*inter) scratch), so
	// its 1024 cell comes from the 0.5B — which is also the model the doc's own prior P=1024 reference
	// (prefill-batched-ttft-2026-09-13.md) used. Recorded, not silently narrowed.
	ps := []int{256, 1024}
	if v := os.Getenv("GOINFER_PREFILL_PROF_P"); v != "" {
		ps = ps[:0]
		for _, f := range strings.Split(v, ",") {
			n, err := strconv.Atoi(strings.TrimSpace(f))
			if err != nil {
				t.Fatalf("GOINFER_PREFILL_PROF_P: bad entry %q", f)
			}
			ps = append(ps, n)
		}
	}
	// WARM-UP, discarded — the decode profiler warms explicitly for the same reason: the first
	// prefill in a process pays one-time driver JIT for every WGSL pipeline it touches, booked into
	// whichever category dispatches each one first. Unwarmed, the first P cell read normsRope at
	// 496 ms (P=256) against 50 ms at P=512 for the 1.5B — O(P) work cannot shrink 10x as P doubles.
	// A single warm call at a small P touches every pipeline the timed cells will use.
	{
		warm := make([][]float32, 64)
		for i := range warm {
			warm[i] = emb()
		}
		rf.Reset()
		if _, e := pf.PrefillLast(context.Background(), warm, 0); e != nil {
			t.Fatalf("warm-up PrefillLast: %v", e)
		}
	}

	var lastAttn, lastNorms time.Duration
	lastP := 0
	for _, p := range ps {
		embs := make([][]float32, p)
		for i := range embs {
			embs[i] = emb()
		}

		// Best of 2 per cell (min, the decode profiler's own noise trim) — each rep is a fresh
		// Reset + full PrefillLast with the profiler's accumulators reset, so the two reps are
		// independent observations of the same cell, not one run's numbers read twice.
		type cell struct {
			wall, gemm, attn, normsRope, kvWrite time.Duration
		}
		var best cell
		for rep := 0; rep < 2; rep++ {
			c.SetPrefillProfForTest(true)
			rf.Reset()
			t0 := time.Now()
			if _, e := pf.PrefillLast(context.Background(), embs, 0); e != nil {
				t.Fatalf("PrefillLast P=%d rep %d: %v", p, rep, e)
			}
			w := time.Since(t0)
			g, a, n, k := c.PrefillProfForTest()
			c.SetPrefillProfForTest(false)
			if rep == 0 || w < best.wall {
				best = cell{w, g, a, n, k}
			}
		}
		wall, gemm, attn, normsRope, kvWrite := best.wall, best.gemm, best.attn, best.normsRope, best.kvWrite

		catSum := gemm + attn + normsRope + kvWrite
		totalFLOPs := float64(flopsPerTokenPerLayer) * float64(nLayers) * float64(p)
		gflops := totalFLOPs / gemm.Seconds() / 1e9

		t.Logf("")
		t.Logf("=== P=%d: wall %.1f ms | category sum %.1f ms (%.1f%% of wall — sync overcount) ===",
			p, float64(wall.Microseconds())/1000, float64(catSum.Microseconds())/1000, 100*float64(catSum)/float64(wall))
		t.Logf("%-12s %10s %8s", "category", "ms", "% of sum")
		for _, row := range []struct {
			name string
			d    time.Duration
		}{{"gemm", gemm}, {"attn", attn}, {"normsRope", normsRope}, {"kvWrite", kvWrite}} {
			t.Logf("%-12s %10.2f %7.1f%%", row.name, float64(row.d.Microseconds())/1000, 100*float64(row.d)/float64(catSum))
		}
		t.Logf("GEMM class: %.1f GFLOP over %s -> %.1f GFLOPS achieved", totalFLOPs/1e9, gemm.Round(time.Microsecond), gflops)
		// Instrument self-check: causal attention is O(P^2), so its booked time MUST grow with P. The
		// first version of this profiler booked 544 ms at P=256 and 55 ms at P=512 — the boundary
		// poll was waiting only for SUBMITTED work while dispatches still sat in the open encoder —
		// and this assertion is what would have caught it on the spot.
		if lastP > 0 && p > lastP && attn <= lastAttn {
			t.Errorf("attention booked %s at P=%d but %s at P=%d — cannot shrink with P for an O(P^2) kernel; the category attribution is scrambled (boundary poll not covering unsubmitted work?)",
				attn.Round(time.Microsecond), p, lastAttn.Round(time.Microsecond), lastP)
		}
		// normsRope is O(P) elementwise (rms, rope, residual, swiglu over M rows): it must not shrink
		// with P either — this is the check the unwarmed first run would have failed.
		if lastP > 0 && p > lastP && normsRope < lastNorms {
			t.Errorf("normsRope booked %s at P=%d but %s at P=%d — O(P) work cannot shrink with P; a one-time cost (driver JIT?) is leaking into a category",
				normsRope.Round(time.Microsecond), p, lastNorms.Round(time.Microsecond), lastP)
		}
		lastAttn, lastNorms, lastP = attn, normsRope, p
	}
}
