//go:build cuda && goinfer_testhooks

package cuda

import (
	"context"
	"fmt"
	"os"
	"strconv"
	"strings"
	"testing"
	"time"

	gc "github.com/eitamring/gocudrv/cuda"
	"github.com/townsendmerino/goinfer/decoder"
)

// TestPrefillLongPrompt probes the batched prefill path at the prompt lengths a deep-context
// benchmark actually uses (up to 8k), which no existing harness covers: TestPrefillTTFT stops at
// M=2048 on a 1.5B model. The M-sized device scratch prefillCore allocates is O(M*inter), so the
// path can pass its LOAD-time report (PrefillPath says "batched") and still decline every real
// long prompt at call time, silently falling back to the ~6 ms/token sequential loop.
//
// Reports, per M: the static decline (if any), whether the call succeeded, its duration, and the
// per-token cost. Diagnostic — it asserts only that the model loaded and that the static gate is
// open; the numbers are the output.
//
//	GOINFER_HEAVY_TESTS=1 go test -tags 'cuda goinfer_testhooks' -run TestPrefillLongPrompt -v -timeout 40m ./cuda/
func TestPrefillLongPrompt(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") == "" {
		t.Skip("set GOINFER_HEAVY_TESTS=1 (loads a 7B model)")
	}
	name := os.Getenv("GOINFER_LONGPROMPT_MODEL")
	if name == "" {
		name = "qwen2.5-7b-instruct-q4_k_m.gguf"
	}
	path := modelPath(name)
	if err := gc.Init(); err != nil {
		t.Skipf("cuInit: %v", err)
	}
	if _, err := gc.GetDevice(0); err != nil {
		t.Skipf("no device: %v", err)
	}
	if _, err := os.Stat(path); err != nil {
		t.Skipf("no fixture at %s", path)
	}
	t0 := time.Now()
	fmt.Fprintf(os.Stderr, "[longprompt] loading %s ...\n", path)
	opts := decoder.Options{Backend: "cuda", Quant: "int4", ResidentContext: 8192}
	if os.Getenv("GOINFER_LONGPROMPT_MOECACHE") != "" {
		// The MoE cells (M35/M26/G20) exceed this card, so the bench launches them with
		// -moe-cache-experts; a probe that omits it is not probing the shipped configuration.
		opts.MoECacheExperts = true
	}
	if os.Getenv("GOINFER_LONGPROMPT_GIW") != "" {
		opts.Quant = "" // a kind-4 .giw bundle is already quantized; -quant does not apply
	}
	mc, err := decoder.Load(path, opts)
	if err != nil {
		t.Fatalf("load (cuda): %v", err)
	}
	defer mc.Close()
	fmt.Fprintf(os.Stderr, "[longprompt] loaded in %s\n", time.Since(t0).Round(time.Millisecond))
	rf, ok := mc.ResidentForwardForTest().(*cudaResident)
	if !ok {
		t.Fatal("resident is not *cudaResident")
	}
	batched, why := rf.PrefillPath()
	fmt.Fprintf(os.Stderr, "[longprompt] PrefillPath(load-time) = %v — %s\n", batched, why)
	// Name EVERY guard that would refuse this model, not just the first. prefillStaticDecline
	// returns on the earliest failure, so a model that trips four of them reads as tripping one —
	// and the cost of fixing it is the count, not the first line.
	fmt.Fprintf(os.Stderr, "[longprompt] guards: prefillReady=%v moe=%v gemma4Moe=%v sandwich=%v qkNorm=%v\n",
		rf.prefillReady, rf.moe, rf.gemma4Moe, rf.sandwich, rf.qkNorm)
	if n := len(rf.layers); n > 0 {
		L0 := &rf.layers[0]
		nonUniform, kEqV, badKind := 0, 0, map[string]int{}
		for l := range rf.layers {
			Ly := &rf.layers[l]
			if Ly.hd != L0.hd || Ly.nKV != L0.nKV || Ly.qDim != L0.qDim || Ly.kvDim != L0.kvDim || Ly.rhalf != L0.rhalf {
				nonUniform++
			}
			if Ly.kEqV {
				kEqV++
			}
			if k := nonBatchableKind(Ly); k != "" {
				badKind[k]++
			}
		}
		fmt.Fprintf(os.Stderr, "[longprompt] layers=%d non-uniform-vs-L0=%d kEqV=%d nonBatchableKinds=%v "+
			"(L0: hd=%d nKV=%d qDim=%d kvDim=%d rhalf=%d window=%d)\n",
			n, nonUniform, kEqV, badKind, L0.hd, L0.nKV, L0.qDim, L0.kvDim, L0.rhalf, L0.window)
	}
	fmt.Fprintf(os.Stderr, "[longprompt] hidden=%d inter=%d layers=%d ctxCap=%d\n",
		rf.hidden, rf.inter, rf.nLayers, rf.ctxCap)
	if free, total, e := rf.dev.Context().MemInfo(); e == nil {
		fmt.Fprintf(os.Stderr, "[longprompt] VRAM free=%.2f GB of %.2f GB after load\n",
			float64(free)/1e9, float64(total)/1e9)
	}

	hidden := rf.hidden
	build := func(n int) [][]float32 {
		embs := make([][]float32, n)
		var s uint32 = 12345
		for i := range embs {
			row := make([]float32, hidden)
			for j := range row {
				s = s*1664525 + 1013904223
				row[j] = float32(int32(s>>16)%2001-1000) / 10000
			}
			embs[i] = row
		}
		return embs
	}

	// C′ attribution. The expert cache DMAs routed experts host→VRAM per token, and on a model that
	// exceeds the card that transfer — not the weight reads batching removes — may be the whole cost.
	// Reported as a DELTA around each arm so the two are comparable; zero unless
	// GOINFER_MOE_CACHE_PROF is set, in which case these lines are the attribution and the timings
	// above them are what it explains.
	cacheProf := func() (time.Duration, time.Duration, time.Duration, uint64) {
		st, ho, dm, c := rf.CacheProfForTest()
		return st, ho, dm, c
	}
	reportCache := func(tag string, st0, ho0, dm0 time.Duration, c0 uint64, wall time.Duration) {
		st, ho, dm, c := cacheProf()
		if c == c0 {
			return // caching off, or profiling not enabled — say nothing rather than print zeros
		}
		bt, sy := rf.BatchProfForTest()
		fmt.Fprintf(os.Stderr, "[longprompt]   %s C′: stall %s host %s dma %s over %d calls "+
			"(batchTime %s, %d syncs) — %.1f%% of the %s arm\n",
			tag, (st - st0).Round(time.Millisecond), (ho - ho0).Round(time.Millisecond),
			(dm - dm0).Round(time.Millisecond), c-c0, bt.Round(time.Millisecond), sy,
			100*float64((st-st0)+(ho-ho0)+(dm-dm0))/float64(wall), tag)
	}

	lengths := []int{512, 2048, 4096, 8012}
	if v := os.Getenv("GOINFER_LONGPROMPT_LENGTHS"); v != "" {
		lengths = nil
		for _, f := range strings.Split(v, ",") {
			n, e := strconv.Atoi(strings.TrimSpace(f))
			if e != nil || n <= 0 {
				t.Fatalf("GOINFER_LONGPROMPT_LENGTHS: bad entry %q", f)
			}
			lengths = append(lengths, n)
		}
	}
	for _, M := range lengths {
		rf.Reset()
		embs := build(M)
		st0, ho0, dm0, c0 := cacheProf()
		start := time.Now()
		_, perr := rf.PrefillLast(context.Background(), embs, 0)
		d := time.Since(start)
		if perr != nil {
			fmt.Fprintf(os.Stderr, "[longprompt] M=%-5d DECLINED after %-10s: %v\n", M, d.Round(time.Millisecond), perr)
			continue
		}
		fmt.Fprintf(os.Stderr, "[longprompt] M=%-5d batched OK %-10s (%.3f ms/token)\n",
			M, d.Round(time.Millisecond), float64(d.Microseconds())/1000/float64(M))
		reportCache("batched", st0, ho0, dm0, c0, d)
	}

	// Sequential reference at the SAME lengths, so the fallback's real cost is measured rather than
	// extrapolated from a shallow point — its per-token cost grows with depth too (each position
	// attends its own prefix), so a 512-token reading understates it badly at 8k.
	for _, M := range lengths {
		rf.Reset()
		embs := build(M)
		st0, ho0, dm0, c0 := cacheProf()
		start := time.Now()
		for i, e := range embs {
			if err := rf.ForwardNoLogits(e, i); err != nil {
				t.Fatalf("sequential ForwardNoLogits(%d): %v", i, err)
			}
		}
		d := time.Since(start)
		fmt.Fprintf(os.Stderr, "[longprompt] M=%-5d SEQUENTIAL %-10s (%.3f ms/token)\n",
			M, d.Round(time.Millisecond), float64(d.Microseconds())/1000/float64(M))
		reportCache("sequential", st0, ho0, dm0, c0, d)
	}
}
