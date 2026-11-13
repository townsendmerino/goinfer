//go:build cuda && goinfer_testhooks

package cuda

import (
	"context"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"slices"
	"sync"
	"testing"
	"time"

	gc "github.com/eitamring/gocudrv/cuda"
	"github.com/townsendmerino/goinfer/decoder"
)

// TestMC3CUDAS0_batchedRowsVsDecode is an EXPLORATORY probe (not a gate) for an MC3 port to CUDA
// (docs/tasks/parked/task-concurrency-2026-09.md): how much a batched M-row pass through the EXACT kernels (gemv_w4a8_rn,
// attn_batched: the verify path, bit-identical to decode per row by construction) costs against M single-token
// decodes. It times, at depth 1024, for M in {2, 4, 8}:
//   - PrefillLastN: every row's full logits, the LM head run once per row (the logits-exact tail);
//   - PrefillLastNArgmax: one batched head over all rows, argmax only (verify's tail);
//   - M sequential Forward calls: the unbatched baseline.
//
// It also checks that PrefillLastN's rows are bit-identical to sequential Forward at the same positions, the identity
// an MC3 step would need. This is one sequence's rows at consecutive positions, not B sequences on B slots: the GEMVs
// are the same, but per-sequence attention and RoPE positions are not exercised. Best of 5.
//
// GOINFER_HEAVY_TESTS=1, GOINFER_CUDA_MODEL=<checkpoint> (default the 1.5B).
func TestMC3CUDAS0_batchedRowsVsDecode(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") == "" {
		t.Skip("set GOINFER_HEAVY_TESTS=1 (loads a real model)")
	}
	path := os.Getenv("GOINFER_CUDA_MODEL")
	if path == "" {
		path = modelPath("qwen2.5-coder-1.5b-instruct-q4_k_m.gguf")
	}
	if err := gc.Init(); err != nil {
		t.Skipf("cuInit: %v", err)
	}
	if _, err := os.Stat(path); err != nil {
		t.Skipf("no fixture at %s", path)
	}
	mc, err := decoder.Load(path, decoder.Options{Backend: "cuda", Quant: "int4"})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	defer mc.Close()
	rf := mustResident(t, mc)
	_, _, _, _, _, _, vocab := mc.Dims()
	emb := func(i int) []float32 { return mc.EmbedResidentForTest((i*2654435761 + 1) % (vocab - 1)) }
	const depth = 1024
	warm := make([][]float32, depth)
	for i := range warm {
		warm[i] = emb(i)
	}
	if _, e := rf.PrefillLast(context.Background(), warm, 0); e != nil {
		t.Fatalf("warm prefill: %v", e)
	}
	best := func(f func()) float64 {
		b := time.Hour
		for range 5 {
			t0 := time.Now()
			f()
			b = min(b, time.Since(t0))
		}
		return float64(b.Microseconds()) / 1000
	}
	one := best(func() {
		if _, e := rf.Forward(emb(5), depth); e != nil {
			t.Fatal(e)
		}
	})
	name := filepath.Base(path)
	fmt.Fprintf(os.Stderr, "[mc3cuda] %s depth %d: one decode %.3f ms\n", name, depth, one)
	for _, k := range []int{2, 4, 8} {
		ek := make([][]float32, k)
		for i := range ek {
			ek[i] = emb(100 + i)
		}
		seq := best(func() {
			for i := range k {
				if _, e := rf.Forward(ek[i], depth+i); e != nil {
					t.Fatal(e)
				}
			}
		})
		lastN := best(func() {
			if _, e := rf.PrefillLastN(ek, depth); e != nil {
				t.Fatal(e)
			}
		})
		argN := best(func() {
			if _, e := rf.PrefillLastNArgmax(ek, depth); e != nil {
				t.Fatal(e)
			}
		})
		// identity: PrefillLastN's rows against sequential Forward at the same positions
		outs, e := rf.PrefillLastN(ek, depth)
		if e != nil {
			t.Fatal(e)
		}
		batched := make([][]float32, k)
		for i := range outs {
			batched[i] = append([]float32(nil), outs[i]...)
		}
		diff := 0
		for i := range k {
			lg, e := rf.Forward(ek[i], depth+i)
			if e != nil {
				t.Fatal(e)
			}
			for j := range lg {
				if math.Float32bits(lg[j]) != math.Float32bits(batched[i][j]) {
					diff++
				}
			}
		}
		line := fmt.Sprintf("k=%d: %d×decode %.3f ms | PrefillLastN %.3f ms = %.2fx cheaper | PrefillLastNArgmax %.3f ms = %.2fx cheaper | per-row logits differing from Forward: %d of %d",
			k, k, seq, lastN, seq/lastN, argN, seq/argN, diff, k*vocab)
		fmt.Fprintf(os.Stderr, "[mc3cuda] %s %s\n", name, line)
		t.Log(line)
	}
}

// TestRF3CUDA_rideIdentity is G-RF3-I's in-process half (docs/tasks/task-forward-and-memory-2026-11.md, "RF3: plan and
// registration"): on the real CUDA resident, two decoders and a newcomer with a long prompt run concurrently through the
// coalescer, the newcomer's prefill chunks ride the decode runs, and every reply equals the same conversation served alone
// (the alternation's own reference). It fails if no chunk rode: RidePasses must be positive, so the test cannot pass on the
// alternation alone.
//
// GOINFER_HEAVY_TESTS=1, GOINFER_CUDA_MODEL=<checkpoint> (default the 1.5B); needs the card, hold the timing lock.
func TestRF3CUDA_rideIdentity(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") == "" {
		t.Skip("set GOINFER_HEAVY_TESTS=1 (loads a real model)")
	}
	path := os.Getenv("GOINFER_CUDA_MODEL")
	if path == "" {
		path = modelPath("qwen2.5-coder-1.5b-instruct-q4_k_m.gguf")
	}
	if err := gc.Init(); err != nil {
		t.Skipf("cuInit: %v", err)
	}
	if _, err := os.Stat(path); err != nil {
		t.Skipf("no fixture at %s", path)
	}
	m, err := decoder.Load(path, decoder.Options{Backend: "cuda", Quant: "int4", ResidentPrefillChunk: 256, ResidentContext: 4096, ResidentKVSlots: 4})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	defer m.Close()
	_ = mustResident(t, m)
	if n := m.EnableResidentConcurrency(4); n != 4 {
		t.Fatalf("EnableResidentConcurrency(4) = %d: no MC3 batcher (the resident needs ResidentKVSlots >= 4)", n)
	}
	// the alone replies take the same MC3 path (no decoders, so one whole prefill)
	gen := func(prompt []int, maxTok int) []int {
		ch, g := m.Generate(context.Background(), prompt, maxTok, decoder.SamplingParams{})
		var ids []int
		for id := range ch {
			ids = append(ids, id)
		}
		if err := g.Err(); err != nil {
			t.Errorf("generate: %v", err)
		}
		return ids
	}
	const decodeTokens = 400
	d0, d1 := []int{1, 2, 3, 40}, []int{1, 2, 3, 50}
	aloneD0, aloneD1 := gen(d0, decodeTokens), gen(d1, decodeTokens)
	// The newcomer's prompt is the model's own greedy text repeated to 1200 tokens: confident, in-distribution continuations,
	// so a near-tie in an out-of-distribution prompt cannot flip a token between two correct paths.
	long := make([]int, 0, 1200)
	for len(long) < 1200 {
		long = append(long, aloneD0...)
	}
	long = long[:1200]
	aloneNew := gen(long, 8)

	// aloneNew left the newcomer's prompt in the bound slot, and a prompt whose prefix the resident holds is prefilled from the
	// divergent suffix only (prefix reuse): a seed token, so no chunk to ride. Forget it so the concurrent newcomer prefills cold.
	m.ForgetResidentForTest()
	before := m.ResidentBatchStats()
	var got0, got1, gotNew []int
	var wg sync.WaitGroup
	wg.Add(3)
	go func() { defer wg.Done(); got0 = gen(d0, decodeTokens) }()
	go func() { defer wg.Done(); got1 = gen(d1, decodeTokens) }()
	go func() {
		defer wg.Done()
		time.Sleep(300 * time.Millisecond) // the two are decoding by now (400 tokens each take about a second)
		gotNew = gen(long, 8)
	}()
	wg.Wait()
	for _, c := range []struct {
		name      string
		got, want []int
	}{{"decoder 0", got0, aloneD0}, {"decoder 1", got1, aloneD1}, {"newcomer", gotNew, aloneNew}} {
		if !slices.Equal(c.got, c.want) {
			t.Errorf("%s: concurrent %v, alone %v", c.name, c.got, c.want)
		}
	}
	st := m.ResidentBatchStats()
	rode := st.RidePasses - before.RidePasses
	t.Logf("ride passes %d, ride rows %d, ride ns %d; runs %d, prefill passes %d, stats %+v", rode, st.RideRows-before.RideRows, st.RideNs-before.RideNs, st.Runs-before.Runs, st.PrefillPasses-before.PrefillPasses, st)
	if rode < 1 {
		t.Errorf("no chunk rode a decode run (RidePasses %d): the check proved nothing about the ride", rode)
	}
}
