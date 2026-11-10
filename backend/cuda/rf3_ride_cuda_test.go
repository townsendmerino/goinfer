//go:build cuda && goinfer_testhooks

package cuda

import (
	"context"
	"os"
	"slices"
	"sync"
	"testing"
	"time"

	gc "github.com/eitamring/gocudrv/cuda"
	"github.com/townsendmerino/goinfer/decoder"
)

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
	m, err := decoder.Load(path, decoder.Options{Backend: "cuda", Quant: "int4", ResidentPrefillChunk: 256, ResidentContext: 4096})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	defer m.Close()
	_ = mustResident(t, m)
	_, _, _, _, _, _, vocab := m.Dims()
	long := make([]int, 1200)
	for i := range long {
		long[i] = (i*7 + 3) % (vocab - 1)
	}
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
	d0, d1 := []int{1, 2, 3, 40}, []int{1, 2, 3, 50}
	aloneD0, aloneD1, aloneNew := gen(d0, 120), gen(d1, 120), gen(long, 8)

	m.EnableResidentConcurrency(4)
	before := m.ResidentBatchStats()
	var got0, got1, gotNew []int
	var wg sync.WaitGroup
	wg.Add(3)
	go func() { defer wg.Done(); got0 = gen(d0, 120) }()
	go func() { defer wg.Done(); got1 = gen(d1, 120) }()
	go func() {
		defer wg.Done()
		time.Sleep(150 * time.Millisecond) // the two are decoding by now
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
	t.Logf("ride passes %d, ride rows %d, ride ns %d", rode, st.RideRows-before.RideRows, st.RideNs-before.RideNs)
	if rode < 1 {
		t.Errorf("no chunk rode a decode run (RidePasses %d): the check proved nothing about the ride", rode)
	}
}
