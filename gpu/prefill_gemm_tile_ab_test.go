//go:build gpu && goinfer_testhooks

package gpu

import (
	"context"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"sort"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestPrefill_gemmTileAB is R10's build gate against the band registered in
// docs/measurements/webgpu-prefill-profile-2026-09-22.md: the 64×64 register-blocked GEMM
// (rb64) against the shipped 16×16 kernel (tiled16, the do-nothing arm), on ONE loaded
// model, arms flipped between calls, ABBA. Three readings: the GEMM class alone through the
// sync-bounded prefill profiler (the band's own quantity), the whole PrefillLast wall
// without the profiler, and the last-row logits' Float32bits, which must be equal.
//
//	GOINFER_HEAVY_TESTS=1 go test -tags 'gpu goinfer_testhooks' ./gpu/ -run TestPrefill_gemmTileAB -v -timeout 30m
func TestPrefill_gemmTileAB(t *testing.T) {
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
	hidden, _, _, _, _, _, _ := m.Dims()
	P := 512
	if v := os.Getenv("GOINFER_PREFILL_AB_P"); v != "" {
		if n, e := parseInt(v); e == nil {
			P = n
		}
	}
	rng := rand.New(rand.NewSource(11))
	embs := make([][]float32, P)
	for i := range embs {
		e := make([]float32, hidden)
		for j := range e {
			e[j] = float32(rng.NormFloat64()) * 0.5
		}
		embs[i] = e
	}
	arm := func(tile int) {
		if e := c.SetGEMMTileForTest(tile); e != nil {
			t.Fatalf("SetGEMMTileForTest(%d): %v", tile, e)
		}
	}
	run := func(tile int, prof bool) (wall, gemm time.Duration, logits []float32) {
		arm(tile)
		c.SetPrefillProfForTest(prof)
		t0 := time.Now()
		lg, e := pf.PrefillLast(context.Background(), embs, 0)
		wall = time.Since(t0)
		if e != nil {
			t.Fatalf("PrefillLast(tile=%d): %v", tile, e)
		}
		if prof {
			gemm, _, _, _ = c.PrefillProfForTest()
		}
		c.SetPrefillProfForTest(false)
		return wall, gemm, lg
	}

	// Warm-up both arms (pipeline JIT, buffer growth), discarded.
	run(16, false)
	run(64, false)

	// Bit-identity of the last-row logits.
	_, _, l16 := run(16, false)
	_, _, l64 := run(64, false)
	if len(l16) != len(l64) {
		t.Fatalf("logit lengths differ: %d vs %d", len(l16), len(l64))
	}
	for i := range l16 {
		if math.Float32bits(l16[i]) != math.Float32bits(l64[i]) {
			t.Fatalf("NOT bit-identical at logit %d: tiled16 %08x vs rb64 %08x", i, math.Float32bits(l16[i]), math.Float32bits(l64[i]))
		}
	}
	t.Logf("bit-identical: %d last-row logits equal (P=%d)", len(l16), P)

	// GEMM class through the profiler, ABBA, 4 pairs.
	const pairs = 4
	var gr, wr []float64
	var g16s, g64s, w16s, w64s time.Duration
	for p := 0; p < pairs; p++ {
		var g16, g64, w16, w64 time.Duration
		if p%2 == 0 {
			_, g16, _ = run(16, true)
			_, g64, _ = run(64, true)
			w16, _, _ = run(16, false)
			w64, _, _ = run(64, false)
		} else {
			_, g64, _ = run(64, true)
			_, g16, _ = run(16, true)
			w64, _, _ = run(64, false)
			w16, _, _ = run(16, false)
		}
		gr = append(gr, g16.Seconds()/g64.Seconds())
		wr = append(wr, w16.Seconds()/w64.Seconds())
		g16s, g64s, w16s, w64s = g16s+g16, g64s+g64, w16s+w16, w64s+w64
		t.Logf("pair %d: GEMM class tiled16 %.1f ms | rb64 %.1f ms | %.3fx    whole prefill %.1f | %.1f ms | %.3fx",
			p, msf(g16), msf(g64), g16.Seconds()/g64.Seconds(), msf(w16), msf(w64), w16.Seconds()/w64.Seconds())
	}
	med := func(x []float64) float64 {
		s := append([]float64(nil), x...)
		sort.Float64s(s)
		return (s[len(s)/2-1] + s[len(s)/2]) / 2
	}
	gm, wm := med(gr), med(wr)
	verdict := "KILL (<1.3x on the GEMM class)"
	switch {
	case gm >= 2.0:
		verdict = "SHIP (>=2.0x on the GEMM class)"
	case gm >= 1.3:
		verdict = "PARK (1.3-2.0x)"
	}
	if gm >= 1.9 && gm < 2.0 || gm >= 1.235 && gm < 1.3 {
		verdict += " — AMBIGUOUS (within 5% of a threshold): parked"
	}
	t.Logf("GEMM class: paired median %.3fx over %d pairs (pooled %.1f -> %.1f ms); whole prefill: paired median %.3fx (pooled %.1f -> %.1f ms) — %s",
		gm, pairs, msf(g16s)/pairs, msf(g64s)/pairs, wm, msf(w16s)/pairs, msf(w64s)/pairs, verdict)
}

func msf(d time.Duration) float64 { return float64(d) / float64(time.Millisecond) }

func parseInt(s string) (int, error) {
	n := 0
	for _, ch := range s {
		if ch < '0' || ch > '9' {
			return 0, os.ErrInvalid
		}
		n = n*10 + int(ch-'0')
	}
	return n, nil
}
