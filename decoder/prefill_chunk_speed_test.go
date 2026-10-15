package decoder

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"testing"
	"time"
)

// TestG40Speed is G40-3 (docs/tasks/task-cpu-prefill-memory-2026-10.md §4, and the amendment before it ran): the CPU text
// prefill chunked against a single pass, in one process, interleaved ABBA across pairs, each arm on a fresh cache. The
// quantity is the median over pairs of chunked ÷ single-pass prefill time. Night only: GOINFER_G40_SPEED=1.
//
// Cells: Qwen2.5-Coder-0.5B and -1.5B at int4, suffixes of 6,000 (5 pairs) and 20,000 tokens (3 pairs). A cell whose
// available memory is under its floor (7 GiB for the 1.5B at 20K, 4 GiB otherwise) is void, not a fail.
func TestG40Speed(t *testing.T) {
	if os.Getenv("GOINFER_G40_SPEED") != "1" {
		t.Skip("set GOINFER_G40_SPEED=1: G40-3's timed grade (night only)")
	}
	home, _ := os.UserHomeDir()
	t0 := time.Now()
	hb := func(format string, a ...any) {
		fmt.Fprintf(os.Stderr, "[g40-3 %6.0fs] %s\n", time.Since(t0).Seconds(), fmt.Sprintf(format, a...))
	}
	for _, c := range []struct {
		model string
		file  string
	}{
		{"0.5B", "qwen2.5-coder-0.5b-instruct-q4_k_m.gguf"},
		{"1.5B", "qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"},
	} {
		path := filepath.Join(home, "models", c.file)
		m, err := Load(path, Options{Quant: "int4"})
		if err != nil {
			t.Fatalf("load %s: %v", path, err)
		}
		_, _, _, _, _, _, vocab := m.Dims()
		for _, cell := range []struct{ n, pairs int }{{6000, 5}, {20000, 3}} {
			floor := int64(4 << 30)
			if c.model == "1.5B" && cell.n == 20000 {
				floor = 7 << 30
			}
			if avail := HostRAMAvailableBytes(); avail > 0 && avail < floor {
				hb("VOID %s %d: %.1f GiB available, under the cell's %.0f GiB floor", c.model, cell.n, float64(avail)/(1<<30), float64(floor)/(1<<30))
				continue
			}
			prompt := chunkPrompt(cell.n, vocab, 1)
			arm := func(chunk int) time.Duration {
				prev := prefillChunkRows
				defer func() { prefillChunkRows = prev }()
				prefillChunkRows = chunk
				cache := m.NewCache(0)
				s := time.Now()
				if _, err := m.prefillLogits(context.Background(), prompt, cache); err != nil {
					t.Fatalf("prefill: %v", err)
				}
				return time.Since(s)
			}
			var ratios []float64
			for p := range cell.pairs {
				var single, chunked time.Duration
				if p%4 == 0 || p%4 == 3 { // ABBA: A = single, B = chunked
					single, chunked = arm(1<<30), arm(2048)
				} else {
					chunked, single = arm(2048), arm(1<<30)
				}
				r := chunked.Seconds() / single.Seconds()
				ratios = append(ratios, r)
				hb("%s %d pair %d/%d: single %.2f s, chunked %.2f s, chunked/single %.4f", c.model, cell.n, p+1, cell.pairs, single.Seconds(), chunked.Seconds(), r)
			}
			s := time.Now()
			releaseHostMemory()
			rel := time.Since(s)
			slices.Sort(ratios)
			med := ratios[len(ratios)/2]
			verdict := "SHIPS"
			switch {
			case med > 1.05:
				verdict = "KILLED at this C"
			case med > 1.03:
				verdict = "PARKED"
			}
			hb("G40-3 METRIC chunked/single %s %d: median %.4f (min %.4f, max %.4f) over %d pairs -> %s; a forced release took %.1f ms",
				c.model, cell.n, med, ratios[0], ratios[len(ratios)-1], len(ratios), verdict, float64(rel.Microseconds())/1000)
		}
		m.Close()
	}
}
