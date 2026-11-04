//go:build darwin && goinfer_testhooks

package metal

import (
	"context"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"slices"
	"sort"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/tokenizer"
)

// TestMEM7_blockFitsTheBudget is BLK-G2 (docs/tasks/task-memory-campaign-2026-10.md, MEM7 gates): the block is the rows
// that fit what the memory guard's budget leaves after the need, capped at prefillBlockScratch and floored at
// prefillBlockRowsMin; the cases that report no block still report none.
func TestMEM7_blockFitsTheBudget(t *testing.T) {
	_, r := mc3FixtureResident(t, 2, 2048)
	const need = int64(3 << 30)
	a := &metalResident{r: r, hidden: r.H, need: func() int64 { return need }}
	free := &metalResident{r: r, hidden: r.H} // no need: today's block
	today := free.PrefillBlockRows()
	defer func(c int64) { metalMemoryCeilingForTest = c }(metalMemoryCeilingForTest)
	for _, c := range []struct {
		name     string
		headroom int64
	}{{"1 GiB", 1 << 30}, {"200 MiB", 200 << 20}, {"1 MiB", 1 << 20}} {
		metalMemoryCeilingForTest = need + c.headroom
		B := a.PrefillBlockRows()
		fmt.Fprintf(os.Stderr, "[BLK-G2] headroom %s: block %d rows (%d bytes of scratch); today %d rows\n", c.name, B, a.PrefillScratchBytes(B), today)
		switch {
		case c.headroom >= prefillBlockScratch:
			if B != today {
				t.Errorf("%s of headroom: %d rows, want today's %d (the 512 MiB ceiling binds)", c.name, B, today)
			}
		case a.PrefillScratchBytes(prefillBlockRowsMin) > c.headroom:
			if B != prefillBlockRowsMin {
				t.Errorf("%s of headroom: %d rows, want the floor %d", c.name, B, prefillBlockRowsMin)
			}
		default:
			if B%64 != 0 || a.PrefillScratchBytes(B) > c.headroom || a.PrefillScratchBytes(B+64) <= c.headroom {
				t.Errorf("%s of headroom: %d rows price at %d bytes; want the most 64-row multiple under %d", c.name, B, a.PrefillScratchBytes(B), c.headroom)
			}
		}
	}
	exact := &metalResident{r: r, hidden: r.H, exact: true, need: func() int64 { return need }}
	if got := exact.PrefillBlockRows(); got != 0 {
		t.Errorf("PrefillBlockRows = %d with --exact-prefill, want 0", got)
	}
	// An int8 cache takes the pass (P3b-2): it gets a block, and a row costs what it does at f16 plus the fixed dequantize
	// scratch (kvI8ScratchRows positions of K and V at f16).
	f16Price := a.PrefillScratchBytes(256)
	r.kvI8 = true
	if got := a.PrefillBlockRows(); got <= 0 {
		t.Errorf("PrefillBlockRows = %d with an int8 KV cache, want a block", got)
	}
	if got, want := a.PrefillScratchBytes(256)-f16Price, 4*int64(r.kvI8ScratchRows())*int64(r.prefillGeom().kvDim); got != want {
		t.Errorf("an int8 cache adds %d bytes of prefill scratch, want %d (the dequantize scratch)", got, want)
	}
	r.kvI8 = false
}

// TestMEM7_speedSmallBlock is BLK-G3 (docs/tasks/task-memory-campaign-2026-10.md, MEM7 gates): on Qwen2.5-Coder-1.5B from
// ~/models, a 12,288-token prose prompt's resident prefill with today's block (the rows prefillBlockScratch holds) against the
// rows a 128 MiB budget gives, five alternating pairs after a warm-up pair, in one process. A timed run: hold the timing lock.
//
//	GOINFER_MEM7_SPEED=1 go test -count=1 -tags goinfer_testhooks -timeout 30m -run '^TestMEM7_speedSmallBlock$' -v ./metal/
func TestMEM7_speedSmallBlock(t *testing.T) {
	if os.Getenv("GOINFER_MEM7_SPEED") == "" {
		t.Skip("set GOINFER_MEM7_SPEED=1 (a timed run, by day, under the timing lock)")
	}
	home, _ := os.UserHomeDir()
	path := filepath.Join(home, "models", "qwen2.5-coder-1.5b-instruct-q4_k_m.gguf")
	if _, err := os.Stat(path); err != nil {
		t.Fatalf("no checkpoint at %s", path)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 25*time.Minute)
	defer cancel()
	giw := decoder.SidecarPath(path, "int4", "metal", false) // the mapped sidecar: a direct .gguf load can be refused by day
	if _, err := os.Stat(giw); err != nil {
		t.Fatalf("no sidecar at %s", giw)
	}
	m, err := decoder.Load(giw, decoder.Options{Backend: "metal", Quant: "int4", ResidentContext: 16384})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	defer m.Close()
	a, ok := m.ResidentForwardForTest().(*metalResident)
	if !ok {
		t.Fatalf("not Metal-resident: %s", m.ResidentDecline())
	}
	per := a.PrefillScratchBytes(64) / 64
	rowsFor := func(budget int64) int { return max(int(budget/per)/64*64, prefillBlockRowsMin) }
	today, small := rowsFor(prefillBlockScratch), rowsFor(128<<20)
	tk, err := tokenizer.LoadGGUF(path)
	if err != nil {
		t.Fatalf("tokenizer: %v", err)
	}
	_, files := decoder.PrefillGatePromptSet()
	const n, pairs = 12288, 5
	ids := decoder.PrefillGateProseIDsForTest(t, tk, files[1], n)[:n]
	defer func(prev int) { prefillBlockRowsForTest = prev }(prefillBlockRowsForTest)
	var ref []float32
	run := func(block int) time.Duration {
		prefillBlockRowsForTest = block
		t0 := time.Now()
		lg, err := m.ResidentPrefillSeedForTest(ctx, ids)
		if err != nil {
			t.Fatalf("prefill with block %d: %v", block, err)
		}
		d := time.Since(t0)
		if ref == nil {
			ref = slices.Clone(lg)
		} else if !slices.EqualFunc(lg, ref, func(x, y float32) bool { return math.Float32bits(x) == math.Float32bits(y) }) {
			t.Fatalf("block %d: the prefill's logits differ from the first run's", block)
		}
		return d
	}
	fmt.Fprintf(os.Stderr, "[BLK-G3] %s, %d tokens: today's block %d rows (%v passes), the 128 MiB block %d rows (%v passes)\n",
		filepath.Base(path), n, today, i8Passes(n, today), small, i8Passes(n, small))
	var ratios []float64
	start := time.Now()
	for p := 0; p <= pairs; p++ { // pair 0 is the warm-up
		var dt, ds time.Duration
		if p%2 == 0 {
			dt, ds = run(today), run(small)
		} else {
			ds = run(small)
			dt = run(today)
		}
		label := fmt.Sprintf("pair %d of %d", p, pairs)
		if p == 0 {
			label = "warm-up"
		} else {
			ratios = append(ratios, ds.Seconds()/dt.Seconds())
		}
		fmt.Fprintf(os.Stderr, "[BLK-G3] %s: today %.2fs, 128 MiB %.2fs (ratio %.4f); elapsed %.0fs\n",
			label, dt.Seconds(), ds.Seconds(), ds.Seconds()/dt.Seconds(), time.Since(start).Seconds())
	}
	sort.Float64s(ratios)
	med := ratios[len(ratios)/2]
	fmt.Fprintf(os.Stderr, "[BLK-G3] median 128 MiB / today %.4f (pairs %v)\n", med, ratios)
	switch {
	case med > 1.30:
		t.Errorf("BLK-G3 FAIL: median ratio %.4f", med)
	case med > 1.15:
		t.Errorf("BLK-G3 PARKED for the owner: median ratio %.4f", med)
	default:
		t.Logf("BLK-G3 PASS: median ratio %.4f", med)
	}
}
