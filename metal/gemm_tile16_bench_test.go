//go:build darwin && goinfer_testhooks

package metal

import (
	"context"
	"fmt"
	"math"
	"os"
	"slices"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestGemmTile16_probe (D-B02): GPU time of one gemm_w4f16_tile dispatch at each tile, over pass sizes 8-256 rows, at the
// Qwen1.5-MoE expert shapes (gate|up N 2816 K 2048, down N 2048 K 1408) and the 1.5B's and 7B's. GOINFER_GEMM16_SHAPES (a substring list) narrows the shapes. 32 dispatches per command
// buffer, best of 15 buffers, tiles interleaved buffer by buffer. A kernel microbenchmark: it picks the selector's
// thresholds, and the pass A/B says what they are worth. GOINFER_GEMM16=1.
func TestGemmTile16_probe(t *testing.T) {
	if os.Getenv("GOINFER_GEMM16") != "1" {
		t.Skip("set GOINFER_GEMM16=1: a timed kernel probe")
	}
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Skipf("no metal device: %v", err)
	}
	defer d.ReleaseObjects()
	defer d.ReleaseAll()
	lib, err := d.CompileLibrary(prefillKernels, MSL3_1)
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	type tile struct {
		name   string
		tm, tn int
		p      Pipeline
	}
	var tiles []tile
	for _, x := range []tile{{name: "m64n64", tm: 64, tn: 64}, {name: "m64n32", tm: 64, tn: 32}, {name: "m32n64", tm: 32, tn: 64},
		{name: "m32n32", tm: 32, tn: 32}, {name: "m16n64", tm: 16, tn: 64}, {name: "m16n32", tm: 16, tn: 32}} {
		p, err := d.NewComputePipeline(lib, "gemm_w4f16_"+x.name)
		if err != nil {
			t.Fatalf("pipeline %s: %v", x.name, err)
		}
		x.p = p
		tiles = append(tiles, x)
	}
	cq := d.NewCommandQueue()
	const per, bufs = 32, 15
	for _, sh := range []struct {
		what string
		N, K int
	}{{"moe gate|up", 2816, 2048}, {"moe down", 2048, 1408}, {"1.5B qkv", 2048, 1536}, {"1.5B o", 1536, 1536},
		{"1.5B gate|up", 17920, 1536}, {"1.5B down", 1536, 8960}, {"7B qkv", 4608, 3584}, {"7B down", 3584, 18944}} {
		if f := os.Getenv("GOINFER_GEMM16_SHAPES"); f != "" && !strings.Contains(f, sh.what) {
			continue
		}
		w, ws := d.NewBufferLen(sh.N*sh.K/8), NewBufferU16s(d, make([]uint16, sh.N*sh.K/32))
		uN, uK, bB, uMode := NewBufferU32(d, uint32(sh.N)), NewBufferU32(d, uint32(sh.K)), NewBufferFloats(d, make([]float32, sh.N)), NewBufferU32(d, 0)
		for _, rows := range []int{8, 16, 24, 32, 40, 48, 56, 64, 72, 96, 128, 160, 192, 256} {
			a, c, uM := NewBufferU16s(d, make([]uint16, rows*sh.K)), NewBufferU16s(d, make([]uint16, rows*sh.N)), NewBufferU32(d, uint32(rows))
			best := make([]time.Duration, len(tiles))
			for i := range best {
				best[i] = time.Hour
			}
			for b := range bufs + 2 {
				for i, tl := range tiles {
					e := cq.Begin()
					for range per {
						e.Dispatch2D(tl.p, (sh.N+tl.tn-1)/tl.tn, (rows+tl.tm-1)/tl.tm, 128, 1, a, w, ws, c, uM, uN, uK, bB, uMode)
					}
					e.End()
					dt := time.Duration((e.GPUEnd() - e.GPUStart()) * 1e9)
					if b >= 2 && dt < best[i] {
						best[i] = dt
					}
				}
			}
			var cells []string
			for i, tl := range tiles {
				cells = append(cells, fmt.Sprintf("%s %.1f", tl.name, float64(best[i])/per/1e3))
			}
			fmt.Fprintf(os.Stderr, "[gemm16] %-12s rows %2d (us/dispatch): %s\n", sh.what, rows, strings.Join(cells, "  "))
		}
	}
}

// TestDB02_tileAB: D-B02's tile rule in the whole pass, in-process. A fresh prompt's PrefillLast, wall time, with
// gemmTile's default rule and with A-P01's ("a01"), arms alternated rep by rep, the last row's logits compared every rep.
// GOINFER_AUDIT_MODEL picks the checkpoint (default the 1.5B), GOINFER_TILE16_MS the prompt lengths. The batched-pass
// floor is 0 so every length takes the pass.
//
//	GOINFER_TILE16=1 go test -tags goinfer_testhooks -count=1 -run '^TestDB02_tileAB$' -v ./metal/
func TestDB02_tileAB(t *testing.T) {
	if os.Getenv("GOINFER_TILE16") != "1" {
		t.Skip("set GOINFER_TILE16=1 (loads a real checkpoint)")
	}
	Ms := []int{16, 24, 40, 72, 100, 160, 512}
	if v := os.Getenv("GOINFER_TILE16_MS"); v != "" {
		Ms = nil
		for _, f := range strings.Split(v, ",") {
			if n, err := strconv.Atoi(strings.TrimSpace(f)); err == nil && n > 0 {
				Ms = append(Ms, n)
			}
		}
	}
	name, a := auditLoad(t, "qwen2.5-coder-1.5b-instruct-q4_k_m.gguf", 1, slices.Max(Ms)+64,
		&decoder.Knobs{"GOINFER_METAL_FAST_PREFILL_FLOOR": "0"})
	defer func() { gemmTilePolicy = "" }()
	t0 := time.Now()
	reps := auditReps(7)
	for _, M := range Ms {
		embs := auditEmbs(a.r, M, M)
		ms := map[string][]float64{}
		var ref []float32
		for rep := range reps + 1 {
			order := []string{"a01", ""}
			if rep%2 == 1 {
				order = []string{"", "a01"}
			}
			for _, pol := range order {
				gemmTilePolicy = pol
				st := time.Now()
				lg, err := a.PrefillLast(context.Background(), embs, 0)
				if err != nil {
					t.Fatalf("M=%d: %v", M, err)
				}
				dt := time.Since(st).Seconds() * 1e3
				if ref == nil {
					ref = append([]float32(nil), lg...)
				}
				for j := range ref {
					if math.Float32bits(ref[j]) != math.Float32bits(lg[j]) {
						t.Fatalf("M=%d rep %d: logit %d differs between the tile rules", M, rep, j)
					}
				}
				if rep > 0 {
					ms[pol] = append(ms[pol], dt)
				}
			}
		}
		ratio := make([]float64, reps)
		above := 0
		for i := range ratio {
			ratio[i] = ms["a01"][i] / ms[""][i]
			if ratio[i] > 1 {
				above++
			}
		}
		auditHB("d-b02-tile", t0, "%s M=%d: A-P01 %.2f ms, padding rule %.2f ms (medians of %d); RESULT a01/new median %.3f, %d of %d reps above 1 (per rep %s)",
			name, M, auditMedian(ms["a01"]), auditMedian(ms[""]), reps, auditMedian(ratio), above, reps, auditFmt3(ratio))
	}
}
