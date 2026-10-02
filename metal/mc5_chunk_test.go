//go:build darwin && goinfer_testhooks

package metal

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestMC5_chunkCost: EXPLORATORY. What chunking itself costs a long prefill, with no decoding in between: one
// PrefillLast over 3000 tokens against the same prompt in chunks of C (cut as mc3Prefill cuts them — only while two
// chunks' worth remains, so the last pass is C..2C-1), wall time per pass, 3 reps each, interleaved.
//
//	GOINFER_METAL_MC3=1 go test -tags goinfer_testhooks -count=1 -run '^TestMC5_chunkCost$' -v ./metal/
func TestMC5_chunkCost(t *testing.T) {
	if os.Getenv("GOINFER_METAL_MC3") != "1" {
		t.Skip("set GOINFER_METAL_MC3=1 (loads a real checkpoint)")
	}
	home, _ := os.UserHomeDir()
	m, err := decoder.Load(filepath.Join(home, "models", "qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"),
		decoder.Options{Backend: "metal", Quant: "int4", ResidentContext: 4096, ResidentKVSlots: 2})
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	a := m.ResidentForwardForTest().(*metalResident)
	r := a.r
	const N = 3000
	seed := uint32(97531)
	embs := make([][]float32, N)
	for i := range embs {
		seed ^= seed << 13
		seed ^= seed >> 17
		seed ^= seed << 5
		embs[i] = mc3Emb(r, int(seed%20000))
	}
	run := func(C int) (total, maxPass float64, passes int) {
		t0 := time.Now()
		from := 0
		for C > 0 && N-from >= 2*C {
			p0 := time.Now()
			if _, err := a.PrefillLast(context.Background(), embs[from:from+C], from); err != nil {
				t.Fatal(err)
			}
			maxPass = max(maxPass, time.Since(p0).Seconds())
			passes++
			from += C
		}
		p0 := time.Now()
		if _, err := a.PrefillLast(context.Background(), embs[from:], from); err != nil {
			t.Fatal(err)
		}
		maxPass = max(maxPass, time.Since(p0).Seconds())
		return time.Since(t0).Seconds(), maxPass, passes + 1
	}
	arms := []int{0, 128, 256, 512}
	res := map[int][]float64{}
	mx := map[int]float64{}
	np := map[int]int{}
	for rep := 0; rep < 3; rep++ {
		for k := range arms {
			C := arms[(k+rep)%len(arms)]
			tot, mp, n := run(C)
			res[C] = append(res[C], tot)
			mx[C], np[C] = max(mx[C], mp), n
		}
	}
	for _, C := range arms {
		v := res[C]
		sort.Float64s(v)
		fmt.Fprintf(os.Stderr, "[mc5-cost] C=%d: %d passes, total %.3f s (median of 3), longest pass %.3f s\n", C, np[C], v[1], mx[C])
	}
}

// TestMC5_passCost: EXPLORATORY. One PrefillLast pass's time against its length at two depths, to separate a pass's
// fixed cost (the intercept) from its per-token cost: C = 16..256 tokens at startPos 64 and 2048, median of 5.
//
//	GOINFER_METAL_MC3=1 go test -tags goinfer_testhooks -count=1 -run '^TestMC5_passCost$' -v ./metal/
func TestMC5_passCost(t *testing.T) {
	if os.Getenv("GOINFER_METAL_MC3") != "1" {
		t.Skip("set GOINFER_METAL_MC3=1 (loads a real checkpoint)")
	}
	home, _ := os.UserHomeDir()
	m, err := decoder.Load(filepath.Join(home, "models", "qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"),
		decoder.Options{Backend: "metal", Quant: "int4", ResidentContext: 4096, ResidentKVSlots: 2})
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	a := m.ResidentForwardForTest().(*metalResident)
	r := a.r
	embs := make([][]float32, 2400)
	for i := range embs {
		embs[i] = mc3Emb(r, 1000+(i*37)%15000)
	}
	if _, err := a.PrefillLast(context.Background(), embs[:2048], 0); err != nil {
		t.Fatal(err)
	}
	for _, at := range []int{64, 2048} {
		line := fmt.Sprintf("startPos %d:", at)
		for _, C := range []int{16, 64, 128, 256} {
			var ts []float64
			for rep := 0; rep < 5; rep++ {
				t0 := time.Now()
				if _, err := a.PrefillLast(context.Background(), embs[at:at+C], at); err != nil {
					t.Fatal(err)
				}
				ts = append(ts, time.Since(t0).Seconds()*1e3)
			}
			sort.Float64s(ts)
			line += fmt.Sprintf("  C=%d %.1f ms (GPU %.1f)", C, ts[2], (r.gpuEnd-r.gpuStart)*1e3)
		}
		fmt.Fprintf(os.Stderr, "[mc5-pass] %s\n", line)
	}
}
