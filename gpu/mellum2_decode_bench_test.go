//go:build gpu

package gpu

import (
	"context"
	"math/rand"
	"os"
	"sort"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/tokenizer"
)

// TestMellum2_decodeThroughput measures decode tok/s of the real Mellum2 (12B-class,
// 64-expert top-8 MoE, 3:1 sliding/full attention, sliding window 1024) on the webgpu
// backend, and reports whether it runs RESIDENT (full-residency DecodeRunner) or STAGED
// (per-matmul backend) + the path reason.
//
// On an 8 GB card Mellum2 stages: int8 MoE experts (the only kind the resident builder
// stacks) make the model ~12 GB > VRAM, while int4 fits VRAM but the resident MoE
// stacking is int8-only (gpu/residency.go). So this measures the STAGED number here;
// set GOINFER_MELLUM_QUANT + a bigger card to get the resident one.
//
//	GOINFER_MELLUM_BENCH=1 [GOINFER_MELLUM_QUANT=int8int8] \
//	  go test -tags gpu -run TestMellum2_decodeThroughput -v -timeout 40m
func TestMellum2_decodeThroughput(t *testing.T) {
	requireHeavyModel(t)
	if os.Getenv("GOINFER_MELLUM_BENCH") == "" {
		t.Skip("set GOINFER_MELLUM_BENCH=1 to run the Mellum2 decode bench")
	}
	dir := os.Getenv("GOINFER_MELLUM_DIR")
	if dir == "" {
		dir = os.ExpandEnv("$HOME/models/mellum2-unq")
	}
	if _, err := os.Stat(dir); err != nil {
		t.Skipf("no Mellum2 checkpoint: %v", err)
	}
	quant := os.Getenv("GOINFER_MELLUM_QUANT")
	if quant == "" {
		quant = "int8int8"
	}
	newOrSkipHW(t).Close() // real-HW gate

	m, err := decoder.Load(dir, decoder.Options{Backend: "webgpu", Quant: quant})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	defer m.Close()

	resident := m.ResidentActive()
	hidden, nLayers, nH, nKV, hd, inter, vocab := m.Dims()
	t.Logf("Mellum2: quant=%s  PATH=%s  (arch-eligible=%v)", quant, map[bool]string{true: "RESIDENT", false: "STAGED"}[resident], m.DecodeRunnerEligible())
	t.Logf("dims: hidden=%d layers=%d heads=%d/%d hd=%d moe_inter=%d vocab=%d", hidden, nLayers, nH, nKV, hd, inter, vocab)

	us := func(d time.Duration) float64 { return float64(d.Microseconds()) }
	med := func(xs []float64) float64 {
		s := append([]float64(nil), xs...)
		sort.Float64s(s)
		return s[len(s)/2]
	}

	if resident {
		// Resident path: drive the DecodeRunner directly for tok/s + TWrite/TEncode/TSync.
		rd := m.ResidentForwardForTest().(*residentDecoder)
		rng := rand.New(rand.NewSource(1))
		emb := make([]float32, hidden)
		for i := range emb {
			emb[i] = float32(rng.NormFloat64()) * 0.5
		}
		measure := func(label string, warmTo, N int) {
			rd.Reset()
			for i := 0; i < warmTo; i++ {
				if _, err := rd.Forward(emb, i); err != nil {
					t.Fatalf("warm[%d]: %v", i, err)
				}
			}
			wall := make([]float64, N)
			tw := make([]float64, N)
			te := make([]float64, N)
			ts := make([]float64, N)
			for i := 0; i < N; i++ {
				t0 := time.Now()
				if _, err := rd.Forward(emb, warmTo+i); err != nil {
					t.Fatalf("Forward[%d]: %v", i, err)
				}
				wall[i] = float64(time.Since(t0).Microseconds())
				r := rd.runner
				tw[i], te[i], ts[i] = us(r.TWrite), us(r.TEncode), us(r.TSync)
			}
			mw, mt, msy, mwall := med(tw), med(te), med(ts), med(wall)
			tot := mw + mt + msy
			t.Logf("=== RESIDENT decode — %s (pos %d..%d) ===", label, warmTo, warmTo+N)
			t.Logf("  %.0f µs/tok = %.2f ms/tok = %.1f tok/s", mwall, mwall/1000, 1e6/mwall)
			t.Logf("  TWE: TWrite %.0f%% | TEncode %.0f%% | TSync %.0f%%", 100*mw/tot, 100*mt/tot, 100*msy/tot)
		}
		measure("short context", 64, 200)
		measure("steady sliding state", 1100, 200)
		return
	}

	// Staged path: measure steady-state decode tok/s via Generate (inter-token rate,
	// TTFT excluded). This is the path Mellum2 actually takes on an 8 GB card.
	tk, err := tokenizer.Load(dir)
	if err != nil {
		t.Fatalf("tokenizer: %v", err)
	}
	ids, _ := tk.Encode("Write a short note about recursion in programming.\n", true)
	greedy := decoder.SamplingParams{Temperature: 0}
	rate := func(maxTokens int) float64 {
		ch, _ := m.Generate(context.Background(), ids, maxTokens, greedy)
		if ch == nil {
			t.Fatal("nil generate channel")
		}
		var ttft time.Duration
		n := 0
		t0 := time.Now()
		for range ch {
			if n == 0 {
				ttft = time.Since(t0)
			}
			n++
		}
		total := time.Since(t0)
		if n < 2 {
			t.Fatalf("produced %d tokens", n)
		}
		return float64(n-1) / (total - ttft).Seconds()
	}
	rate(4) // warm
	const rounds, gen = 4, 32
	best := 0.0
	for i := 0; i < rounds; i++ {
		if tps := rate(gen); tps > best {
			best = tps
		}
	}
	t.Logf("=== STAGED webgpu decode (per-matmul backend, quant=%s) ===", quant)
	t.Logf("  steady-state %.2f tok/s = %.1f ms/token (best of %d × %d-tok greedy)", best, 1000/best, rounds, gen)
}
