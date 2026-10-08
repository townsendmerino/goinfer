//go:build darwin && goinfer_testhooks

package metal

import (
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestQ4KLaneResidentQwen is the q4k lane's by-day wiring smoke (docs/tasks/task-metal-q4k-2026-10.md), not G-Q2: a
// Qwen2.5-Coder q4_k_m at --quant q4k, resident on Metal, against the CPU at q4k on the same file, teacher-forced over
// 32 positions. The two files cover the lane's shapes between them:
//   - 0.5B: ffn_down Q4_K in half the layers and Q6_K (int8) in the rest, as Phi-3's; everything else int8 (Q5_0/Q8_0);
//     q/k/v biases; an int8 LM head. It fits by day.
//   - 1.5B: also Q4_K QKV, gate|up and o-proj, with Q4_K and int8 mixed inside the fused QKV (attn_v is Q6_K in half the
//     layers), so a projection runs as two segments. It loads about 3 GB, so it runs under GOINFER_HEAVY_TESTS=1.
//
// The engines load one at a time. A fit-guard refusal on this machine's free memory is a SKIP naming it, never a pass.
//
// The two paths differ by arithmetic only (f32 activations here against the CPU's per-32 int8 ones), so the bar, set
// before the first run, is a per-position last-row logits cosine of at least 0.99 and no NaN, on a decode path that
// reads metal-resident (q4k). A wiring defect (a wrong segment offset, a missing bias, the wrong kind's kernel) puts a
// position far below it. GOINFER_QWEN05_GGUF / GOINFER_QWEN15_GGUF override the default ~/models paths.
func TestQ4KLaneResidentQwen(t *testing.T) {
	for _, c := range []struct {
		name, env, file string
		heavy           bool
	}{
		{"0.5B", "GOINFER_QWEN05_GGUF", "qwen2.5-coder-0.5b-instruct-q4_k_m.gguf", false},
		{"1.5B", "GOINFER_QWEN15_GGUF", "qwen2.5-coder-1.5b-instruct-q4_k_m.gguf", true},
	} {
		t.Run(c.name, func(t *testing.T) {
			if c.heavy {
				requireHeavyModel(t)
			}
			path := os.Getenv(c.env)
			if path == "" {
				home, err := os.UserHomeDir()
				if err != nil {
					t.Skipf("no home directory: %v", err)
				}
				path = filepath.Join(home, "models", c.file)
			}
			if _, err := os.Stat(path); err != nil {
				t.Skipf("no model at %s", path)
			}
			testQ4KLaneResident(t, path)
		})
	}
}

func testQ4KLaneResident(t *testing.T, path string) {
	const positions = 32
	tok := func(p int) int { return 100 + 37*p%5000 }

	load := func(backend string) *decoder.Model {
		m, err := decoder.Load(path, decoder.Options{Backend: backend, Quant: "q4k", ResidentContext: 256})
		if err != nil && strings.Contains(err.Error(), "was NOT loaded") {
			t.Skipf("SKIP, not a pass: the fit guard refused the %s load on this machine's free memory: %v", backend, err)
		}
		if err != nil {
			t.Fatalf("load %s: %v", backend, err)
		}
		return m
	}
	mm := load("metal")
	dp := mm.DecodePath()
	if !strings.HasPrefix(dp, "metal-resident (q4k") {
		decline := mm.ResidentDecline()
		mm.Close()
		t.Fatalf("decode path %q, want metal-resident (q4k); decline: %s", dp, decline)
	}
	rf := mm.ResidentForwardForTest()
	rf.Reset()
	var got [][]float32
	for p := range positions {
		lg, err := rf.Forward(mm.EmbedResidentForTest(tok(p)), p)
		if err != nil {
			t.Fatal(err)
		}
		got = append(got, append([]float32(nil), lg...))
	}
	mm.Close()

	mc := load("cpu")
	defer mc.Close()
	cache := mc.NewCache(positions)
	worst, agree := 1.0, 0
	for p := range positions {
		want, err := mc.ForwardForTest(tok(p), cache)
		if err != nil {
			t.Fatal(err)
		}
		var dot, na, nb float64
		ga, wa := 0, 0
		for i := range want {
			g, w := float64(got[p][i]), float64(want[i])
			if math.IsNaN(g) || math.IsInf(g, 0) {
				t.Fatalf("position %d logit %d is %v", p, i, g)
			}
			dot, na, nb = dot+g*w, na+g*g, nb+w*w
			if got[p][i] > got[p][ga] {
				ga = i
			}
			if want[i] > want[wa] {
				wa = i
			}
		}
		c := dot / math.Sqrt(na*nb)
		worst = math.Min(worst, c)
		if ga == wa {
			agree++
		}
	}
	fmt.Printf("[q4k lane] %s, Metal vs CPU over %d positions: worst cosine %.6f, argmax agree %d/%d (%s)\n",
		filepath.Base(path), positions, worst, agree, positions, dp)
	if worst < 0.99 {
		t.Errorf("worst per-position cosine %.6f, bar 0.99", worst)
	}
}
