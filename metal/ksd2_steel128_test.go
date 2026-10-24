//go:build darwin && goinfer_testhooks

package metal

import (
	"context"
	"crypto/sha256"
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"slices"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestKSD2_steel128Logits is D2-G1's instrument (docs/tasks/task-kernel-survey-2026-10.md, "KS-D2 gates"): Qwen2.5-Coder-1.5B
// (head dim 128, the Metal .giw), PrefillLast at K=512 and 3,900, the SHA-256 of the last logits' bits. Run from a test binary
// at the change and one at its base; the hashes must match. A real checkpoint (GOINFER_HEAVY_TESTS=1).
func TestKSD2_steel128Logits(t *testing.T) {
	requireHeavyModel(t)
	path := os.ExpandEnv("$HOME/models/qwen2.5-coder-1.5b-instruct-q4_k_m.int4.metal.v17.giw")
	m, err := decoder.Load(path, decoder.Options{Backend: "metal", ResidentContext: 4096})
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	a, ok := m.ResidentForwardForTest().(*metalResident)
	if !ok {
		t.Fatalf("not Metal-resident: %s", m.ResidentDecline())
	}
	_, _, _, _, _, _, vocab := m.Dims()
	for _, n := range []int{512, 3900} {
		embs := make([][]float32, n)
		for i := range embs {
			embs[i] = m.EmbedResidentForTest((i*7919 + 13) % vocab)
		}
		a.Reset()
		lg, err := a.PrefillLast(context.Background(), embs, 0)
		if err != nil {
			t.Fatal(err)
		}
		h := sha256.New()
		for _, v := range lg {
			_ = binary.Write(h, binary.LittleEndian, math.Float32bits(v))
		}
		fmt.Fprintf(os.Stderr, "[D2-G1] K=%d last-logits sha256 %x (%d values)\n", n, h.Sum(nil), len(lg))
	}
}

// TestKSD2_prefillSpeed is D2-G4: Qwen2.5-Coder-0.5B (head dim 64, the .gguf at int4), PrefillLast at K=3,900 and at K=512,
// attention_prefill_steel against attention_prefill_fused (prefillSteelAttnOff) on one resident, A B B A over 10 pairs after 2
// warm runs each. A timed run: hold the timing lock (GOINFER_KSD2_TIMING=1).
func TestKSD2_prefillSpeed(t *testing.T) {
	if os.Getenv("GOINFER_KSD2_TIMING") == "" {
		t.Skip("set GOINFER_KSD2_TIMING=1 (a timed run under the timing lock)")
	}
	path := os.ExpandEnv("$HOME/models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf")
	m, err := decoder.Load(path, decoder.Options{Backend: "metal", Quant: "int4", DirectLoad: true, ResidentContext: 4096})
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	a, ok := m.ResidentForwardForTest().(*metalResident)
	if !ok {
		t.Fatalf("not Metal-resident: %s", m.ResidentDecline())
	}
	if _, steel := a.r.prefillAttnKernels(); !steel {
		t.Fatal("the 0.5B's prefill does not take attention_prefill_steel")
	}
	_, _, _, _, _, _, vocab := m.Dims()
	for _, n := range []int{3900, 512} {
		embs := make([][]float32, n)
		for i := range embs {
			embs[i] = m.EmbedResidentForTest((i*7919 + 13) % vocab)
		}
		run := func(fused bool) float64 {
			prev := prefillSteelAttnOff
			prefillSteelAttnOff = fused
			defer func() { prefillSteelAttnOff = prev }()
			a.Reset()
			t0 := time.Now()
			if _, err := a.PrefillLast(context.Background(), embs, 0); err != nil {
				t.Fatal(err)
			}
			return time.Since(t0).Seconds()
		}
		for range 2 {
			run(true)
			run(false)
		}
		var ratios []float64
		below := 0
		for i := range 10 {
			var fused, steel float64
			if i%2 == 0 {
				fused, steel = run(true), run(false)
			} else {
				steel, fused = run(false), run(true)
			}
			ratios = append(ratios, steel/fused)
			if steel < fused {
				below++
			}
			fmt.Fprintf(os.Stderr, "[D2-G4] K=%d pair %d: fused %.1f ms, steel %.1f ms, steel/fused %.4f\n", n, i, fused*1e3, steel*1e3, steel/fused)
		}
		slices.Sort(ratios)
		fmt.Fprintf(os.Stderr, "[D2-G4] K=%d median steel/fused %.4f, %d of 10 pairs below 1\n", n, (ratios[4]+ratios[5])/2, below)
	}
}
