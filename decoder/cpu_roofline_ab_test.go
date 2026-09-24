//go:build goinfer_testhooks

package decoder

import (
	"context"
	"os"
	"sort"
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/tokenizer"
)

// cpuDecodeAB is the paired, in-process, ABBA harness for CPU-decode knobs on one loaded model
// (docs/measurements/cpu-decode-roofline-2026-09-23.md). Same discipline as TestR9_cpuTuningAB:
// one load, the arm flipped between generations so drift cannot pose as an effect, the order of
// the two arms alternating per pair, the paired ratio (not pooled means) reported. It reads
// lastDecodeSplit.fwd — the forward ms/token DECODE TIMING already computes — so nothing here
// times anything of its own.
type cpuDecodeAB struct {
	t   *testing.T
	m   *Model
	ids []int
}

func newCPUDecodeAB(t *testing.T, envKey, def string, depth int) *cpuDecodeAB {
	t.Helper()
	if os.Getenv("GOINFER_HEAVY_TESTS") == "" {
		t.Skip("set GOINFER_HEAVY_TESTS=1 (loads a real checkpoint on CPU)")
	}
	path := os.Getenv(envKey)
	if path == "" {
		path = os.ExpandEnv(def)
	}
	if _, err := os.Stat(path); err != nil {
		t.Skipf("no fixture at %s", path)
	}
	prev := decodeTiming
	decodeTiming = true
	t.Cleanup(func() { decodeTiming = prev })
	m, err := Load(path, Options{Backend: "cpu", Quant: "int4"})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	t.Cleanup(func() { m.Close() })
	tk, err := tokenizer.LoadGGUF(path)
	if err != nil {
		t.Fatalf("tokenizer: %v", err)
	}
	ids, err := tk.Encode("Continue this text. "+strings.Repeat(" the", depth), true)
	if err != nil {
		t.Fatalf("encode: %v", err)
	}
	if len(ids) > depth {
		ids = ids[:depth]
	}
	return &cpuDecodeAB{t: t, m: m, ids: ids}
}

// gen decodes 24 greedy tokens and returns the forward ms/token plus the greedy token stream, so an
// arm change that altered numerics shows up as a different stream and not just a different time.
func (h *cpuDecodeAB) gen() (float64, []int) {
	h.t.Helper()
	out, g := h.m.Generate(context.Background(), h.ids, 24, SamplingParams{Temperature: 0})
	var toks []int
	for id := range out {
		toks = append(toks, id)
	}
	if g.err != nil {
		h.t.Fatalf("generate: %v", g.err)
	}
	return lastDecodeSplit.fwd, toks
}

// run flips set(true)/set(false) ABBA for `pairs` pairs and logs each pair and the median paired
// ratio OFF/ON (>1 means ON is faster). It fails if the two arms ever emit different tokens: every
// knob measured here is claimed bit-identical, so a stream difference is a defect, not a result.
func (h *cpuDecodeAB) run(name string, pairs int, set func(on bool)) (median float64) {
	h.t.Helper()
	h.gen() // warm-up
	set(false)
	_, ref := h.gen()
	var ratios []float64
	var sumOn, sumOff float64
	for p := 0; p < pairs; p++ {
		var on, off float64
		var tOn, tOff []int
		if p%2 == 0 {
			set(true)
			on, tOn = h.gen()
			set(false)
			off, tOff = h.gen()
		} else {
			set(false)
			off, tOff = h.gen()
			set(true)
			on, tOn = h.gen()
		}
		for i := range ref {
			if tOn[i] != ref[i] || tOff[i] != ref[i] {
				h.t.Fatalf("%s pair %d: greedy stream diverged at token %d (on=%d off=%d ref=%d) — not bit-identical",
					name, p, i, tOn[i], tOff[i], ref[i])
			}
		}
		ratios = append(ratios, off/on)
		sumOn, sumOff = sumOn+on, sumOff+off
		h.t.Logf("%s pair %d: ON %.2f | OFF %.2f ms/tok | ON is %.3fx faster", name, p, on, off, off/on)
	}
	set(false)
	sort.Float64s(ratios)
	median = ratios[len(ratios)/2]
	h.t.Logf("%s: mean ON %.2f OFF %.2f ms/tok | paired median %.3fx (min %.3f max %.3f, n=%d)",
		name, sumOn/float64(pairs), sumOff/float64(pairs), median, ratios[0], ratios[len(ratios)-1], pairs)
	return median
}

// TestCPURoofline_w4a8Batch re-measures R-06's parked GOINFER_W4A8_BATCH (q/k/v as one fork/join,
// gate/up as one) against the roofline accounting: the fork/join cost the per-component GB/s table
// attributes to the small projections predicts a win of about 3-4 ms of a 54 ms 1.5B token.
//
//	GOINFER_HEAVY_TESTS=1 go test -tags goinfer_testhooks ./decoder/ -run TestCPURoofline_w4a8Batch -v -count=1 -timeout 30m
func TestCPURoofline_w4a8Batch(t *testing.T) {
	for _, mc := range []struct{ name, env, def string }{
		{"1.5B", "GOINFER_CPU_MODEL", "$HOME/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"},
		{"0.5B", "GOINFER_CPU_MODEL_05B", "$HOME/models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf"},
	} {
		t.Run(mc.name, func(t *testing.T) {
			h := newCPUDecodeAB(t, mc.env, mc.def, 128)
			orig := w4a8BatchEnabled
			t.Cleanup(func() { w4a8BatchEnabled = orig })
			h.run("w4a8 batch ("+mc.name+")", 5, func(on bool) { w4a8BatchEnabled = on })
		})
	}
}

// TestCPURoofline_fusedGateUp measures the fused gate+up+SwiGLU fork/join (cpu_gateup_fused.go) two
// ways: against the plain baseline (batch OFF), and as the marginal step on top of R-06's batch (batch ON in
// both arms) — the second is what a shipped stack would actually add.
//
//	GOINFER_HEAVY_TESTS=1 go test -tags goinfer_testhooks ./decoder/ -run TestCPURoofline_fusedGateUp -v -count=1 -timeout 30m
func TestCPURoofline_fusedGateUp(t *testing.T) {
	for _, mc := range []struct{ name, env, def string }{
		{"1.5B", "GOINFER_CPU_MODEL", "$HOME/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"},
		{"0.5B", "GOINFER_CPU_MODEL_05B", "$HOME/models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf"},
		{"7B", "GOINFER_CPU_MODEL_D7", "$HOME/models/qwen2.5-7b-instruct-q4_k_m.gguf"},
	} {
		t.Run(mc.name, func(t *testing.T) {
			h := newCPUDecodeAB(t, mc.env, mc.def, 128)
			origB, origF := w4a8BatchEnabled, cpuFusedGateUp
			t.Cleanup(func() { w4a8BatchEnabled, cpuFusedGateUp = origB, origF })
			w4a8BatchEnabled = false
			h.run("fused gate+up vs plain ("+mc.name+")", 5, func(on bool) { cpuFusedGateUp = on })
			w4a8BatchEnabled = true
			h.run("fused gate+up on top of batch ("+mc.name+")", 5, func(on bool) { cpuFusedGateUp = on })
		})
	}
}

// TestCPURoofline_fusedGateUp_logitsBitIdentical is the strong form of the A/B harness's token check: a
// matching greedy stream proves little about numerics (CLAUDE.md: LFM2's two bugs both held argmax at
// logit cosine 0.897), so this captures the FULL logits vector at every decode step through the
// sampler's own LogitProcessor hook, in both arms, and compares them with != on every element.
//
//	GOINFER_HEAVY_TESTS=1 GOINFER_NO_FIT_GUARD=1 go test -tags goinfer_testhooks ./decoder/ -run TestCPURoofline_fusedGateUp_logitsBitIdentical -v -count=1 -timeout 30m
func TestCPURoofline_fusedGateUp_logitsBitIdentical(t *testing.T) {
	for _, mc := range []struct{ name, env, def string }{
		{"0.5B", "GOINFER_CPU_MODEL_05B", "$HOME/models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf"},
		{"1.5B", "GOINFER_CPU_MODEL", "$HOME/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"},
		{"7B", "GOINFER_CPU_MODEL_D7", "$HOME/models/qwen2.5-7b-instruct-q4_k_m.gguf"},
	} {
		t.Run(mc.name, func(t *testing.T) {
			h := newCPUDecodeAB(t, mc.env, mc.def, 128)
			orig := cpuFusedGateUp
			t.Cleanup(func() { cpuFusedGateUp = orig })
			capture := func(on bool) [][]float32 {
				cpuFusedGateUp = on
				var steps [][]float32
				sp := SamplingParams{Temperature: 0, LogitProcessor: func(_ []int, logits []float32) {
					steps = append(steps, append([]float32(nil), logits...))
				}}
				out, g := h.m.Generate(context.Background(), h.ids, 48, sp)
				for range out {
				}
				if g.err != nil {
					t.Fatalf("generate: %v", g.err)
				}
				return steps
			}
			off := capture(false)
			on := capture(true)
			if len(on) != len(off) || len(on) == 0 {
				t.Fatalf("captured %d steps ON vs %d OFF", len(on), len(off))
			}
			bad := 0
			for s := range off {
				for i := range off[s] {
					if on[s][i] != off[s][i] {
						bad++
						if bad <= 3 {
							t.Errorf("step %d logit %d: fused %v != unfused %v", s, i, on[s][i], off[s][i])
						}
					}
				}
			}
			t.Logf("%s: %d decode steps x %d logits compared, %d differ", mc.name, len(off), len(off[0]), bad)
		})
	}
}
