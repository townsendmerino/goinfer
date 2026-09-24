//go:build goinfer_testhooks

package decoder

import (
	"context"
	"fmt"
	"os"
	"runtime"
	"sort"
	"strings"
	"testing"
	"time"

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

// procStatusMB reads one "Rss*:" line from /proc/self/status in MB (0 if absent, e.g. off linux).
func procStatusMB(key string) float64 {
	b, err := os.ReadFile("/proc/self/status")
	if err != nil {
		return 0
	}
	for _, line := range strings.Split(string(b), "\n") {
		if strings.HasPrefix(line, key+":") {
			var kb float64
			fmt.Sscanf(strings.TrimSpace(strings.TrimPrefix(line, key+":")), "%f", &kb)
			return kb / 1024
		}
	}
	return 0
}

// TestCPURoofline_giwVsDirect times CPU decode with the weights loaded through a .giw sidecar (zero-copy
// aliases of a page-cache mapping) against a direct .gguf load (heap copies), with a second direct load as
// the do-nothing arm. Pre-registered in docs/measurements/cpu-giw-vs-direct-2026-09-24.md.
//
//	GOINFER_HEAVY_TESTS=1 go test -tags goinfer_testhooks ./decoder/ -run TestCPURoofline_giwVsDirect -v -count=1 -timeout 60m
func TestCPURoofline_giwVsDirect(t *testing.T) {
	for _, mc := range []struct{ name, env, def string }{
		{"1.5B", "GOINFER_CPU_MODEL", "$HOME/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"},
		{"7B", "GOINFER_CPU_MODEL_D7", "$HOME/models/qwen2.5-7b-instruct-q4_k_m.gguf"},
	} {
		t.Run(mc.name, func(t *testing.T) {
			var promptIDs []int // every arm decodes the SAME ids, tokenized once from the .gguf
			load := func(label, path string) *cpuDecodeAB {
				var before runtime.MemStats
				runtime.GC()
				runtime.ReadMemStats(&before)
				anon0, file0 := procStatusMB("RssAnon"), procStatusMB("RssFile")
				t0 := time.Now()
				var h *cpuDecodeAB
				if strings.HasSuffix(path, ".giw") {
					// a .giw bundle carries its own tokenizer, not a GGUF one, so it cannot go through
					// newCPUDecodeAB's tokenizer.LoadGGUF; load the weights directly and reuse the ids
					m, err := Load(path, Options{Backend: "cpu", Quant: "int4"})
					if err != nil {
						t.Fatalf("load %s: %v", path, err)
					}
					t.Cleanup(func() { m.Close() })
					h = &cpuDecodeAB{t: t, m: m, ids: promptIDs}
				} else {
					t.Setenv(mc.env, path)
					h = newCPUDecodeAB(t, mc.env, mc.def, 128)
					if promptIDs == nil {
						promptIDs = h.ids
					}
				}
				el := time.Since(t0)
				var after runtime.MemStats
				runtime.GC()
				runtime.ReadMemStats(&after)
				fmt.Fprintf(os.Stderr, "[giw-vs-direct %s] loaded %-7s in %6.2fs | heap-in-use +%.0f MB | RssAnon +%.0f MB | RssFile +%.0f MB\n",
					mc.name, label, el.Seconds(), float64(after.HeapInuse-before.HeapInuse)/(1<<20),
					procStatusMB("RssAnon")-anon0, procStatusMB("RssFile")-file0)
				return h
			}
			gguf := os.ExpandEnv(mc.def)
			giw := strings.TrimSuffix(gguf, ".gguf") + ".int4.cpu-amd64.giw"
			if _, err := os.Stat(giw); err != nil {
				t.Skipf("no sidecar at %s (build it with cmd/prequant -quant int4 -target cpu)", giw)
			}
			direct := load("direct", gguf)
			direct2 := load("direct'", gguf)
			side := load("giw", giw)
			// the token hard gate is inside pairAB; it compares against direct's stream
			pairAB(t, mc.name+" A/A direct'/direct", direct, direct2, 7)
			pairAB(t, mc.name+" giw/direct", direct, side, 7)
		})
	}
}

// pairAB alternates generations between two loaded models ABBA and logs per-pair ratios b_ms/a_ms and
// their median, min and max. Every generation's greedy stream must equal a's first one.
func pairAB(t *testing.T, name string, a, b *cpuDecodeAB, pairs int) {
	t.Helper()
	a.gen()
	b.gen() // warm-ups: page in the mapping, settle the scratch
	_, ref := a.gen()
	check := func(which string, p int, toks []int) {
		if len(toks) != len(ref) {
			t.Fatalf("%s pair %d (%s): %d tokens vs %d", name, p, which, len(toks), len(ref))
		}
		for i := range ref {
			if toks[i] != ref[i] {
				t.Fatalf("%s pair %d (%s): greedy stream diverged at token %d (%d vs %d) — speed result void", name, p, which, i, toks[i], ref[i])
			}
		}
	}
	var ratios []float64
	for p := 0; p < pairs; p++ {
		var ma, mb float64
		var ta, tb []int
		if p%2 == 0 {
			ma, ta = a.gen()
			mb, tb = b.gen()
		} else {
			mb, tb = b.gen()
			ma, ta = a.gen()
		}
		check("a", p, ta)
		check("b", p, tb)
		ratios = append(ratios, mb/ma)
		fmt.Fprintf(os.Stderr, "[%s] pair %d: a %.2f | b %.2f ms/tok | b/a %.4f\n", name, p, ma, mb, mb/ma)
	}
	sort.Float64s(ratios)
	fmt.Fprintf(os.Stderr, "[%s] median b/a %.4f (min %.4f max %.4f, n=%d)\n", name, ratios[len(ratios)/2], ratios[0], ratios[len(ratios)-1], pairs)
}
