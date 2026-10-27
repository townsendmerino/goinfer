//go:build darwin && goinfer_testhooks

package metal

import (
	"context"
	"fmt"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"sort"
	"strings"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/internal/mem/probe"
	"github.com/townsendmerino/goinfer/tokenizer"
)

// d7Load1 is the 1-minute load average (sysctl vm.loadavg), the void rule's first term.
func d7Load1() float64 {
	out, err := exec.Command("sysctl", "-n", "vm.loadavg").Output()
	var l1 float64
	if err != nil {
		return math.NaN()
	}
	if _, err := fmt.Sscanf(strings.TrimSpace(strings.Trim(strings.TrimSpace(string(out)), "{}")), "%g", &l1); err != nil {
		return math.NaN()
	}
	return l1
}

// TestD7_int8StepVsF16Pass is D7's measurement and P3B2-G3's instrument (docs/tasks/task-kv-precision-everywhere-2026-10.md, "D7
// measurement", "P3b-2 gates"): on Qwen2.5-Coder-1.5B from ~/models, the resident prefill of a 2,048- and an 8,192-token prose
// prompt with an f16 cache against an int8 cache, a warm-up pair and five order-alternating pairs per length, in one process. D7
// ran with the int8 arm on the step kernels (main 725c840c); since P3b-2 both arms take the MMA pass, and the median int8 / f16
// ratio must be at most 1.25 at both lengths. A timed run: hold the timing lock.
//
//	GOINFER_D7_SPEED=1 go test -count=1 -tags goinfer_testhooks -timeout 40m -run '^TestD7_int8StepVsF16Pass$' -v ./metal/
func TestD7_int8StepVsF16Pass(t *testing.T) {
	if os.Getenv("GOINFER_D7_SPEED") == "" {
		t.Skip("set GOINFER_D7_SPEED=1 (a timed run, by day, under the timing lock)")
	}
	home, _ := os.UserHomeDir()
	path := filepath.Join(home, "models", "qwen2.5-coder-1.5b-instruct-q4_k_m.gguf")
	giw := decoder.SidecarPath(path, "int4", "metal", false)
	if _, err := os.Stat(giw); err != nil {
		t.Fatalf("no sidecar at %s", giw)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 35*time.Minute)
	defer cancel()
	load := func(kv string) *decoder.Model {
		m, err := decoder.Load(giw, decoder.Options{Backend: "metal", Quant: "int4", KVPrecision: kv, KVQuant: map[string]string{"": "f32", "i8": "i8"}[kv], ResidentContext: 8448})
		if err != nil {
			t.Fatalf("load --kv %q: %v", kv, err)
		}
		t.Cleanup(func() { m.Close() })
		a, ok := m.ResidentForwardForTest().(*metalResident)
		if !ok {
			t.Fatalf("--kv %q: not Metal-resident: %s", kv, m.ResidentDecline())
		}
		_, why := a.PrefillPath()
		fmt.Fprintf(os.Stderr, "[D7] --kv %-4q prefill path: %s\n", kv, why)
		if !strings.Contains(why, "MMA") {
			t.Fatalf("--kv %q: the prefill path does not name the MMA pass: %s", kv, why)
		}
		return m
	}
	f16, i8 := load(""), load("i8")
	tk, err := tokenizer.LoadGGUF(path)
	if err != nil {
		t.Fatalf("tokenizer: %v", err)
	}
	_, files := decoder.PrefillGatePromptSet()
	const pairs = 5
	start := time.Now()
	for _, n := range []int{2048, 8192} {
		ids := decoder.PrefillGateProseIDsForTest(t, tk, files[1], n)[:n]
		refs := map[*decoder.Model][]float32{}
		run := func(m *decoder.Model) time.Duration {
			t0 := time.Now()
			lg, err := m.ResidentPrefillSeedForTest(ctx, ids)
			if err != nil {
				t.Fatalf("%d tokens: prefill: %v", n, err)
			}
			d := time.Since(t0)
			if refs[m] == nil {
				refs[m] = slices.Clone(lg)
			} else if !slices.EqualFunc(lg, refs[m], func(x, y float32) bool { return math.Float32bits(x) == math.Float32bits(y) }) {
				t.Fatalf("%d tokens: a run's logits differ from that arm's first run", n)
			}
			return d
		}
		var ratios []float64
		for p := 0; p <= pairs; p++ { // pair 0 is the warm-up
			l1, avail := d7Load1(), probe.Available()
			var df, di time.Duration
			if p%2 == 0 {
				df, di = run(f16), run(i8)
			} else {
				di = run(i8)
				df = run(f16)
			}
			label := fmt.Sprintf("pair %d of %d", p, pairs)
			if p == 0 {
				label = "warm-up"
			} else {
				ratios = append(ratios, di.Seconds()/df.Seconds())
			}
			fmt.Fprintf(os.Stderr, "[D7] %d tokens, %s (load1 %.2f, available %.2f GB at start): f16 %.2fs (%.0f tok/s), i8 %.2fs (%.0f tok/s), i8/f16 %.3f; elapsed %.0fs\n",
				n, label, l1, float64(avail)/1e9, df.Seconds(), float64(n)/df.Seconds(), di.Seconds(), float64(n)/di.Seconds(), di.Seconds()/df.Seconds(), time.Since(start).Seconds())
		}
		sort.Float64s(ratios)
		med := ratios[len(ratios)/2]
		fmt.Fprintf(os.Stderr, "[D7] %d tokens: median i8 / f16 %.3f (pairs %.3f)\n", n, med, ratios)
		if med > 1.25 {
			t.Errorf("P3B2-G3: at %d tokens the int8 prefill takes %.3f× the f16 one, over 1.25", n, med)
		}
	}
}
