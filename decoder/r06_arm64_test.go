//go:build goinfer_testhooks && !windows

package decoder

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"runtime"
	"slices"
	"strconv"
	"strings"
	"syscall"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/internal/hostmem"
)

// TestR06_speed is R-06's re-measure on the current tree (docs/measurements/r06-arm64-2026-10-25.md): w4a8BatchEnabled on
// against off, in one process per model (0.5B, 1.5B, 7B), A B B A over 10 valid pairs after a warm-up, each arm 96 greedy
// tokens at depth 128 timed by the decoder's forward clock. Every arm's tokens must equal the reference stream (the change is
// bit-identical). CPU decode keeps the cores busy itself, so an arm is void when OTHER processes' CPU, averaged over the arm
// (the machine's busy cores less this process's own CPU time), exceeds 1.0 core, or reclaimable memory is under 2 GiB at either
// end, or the machine paged out or swapped during it (r06Paging); a void pair is re-run, up to 10 times a model. Before each load it requires the checkpoint's size plus 2 GiB reclaimable
// (the void rule's memory term), and fails naming R06-VOID-MEMORY otherwise. It prints each model's median off/on and pairs faster, and its grade
// against the merge rule (median ≥ 1.05× with at least 9 of 10 faster); the grade is a report, not a failure. A timed run: it
// refuses to run outside the timing lock (GOINFER_R06_TIMING=1, GOINFER_HEAVY_TESTS=1).
func TestR06_speed(t *testing.T) {
	if os.Getenv("GOINFER_R06_TIMING") == "" {
		t.Skip("set GOINFER_R06_TIMING=1 (a timed run under the timing lock)")
	}
	if os.Getenv("GOINFER_TIMING_LOCK") == "" {
		t.Fatal("not under the timing lock: run through scripts/timing_lock.py run")
	}
	for _, mc := range []struct{ name, env, def string }{
		{"0.5B", "GOINFER_CPU_MODEL_05B", "$HOME/models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf"},
		{"1.5B", "GOINFER_CPU_MODEL", "$HOME/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"},
		{"7B", "GOINFER_CPU_MODEL_D7", "$HOME/models/qwen2.5-7b-instruct-q4_k_m.gguf"},
	} {
		t.Run(mc.name, func(t *testing.T) {
			path := os.Getenv(mc.env)
			if path == "" {
				path = os.ExpandEnv(mc.def)
			}
			// The owner's waiver (GOINFER_R06_WAIVE_MARGIN names a model): that model's load needs only its own size reclaimable,
			// and its arms read no swap activity in place of the 2 GiB line.
			waived := os.Getenv("GOINFER_R06_WAIVE_MARGIN") == mc.name
			margin := int64(2 << 30)
			if waived {
				margin = 0
			}
			if fi, err := os.Stat(path); err == nil {
				if r, ok := hostmem.Read(); !ok || r.Reclaimable < fi.Size()+margin {
					t.Fatalf("R06-VOID-MEMORY: reclaimable %.2f GiB before loading %s (%.2f GiB), under its size plus %.0f GiB",
						float64(r.Reclaimable)/(1<<30), mc.name, float64(fi.Size())/(1<<30), float64(margin)/(1<<30))
				}
			}
			h := newCPUDecodeAB(t, mc.env, mc.def, 128)
			orig := w4a8BatchEnabled
			t.Cleanup(func() { w4a8BatchEnabled = orig })
			gen := func(on bool) (float64, []int) {
				w4a8BatchEnabled = on
				out, g := h.m.Generate(context.Background(), h.ids, 96, SamplingParams{Temperature: 0})
				var toks []int
				for id := range out {
					toks = append(toks, id)
				}
				if g.err != nil {
					t.Fatalf("generate: %v", g.err)
				}
				return lastDecodeSplit.fwd, toks
			}
			gen(false) // warm-up
			gen(true)
			_, ref := gen(false)
			arm := func(on bool) (ms float64, valid bool, note string) {
				p0 := r06Paging(t)
				c0, r0, u0, t0 := r06Sample(t)
				ms, toks := gen(on)
				c1, r1, u1, t1 := r06Sample(t)
				p1 := r06Paging(t)
				if !slices.Equal(toks, ref) {
					t.Fatalf("arm on=%v: the greedy stream differs from the reference: not bit-identical", on)
				}
				busy, ok := hostmem.BusyCores(c0, c1, runtime.NumCPU())
				own := (u1 - u0).Seconds() / t1.Sub(t0).Seconds()
				others := busy - own
				gib := func(r hostmem.Reading) float64 { return float64(r.Reclaimable) / (1 << 30) }
				paged := p1 - p0
				memOK := paged == 0 && (waived || min(r0.Reclaimable, r1.Reclaimable) >= 2<<30)
				valid = ok && others <= 1.0 && memOK
				return ms, valid, fmt.Sprintf("others %.2f cores (all %.2f, own %.2f), reclaimable %.2f→%.2f GiB, pageouts+swapouts %d", others, busy, own, gib(r0), gib(r1), paged)
			}
			var ratios []float64
			faster, voids := 0, 0
			for len(ratios) < 10 {
				var on, off float64
				var vOn, vOff bool
				var nOn, nOff string
				if len(ratios)%2 == 0 {
					on, vOn, nOn = arm(true)
					off, vOff, nOff = arm(false)
				} else {
					off, vOff, nOff = arm(false)
					on, vOn, nOn = arm(true)
				}
				if !vOn || !vOff {
					voids++
					fmt.Fprintf(os.Stderr, "[R-06 %s] VOID pair (on: %s; off: %s)\n", mc.name, nOn, nOff)
					if voids > 10 {
						t.Fatalf("over 10 void pairs")
					}
					time.Sleep(20 * time.Second)
					continue
				}
				ratios = append(ratios, off/on)
				if on < off {
					faster++
				}
				fmt.Fprintf(os.Stderr, "[R-06 %s] pair %d (on: %s; off: %s): on %.2f, off %.2f ms/token, off/on %.4f\n",
					mc.name, len(ratios)-1, nOn, nOff, on, off, off/on)
			}
			slices.Sort(ratios)
			med := (ratios[4] + ratios[5]) / 2
			grade := "BELOW the merge rule"
			if med >= 1.05 && faster >= 9 {
				grade = "MEETS the merge rule"
			}
			fmt.Fprintf(os.Stderr, "[R-06 %s] median off/on %.4f (pairs %.4f–%.4f), %d of 10 faster, %d void pairs: %s\n",
				mc.name, med, ratios[0], ratios[9], faster, voids, grade)
		})
	}
}

// r06Paging is the machine's pageouts plus swapouts so far, from vm_stat (darwin) or /proc/vmstat's pswpout (linux); a rise
// across an arm means it paged. The test fails if it cannot be read.
func r06Paging(t *testing.T) int64 {
	t.Helper()
	var n int64
	if runtime.GOOS == "linux" {
		b, err := os.ReadFile("/proc/vmstat")
		if err != nil {
			t.Fatalf("vmstat: %v", err)
		}
		for _, ln := range strings.Split(string(b), "\n") {
			if f := strings.Fields(ln); len(f) == 2 && f[0] == "pswpout" {
				n, _ = strconv.ParseInt(f[1], 10, 64)
				return n
			}
		}
		t.Fatal("no pswpout in /proc/vmstat")
	}
	out, err := exec.Command("vm_stat").Output()
	if err != nil {
		t.Fatalf("vm_stat: %v", err)
	}
	found := 0
	for _, ln := range strings.Split(string(out), "\n") {
		k, v, ok := strings.Cut(ln, ":")
		if !ok || (k != "Pageouts" && k != "Swapouts") {
			continue
		}
		x, err := strconv.ParseInt(strings.TrimSuffix(strings.TrimSpace(v), "."), 10, 64)
		if err != nil {
			t.Fatalf("vm_stat %s %q: %v", k, v, err)
		}
		n += x
		found++
	}
	if found != 2 {
		t.Fatalf("vm_stat: found %d of Pageouts and Swapouts", found)
	}
	return n
}

// r06Sample reads the machine's CPU ticks, reclaimable memory, this process's CPU time and the wall clock; the test fails if a
// reading cannot be taken.
func r06Sample(t *testing.T) (hostmem.CPUTicks, hostmem.Reading, time.Duration, time.Time) {
	t.Helper()
	c, cok := hostmem.ReadCPU()
	r, rok := hostmem.Read()
	var ru syscall.Rusage
	if !cok || !rok || syscall.Getrusage(syscall.RUSAGE_SELF, &ru) != nil {
		t.Fatalf("no CPU, memory or rusage reading (cpu %v, memory %v)", cok, rok)
	}
	return c, r, time.Duration(ru.Utime.Nano() + ru.Stime.Nano()), time.Now()
}
