//go:build cuda && goinfer_testhooks

package cuda

import (
	"fmt"
	"os"
	"slices"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/decoder"
)

// kso1cLoad1 is the 1-minute load average.
func kso1cLoad1() float64 {
	b, _ := os.ReadFile("/proc/loadavg")
	f, _ := strconv.ParseFloat(strings.Fields(string(b) + " 0")[0], 64)
	return f
}

// kso1cMemAvailGB is /proc/meminfo's MemAvailable in GB.
func kso1cMemAvailGB() float64 {
	b, _ := os.ReadFile("/proc/meminfo")
	for _, l := range strings.Split(string(b), "\n") {
		if f := strings.Fields(l); len(f) >= 2 && f[0] == "MemAvailable:" {
			kb, _ := strconv.ParseFloat(f[1], 64)
			return kb / (1 << 20)
		}
	}
	return 0
}

// kso1cLoadBothArms loads path with kquantKeepInt8ForTest held for the load only, so every native weight's int8 copy is on the
// device beside it (kso1cArms switches between them).
func kso1cLoadBothArms(path string, opts decoder.Options) (*decoder.Model, error) {
	kquantKeepInt8ForTest = true
	defer func() { kquantKeepInt8ForTest = false }()
	return decoder.Load(path, opts)
}

// kso1cArms returns a switch between a resident's two arms, built under kquantKeepInt8ForTest: use(true) binds every native weight,
// use(false) its int8 copy (cudaResident.kquantAlt). It checks the round trip lands every native weight back, and returns their count.
func kso1cArms(t *testing.T, r *cudaResident, label string) (use func(native bool), nNative int) {
	t.Helper()
	slots := []*cudaWQ{&r.lmW}
	for l := range r.layers {
		Ly := &r.layers[l]
		slots = append(slots, &Ly.q, &Ly.k, &Ly.v, &Ly.o, &Ly.g, &Ly.u, &Ly.d)
	}
	use = func(native bool) {
		from, to := 1, 0
		if !native {
			from, to = 0, 1
		}
		for _, p := range r.kquantAlt {
			for _, s := range slots {
				if *s == p[from] {
					*s = p[to]
				}
			}
		}
	}
	use(false)
	for _, s := range slots {
		if s.kind == "q5k" || s.kind == "q6k" {
			t.Fatalf("%s: a native weight is still bound on the int8 arm", label)
		}
	}
	use(true)
	for _, s := range slots {
		if s.kind == "q5k" || s.kind == "q6k" {
			nNative++
		}
	}
	if nNative != len(r.kquantAlt) || nNative == 0 {
		t.Fatalf("%s: %d native slots after the swap round trip, %d native weights", label, nNative, len(r.kquantAlt))
	}
	return use, nNative
}

// kso1cSpeed is O1C-G7's A/B on one model: one CUDA q4k resident holding both the native and the int8 buffers of every natively
// uploaded weight (kquantKeepInt8ForTest), swapped per run; 128 greedy decode tokens after a 32-token prompt, A B B A over 10 pairs
// after 2 warm runs each. A pair whose cell starts with the 1-minute load above 8.0 (half nobara's 16 cores) is void and re-run in
// the same order (CLAUDE.md, "Run budget"). It returns the median native/int8 per-token time and the pairs faster.
func kso1cSpeed(t *testing.T, path, label string) (float64, int) {
	m, err := kso1cLoadBothArms(path, decoder.Options{Backend: "cuda", Quant: "q4k", ResidentContext: 512})
	if err != nil {
		t.Fatalf("%s: %v", label, err)
	}
	defer m.Close()
	r, ok := m.ResidentForwardForTest().(*cudaResident)
	if !ok {
		t.Fatalf("%s: no CUDA resident: %s", label, m.ResidentDecline())
	}
	if len(r.kquantAlt) == 0 {
		t.Fatalf("%s: no weight was uploaded natively", label)
	}
	use, nNative := kso1cArms(t, r, label)
	fmt.Fprintf(os.Stderr, "[O1C-G7] %s: %d weights native (the int8 copies beside them); load %.2f, %.1f GB available\n",
		label, nNative, kso1cLoad1(), kso1cMemAvailGB())
	defer use(true)
	const prompt, gen = 32, 128
	_, _, _, _, _, _, vocab := m.Dims()
	toks := map[bool][]int{}
	run := func(native bool) float64 {
		use(native)
		r.Reset()
		for pos := range prompt {
			if _, err := r.Forward(m.EmbedResidentForTest((pos*7919+13)%vocab), pos); err != nil {
				t.Fatal(err)
			}
		}
		tok := 13
		seq := make([]int, 0, gen)
		t0 := time.Now()
		for pos := prompt; pos < prompt+gen; pos++ {
			lg, err := r.Forward(m.EmbedResidentForTest(tok), pos)
			if err != nil {
				t.Fatal(err)
			}
			tok = 0
			for i, v := range lg {
				if v > lg[tok] {
					tok = i
				}
			}
			seq = append(seq, tok)
		}
		s := time.Since(t0).Seconds() / gen
		if prev, ok := toks[native]; ok && !slices.Equal(prev, seq) {
			t.Errorf("%s: the native=%v arm's tokens changed between runs", label, native)
		}
		toks[native] = seq
		return s
	}
	for range 2 {
		run(false)
		run(true)
	}
	var ratios []float64
	faster, voids := 0, 0
	for i := 0; len(ratios) < 10; i++ {
		if voids > 5 {
			t.Fatalf("%s: %d void pairs (load above 8.0): the box is busy", label, voids)
		}
		la := kso1cLoad1()
		if la > 8.0 {
			voids++
			fmt.Fprintf(os.Stderr, "[O1C-G7] %s pair %d VOID: load %.2f above 8.0 at its start; re-run\n", label, len(ratios), la)
			time.Sleep(30 * time.Second)
			continue
		}
		var o, n float64
		if len(ratios)%2 == 0 {
			o, n = run(false), run(true)
		} else {
			n, o = run(true), run(false)
		}
		ratios = append(ratios, n/o)
		if n < o {
			faster++
		}
		fmt.Fprintf(os.Stderr, "[O1C-G7] %s pair %d (load %.2f, %.1f GB available): int8 %.3f ms/token (%.1f tok/s), native %.3f (%.1f), native/int8 %.4f\n",
			label, len(ratios)-1, la, kso1cMemAvailGB(), o*1e3, 1/o, n*1e3, 1/n, n/o)
	}
	sorted := slices.Clone(ratios)
	slices.Sort(sorted)
	med := (sorted[4] + sorted[5]) / 2
	same := 0
	for i := range toks[true] {
		if toks[true][i] != toks[false][i] {
			break
		}
		same++
	}
	fmt.Fprintf(os.Stderr, "[O1C-G7] %s median native/int8 %.4f, %d of 10 pairs faster, %d void; the arms' greedy tokens agree for the first %d of %d\n",
		label, med, faster, voids, same, gen)
	return med, faster
}

// TestKSO1C_speed is O1C-G7 (docs/tasks/task-kernel-survey-2026-10.md, "KS-O1 on CUDA: gates"): kso1cSpeed on Phi-3 mini q4, then
// on Qwen2.5-Coder-1.5B q4_k_m. SHIP iff Phi-3 is faster on at least 9 of 10 pairs and the 1.5B's median native/int8 is <= 1.00;
// PARK if Phi-3 is faster on 6-8; KILL if on 5 or fewer, or if the 1.5B's median is above 1.00. A timed run: hold the timing lock.
//
//	GOINFER_KSO1C_TIMING=1 python3 scripts/timing_lock.py run --label o1c-g7 -- go test -count=1 -timeout 30m -tags 'cuda goinfer_testhooks' -run '^TestKSO1C_speed$' -v ./cuda/
func TestKSO1C_speed(t *testing.T) {
	if os.Getenv("GOINFER_KSO1C_TIMING") == "" {
		t.Skip("set GOINFER_KSO1C_TIMING=1 (a timed run under the timing lock)")
	}
	pm, pf := kso1cSpeed(t, modelPath("phi3-mini-4k-gguf/Phi-3-mini-4k-instruct-q4.gguf"), "Phi-3 mini")
	qm, qf := kso1cSpeed(t, modelPath("qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"), "Qwen2.5-Coder-1.5B")
	verdict := "PARK"
	switch {
	case pf <= 5 || qm > 1.00:
		verdict = "KILL"
	case pf >= 9:
		verdict = "SHIP"
	}
	fmt.Fprintf(os.Stderr, "[O1C-G7] Phi-3 median %.4f (%d of 10 faster), 1.5B median %.4f (%d of 10): %s\n", pm, pf, qm, qf, verdict)
	if verdict != "SHIP" {
		t.Errorf("O1C-G7 %s: Phi-3 median %.4f (%d of 10 pairs faster), the 1.5B's %.4f", verdict, pm, pf, qm)
	}
}
