//go:build goinfer_testhooks

package decoder

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/tokenizer"
)

// TestGptOssCPUPrefillGate is the measurement tool for gpt-oss's batched CPU prefill (docs/measurements/gptoss-cpu-prefill-2026-11-23,
// which registers its gates and bars). One CPU load of GPTOSS_PF_MODEL at int4 (serve's default) serves every arm:
//
//	S  the sequential prefill (the per-token walker), today's path;
//	E  the batched walker with exact attention (ExactPrefill), the exact arm, which must equal S bit for bit;
//	B  the batched walker at the default (CPU fast attention from fastAttnMinPrompt rows), the candidate.
//
// GPTOSS_PF_MODE=fit prints the load's fit plan (FitQuery) and the process's resident memory after the load. GPTOSS_PF_MODE=fidelity scores B against E over GPTOSS_PF_N prompts of prompt set A at each depth, teacher-forced on E's greedy
// continuation (gptossPFPositions positions). GPTOSS_PF_MODE=ttft times S against B, A B B A per block over GPTOSS_PF_BLOCKS blocks
// at each depth (block b uses prompt b), and checks E against S on every S cell's prompt; a cell starting above load 8.0 or under
// GPTOSS_PF_MIN_AVAIL_MIB available is VOID and recorded as such. Records go to GPTOSS_PF_OUT. A measurement tool: it skips unless
// GPTOSS_PF_MODEL is set.
func TestGptOssCPUPrefillGate(t *testing.T) {
	model := os.Getenv("GPTOSS_PF_MODEL")
	if model == "" {
		t.Skip("set GPTOSS_PF_MODEL, GPTOSS_PF_MODE and GPTOSS_PF_OUT")
	}
	mode, out := os.Getenv("GPTOSS_PF_MODE"), os.Getenv("GPTOSS_PF_OUT")
	if mode != "fidelity" && mode != "ttft" && mode != "fit" {
		t.Fatalf("GPTOSS_PF_MODE %q: want fit, fidelity or ttft", mode)
	}
	if err := os.MkdirAll(out, 0o755); err != nil {
		t.Fatal(err)
	}
	depths := gptossPFInts(t, "GPTOSS_PF_DEPTHS", "1024,3900")
	n := gptossPFInts(t, "GPTOSS_PF_N", "8")[0]
	blocks := gptossPFInts(t, "GPTOSS_PF_BLOCKS", "2")[0]
	minAvail := gptossPFInts(t, "GPTOSS_PF_MIN_AVAIL_MIB", "35000")[0] // the registered void bar: the load's need plus 2 GB
	start := time.Now()
	logf, err := os.OpenFile(filepath.Join(out, mode+".log"), os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o644)
	if err != nil {
		t.Fatal(err)
	}
	defer logf.Close()
	say := func(format string, a ...any) {
		line := fmt.Sprintf("gptoss-pf [%6.0fs] %s", time.Since(start).Seconds(), fmt.Sprintf(format, a...))
		fmt.Fprintln(os.Stderr, line)
		fmt.Fprintln(logf, line)
	}

	tk, err := tokenizer.LoadGGUF(model)
	if err != nil {
		t.Fatalf("tokenizer: %v", err)
	}
	maxK := 0
	for _, k := range depths {
		maxK = max(maxK, k)
	}
	files := PrefillGatePromptSetFor("a")
	if mode == "fit" { // the load's own plan, then the load and the process's peak resident memory after it
		a, err := FitQuery(model, Options{Backend: "cpu", Quant: "int4"})
		if err != nil {
			t.Fatalf("fit query: %v", err)
		}
		raw, _ := json.Marshal(a)
		say("fit: %s", raw)
	}
	t0 := time.Now()
	m, err := Load(model, Options{Backend: "cpu", Quant: "int4"}) // serve's default quant
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	defer m.Close()
	if m.w.arch.gptoss == nil {
		t.Fatalf("%s is not gpt-oss", model)
	}
	say("loaded %s in %.0fs; mode %s, depths %v, prompt set a, git %s", filepath.Base(model), time.Since(t0).Seconds(), mode, depths, os.Getenv("GPTOSS_PF_GIT"))
	if !m.canBatchN(maxK) {
		t.Fatalf("canBatchN(%d) = false: this build does not batch gpt-oss's prefill", maxK)
	}
	if mode == "fit" {
		raw, _ := os.ReadFile("/proc/self/status")
		for _, ln := range strings.Split(string(raw), "\n") {
			if strings.HasPrefix(ln, "VmHWM") || strings.HasPrefix(ln, "VmRSS") || strings.HasPrefix(ln, "VmSwap") {
				say("after load: %s", strings.Join(strings.Fields(ln), " "))
			}
		}
		return
	}
	ids := func(i, k int) []int {
		return PrefillGateProseIDsForTest(t, tk, files[i%len(files)], k)[:k]
	}

	switch mode {
	case "fidelity":
		type line struct {
			Depth     int       `json:"depth"`
			Prompt    string    `json:"prompt"`
			Agree     []bool    `json:"agree"`
			HardFlip  []bool    `json:"hard_flip"`
			GapPct    []float64 `json:"gap_pct"`
			KL        []float64 `json:"kl"`
			ExpertMaj int64     `json:"expert_major_chunks"`
		}
		f, err := os.Create(filepath.Join(out, "fidelity-positions.jsonl"))
		if err != nil {
			t.Fatal(err)
		}
		defer f.Close()
		enc := json.NewEncoder(f)
		var pos, agree, hard int
		var klSum, klMax float64
		for _, k := range depths {
			for i := range n {
				p := ids(i, k)
				var eL, bL [][]float32
				var teacher []int
				runE := func() { eL, teacher = gptossPFRun(t, m, p, true, nil) }
				var em int64
				runB := func() {
					r0 := gptOssExpertMajorRuns.Load()
					bL, _ = gptossPFRun(t, m, p, false, teacher)
					em = gptOssExpertMajorRuns.Load() - r0
				}
				runE() // B is forced on E's tokens, so E runs first
				runB()
				l := line{Depth: k, Prompt: filepath.Base(files[i%len(files)]), ExpertMaj: em}
				for j := range eL {
					a, gap, hf := NearTieArgmaxForTest(eL[j], bL[j])
					kl := KLDivergenceForTest(eL[j], bL[j])
					l.Agree, l.HardFlip, l.GapPct, l.KL = append(l.Agree, a), append(l.HardFlip, hf), append(l.GapPct, gap), append(l.KL, kl)
					pos++
					if a {
						agree++
					}
					if hf {
						hard++
					}
					klSum += kl
					klMax = math.Max(klMax, kl)
				}
				if err := enc.Encode(l); err != nil {
					t.Fatal(err)
				}
				say("fidelity K=%d prompt %d/%d (%s): agree %d/%d, hard flips %d, expert-major chunks %d; pooled %d/%d agree, %d hard",
					k, i+1, n, l.Prompt, gptossPFCount(l.Agree), len(l.Agree), gptossPFCount(l.HardFlip), em, agree, pos, hard)
			}
		}
		say("FIDELITY pooled: positions %d, top-1 agreement %.2f%%, hard flips %d, mean KL %.3g, max KL %.3g", pos, 100*float64(agree)/float64(pos), hard, klSum/float64(pos), klMax)
	case "ttft":
		f, err := os.OpenFile(filepath.Join(out, "ttft-cells.tsv"), os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o644)
		if err != nil {
			t.Fatal(err)
		}
		defer f.Close()
		fmt.Fprintln(f, "start\tdepth\tblock\tcell\tarm\tload1\tmemavail_mib\tttft_s\theap_peak_mib\te_equals_s")
		for _, k := range depths {
			for b := range blocks {
				p := ids(b, k)
				for c, arm := range "ABBA" {
					l1, avail := gptossPFHostState()
					st := time.Now().Format(time.RFC3339)
					void := l1 > 8.0 || avail < minAvail
					var wall time.Duration
					var peak uint64
					var last []float32
					identical := "-"
					if !void {
						wall, peak, last = gptossPFTimed(t, m, p, arm == 'B')
						if arm == 'A' {
							eL, _ := gptossPFRun(t, m, p, true, nil)
							identical = strconv.FormatBool(gptossPFEqual(eL[0], last))
						}
					}
					ttft := "VOID"
					if !void {
						ttft = fmt.Sprintf("%.2f", wall.Seconds())
					}
					fmt.Fprintf(f, "%s\t%d\t%d\t%d\t%c\t%.2f\t%d\t%s\t%d\t%s\n", st, k, b+1, c+1, arm, l1, avail, ttft, peak>>20, identical)
					say("ttft K=%d block %d cell %d %c: load %.2f avail %d MiB -> %s s (heap peak %d MiB, E==S %s)", k, b+1, c+1, arm, l1, avail, ttft, peak>>20, identical)
				}
			}
		}
	}
}

// gptossPFPositions is how many positions the fidelity mode scores per prompt: the one after the prompt and 32 teacher-forced.
const gptossPFPositions = 33

// gptossPFRun prefills p on a fresh cache, exact (ExactPrefill: the batched walker's f64 attention) or at the default, then feeds
// gptossPFPositions-1 tokens (forced when given, else its own greedy picks). It returns each position's logits and the tokens fed.
func gptossPFRun(t *testing.T, m *Model, p []int, exact bool, forced []int) ([][]float32, []int) {
	t.Helper()
	m.opts.ExactPrefill = exact
	defer func() { m.opts.ExactPrefill = false }()
	cache := m.NewCache(len(p) + gptossPFPositions)
	lg, err := m.prefillLogits(context.Background(), p, cache)
	if err != nil {
		t.Fatal(err)
	}
	logits := [][]float32{append([]float32(nil), lg...)}
	var toks []int
	for i := range gptossPFPositions - 1 {
		next := argmaxF32(logits[len(logits)-1])
		if forced != nil {
			next = forced[i]
		}
		toks = append(toks, next)
		lg, err := m.forward(next, cache)
		if err != nil {
			t.Fatal(err)
		}
		logits = append(logits, append([]float32(nil), lg...))
	}
	return logits, toks
}

// gptossPFTimed runs one TTFT cell: the prompt's prefill to its first logits, sequential (batched false: the per-token walker) or
// through prefillLogits at the default, on a fresh cache. It returns the wall time, the Go heap's peak in use while it ran (sampled
// every 20 ms), and the logits.
func gptossPFTimed(t *testing.T, m *Model, p []int, batched bool) (time.Duration, uint64, []float32) {
	t.Helper()
	runtime.GC()
	cache := m.NewCache(len(p) + 1)
	var peak atomic.Uint64
	stop := make(chan struct{})
	done := make(chan struct{})
	go func() {
		defer close(done)
		var ms runtime.MemStats
		tick := time.NewTicker(20 * time.Millisecond)
		defer tick.Stop()
		for {
			runtime.ReadMemStats(&ms)
			if ms.HeapInuse > peak.Load() {
				peak.Store(ms.HeapInuse)
			}
			select {
			case <-stop:
				return
			case <-tick.C:
			}
		}
	}()
	t0 := time.Now()
	var lg []float32
	var err error
	if batched {
		lg, err = m.prefillLogits(context.Background(), p, cache)
	} else {
		for _, id := range p[:len(p)-1] {
			if _, err = m.runLayers(id, cache); err != nil {
				break
			}
		}
		if err == nil {
			lg, err = m.forward(p[len(p)-1], cache)
		}
	}
	wall := time.Since(t0)
	close(stop)
	<-done
	if err != nil {
		t.Fatal(err)
	}
	return wall, peak.Load(), append([]float32(nil), lg...)
}

// gptossPFHostState is the 1-minute load average and MemAvailable (MiB) at a cell's start.
func gptossPFHostState() (float64, int) {
	var l1 float64
	if raw, err := os.ReadFile("/proc/loadavg"); err == nil {
		l1, _ = strconv.ParseFloat(strings.Fields(string(raw))[0], 64)
	}
	avail := 0
	if raw, err := os.ReadFile("/proc/meminfo"); err == nil {
		for _, ln := range strings.Split(string(raw), "\n") {
			if f := strings.Fields(ln); len(f) >= 2 && f[0] == "MemAvailable:" {
				kb, _ := strconv.Atoi(f[1])
				avail = kb / 1024
			}
		}
	}
	return l1, avail
}

func gptossPFEqual(a, b []float32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func gptossPFCount(bs []bool) int {
	n := 0
	for _, b := range bs {
		if b {
			n++
		}
	}
	return n
}

func gptossPFInts(t *testing.T, name, def string) []int {
	t.Helper()
	v := os.Getenv(name)
	if v == "" {
		v = def
	}
	var out []int
	for _, s := range strings.Split(v, ",") {
		x, err := strconv.Atoi(strings.TrimSpace(s))
		if err != nil || x < 1 {
			t.Fatalf("%s %q", name, v)
		}
		out = append(out, x)
	}
	return out
}
