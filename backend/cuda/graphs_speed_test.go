//go:build cuda && goinfer_testhooks

package cuda

import (
	"bufio"
	"context"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"os"
	"runtime/debug"
	"slices"
	"sort"
	"strconv"
	"strings"
	"syscall"
	"testing"
	"time"

	gc "github.com/eitamring/gocudrv/cuda"
	"github.com/townsendmerino/goinfer/decoder"
)

// TestGraphsDecodeSpeedup measures the real decode tok/s of graph replay vs live launch on a real
// model, end-to-end through the safe-gate. Graphs are enabled via the UNSAFE override (this idle box is
// DEFAULT compute mode; with no churn, replay is bit-exact — the self-test in admitGraphs confirms
// it). Heavy; gated.
//
//	GOINFER_HEAVY_TESTS=1 go test -tags cuda -run TestGraphsDecodeSpeedup -v
func TestGraphsDecodeSpeedup(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") == "" {
		t.Skip("set GOINFER_HEAVY_TESTS=1 (loads a 1.5B model)")
	}
	path := modelPath("qwen2.5-coder-1.5b-instruct-q4_k_m.gguf")
	if err := gc.Init(); err != nil {
		t.Skipf("cuInit: %v", err)
	}
	if _, err := gc.GetDevice(0); err != nil {
		t.Skipf("no device: %v", err)
	}
	if _, err := os.Stat(path); err != nil {
		t.Skipf("no fixture at %s", path)
	}

	measure := func(graphs bool) (float64, bool) {
		if graphs {
			t.Setenv("GOINFER_CUDA_GRAPHS", "1")
			t.Setenv("GOINFER_CUDA_GRAPHS_UNSAFE", "1")
		} else {
			t.Setenv("GOINFER_CUDA_GRAPHS", "")
			t.Setenv("GOINFER_CUDA_GRAPHS_UNSAFE", "")
		}
		mc, err := decoder.Load(path, decoder.Options{Backend: "cuda", Quant: "int4"})
		if err != nil {
			t.Fatalf("load: %v", err)
		}
		defer mc.Close()
		rf := mustResident(t, mc)
		if rf.graphs != graphs {
			t.Fatalf("graphs=%v requested, resident has %v", graphs, rf.graphs)
		}
		_, _, _, _, _, _, vocab := mc.Dims()
		emb := func(i int) []float32 { return mc.EmbedResidentForTest((i*2654435761 + 1) % (vocab - 1)) }
		// warm + establish KV over a short prefix
		for i := range 16 {
			if _, err := rf.ForwardArgmax(emb(i), i); err != nil {
				t.Fatalf("warm: %v", err)
			}
		}
		const steps = 256
		best := time.Hour
		for range 3 {
			t0 := time.Now()
			for i := range steps {
				if _, err := rf.ForwardArgmax(emb(1000+i), 16+i); err != nil {
					t.Fatalf("decode: %v", err)
				}
			}
			if d := time.Since(t0); d < best {
				best = d
			}
		}
		return float64(steps) / best.Seconds(), rf.graphs
	}

	base, _ := measure(false)
	g, on := measure(true)
	if !on {
		t.Fatal("graphs did not enable under the UNSAFE override")
	}
	t.Logf("decode tok/s (1.5B, greedy): live %.1f  |  graphs %.1f  →  %.2f×", base, g, g/base)
}

// TestGraphsDecode26B measures CUDA graphs on the 26B C′ path, where they are not free to turn on:
// they force the DMA overlap off (a captured segment cannot wait per miss) and block compute-time
// LoRA. So the comparison is the trade itself — today's default (graphs off, overlap on) against
// graphs on (overlap off) — not graphs vs a strawman (docs/cuda-graphs-investigation.md has the
// dense-1.5B result). Greedy decode feeds each argmax back, so MoE routing and the expert cache see
// a real continuation. Arms are separate loads (two 26B residents do not fit 8 GB), interleaved ABBA.
//
// Pre-registered decision rule: graphs >= 1.05x default → worth keeping; <= 1.00x → remove graphs;
// between → ambiguous, back to the owner.
//
//	GOINFER_HEAVY_TESTS=1 go test -tags 'cuda goinfer_testhooks' -run TestGraphsDecode26B -v -timeout 30m
func TestGraphsDecode26B(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") == "" {
		t.Skip("set GOINFER_HEAVY_TESTS=1 (loads the real 26B four times)")
	}
	path := os.Getenv("GOINFER_GEMMA4_26B_GIW")
	if path == "" {
		path = modelPath("gemma4-26b-int4.giw")
	}
	if _, err := os.Stat(path); err != nil {
		t.Skipf("no 26B .giw at %s: %v", path, err)
	}
	t.Setenv("GOINFER_MOE_CACHE_EXPERTS", "1")
	start := time.Now()

	const warm, steps = 32, 128
	run := func(graphs bool, arm int) (float64, []int) {
		if graphs {
			t.Setenv("GOINFER_CUDA_GRAPHS", "1")
			t.Setenv("GOINFER_CUDA_GRAPHS_UNSAFE", "1")
		} else {
			t.Setenv("GOINFER_CUDA_GRAPHS", "")
			t.Setenv("GOINFER_CUDA_GRAPHS_UNSAFE", "")
		}
		fmt.Fprintf(os.Stderr, "[graphs26b %s] arm %d/4 graphs=%v: loading (elapsed %s)\n",
			time.Now().Format("15:04:05"), arm, graphs, time.Since(start).Round(time.Second))
		m, err := decoder.Load(path, decoder.Options{Backend: "cuda"})
		if err != nil {
			t.Fatalf("load: %v", err)
		}
		defer m.Close()
		rf, ok := m.ResidentForwardForTest().(*cudaResident)
		if !ok || rf == nil {
			t.Fatalf("cuda resident declined the 26B: %s", m.ResidentDecline())
		}
		if rf.graphs != graphs {
			t.Fatalf("graphs=%v requested, resident has %v", graphs, rf.graphs)
		}
		if graphs && rf.overlap {
			t.Fatal("graphs on but the DMA overlap is still on — the trade this measures is not in effect")
		}
		_, _, _, _, _, _, vocab := m.Dims()
		id := 2
		for i := range warm { // a fixed pseudo-random prompt, teacher-forced
			next, err := rf.ForwardArgmax(m.EmbedResidentForTest((i*2654435761+7)%(vocab-1)), i)
			if err != nil {
				t.Fatalf("warm: %v", err)
			}
			id = next
		}
		out := make([]int, steps)
		t0 := time.Now()
		for i := range steps { // greedy continuation: routing follows the model's own output
			next, err := rf.ForwardArgmax(m.EmbedResidentForTest(id), warm+i)
			if err != nil {
				t.Fatalf("decode: %v", err)
			}
			out[i], id = next, next
		}
		tps := float64(steps) / time.Since(t0).Seconds()
		fmt.Fprintf(os.Stderr, "[graphs26b %s] arm %d/4 graphs=%v overlap=%v slots=%d: %.2f tok/s\n",
			time.Now().Format("15:04:05"), arm, graphs, rf.overlap, rf.cacheSlots, tps)
		return tps, out
	}

	d1, outD := run(false, 1)
	g1, outG := run(true, 2)
	g2, _ := run(true, 3)
	d2, _ := run(false, 4)
	def, gr := (d1+d2)/2, (g1+g2)/2
	same := len(outD) == len(outG)
	for i := range outD {
		if outD[i] != outG[i] {
			same = false
			break
		}
	}
	verdict := "AMBIGUOUS — back to the owner"
	switch {
	case gr >= 1.05*def:
		verdict = "KEEP (>= 1.05x)"
	case gr <= def:
		verdict = "REMOVE (<= 1.00x)"
	}
	t.Logf("26B C′ decode tok/s, ABBA: default(overlap on) %.2f / %.2f, graphs(overlap off) %.2f / %.2f → graphs/default %.3fx; token streams identical=%v; rule: %s",
		d1, d2, g1, g2, gr/def, same, verdict)
}

// TestMEM10_pageableExpertSourceAB is MEM10's tier 1 A/B (docs/measurements/mem10-2026-11/ab-registration.md): the 26B with paged experts, its
// expert slots uploaded from today's pinned copy (A) or from pageable file pages (B, moePageableDirForTest), A B B A A B B A, one load per arm.
// Each load times 256 greedy decode tokens after a 32-token prompt and after a 4,096-token prefill, and reads the process's memory and swap. The
// first MoE layer's source bytes must hash the same in every arm, and the first B load's first 64 tokens must equal A's. It prints the paired
// B/A medians and the registered verdict; GOINFER_MEM10_AB_OUT names a JSON file for the record.
//
//	GOINFER_HEAVY_TESTS=1 go test -c -tags 'cuda goinfer_testhooks' -o cuda.test . && ./cuda.test -test.run '^TestMEM10_pageableExpertSourceAB$' -test.v -test.timeout 90m
func TestMEM10_pageableExpertSourceAB(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") == "" {
		t.Skip("set GOINFER_HEAVY_TESTS=1 (loads the real 26B eight times)")
	}
	// The sidecar serve opens for gemma4-26b-q4_k_m.gguf on CUDA (int4, embed-int4 head): a test binary does not take Load's sidecar default
	// (sidecarInTests), so the test opens it by name, the load MEM10-G0 read.
	path := modelPath("gemma4-26b-q4_k_m.int4.e4h.cuda.v17.giw")
	if _, err := os.Stat(path); err != nil {
		t.Fatalf("no 26B sidecar at %s: %v", path, err)
	}
	dir := os.Getenv("GOINFER_MEM10_PAGEABLE_DIR")
	if dir == "" {
		dir = os.ExpandEnv("$HOME/goinfer-bench/mem10-pageable")
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	var sfs syscall.Statfs_t
	if err := syscall.Statfs(dir, &sfs); err != nil || sfs.Type == 0x01021994 {
		t.Fatalf("%s is tmpfs (or unreadable: %v): its pages would be shared memory, not the file pages arm B measures", dir, err)
	}
	const shortPrompt, longPrompt, steps, checkToks = 32, 4096, 256, 64
	opts := decoder.Options{Backend: "cuda", MoECacheExperts: true, ResidentContext: longPrompt + steps}
	type arm struct {
		Name                                    string
		Short, Long                             float64
		Load1, AvailMiB                         float64
		RssAnon, RssShmem, RssFile, VmSwap, Hwm int // MiB, after the long decode
		ShmemAtLoad                             int
		SrcRssMiB                               map[string]int
		Pageable                                int
		Hash                                    string
		Toks                                    []int
		Void                                    []string
	}
	start := time.Now()
	run := func(name string, n int) arm {
		a := arm{Name: name}
		for {
			la, av := loadAvg1(), memAvailableMiB()
			if la <= 8.0 && av >= 30*1024 {
				a.Load1, a.AvailMiB = la, float64(av)
				break
			}
			a.Void = append(a.Void, fmt.Sprintf("load %.2f, available %d MiB at %s", la, av, time.Now().Format("15:04:05")))
			fmt.Fprintf(os.Stderr, "[mem10ab] load %d/8 (%s) VOID at its start (load %.2f, available %d MiB); waiting\n", n, name, la, av)
			time.Sleep(30 * time.Second)
		}
		if name == "B" {
			moePageableDirForTest = dir
		}
		defer func() { moePageableDirForTest = "" }()
		fmt.Fprintf(os.Stderr, "[mem10ab %s] load %d/8 arm %s: loading (load %.2f, available %.0f MiB, elapsed %s)\n",
			time.Now().Format("15:04:05"), n, name, a.Load1, a.AvailMiB, time.Since(start).Round(time.Second))
		m, err := decoder.Load(path, opts)
		if err != nil {
			t.Fatalf("load %d (%s): %v", n, name, err)
		}
		defer func() { m.Close(); debug.FreeOSMemory() }()
		rf, ok := m.ResidentForwardForTest().(*cudaResident)
		if !ok || rf == nil {
			t.Fatalf("load %d (%s): the cuda resident declined the 26B: %s", n, name, m.ResidentDecline())
		}
		a.Pageable = rf.pageableStacks
		moe := 0
		h := sha256.New()
		for i := range rf.layers {
			if rf.layers[i].expGU.srcW != nil {
				moe++
				if moe == 1 {
					for _, b := range []*cudaWQ{&rf.layers[i].expGU, &rf.layers[i].expDown} {
						h.Write(b.srcW.Bytes())
						h.Write(b.srcS.Bytes())
					}
				}
			}
		}
		a.Hash = fmt.Sprintf("%x", h.Sum(nil))[:16]
		if name == "A" && a.Pageable != 0 || name == "B" && a.Pageable != 2*moe {
			t.Fatalf("load %d (%s): %d of %d expert stacks pageable", n, name, a.Pageable, 2*moe)
		}
		a.ShmemAtLoad = procStatusMiB("RssShmem")
		_, _, _, _, _, _, vocab := m.Dims()
		tok := func(i int) int { return (i*2654435761 + 7) % (vocab - 1) }
		ctx := context.Background()
		// Short context: a 32-token prompt, teacher-forced, then 256 greedy tokens.
		id := 2
		for i := range shortPrompt {
			if id, err = rf.ForwardArgmax(m.EmbedResidentForTest(tok(i)), i); err != nil {
				t.Fatalf("short prompt: %v", err)
			}
		}
		t0 := time.Now()
		for i := range steps {
			if id, err = rf.ForwardArgmax(m.EmbedResidentForTest(id), shortPrompt+i); err != nil {
				t.Fatalf("short decode: %v", err)
			}
			if i < checkToks {
				a.Toks = append(a.Toks, id)
			}
		}
		a.Short = float64(steps) / time.Since(t0).Seconds()
		// Long context: a 4,096-token prompt through the batched prefill, then 256 greedy tokens.
		embs := make([][]float32, longPrompt)
		for i := range embs {
			embs[i] = m.EmbedResidentForTest(tok(i))
		}
		lg, err := rf.PrefillLast(ctx, embs, 0)
		if err != nil {
			t.Fatalf("long prefill: %v (a declined prefill would time a per-token prompt, not this cell)", err)
		}
		id = argmaxF32(lg)
		t0 = time.Now()
		for i := range steps {
			if id, err = rf.ForwardArgmax(m.EmbedResidentForTest(id), longPrompt+i); err != nil {
				t.Fatalf("long decode: %v", err)
			}
		}
		a.Long = float64(steps) / time.Since(t0).Seconds()
		a.RssAnon, a.RssShmem, a.RssFile = procStatusMiB("RssAnon"), procStatusMiB("RssShmem"), procStatusMiB("RssFile")
		a.VmSwap, a.Hwm = procStatusMiB("VmSwap"), procStatusMiB("VmHWM")
		a.SrcRssMiB = smapsRssMiB(map[string]string{"giw": ".giw", "pageable": "mem10-stack-"})
		fmt.Fprintf(os.Stderr, "[mem10ab %s] load %d/8 arm %s: short %.2f tok/s, long %.2f tok/s; RssShmem %d MiB (at load %d), RssAnon %d, RssFile %d, VmSwap %d, VmHWM %d; Rss of .giw %d, pageable files %d; first-layer hash %s\n",
			time.Now().Format("15:04:05"), n, name, a.Short, a.Long, a.RssShmem, a.ShmemAtLoad, a.RssAnon, a.RssFile, a.VmSwap, a.Hwm,
			a.SrcRssMiB["giw"], a.SrcRssMiB["pageable"], a.Hash)
		return a
	}
	order := []string{"A", "B", "B", "A", "A", "B", "B", "A"}
	var arms []arm
	for i, name := range order {
		arms = append(arms, run(name, i+1))
	}
	for _, a := range arms[1:] {
		if a.Hash != arms[0].Hash {
			t.Fatalf("arm %s's first MoE layer source hashes %s, arm A's %s: the arms do not upload the same bytes", a.Name, a.Hash, arms[0].Hash)
		}
	}
	if !slices.Equal(arms[1].Toks, arms[0].Toks) {
		t.Errorf("the first B load's first %d greedy tokens differ from A's: %v vs %v", checkToks, arms[1].Toks[:8], arms[0].Toks[:8])
	}
	med := func(f func(arm) float64) (float64, []float64) {
		var r []float64
		for blk := 0; blk < len(order); blk += 4 { // A B B A: each B with its adjacent A
			r = append(r, f(arms[blk+1])/f(arms[blk]), f(arms[blk+2])/f(arms[blk+3]))
		}
		s := slices.Clone(r)
		sort.Float64s(s)
		return (s[1] + s[2]) / 2, r
	}
	ms, rs := med(func(a arm) float64 { return a.Short })
	ml, rl := med(func(a arm) float64 { return a.Long })
	var shA, shB, swapB int
	for _, a := range arms {
		if a.Name == "A" {
			shA = max(shA, a.RssShmem)
		} else {
			shB = max(shB, a.RssShmem)
			swapB = max(swapB, a.VmSwap)
		}
	}
	saved := shA - shB
	verdict := "AMBIGUOUS, PARKED"
	switch {
	case ms < 0.90 || ml < 0.90:
		verdict = "KILL"
	case ms >= 0.97 && ml >= 0.97 && saved >= 11*1024 && swapB == 0:
		verdict = "SHIP-WORTHY"
	}
	fmt.Fprintf(os.Stderr, "[mem10ab] short B/A %v median %.4f; long B/A %v median %.4f; RssShmem saved %d MiB (A max %d, B max %d); B's VmSwap max %d MiB => %s\n",
		mem10Ratios(rs), ms, mem10Ratios(rl), ml, saved, shA, shB, swapB, verdict)
	if out := os.Getenv("GOINFER_MEM10_AB_OUT"); out != "" {
		b, _ := json.MarshalIndent(map[string]any{"arms": arms, "short_ratios": rs, "long_ratios": rl, "short_median": ms, "long_median": ml,
			"shmem_saved_mib": saved, "b_vmswap_max_mib": swapB, "verdict": verdict}, "", " ")
		if err := os.WriteFile(out, b, 0o644); err != nil {
			t.Error(err)
		}
	}
}

func mem10Ratios(r []float64) string {
	s := make([]string, len(r))
	for i, v := range r {
		s[i] = strconv.FormatFloat(v, 'f', 4, 64)
	}
	return strings.Join(s, ", ")
}

// procStatusMiB is a /proc/self/status field in MiB (0 when absent).
func procStatusMiB(field string) int {
	b, err := os.ReadFile("/proc/self/status")
	if err != nil {
		return 0
	}
	for _, l := range strings.Split(string(b), "\n") {
		if rest, ok := strings.CutPrefix(l, field+":"); ok {
			f := strings.Fields(rest)
			if len(f) > 0 {
				v, _ := strconv.Atoi(f[0])
				return v / 1024
			}
		}
	}
	return 0
}

// smapsRssMiB sums /proc/self/smaps's Rss over the mappings whose pathname contains each pattern, in MiB.
func smapsRssMiB(patterns map[string]string) map[string]int {
	out := map[string]int{}
	f, err := os.Open("/proc/self/smaps")
	if err != nil {
		return out
	}
	defer f.Close()
	cur := ""
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		l := sc.Text()
		if fs := strings.Fields(l); len(fs) >= 5 && strings.Contains(fs[0], "-") && !strings.HasSuffix(fs[0], ":") {
			cur = ""
			if len(fs) >= 6 {
				for k, p := range patterns {
					if strings.Contains(strings.Join(fs[5:], " "), p) {
						cur = k
					}
				}
			}
			continue
		}
		if rest, ok := strings.CutPrefix(l, "Rss:"); ok && cur != "" {
			v, _ := strconv.Atoi(strings.Fields(rest)[0])
			out[cur] += v / 1024
		}
	}
	return out
}

func loadAvg1() float64 {
	b, _ := os.ReadFile("/proc/loadavg")
	f := strings.Fields(string(b))
	if len(f) == 0 {
		return 0
	}
	v, _ := strconv.ParseFloat(f[0], 64)
	return v
}

func memAvailableMiB() int {
	b, _ := os.ReadFile("/proc/meminfo")
	for _, l := range strings.Split(string(b), "\n") {
		if rest, ok := strings.CutPrefix(l, "MemAvailable:"); ok {
			v, _ := strconv.Atoi(strings.Fields(rest)[0])
			return v / 1024
		}
	}
	return 0
}
