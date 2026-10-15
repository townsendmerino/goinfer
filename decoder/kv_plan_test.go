package decoder

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"runtime"
	"strconv"
	"testing"
	"time"
)

// TestG40KV_decodeWithinBudgetNeverRegrows is G40's KV gate (a) (docs/tasks/task-cpu-prefill-memory-2026-10.md, "The KV
// reserve fix"): through Session.Generate, serve's route, whose cache starts with no capacity, a chunked prefill reserves
// the prompt plus max_tokens, so decoding up to the budget grows no KV layer. It holds for the f32 and the int8 cache.
// Planting a reserve of the prompt alone must turn it red (gate (c)).
func TestG40KV_decodeWithinBudgetNeverRegrows(t *testing.T) {
	for _, kvq := range []string{"", "i8"} {
		t.Run("kv="+kvq, func(t *testing.T) {
			m, err := Load(filepath.Join("..", "testdata", "llama-tiny"), Options{KVQuant: kvq})
			if err != nil {
				t.Fatalf("load: %v", err)
			}
			defer m.Close()
			prevC, prevHook := prefillChunkRows, prefillChunkDone
			defer func() { prefillChunkRows, prefillChunkDone = prevC, prevHook }()
			prefillChunkRows = 256
			var atPrefillEnd int64
			chunks := 0
			prefillChunkDone = func(chunk int) {
				if chunk >= 0 {
					chunks++
				}
				atPrefillEnd = kvRegrows.Load()
			}
			_, _, _, _, _, _, vocab := m.Dims()
			prompt := make([]int, 1000)
			for i := range prompt {
				prompt[i] = (i*7919 + 13) % vocab
			}
			const budget = 300
			s := m.NewSession(0)
			out, g := s.Generate(context.Background(), prompt, budget, SamplingParams{})
			n := 0
			for range out {
				n++
			}
			if err := g.Err(); err != nil {
				t.Fatalf("generate: %v", err)
			}
			if chunks < 2 {
				t.Fatalf("the prefill ran in %d chunk(s): the gate needs a chunked prefill", chunks)
			}
			// A rounding slack of up to a few dozen rows could hide a missing reserve; the decode has to run past it.
			if n < 100 {
				t.Fatalf("generation stopped after %d tokens: too few to cross the reserve's rounding slack, so the gate cannot see a missing reserve", n)
			}
			if d := kvRegrows.Load() - atPrefillEnd; d != 0 {
				t.Errorf("decoding %d tokens within a budget of %d grew a KV layer %d time(s): the prefill did not reserve the budget", n, budget, d)
			}
			fmt.Fprintf(os.Stderr, "[g40-kv] kv=%q: %d chunks, %d tokens decoded, regrows during decode %d\n", kvq, chunks, n, kvRegrows.Load()-atPrefillEnd)
		})
	}
}

// processMemory is the process's resident memory as the OS counts it: footprint(1)'s phys_footprint on macOS, VmRSS on
// Linux, and -1 where neither is available.
func processMemory() int64 {
	switch runtime.GOOS {
	case "darwin":
		o, err := exec.Command("footprint", "-p", strconv.Itoa(os.Getpid())).Output()
		if err != nil {
			return -1
		}
		mm := regexp.MustCompile(`Footprint: ([\d.]+) (KB|MB|GB)`).FindSubmatch(o)
		if mm == nil {
			return -1
		}
		v, _ := strconv.ParseFloat(string(mm[1]), 64)
		return int64(v * map[string]float64{"KB": 1 << 10, "MB": 1 << 20, "GB": 1 << 30}[string(mm[2])])
	case "linux":
		b, err := os.ReadFile("/proc/self/status")
		if err != nil {
			return -1
		}
		mm := regexp.MustCompile(`VmRSS:\s+(\d+) kB`).FindSubmatch(b)
		if mm == nil {
			return -1
		}
		v, _ := strconv.ParseInt(string(mm[1]), 10, 64)
		return v << 10
	}
	return -1
}

// TestG40KV_forcedRegrowReleases is G40's KV gate (b): on the 0.5B's KV geometry (24 layers, 2 KV heads of 64, f32), a
// cache planned for 20,189 rows and filled to its capacity takes one row more. Every layer grows by the policy, the old arrays are dropped,
// and memory returns to the OS: afterwards the live heap and the process's memory (processMemory) are within the grown KV
// plus a margin (16 MiB live, 64 MiB process) over their readings before the cache existed. No forward pass runs. Where the
// OS gives no reading (Windows), only the live-heap bound is read.
func TestG40KV_forcedRegrowReleases(t *testing.T) {
	const layers, nKV, hd, rows = 24, 2, 64, 20189
	live := func() int64 {
		runtime.GC()
		var s runtime.MemStats
		runtime.ReadMemStats(&s)
		return int64(s.HeapAlloc)
	}
	fp := processMemory
	releaseHostMemory()
	liveBase, fpBase := live(), fp()
	c := NewKVCache(layers, nKV, hd, 0, 0, nil)
	c.planRequest(rows, 0, 32768)
	row := make([]float32, nKV*hd)
	app := func(n int) {
		for range n {
			for l := range layers {
				c.Append(l, row, row)
			}
		}
	}
	app(2048)
	c.reserveRows(c.planRows - c.Pos())
	app(rows - 2048)
	app(cap(c.keys[0])/(nKV*hd) - c.Pos()) // the reserve's rounding slack: still no growth
	before := kvRegrows.Load()
	app(1) // past the capacity: a forced regrow of every layer
	if d := kvRegrows.Load() - before; d != layers {
		t.Fatalf("one row past the plan grew %d layers, want %d", d, layers)
	}
	var capBytes int64
	for l := range layers {
		capBytes += 4 * int64(cap(c.keys[l])+cap(c.vals[l]))
	}
	time.Sleep(2 * time.Second)
	liveAfter, fpAfter := live()-liveBase, fp()
	const MiB = 1 << 20
	fmt.Fprintf(os.Stderr, "[g40-kv] forced regrow: KV capacity %.1f MiB (%d rows); live heap over base %.1f MiB (bound +16); footprint %.1f -> %.1f MiB (bound base + KV + 64)\n",
		float64(capBytes)/MiB, cap(c.keys[0])/(nKV*hd), float64(liveAfter)/MiB, float64(fpBase)/MiB, float64(fpAfter)/MiB)
	if liveAfter > capBytes+16*MiB {
		t.Errorf("live heap %d bytes over its base after the regrow; the grown KV is %d: the old arrays are still referenced", liveAfter, capBytes)
	}
	if fpBase < 0 || fpAfter < 0 {
		if runtime.GOOS == "darwin" || runtime.GOOS == "linux" {
			t.Fatalf("the process's memory is unreadable on %s: the gate cannot read its second bound", runtime.GOOS)
		}
		t.Logf("no process-memory reading on %s: the footprint bound is not read here; the live-heap bound above holds", runtime.GOOS)
		return
	}
	if fpAfter > fpBase+capBytes+64*MiB {
		t.Errorf("footprint %d bytes after the regrow, over base %d + KV %d + 64 MiB: the old arrays' memory was not returned", fpAfter, fpBase, capBytes)
	}
	runtime.KeepAlive(c)
}
