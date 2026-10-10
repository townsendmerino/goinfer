package decoder

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"runtime"
	"runtime/pprof"
	"strings"
	"testing"
	"time"
)

// Prefill profiler: time and profile ONE batched CPU prefill at a chosen prompt length and quantization (history:
// docs/code-notes/decoder.md#TestG15PrefillProfile.file).
//
// It records MACHINE STATE beside the number (load average before and after), because a timing with no recorded state
// cannot be argued with later; both are required by docs/benchmarks.md's methodology list. It REFUSES to run when another
// goinfer serve or decoder test binary is running, unless overridden: an abandoned prefill still burning a core is what
// corrupts a prefill timing.
//
// The profile covers the PREFILL ONLY. Model load is excluded deliberately: it is seconds of unrelated I/O and quantization
// that would dominate a short profile and differ between arms by construction.
//
// Run (one quant per process: loadBenchModel is a sync.Once):
//
//	GOINFER_PREQUANT_GGUF=<model.gguf> GOINFER_BENCH_QUANT=int4 \
//	GOINFER_G15_K=3020 GOINFER_G15_PROF=/tmp/int4-3020.prof \
//	go test ./decoder/ -run TestG15PrefillProfile -timeout 3600s -v
//
// Then: go tool pprof -top -nodecount=30 <prof>
//
// Set GOINFER_G15_ALLOW_BUSY=1 to run anyway on a loaded box, and then say so beside any number it produces.

// loadAvg returns the 1/5/15-minute load averages as printed by uptime, or "" if
// they cannot be read. Best-effort by design: an unreadable load average must
// degrade the record, never fail the measurement.
func loadAvg() string {
	out, err := exec.Command("uptime").Output()
	if err != nil {
		return ""
	}
	line := string(out)
	i := strings.LastIndex(line, ":")
	if i < 0 {
		return strings.TrimSpace(line)
	}
	return strings.TrimSpace(line[i+1:])
}

// competingRun reports whether another goinfer serve or decoder test binary is
// running — the specific thing that corrupts a prefill timing, as opposed to
// ambient desktop load. Best-effort: if the process list cannot be read, it
// reports nothing found rather than blocking a measurement.
func competingRun() (string, bool) {
	out, err := exec.Command("ps", "-eo", "pid,comm").Output()
	if err != nil {
		return "", false
	}
	self := fmt.Sprint(os.Getpid())
	for line := range strings.SplitSeq(string(out), "\n") {
		f := strings.Fields(line)
		if len(f) < 2 || f[0] == self {
			continue
		}
		base := f[1][strings.LastIndex(f[1], "/")+1:]
		if strings.HasPrefix(base, "goinfer") || strings.HasPrefix(base, "decoder.test") {
			return line, true
		}
	}
	return "", false
}

// firstLoad parses the 1-minute load average out of loadAvg()'s text.
func firstLoad(s string) (float64, bool) {
	f := strings.FieldsFunc(s, func(r rune) bool { return r == ' ' || r == ',' })
	if len(f) == 0 {
		return 0, false
	}
	var v float64
	if _, err := fmt.Sscanf(f[0], "%f", &v); err != nil {
		return 0, false
	}
	return v, true
}

func TestG15PrefillProfile(t *testing.T) {
	out := os.Getenv("GOINFER_G15_PROF")
	if out == "" {
		t.Skip("set GOINFER_G15_PROF=<path> (and GOINFER_G15_K, GOINFER_BENCH_QUANT) to profile a prefill")
	}
	K := 3020
	if v := os.Getenv("GOINFER_G15_K"); v != "" {
		if _, err := fmt.Sscanf(v, "%d", &K); err != nil {
			t.Fatalf("GOINFER_G15_K=%q: %v", v, err)
		}
	}
	// Pre-flight: refuse on a COMPETING PROCESS OF OURS (the real hazard, precisely reproducible) and always RECORD the
	// load rather than gating on it. An absolute load threshold is wrong for a developer Mac, which idles above it with an
	// editor open, so such a check skips every real measurement and only trains people to set ALLOW_BUSY=1.
	loadBefore := loadAvg()
	// Ambient load does not invalidate the number, but it belongs ON it: a timing taken at load 5 deserves more suspicion
	// than one at load 0, and the reader can only apply that if it is written down.
	if v, ok := firstLoad(loadBefore); ok && v > 2.0 {
		t.Logf("NOTE: ambient load is %.2f — not our own work (that is refused below), but this "+
			"timing is noisier than an idle one. Quote it with the load.", v)
	}
	if other, found := competingRun(); found && os.Getenv("GOINFER_G15_ALLOW_BUSY") == "" {
		t.Skipf("another goinfer process is running (%s): a prefill timing taken beside our own "+
			"competing work is how the withdrawn G15 cliff happened. Wait for it, or set "+
			"GOINFER_G15_ALLOW_BUSY=1 and say so beside the number.", other)
	}
	m, err := loadBenchModel()
	if err != nil {
		t.Skipf("no model (%v); set GOINFER_PREQUANT_GGUF", err)
	}
	if !m.canBatchN(K) {
		t.Skipf("model has no batched prefill at K=%d", K)
	}

	ids := make([]int, K)
	for i := range ids {
		ids[i] = 785 // any valid id; content is irrelevant to prefill cost
	}
	cache := m.NewCache(K + 8)

	f, err := os.Create(out)
	if err != nil {
		t.Fatalf("create profile: %v", err)
	}
	defer f.Close()

	// Report the allocation delta alongside the profile: it costs nothing here and answers the GC question without a
	// second run.
	var before, after runtime.MemStats
	runtime.GC()
	runtime.ReadMemStats(&before)

	if err := pprof.StartCPUProfile(f); err != nil {
		t.Fatalf("start profile: %v", err)
	}
	start := time.Now()
	_, err = m.forwardLayersN(context.Background(), ids, cache, cpuFastAttention())
	elapsed := time.Since(start)
	pprof.StopCPUProfile()
	if err != nil {
		t.Fatalf("prefill K=%d: %v", K, err)
	}
	runtime.ReadMemStats(&after)

	quant := os.Getenv("GOINFER_BENCH_QUANT")
	if quant == "" {
		quant = "int8int8 (bench default)"
	}
	fmt.Fprintf(os.Stderr,
		"prefill: quant=%s K=%d elapsed=%.1fs (%.1f tok/s) alloc=%.1f GB in %d GCs -> %s\n"+
			"  machine: load before [%s] after [%s] cpus=%d\n",
		quant, K, elapsed.Seconds(), float64(K)/elapsed.Seconds(),
		float64(after.TotalAlloc-before.TotalAlloc)/(1<<30), after.NumGC-before.NumGC, out,
		loadBefore, loadAvg(), runtime.NumCPU())
}
