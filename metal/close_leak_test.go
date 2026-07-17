//go:build darwin

package metal

import (
	"os"
	"os/exec"
	"runtime"
	"strconv"
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// rssMB reports this process's resident size in MB. Metal buffers are shared/UMA — system
// memory — so a leaked MTLBuffer shows up here. (The Go heap side of the model is GC'd, so what
// ratchets across cycles is the objc allocations.)
func rssMB(t *testing.T) int {
	t.Helper()
	out, err := exec.Command("ps", "-o", "rss=", "-p", strconv.Itoa(os.Getpid())).Output()
	if err != nil {
		t.Skipf("ps: %v", err)
	}
	kb, err := strconv.Atoi(strings.TrimSpace(string(out)))
	if err != nil {
		t.Skipf("parse rss: %v", err)
	}
	return kb / 1024
}

// TestMetal_CloseFreesMemory is the leak gate: load a model resident, use it, Close it, repeat.
//
// The signal (the one the CUDA leak hunt trusted): does memory COME BACK between cycles, or
// ratchet? A sawtooth means Close frees; a staircase that never descends means it leaks. The
// shape is the diagnosis, so the trajectory is logged, not just the peak.
//
// Why this matters: purego has no ARC and Metal has no context-destroy to reclaim in bulk, so
// every MTLBuffer must be released explicitly. Close() used to free NOTHING — it closed the
// executor channel and returned, on the documented assumption of a "single-model lifetime".
// cmd/serve is multi-model with /admin/models/unload, so that assumption leaked a whole model
// (weights + per-layer KV + MoE experts) per load. Invisible in a one-model run — which is
// exactly why it survived.
func TestMetal_CloseFreesMemory(t *testing.T) {
	if testing.Short() {
		t.Skip("loads a real model repeatedly")
	}
	if _, err := CreateSystemDefaultDevice(); err != nil {
		t.Skipf("no metal device: %v", err)
	}
	path := os.ExpandEnv("$HOME/models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf")
	if _, err := os.Stat(path); err != nil {
		t.Skipf("no checkpoint at %s", path)
	}

	cycle := func() {
		m, err := decoder.Load(path, decoder.Options{Quant: "int8int8"})
		if err != nil {
			t.Fatalf("load: %v", err)
		}
		r, err := BuildResident(m)
		if err != nil {
			t.Fatalf("BuildResident: %v", err)
		}
		r.Forward(1, 0) // touch it so the buffers are real, not lazily unfaulted
		r.Close()
		runtime.GC() // drop the Go-heap half so what remains is the objc side
	}

	cycle() // warm: first load pays one-time costs (library compile, pipelines)
	runtime.GC()
	base := rssMB(t)
	const cycles = 4
	peak := base
	for i := 0; i < cycles; i++ {
		cycle()
		got := rssMB(t)
		if got > peak {
			peak = got
		}
		t.Logf("cycle %d: rss %d MB (%+d vs base %d)", i+1, got, got-base, base)
	}
	end := rssMB(t)
	t.Logf("trajectory: base %d MB → peak %d MB → end %d MB (growth %+d MB over %d cycles)",
		base, peak, end, end-base, cycles)

	// Each cycle allocates ~0.7 GB of Metal buffers (int8 weights re-quantized to int4 + KV).
	// If Close frees, growth across 4 cycles stays near zero; if it leaks, it is GBs.
	if grow := end - base; grow > 400 {
		t.Errorf("LEAK: rss grew %+d MB over %d load/Close cycles — Close() is not freeing "+
			"(a staircase, not a sawtooth)", grow, cycles)
	}
}
