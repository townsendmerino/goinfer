//go:build cuda && goinfer_testhooks

package cuda

import (
	"os"
	"testing"

	gc "github.com/eitamring/gocudrv/cuda"
	"github.com/townsendmerino/goinfer/decoder"
)

// TestResidentCloseFreesVRAM is the lifecycle gate.
//
// WHY THIS EXISTS. cudaResident.Close() used to free the page-locked HOST buffer and close the
// executor channel — and nothing else. It never freed a single DEVICE allocation and never
// released the context, so every decoder.Load(Backend:"cuda") + Close() leaked the ENTIRE model
// (weights + per-layer KV cache — gigabytes on a real checkpoint) until the process exited
// (d8e81cb).
//
// It hid because it is invisible in a one-model run. It only bites a model zoo, an
// /admin/models/unload, or a test binary that loads several models in sequence — and it bit all
// three. It reddened the whole CUDA suite: VRAM ratcheted 421 -> 1801 -> 3077 -> 4783 -> 7733 MiB
// and pinned at the 8192 ceiling, after which every Alloc/NewStream returned nil, the tests
// DROPPED those errors, and the resulting zero-filled buffers surfaced as "cosine 0.000000 —
// layout/unpack mismatch". An OOM wore a parity bug's clothes for long enough that two people
// independently concluded "the tests just interfere; they pass individually" and moved on.
//
// The gate is the SHAPE of memory across load/close cycles, which is the signal that actually
// found it: a sawtooth means Close frees; a staircase means it leaks. Peak alone proves nothing,
// and memory measured AFTER the process is worthless — it always looks clean, because the
// process exited.
func TestResidentCloseFreesVRAM(t *testing.T) {
	requireHeavyModel(t)
	gguf := os.ExpandEnv("$HOME/models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf")
	if _, err := os.Stat(gguf); err != nil {
		t.Skipf("no 0.5B gguf at %s", gguf)
	}
	if err := gc.Init(); err != nil {
		t.Skipf("cuInit: %v", err)
	}
	dev, err := gc.GetDevice(0)
	if err != nil {
		t.Skipf("no device: %v", err)
	}
	// A context of our own purely to read MemInfo. Held for the whole test so the driver stays
	// initialized between cycles and the readings are comparable.
	probe, err := dev.Primary()
	if err != nil {
		t.Skipf("primary ctx: %v", err)
	}
	defer probe.Close()

	used := func() uint64 {
		free, total, e := probe.MemInfo()
		if e != nil {
			t.Fatalf("MemInfo: %v", e)
		}
		return (total - free) / (1 << 20) // MiB
	}

	base := used()
	t.Logf("baseline %d MiB", base)

	const cycles = 3
	var loaded, closed []uint64
	for i := range cycles {
		m, err := decoder.Load(gguf, decoder.Options{Backend: "cuda", Quant: "int4"})
		if err != nil {
			t.Fatalf("cycle %d: load: %v", i, err)
		}
		if !m.ResidentActive() {
			_ = m.Close()
			t.Skip("cuda declined this model — nothing resident to leak")
		}
		// Touch the model so the KV cache and scratch actually allocate; a model that is loaded
		// but never stepped would under-report what Close has to reclaim.
		if _, e := m.ResidentForwardForTest().Forward(m.EmbedResidentForTest(1), 0); e != nil {
			t.Fatalf("cycle %d: forward: %v", i, e)
		}
		l := used()
		if e := m.Close(); e != nil {
			t.Fatalf("cycle %d: close: %v", i, e)
		}
		c := used()
		loaded, closed = append(loaded, l), append(closed, c)
		t.Logf("cycle %d: loaded %d MiB (+%d), after Close %d MiB", i, l, int64(l)-int64(base), c)
	}

	// 1. Close must give the memory back. Allow slack for driver/context bookkeeping that is not
	//    the model, but the model itself is ~700+ MiB at int4 — a leak is unmissable at this bar.
	const slackMiB = 128
	for i, c := range closed {
		if c > base+slackMiB {
			t.Errorf("cycle %d: %d MiB still used after Close (baseline %d, +%d MiB) — Close did not "+
				"free the model's device memory. Every Load+Close leaks the whole model: weights and "+
				"the per-layer KV cache. This is d8e81cb; it is a PRODUCTION leak (serve leaks a model "+
				"on every /admin/models/unload), not a test artifact.",
				i, c, base, int64(c)-int64(base))
		}
	}

	// 2. The SHAPE across cycles is the real tell. If each cycle's post-Close floor climbs, memory
	//    is ratcheting — the staircase that saturated the card and made an OOM look like a parity
	//    bug. Cycle-to-cycle growth, not absolute peak, is what distinguishes a leak from a
	//    legitimately large model.
	if len(closed) >= 2 {
		drift := int64(closed[len(closed)-1]) - int64(closed[0])
		if drift > slackMiB {
			t.Errorf("post-Close memory RATCHETED %d MiB across %d load/close cycles (%v) — a staircase, "+
				"not a sawtooth. Each cycle keeps some of the model. This is the shape that pinned an "+
				"8 GB card mid-suite and turned every later Alloc into a silent nil.",
				drift, cycles, closed)
		}
	}
	if len(loaded) > 0 && loaded[0] <= base {
		t.Errorf("loading a resident model did not increase device memory (%d -> %d MiB) — the model "+
			"is not actually on the GPU, so this test is not measuring what it claims", base, loaded[0])
	}
	t.Logf("VRAM sawtooths across %d cycles: loaded %v, post-Close %v (baseline %d) — Close frees",
		cycles, loaded, closed, base)
}
