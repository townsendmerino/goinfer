package decoder

import (
	"strings"
	"testing"
)

// injectHostRAMAvailable replaces the machine's CURRENTLY AVAILABLE memory figure for one test —
// the prefill_budget.go counterpart to fitguard_test.go's injectHostRAM (total RAM).
func injectHostRAMAvailable(t *testing.T, bytes int64) func() {
	t.Helper()
	prev := hostRAMAvailable
	hostRAMAvailable = func() int64 { return bytes }
	restore := func() { hostRAMAvailable = prev }
	t.Cleanup(restore)
	return restore
}

// R13 (docs/measurements/cold-user-2026-09-07-macbook-arm64.md): the load-time guard prices the
// worst case a request COULD reach; it cannot see the request that actually arrives. This is the
// request-time counterpart — driven with numbers shaped like the actual failure (a 7B-class
// model whose weights alone fit, but whose KV+scratch for a real agent-sized prompt does not).
//
// R13-follow-on: total RAM here is AMPLE (so Load succeeds and the load-time guard has nothing to
// say) but currently-AVAILABLE RAM is tight — the shape of the live re-run's actual failure: the
// load-time guard correctly auto-pinned a smaller context against total RAM, and the request
// still swapped, because other processes on the real machine had already claimed most of what
// the guard assumed was free.
func TestAdmitPrefillMemory_refusesAnOversizedRequest(t *testing.T) {
	const gguf = "testdata/gptoss_tiny.gguf"
	restore := injectHostRAM(t, 64<<30) // 64 GB total: ample, so Load succeeds regardless
	defer restore()
	restoreAvail := injectHostRAMAvailable(t, 4<<20) // but only 4 MiB actually available right now
	defer restoreAvail()

	m, err := Load(gguf, Options{Quant: "int4"})
	if err != nil {
		t.Fatalf("Load refused at 64 GB total RAM (weights alone should fit): %v", err)
	}
	defer m.Close()

	before := prefillEnters.Load()
	// A prompt+max_tokens shaped like the run's own opencode request: tens of thousands of
	// positions, which this fixture's KV rate (measured elsewhere: 512 B/position) alone already
	// exceeds a few-MiB available-memory margin.
	err = m.AdmitPrefillMemory(20000, 4096)
	if err == nil {
		t.Fatal("AdmitPrefillMemory admitted a request that cannot fit the currently-available memory")
	}
	if got := prefillEnters.Load() - before; got != 0 {
		t.Errorf("AdmitPrefillMemory allocated a KV cache (prefillEnters +%d) while REFUSING the request — "+
			"admission must be a pure check with no allocation side effect", got)
	}
	for _, want := range []string{"GB", "available", "-ctx", "-stream-weights", "GOINFER_NO_FIT_GUARD"} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("refusal does not mention %q:\n%s", want, err.Error())
		}
	}
}

// The other side: a request that genuinely fits must not be refused, and must not be charged a
// false positive from an overly conservative scratch estimate.
func TestAdmitPrefillMemory_admitsARequestThatFits(t *testing.T) {
	const gguf = "testdata/gptoss_tiny.gguf"
	restore := injectHostRAM(t, 64<<30) // 64 GB: ample for a small prompt against a tiny model
	defer restore()
	restoreAvail := injectHostRAMAvailable(t, 64<<30)
	defer restoreAvail()

	m, err := Load(gguf, Options{Quant: "int4"})
	if err != nil {
		t.Fatalf("Load refused at 64 GB RAM: %v", err)
	}
	defer m.Close()

	if err := m.AdmitPrefillMemory(64, 128); err != nil {
		t.Errorf("AdmitPrefillMemory refused a small request on an ample machine: %v", err)
	}
}

// GOINFER_NO_FIT_GUARD is the same escape hatch the load-time guard uses — one flag for both, not
// two names to remember.
func TestAdmitPrefillMemory_envOverrideAdmits(t *testing.T) {
	const gguf = "testdata/gptoss_tiny.gguf"
	restore := injectHostRAM(t, 64<<30)
	defer restore()
	restoreAvail := injectHostRAMAvailable(t, 4<<20)
	defer restoreAvail()
	t.Setenv("GOINFER_NO_FIT_GUARD", "1")

	m, err := Load(gguf, Options{Quant: "int4"})
	if err != nil {
		t.Fatalf("Load refused under the override: %v", err)
	}
	defer m.Close()

	if err := m.AdmitPrefillMemory(20000, 4096); err != nil {
		t.Errorf("GOINFER_NO_FIT_GUARD=1 did not bypass AdmitPrefillMemory: %v", err)
	}
}

// AdmitPrefillMemory must proceed (not refuse) when availability is unknown — the same "unknown
// ⇒ proceed" rule the load-time guard uses, and the platform-unsupported case (Windows/BSD).
func TestAdmitPrefillMemory_unknownAvailabilityProceeds(t *testing.T) {
	const gguf = "testdata/gptoss_tiny.gguf"
	restore := injectHostRAM(t, 64<<30)
	defer restore()
	restoreAvail := injectHostRAMAvailable(t, 0)
	defer restoreAvail()

	m, err := Load(gguf, Options{Quant: "int4"})
	if err != nil {
		t.Fatalf("Load refused: %v", err)
	}
	defer m.Close()

	if err := m.AdmitPrefillMemory(20000, 4096); err != nil {
		t.Errorf("AdmitPrefillMemory refused with unknown (0) availability: %v — unknown must proceed", err)
	}
}

// prefillScratchBytes must scale with prompt length (the MLP term) and never fall below the
// real, already-enforced attention scratch cap.
func TestPrefillScratchBytes_scalesWithPromptLength(t *testing.T) {
	cfg := &Config{IntermediateDim: 4096}
	small := prefillScratchBytes(cfg, 64)
	large := prefillScratchBytes(cfg, 8192)
	if small < prefillAttnScratchBudget {
		t.Errorf("scratch estimate %d is below the real attention scratch cap %d", small, prefillAttnScratchBudget)
	}
	if large <= small {
		t.Errorf("scratch estimate did not grow with prompt length: %d (64 tok) vs %d (8192 tok)", small, large)
	}
	// Unknown dims must still return the real floor, not zero.
	if got := prefillScratchBytes(nil, 100); got != prefillAttnScratchBudget {
		t.Errorf("nil config: got %d, want the attention scratch floor %d", got, prefillAttnScratchBudget)
	}
}
