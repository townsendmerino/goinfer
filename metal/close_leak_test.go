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
	requireHeavyModel(t)
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
		r, err := buildResident(m)
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
	for i := range cycles {
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

// TestMetal_PrefillScratchDoesNotLeak is the C5 gate: PrefillLast used to allocate ~24 per-call
// scratch/uniform buffers onto the device ledger and free NONE until Close, so every request
// leaked ~100–150 MB (7B) of unified memory — a ratchet, since cmd/serve calls PrefillLast once
// per request. The mustBuf OOM panic that eventually followed is recovered only on BuildResident's
// path, not prefill's, so it killed serve. The existing close_leak tests pin load/Forward/Close
// cycles, not per-REQUEST prefill growth — which is why this class was invisible.
//
// Signal (same as the sibling gates): run many PrefillLast calls against ONE resident model and
// watch the trajectory. Per-call release → flat; the old leak → a staircase of ~24 buffers/call.
func TestMetal_PrefillScratchDoesNotLeak(t *testing.T) {
	requireHeavyModel(t)
	if testing.Short() {
		t.Skip("loads a real model and runs many prefills")
	}
	if _, err := CreateSystemDefaultDevice(); err != nil {
		t.Skipf("no metal device: %v", err)
	}
	path := os.ExpandEnv("$HOME/models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf")
	if _, err := os.Stat(path); err != nil {
		t.Skipf("no checkpoint at %s", path)
	}
	m, err := decoder.Load(path, decoder.Options{Quant: "int8int8"})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	r, err := buildResident(m)
	if err != nil {
		t.Fatalf("BuildResident: %v", err)
	}
	defer r.Close()
	if !r.prefillOK {
		t.Skip("model has no dense-prefill shape (nothing to leak here)")
	}

	// A 128-token prompt: big enough that a leaked call's scratch (guF alone is Mpad*2I*2) is a
	// clear multi-MB step, so 30 calls would leak >100 MB if the release regressed.
	const M = 128
	embs := make([][]float32, M)
	for i := range embs {
		e := make([]float32, r.H)
		for j := range e {
			e[j] = 0.02 * float32((j%7)-3) // small non-zero; output is not checked, only memory
		}
		embs[i] = e
	}

	r.PrefillLast(embs, 0)         // warm: pays library compile + first scratch alloc
	ledgerBase, _ := ledgerLens(r) // buffers on the device ledger after warm-up
	sizeBase := r.d.CurrentAllocatedSize()
	runtime.GC()
	base := rssMB(t)
	const iters = 30
	peak := base
	for range iters {
		r.PrefillLast(embs, 0)
		if got := rssMB(t); got > peak {
			peak = got
		}
	}
	ledgerEnd, _ := ledgerLens(r)
	sizeEnd := r.d.CurrentAllocatedSize()
	runtime.GC()
	end := rssMB(t)
	t.Logf("prefill: device ledger %d → %d buffers, CurrentAllocatedSize %d → %d bytes over %d prefills; RSS base %d → peak %d → end %d MB (informational)",
		ledgerBase, ledgerEnd, sizeBase, sizeEnd, iters, base, peak, end)

	// The GATE is the ledger, not RSS: with the C5 fix each PrefillLast releaseBuf's every scratch
	// buffer it allocated, so the device ledger returns to its warm-up length after each call and is
	// INVARIANT across the loop. The pre-fix leak appended ~24 buffers/call → +720 over 30. This is
	// deterministic and compression-immune; on a loaded macOS box RSS is not (leaked pages get
	// compressed straight out, so an RSS-only gate can read a real leak as flat — see
	// TestMetal_CloseWithSecondModelAlive).
	if ledgerEnd != ledgerBase {
		t.Errorf("PREFILL LEAK: device ledger grew %d → %d buffers over %d PrefillLast calls (%+d) — per-call "+
			"scratch is not being released", ledgerBase, ledgerEnd, iters, ledgerEnd-ledgerBase)
	}
	// CurrentAllocatedSize (G-06, audit-metal-2026-09-12.md): the ledger above proves this
	// package's OWN bookkeeping returns to baseline, but ReleaseBuf/ReleaseAll clear that
	// bookkeeping unconditionally before (not because) the native release actually happens — so
	// a ledger-only gate cannot tell "everything was truly freed" from "the release call silently
	// did nothing". This reads MTLDevice's own reported allocation instead, independent of the
	// ledger. Some allocator slack is expected (rounding, fragmentation from 30 alloc/free
	// cycles) — the bound is "close to baseline", not exact equality.
	if sizeEnd > sizeBase+sizeBase/4+(1<<20) { // 25% slack + 1 MiB floor for a near-zero base
		t.Errorf("PREFILL LEAK (device-reported): CurrentAllocatedSize %d → %d bytes over %d "+
			"PrefillLast calls (%+d) — native GPU memory is not actually being freed even though "+
			"the ledger (allocs=%d→%d) says it is", sizeBase, sizeEnd, iters, int64(sizeEnd)-int64(sizeBase), ledgerBase, ledgerEnd)
	}
}

// TestMetal_CloseWithSecondModelAlive is the condition the Linux box warned about: their first
// CUDA fix looked correct under a single load/close cycle and was NOT — the bug only showed with
// a second context/model alive. Two hazards it covers that a sequential test cannot:
//
//  1. USE-AFTER-FREE: Close(A) must not free anything B is still using. Metal has no context to
//     destroy, so we free per-Device — and BuildResident makes a fresh Device per model, which is
//     what should keep the ledgers disjoint. This asserts that rather than trusting it: B's
//     logits must be IDENTICAL before and after A is closed.
//  2. The free must still actually happen with another model resident.
func TestMetal_CloseWithSecondModelAlive(t *testing.T) {
	requireHeavyModel(t)
	if testing.Short() {
		t.Skip("loads real models")
	}
	if _, err := CreateSystemDefaultDevice(); err != nil {
		t.Skipf("no metal device: %v", err)
	}
	path := os.ExpandEnv("$HOME/models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf")
	if _, err := os.Stat(path); err != nil {
		t.Skipf("no checkpoint at %s", path)
	}
	load := func() *resident {
		m, err := decoder.Load(path, decoder.Options{Quant: "int8int8"})
		if err != nil {
			t.Fatalf("load: %v", err)
		}
		r, err := buildResident(m)
		if err != nil {
			t.Fatalf("BuildResident: %v", err)
		}
		return r
	}
	a, b := load(), load() // BOTH alive

	// Snapshot B's device ledgers so we can prove closing A leaves them untouched.
	bBufs0, bObjs0 := ledgerLens(b)
	// CurrentAllocatedSize (G-06) reflects the whole PHYSICAL device, not a per-Device-handle
	// value — confirmed directly (allocating through one *Device's handle shows up identically
	// through another's, since MTLCreateSystemDefaultDevice hands back retains of the same
	// underlying GPU). So this total includes both A's and B's buffers; read via b.d since a's
	// id goes to 0 once Close runs below.
	sizeBoth := b.d.CurrentAllocatedSize()
	want := append([]float32(nil), b.Forward(7, 0)...) // B's output while A is alive

	a.Close() // free A only

	// 1. B must be untouched — a shared/over-broad free shows up as changed logits or a crash.
	got := b.Forward(7, 0)
	if len(got) != len(want) {
		t.Fatalf("B logits length changed after closing A: %d vs %d", len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("USE-AFTER-FREE: closing A changed B's logits at %d (%v vs %v) — the free is not per-model",
				i, got[i], want[i])
		}
	}

	// 2. The free must actually HAPPEN with B resident — asserted on the device LEDGERS, not RSS.
	//
	// RSS cannot answer this on a loaded machine: an earlier version loaded a C after closing A and
	// checked RSS stayed flat, but macOS returns freed MTLBuffer pages to the allocator (not the OS)
	// and COMPRESSES inactive pages under memory pressure — so the reuse probe read a clean free as a
	// leak when the box was busy (swinging +2 MB idle to +560 MB loaded on identical, correct code),
	// and, worse, a real leak (ReleaseAll neutered) DID NOT ratchet RSS because the leaked pages were
	// compressed straight back out. RSS is unreliable in both directions here.
	//
	// The ledger is the ground truth: every MTLBuffer/pipeline/library goes on the owning Device's
	// allocs/objs list, and Close must empty A's while leaving B's exactly as they were. This is
	// deterministic, compression-immune, and is precisely M24's contract (release every tracked
	// handle, per model). (RSS is logged for a human, but not asserted.)
	aBufs, aObjs := ledgerLens(a)
	if aBufs != 0 || aObjs != 0 {
		// The device-id-nilled half of M24's contract is asserted in aikit/gpu's own pure-device
		// leak test (it needs Device's private id field); here the 0/0 ledger is the leak signal.
		t.Errorf("closing A left resources on its ledger: %d buffers, %d objc objects — Close did not free with B resident",
			aBufs, aObjs)
	}
	bBufs1, bObjs1 := ledgerLens(b)
	if bBufs1 != bBufs0 || bObjs1 != bObjs0 {
		t.Errorf("closing A changed B's ledger (%d→%d buffers, %d→%d objc) — the free is not per-model",
			bBufs0, bBufs1, bObjs0, bObjs1)
	}
	// CurrentAllocatedSize (G-06): with A's ledger at 0/0, closing A SHOULD have shrunk the
	// device's real allocation by roughly A's own footprint, not just cleared A's bookkeeping —
	// this is the check LedgerLen alone cannot make. Bound loosely (some free/small-buffer slack)
	// rather than pin an exact byte count, since B's own resident weights/KV/scratch still
	// dominate the total and this is only proving the total moved in the right direction.
	sizeAfter := b.d.CurrentAllocatedSize()
	if sizeAfter >= sizeBoth {
		t.Errorf("device-reported total after closing A: %d bytes, want < %d (before) — A's native "+
			"GPU memory does not appear to have actually been freed, even though its ledger reads 0/0",
			sizeAfter, sizeBoth)
	}
	runtime.GC()
	t.Logf("A+B alive → closed A: A ledger now %d buf/%d obj (want 0/0), B ledger %d buf/%d obj (unchanged from %d/%d); "+
		"device CurrentAllocatedSize %d → %d bytes; B bit-identical. RSS %d MB (informational)",
		aBufs, aObjs, bBufs1, bObjs1, bBufs0, bObjs0, sizeBoth, sizeAfter, rssMB(t))
	b.Close()
}

// ledgerLens reports how many MTLBuffers and non-buffer objc objects a resident's Device still owns
// — the compression-immune ground truth for "did Close free it". Via aikit/gpu's exported
// LedgerLen accessor (the device layer now lives there; Device's ledgers are private to it).
func ledgerLens(r *resident) (bufs, objs int) {
	return r.d.LedgerLen()
}
