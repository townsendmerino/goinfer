//go:build darwin

package gpudev

import (
	"bufio"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"syscall"
	"testing"
	"unsafe"
)

// touchSrc reads one float per page of a no-copy buffer, faulting every page in — the minimal way to
// make the GPU actually READ across the whole mapping (a buffer Metal never touches says nothing
// about wiring). Grid = nPages; thread gid reads buf[gid*strideFloats], writes out[gid].
const touchSrc = `
#include <metal_stdlib>
using namespace metal;
kernel void touch_pages(device const float* buf[[buffer(0)]], device float* out[[buffer(1)]],
    constant uint& strideFloats[[buffer(2)]], uint gid[[thread_position_in_grid]]) {
    out[gid] = buf[(uint)gid*strideFloats];
}
`

func wiredPages(t *testing.T) int64 {
	t.Helper()
	out, err := exec.Command("vm_stat").Output()
	if err != nil {
		t.Fatalf("vm_stat: %v", err)
	}
	sc := bufio.NewScanner(strings.NewReader(string(out)))
	for sc.Scan() {
		line := sc.Text()
		if strings.Contains(line, "Pages wired down") {
			f := strings.TrimSpace(line[strings.Index(line, ":")+1:])
			f = strings.TrimSuffix(f, ".")
			n, err := strconv.ParseInt(f, 10, 64)
			if err != nil {
				t.Fatalf("parse wired %q: %v", f, err)
			}
			return n
		}
	}
	t.Fatal("vm_stat: no 'Pages wired down' line")
	return 0
}

// TestNoCopyWiring is the DECISIVE Step-6 pre-flight for goinfer's no-copy-resident 26B plan: does
// Metal WIRE the pages of a newBufferWithBytesNoCopy shared buffer that a command buffer touches? If
// it does, mmap'd .giw weight pages become unevictable and the plan fails for a reason distinct from
// capacity — "wrong approach", not "need a bigger Mac". The wiring MECHANISM is scale-invariant, so
// this maps a BOUNDED ~1.2 GB prefix of a real file (file-backed read-only, so faulted pages are
// genuinely evictable back to disk) rather than a full 15 GB — decisive without the swap-storm risk
// on a 16 GB machine. Env-gated: GOINFER_NOCOPY_PROBE=1, with GOINFER_NOCOPY_FILE pointing at any
// large file (defaults to the 26B .giw). It lives in gpu/ because NewBufferNoCopy lives here.
//
// READ: after the dispatch, "wired" rising by ~the mapped size ⇒ Metal wires no-copy resources ⇒
// plan fails. "wired" ~flat (pages land in active/inactive, i.e. evictable) ⇒ plan works.
func TestNoCopyWiring(t *testing.T) {
	if os.Getenv("GOINFER_NOCOPY_PROBE") == "" {
		t.Skip("set GOINFER_NOCOPY_PROBE=1 to run the no-copy wiring probe (maps ~1.2 GB, shells vm_stat)")
	}
	path := os.Getenv("GOINFER_NOCOPY_FILE")
	if path == "" {
		path = "/Users/francistownsend-merino/models/gemma4-26b-int4.giw"
	}
	f, err := os.Open(path)
	if err != nil {
		t.Skipf("no probe file (%s): %v", path, err)
	}
	defer f.Close()

	const pageSize = 16384            // arm64 macOS
	const mapBytes = 76800 * pageSize // 1.2 GB, exact page multiple
	const strideFloats = pageSize / 4 // one float per page

	mapping, err := syscall.Mmap(int(f.Fd()), 0, mapBytes, syscall.PROT_READ, syscall.MAP_PRIVATE)
	if err != nil {
		t.Fatalf("mmap %d bytes (file too small?): %v", mapBytes, err)
	}
	defer func() { _ = syscall.Munmap(mapping) }()
	if uintptr(unsafe.Pointer(&mapping[0]))%pageSize != 0 {
		t.Fatalf("mmap base not page-aligned: %p", unsafe.Pointer(&mapping[0]))
	}

	d, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Skipf("device: %v", err)
	}
	lib, err := d.CompileLibrary(touchSrc, MSL3_1)
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	pipe, err := d.NewComputePipeline(lib, "touch_pages")
	if err != nil {
		t.Fatalf("pipeline: %v", err)
	}

	wBefore := wiredPages(t)
	buf := d.NewBufferNoCopy(unsafe.Pointer(&mapping[0]), mapBytes) // page-aligned base + page-multiple len
	wAfterCreate := wiredPages(t)

	nPages := mapBytes / pageSize
	out := d.NewBufferLen(nPages)
	q := d.NewCommandQueue()
	q.Run1D(pipe, nPages, 256, buf, out, NewBufferOf(d, []uint32{uint32(strideFloats)}))
	_ = out.Floats()[0] // completion barrier (Run1D waits) + force the readback
	wAfterTouch := wiredPages(t)
	_ = mapping[0] // keep the mapping alive across the GPU read (deallocator is nil)

	mapPages := int64(nPages)
	gb := func(pages int64) float64 { return float64(pages) * pageSize / (1 << 30) }
	dCreate, dTouch, dTotal := wAfterCreate-wBefore, wAfterTouch-wAfterCreate, wAfterTouch-wBefore
	t.Logf("mapped %.2f GB (%d pages) from %s", gb(mapPages), mapPages, path)
	t.Logf("wired pages: before=%d after-create=%d(Δ%+d) after-touch=%d(Δ%+d) | total Δ%+d = %.2f GB",
		wBefore, wAfterCreate, dCreate, wAfterTouch, dTouch, dTotal, gb(dTotal))

	// ESTABLISHED FINDING (2026-08, this box): Metal WIRES the pages of a no-copy buffer that a
	// command buffer touches, and they STAY wired after completion (measured +1.08 GB for a 1.17 GB
	// mapping). So a whole-model mmap-resident plan is dead — a full decode touches every weight and
	// would wire all 15 GB on a 16 GB Mac. This test now DOCUMENTS that behavior: a green run
	// confirms it; a failure means Metal's residency changed (e.g. a new OS made no-copy pages
	// evictable), which would REOPEN the whole-model no-copy option — re-evaluate the paging design.
	if dTotal < mapPages/2 {
		t.Errorf("Metal's no-copy behavior CHANGED: wired grew only %.2f GB after touching the whole %.2f GB "+
			"mapping (was ~full wiring). No-copy pages now look evictable — re-evaluate the whole-model resident plan.",
			gb(dTotal), gb(mapPages))
	} else {
		t.Logf("CONFIRMED: Metal wires no-copy pages a command buffer touches (+%.2f GB for a %.2f GB mapping, "+
			"still wired post-completion). Whole-model mmap-resident is not viable; expert demand-paging (only the "+
			"routed experts wired per token) is the path.", gb(dTotal), gb(mapPages))
	}
}
