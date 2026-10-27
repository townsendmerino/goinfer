//go:build gpu

package kernels

import (
	"fmt"
	"runtime"
	"sync"
	"time"
	"weak"

	"github.com/oliverbestmann/webgpu/wgpu"
	"github.com/townsendmerino/goinfer/internal/mem"
)

// Every buffer this package or the gpu package creates is a lease on its Context's mem.Ledger (MM3), made by the lined
// wrappers below; releaseBuf closes it. A typed source scan holds that no buffer is created or freed any other way
// (TestLedger_everyBufferIsLeased).

// bufLeases maps each live leased buffer, weakly, to its lease's release. Weak because some buffers are never released
// explicitly and are freed by the binding's finalizer when the GC collects them: a strong key would keep them, and the
// device they pin, alive. A cleanup closes such a buffer's lease when it is collected, and counts it on its Context
// (Collected): a buffer nobody released.
var bufLeases sync.Map // weak.Pointer[wgpu.Buffer] → leaseEntry

// leaseEntry is a live buffer's lease: its release, its Context and its bytes.
type leaseEntry struct {
	release func()
	c       *Context
	n       int64
}

// Collected is how many of this Context's buffers, and how many bytes, were freed by the GC with their lease still open: buffers
// no code released. The I2 check reads it (LedgerBreaks).
func (c *Context) Collected() (buffers, bytes int64) {
	return c.collected.Load(), c.collectedBytes.Load()
}

// Ledger is this Context's account of the buffers created on it.
func (c *Context) Ledger() *mem.Ledger { return &c.ledger }

// ClosedForTest, when set, is called at the end of Close's release with the Context, so a test can read its ledger then (I1, I2).
// A test seam, nil in production.
var ClosedForTest func(*Context)

func (c *Context) lease(line string, b *wgpu.Buffer) *wgpu.Buffer {
	if b == nil {
		return nil
	}
	n := int64(b.GetSize())
	wp := weak.Make(b)
	bufLeases.Store(wp, leaseEntry{c.ledger.Reserve(mem.VRAM, mem.LineID(line), n, n).Release, c, n})
	if DebugCollectedForTest != nil {
		DebugCollectedForTest(wp, true)
	}
	runtime.AddCleanup(b, collected, wp)
	return b
}

func (c *Context) tryCreateBuffer(line string, d *wgpu.BufferDescriptor) (*wgpu.Buffer, error) {
	b, err := c.device.TryCreateBuffer(d)
	if err != nil {
		return nil, err
	}
	return c.lease(line, b), nil
}

func (c *Context) tryCreateBufferInit(line string, d *wgpu.BufferInitDescriptor) (*wgpu.Buffer, error) {
	b, err := c.device.TryCreateBufferInit(d)
	if err != nil {
		return nil, err
	}
	return c.lease(line, b), nil
}

// unlease closes b's lease, if it has one.
func unlease(b *wgpu.Buffer) { unleaseKey(weak.Make(b)) }

func unleaseKey(wp weak.Pointer[wgpu.Buffer]) {
	if e, ok := bufLeases.LoadAndDelete(wp); ok {
		e.(leaseEntry).release()
	}
}

// collected is the cleanup of a leased buffer the GC freed: its lease, if still open, is closed and counted.
func collected(wp weak.Pointer[wgpu.Buffer]) {
	if e, ok := bufLeases.LoadAndDelete(wp); ok {
		le := e.(leaseEntry)
		le.c.collected.Add(1)
		le.c.collectedBytes.Add(le.n)
		if DebugCollectedForTest != nil {
			DebugCollectedForTest(wp, false)
		}
		le.release()
	}
}

// DebugCollectedForTest, when set, is told of each lease made (made=true) and each one the GC closed (made=false). A test seam.
var DebugCollectedForTest func(wp weak.Pointer[wgpu.Buffer], made bool)

// releaseBuf closes b's lease and releases it.
func releaseBuf(b *wgpu.Buffer) {
	unlease(b)
	b.Release()
}

// leasedBuf is a buffer kept for a later Release through an interface (a release list): its Release goes through releaseBuf,
// which a *wgpu.Buffer's own Release, reached through the interface, would not.
type leasedBuf struct{ b *wgpu.Buffer }

// Release closes the buffer's lease and releases it.
func (l leasedBuf) Release() { releaseBuf(l.b) }

// releaser is b's release as a func, b evaluated now as a method value would be: what a deferred or kept release holds.
func releaser(b *wgpu.Buffer) func() { return func() { releaseBuf(b) } }

// BufferReleaser is releaser, for the gpu package.
func BufferReleaser(b *wgpu.Buffer) func() { return releaser(b) }

// CreateBufferOn creates a buffer on c leased on plan line line: the gpu package's way to create one.
func CreateBufferOn(c *Context, line string, d *wgpu.BufferDescriptor) (*wgpu.Buffer, error) {
	return c.tryCreateBuffer(line, d)
}

// ReleaseBuffer closes b's lease and releases it: the gpu package's way to free a buffer.
func ReleaseBuffer(b *wgpu.Buffer) { releaseBuf(b) }

// BuffersHeld is wgpu-native's own count of the buffers this Context's instance holds for the user: what the ledger's open
// leases must equal (I3). Diagnostics; nothing decides on it.
func (c *Context) BuffersHeld() uint64 {
	r := c.instance.GenerateReport()
	if r.Hub == nil {
		return 0
	}
	return r.Hub.Buffers.NumKeptFromUser
}

// LedgerBreaks says how each closed Context breaks I1 (an unplanned byte ever) or I2 (a lease open after Close, or a buffer the
// GC freed with its lease open: one no code released): the check every test package runs once, after its tests, on the Contexts
// its ClosedForTest hook kept. It first lets the GC collect (up to about a second) so the cleanups have run. It must not run
// inside Close: forcing collection there starts finalizer-driven device releases beside the caller's.
func LedgerBreaks(cs []*Context) []string {
	open := func() int {
		n := 0
		for _, c := range cs {
			n += c.Ledger().Usage().Leases
		}
		return n
	}
	for i := 0; i < 50 && open() > 0; i++ {
		runtime.GC()
		time.Sleep(20 * time.Millisecond)
	}
	var out []string
	for _, c := range cs {
		u := c.Ledger().Usage()
		nc, bc := c.Collected()
		if u.PeakUnplannedCharged > 0 || u.Leases > 0 || nc > 0 {
			out = append(out, fmt.Sprintf("unplanned peak %d bytes; %d leases (%d bytes) open after Close, by line %v; %d buffers (%d bytes) never released, freed by the GC",
				u.PeakUnplannedCharged, u.Leases, u.Charged, u.ByLine, nc, bc))
		}
	}
	return out
}
