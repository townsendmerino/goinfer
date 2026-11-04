//go:build darwin

package gpudev

import (
	"testing"

	"github.com/ebitengine/purego/objc"
)

// TestMetal_smallBufferSuballocation is a diagnostic for G-MM3c-4 (docs/tasks/task-subsystems-2026-10.md, MM3c): on llama-tiny
// the ledger's Unified charge, the sum of each buffer's allocatedSize, exceeds the growth of the device's CurrentAllocatedSize
// over the build by a few KB. It allocates small buffers one at a time and logs, for each, its length, its allocatedSize and
// how far CurrentAllocatedSize moved. If small buffers move it in whole pages, or not at all while a page has room, then
// the charge per buffer and the device's growth disagree by up to a page whenever a page was opened before the baseline, which
// is the hypothesis for the gap. It asserts only that the instrument read something; the reading is the record.
func TestMetal_smallBufferSuballocation(t *testing.T) {
	d := newLeakDevice(t)
	sizes := []int{256, 256, 256, 1024, 1024, 3692, 4096, 16384, 16384, 65536}
	var bufs []Buffer
	defer func() {
		for _, b := range bufs {
			d.ReleaseBuf(b)
		}
	}()
	start := d.CurrentAllocatedSize()
	prev := start
	var charged int64
	for i, n := range sizes {
		b := d.NewBufferBytes(n)
		bufs = append(bufs, b)
		alloc := int64(objc.Send[uintptr](b.id, selAllocatedSize))
		cur := d.CurrentAllocatedSize()
		charged += alloc
		t.Logf("buffer %d: length %d, allocatedSize %d, CurrentAllocatedSize moved %d (to %d)", i, n, alloc, int64(cur)-int64(prev), cur)
		prev = cur
	}
	grown := int64(prev) - int64(start)
	t.Logf("in all: allocatedSize sums to %d, CurrentAllocatedSize grew %d; charged - grown = %d", charged, grown, charged-grown)
	if charged == 0 {
		t.Fatal("no buffer reported an allocatedSize: the instrument read nothing")
	}
}
