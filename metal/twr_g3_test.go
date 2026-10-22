//go:build darwin

package metal

import (
	"fmt"
	"os"
	"runtime"
	"testing"

	"github.com/townsendmerino/aikit/vision"
)

// TestTWRG3_qwen3VLHostHeap is TWR-G3 (docs/tasks/task-memory-campaign-2026-10.md, MEM4(b) gates): the Qwen3-VL-2B tower
// attached on Metal, float32; the Go heap in use after runtime.GC with the attach alone (main) and after the release serve
// now calls (ReleaseBlocks). The heap must fall by at least 80% of the blocks' float32 bytes.
func TestTWRG3_qwen3VLHostHeap(t *testing.T) {
	requireHeavyModel(t)
	dir := os.ExpandEnv("$HOME/models/qwen3-vl-2b-instruct")
	if _, err := os.Stat(dir); err != nil {
		t.Skipf("no checkpoint at %s", dir)
	}
	heap := func() int64 {
		runtime.GC()
		runtime.GC()
		var m runtime.MemStats
		runtime.ReadMemStats(&m)
		return int64(m.HeapAlloc)
	}
	enc, err := vision.LoadQwen3VisionEncoder(dir, false)
	if err != nil {
		t.Fatal(err)
	}
	w, err := enc.Weights()
	if err != nil {
		t.Fatal(err)
	}
	var blockBytes int64
	for _, b := range w.Blocks {
		for _, s := range [][]float32{b.Norm1W, b.Norm1B, b.Norm2W, b.Norm2B, b.QKV.W, b.QKV.B, b.Proj.W, b.Proj.B, b.FC1.W, b.FC1.B, b.FC2.W, b.FC2.B} {
			blockBytes += 4 * int64(len(s))
		}
	}
	w = vision.Qwen3Weights{}
	a, err := newQwen3VAccel(enc)
	if err != nil {
		t.Fatal(err)
	}
	defer a.Close()
	attached := heap()
	if err := enc.ReleaseBlocks(); err != nil {
		t.Fatal(err)
	}
	after := heap()
	const MiB = 1 << 20
	fmt.Fprintf(os.Stderr, "[TWR-G3] Qwen3-VL-2B: blocks %.1f MiB (float32); Go heap after the attach %.1f MiB, after the release %.1f MiB (-%.1f MiB, %.1f%%)\n",
		float64(blockBytes)/MiB, float64(attached)/MiB, float64(after)/MiB, float64(attached-after)/MiB, 100*float64(attached-after)/float64(blockBytes))
	if float64(attached-after) < 0.8*float64(blockBytes) {
		t.Errorf("TWR-G3: the release freed %.1f MiB of the blocks' %.1f MiB", float64(attached-after)/MiB, float64(blockBytes)/MiB)
	}
	runtime.KeepAlive(enc)
}
