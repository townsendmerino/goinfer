//go:build darwin

package metal

import (
	"fmt"
	"math"
	"os"
	"testing"
	"time"

	"github.com/townsendmerino/aikit/vision"
)

// TestKSD6H4_gridTowerScratch is KS-D6's and KS-H4's first measurement (docs/tasks/task-kernel-survey-2026-10.md), on the
// Qwen3-VL-2B grid tower (MEM2-G4's), float32, a 4K grid (32,160 patches):
//   - KS-D6: each scratch buffer's allocated bytes after the forward, and a second forward with the buffers the survey calls
//     unread (the unfused attention's qh, kh, vh, sc, vt, ob, and f2) filled with NaN first: its output must equal the
//     first forward's byte for byte, which shows nothing reads them;
//   - KS-H4: Metal's allocated size net of the build, right after the forward and 10 s later, with nothing released.
//
// It changes no product code. Not timed; a real checkpoint (GOINFER_HEAVY_TESTS=1).
func TestKSD6H4_gridTowerScratch(t *testing.T) {
	requireHeavyModel(t)
	dir := os.ExpandEnv("$HOME/models/qwen3-vl-2b-instruct")
	if _, err := os.Stat(dir); err != nil {
		t.Skipf("no checkpoint at %s", dir)
	}
	enc, err := vision.LoadQwen3VisionEncoder(dir, false)
	if err != nil {
		t.Fatal(err)
	}
	probe, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Skipf("no metal device: %v", err)
	}
	defer probe.ReleaseObjects()
	c := enc.Cfg
	pd := c.InChannels * c.TemporalPatchSize * c.PatchSize * c.PatchSize
	g := [3]int{1, 134, 240}
	px := make([]float32, g[0]*g[1]*g[2]*pd)
	for i := range px {
		px[i] = float32(math.Sin(float64(i) * 0.001))
	}
	const MiB = 1 << 20
	size := func() int64 { return int64(probe.CurrentAllocatedSize()) }
	a, err := newQwen3VAccel(enc)
	if err != nil {
		t.Fatal(err)
	}
	defer a.Close()
	built := size()
	out1, err := a.Hidden(px, [][3]int{g})
	if err != nil {
		t.Fatal(err)
	}
	out1 = append([]float32(nil), out1...)
	after := size()

	bufs := []struct {
		name   string
		b      Buffer
		unread bool
	}{
		{"x", a.x, false}, {"xn", a.xn, false}, {"pt", a.pt, false}, {"pos", a.pos, false},
		{"q", a.q, false}, {"k", a.k, false}, {"v", a.v, false}, {"ctx", a.ctx, false}, {"o", a.o, false},
		{"f1", a.f1, false}, {"f2", a.f2, true}, {"ropeCos", a.ropeCos, false}, {"ropeSin", a.ropeSin, false},
		{"qh", a.qh, true}, {"kh", a.kh, true}, {"vh", a.vh, true}, {"sc", a.sc, true}, {"vt", a.vt, true}, {"ob", a.ob, true},
	}
	var total, unread int64
	for _, b := range bufs {
		n := int64(4 * b.b.Len())
		total += n
		if b.unread {
			unread += n
		}
		fmt.Fprintf(os.Stderr, "[KS-D6] %-8s %9.1f MiB%s\n", b.name, float64(n)/MiB, map[bool]string{true: "  (called unread)", false: ""}[b.unread])
	}
	fmt.Fprintf(os.Stderr, "[KS-D6] scratch buffers %.1f MiB, of which called unread %.1f MiB (%.1f%%); pt %.1f MiB; Metal's allocated size net of the build %.1f MiB\n",
		float64(total)/MiB, float64(unread)/MiB, 100*float64(unread)/float64(total), float64(4*a.pt.Len())/MiB, float64(after-built)/MiB)

	nan := float32(math.NaN())
	for _, b := range bufs {
		if b.unread && b.b.Len() > 0 {
			f := b.b.Floats()
			for i := range f {
				f[i] = nan
			}
		}
	}
	out2, err := a.Hidden(px, [][3]int{g})
	if err != nil {
		t.Fatal(err)
	}
	differ := 0
	for i := range out1 {
		if math.Float32bits(out1[i]) != math.Float32bits(out2[i]) {
			differ++
		}
	}
	fmt.Fprintf(os.Stderr, "[KS-D6] the forward with the unread buffers filled with NaN: %d of %d outputs differ\n", differ, len(out1))
	if differ != 0 {
		t.Errorf("%d outputs differ once the buffers called unread hold NaN: something reads them", differ)
	}

	s0 := size()
	time.Sleep(10 * time.Second)
	s1 := size()
	fmt.Fprintf(os.Stderr, "[KS-H4] allocated net of the build: right after the forward %.1f MiB, 10 s later %.1f MiB\n", float64(s0-built)/MiB, float64(s1-built)/MiB)
}
