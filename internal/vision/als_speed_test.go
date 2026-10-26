package vision

import (
	"fmt"
	"math"
	"math/rand"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"runtime/debug"
	"slices"
	"testing"
	"time"

	"github.com/townsendmerino/aikit/linalg"
)

// alsBF16 rebuilds a projection weight as 16-bit storage: QuantizeBF16 of its f32 values, exact for a weight that came from a
// bf16 checkpoint, so the matmul reads the bytes WrapBF16 would alias. Empty and non-f32 weights are left as they are.
func alsBF16(t *testing.T, w *linalg.WeightMat) {
	f, ok := w.F32()
	if !ok {
		return
	}
	for _, v := range f {
		if linalg.BF16ToF32(linalg.F32ToBF16(v)) != v {
			t.Fatal("a weight is not exactly bf16: the checkpoint is not a bf16 one")
		}
	}
	*w = linalg.QuantizeBF16(f, w.Rows(), w.Cols())
}

// TestALS_G0_speed is MEM4(a)'s ALS-G0 (goinfer docs/tasks/task-memory-campaign-2026-10.md, "MEM4(a) gates"): the CPU tower
// forward with f32-held projection weights against 16-bit-held ones, on Qwen2.5-VL-3B's tower (random patches at grid
// (1, 32, 44)) and Gemma-3-4B's SigLIP (goinfer's gemma3_preprocess_image.png, named by ALS_G0_IMAGE), A B B A over 6 pairs
// after one warm run each, both arms' features compared bit for bit. Armed by ALS_G0=1 (real checkpoints from ~/models; a timed
// run under goinfer's timing lock).
func TestALS_G0_speed(t *testing.T) {
	if os.Getenv("ALS_G0") != "1" {
		t.Skip("set ALS_G0=1 (real checkpoints; timed)")
	}
	home, err := os.UserHomeDir()
	if err != nil {
		t.Fatal(err)
	}
	// idle waits until the 1-minute load average is under 4.0 (half this Mac's 8 cores, the void line), at most 5 minutes, and
	// returns the load it starts at: the tower forward runs on every core, so the previous run holds the average up for a minute.
	idle := func() string {
		for range 60 {
			la, err := exec.Command("sysctl", "-n", "vm.loadavg").Output()
			if err != nil {
				t.Fatal(err)
			}
			var l1 float64
			if _, err := fmt.Sscanf(string(la), "{ %f", &l1); err != nil {
				t.Fatalf("parse %q: %v", la, err)
			}
			if l1 < 4.0 {
				return fmt.Sprintf("%.2f", l1)
			}
			time.Sleep(5 * time.Second)
		}
		t.Fatal("the load never fell under 4.0 in 5 minutes: void")
		return ""
	}
	ab := func(name string, fwd func() ([]float32, error), toBF16, toF32 func()) {
		run := func(bf16 bool) (float64, []float32) {
			l := idle()
			fmt.Fprintf(os.Stderr, "[ALS-G0] %s %s run starts at load %s\n", name, map[bool]string{true: "16-bit", false: "f32"}[bf16], l)
			if bf16 {
				toBF16()
			} else {
				toF32()
			}
			t0 := time.Now()
			out, err := fwd()
			if err != nil {
				t.Fatal(err)
			}
			return time.Since(t0).Seconds(), slices.Clone(out)
		}
		run(false)
		run(true)
		var ratios []float64
		for i := range 6 {
			la, err := exec.Command("sysctl", "-n", "vm.loadavg").Output()
			if err != nil {
				t.Fatal(err)
			}
			var tf, th float64
			var of, oh []float32
			if i%2 == 0 {
				tf, of = run(false)
				th, oh = run(true)
			} else {
				th, oh = run(true)
				tf, of = run(false)
			}
			for j := range of {
				if math.Float32bits(of[j]) != math.Float32bits(oh[j]) {
					t.Fatalf("%s pair %d: the 16-bit arm's feature %d differs from the f32 arm's", name, i, j)
				}
			}
			ratios = append(ratios, th/tf)
			fmt.Fprintf(os.Stderr, "[ALS-G0] %s pair %d (load %s): f32 %.2f s, 16-bit %.2f s, 16-bit/f32 %.4f\n", name, i, la[:len(la)-1], tf, th, th/tf)
		}
		slices.Sort(ratios)
		fmt.Fprintf(os.Stderr, "[ALS-G0] %s median 16-bit/f32 %.4f (features equal in every pair)\n", name, (ratios[2]+ratios[3])/2)
	}

	t.Run("qwen2.5-vl", func(t *testing.T) {
		e, err := LoadQwenVisionEncoder(filepath.Join(home, "models", "qwen25vl-3b-instruct"), false)
		if err != nil {
			t.Fatal(err)
		}
		c := e.Cfg
		grid := [][3]int{{1, 32, 44}}
		pd := c.InChans * c.TemporalPatchSize * c.PatchSize * c.PatchSize
		rng := rand.New(rand.NewSource(61))
		px := make([]float32, 32*44*pd)
		for i := range px {
			px[i] = float32(rng.NormFloat64())
		}
		var f32w []linalg.WeightMat
		ptrs := []*linalg.WeightMat{&e.merger0w, &e.merger2w}
		for i := range e.blocks {
			b := &e.blocks[i]
			ptrs = append(ptrs, &b.qkvw, &b.projw, &b.gatew, &b.upw, &b.downw)
		}
		for _, p := range ptrs {
			f32w = append(f32w, *p)
		}
		var bf []linalg.WeightMat
		for _, w := range f32w {
			w := w
			alsBF16(t, &w)
			bf = append(bf, w)
		}
		ab("qwen2.5-vl", func() ([]float32, error) { return e.Forward(px, grid) },
			func() {
				for i, p := range ptrs {
					*p = bf[i]
				}
			}, func() {
				for i, p := range ptrs {
					*p = f32w[i]
				}
			})
	})
	t.Run("siglip", func(t *testing.T) {
		e, err := LoadEncoder(filepath.Join(home, "models", "gemma-3-4b-it"), false)
		if err != nil {
			t.Fatal(err)
		}
		img, err := os.ReadFile(os.Getenv("ALS_G0_IMAGE")) // goinfer's testdata/gemma3_preprocess_image.png
		if err != nil {
			t.Fatalf("ALS_G0_IMAGE: %v", err)
		}
		pv, err := Preprocess(img, Gemma3())
		if err != nil {
			t.Fatal(err)
		}
		var ptrs []*linalg.WeightMat
		for i := range e.layers {
			l := &e.layers[i]
			ptrs = append(ptrs, &l.qw, &l.kw, &l.vw, &l.ow, &l.fc1w, &l.fc2w)
		}
		var f32w, bf []linalg.WeightMat
		for _, p := range ptrs {
			f32w = append(f32w, *p)
			w := *p
			alsBF16(t, &w)
			bf = append(bf, w)
		}
		ab("siglip", func() ([]float32, error) { return e.Forward(pv.Data) },
			func() {
				for i, p := range ptrs {
					*p = bf[i]
				}
			}, func() {
				for i, p := range ptrs {
					*p = f32w[i]
				}
			})
	})
}

// TestALS_G2_heap is MEM4(a)'s ALS-G2: Qwen2.5-VL-3B's tower (~/models/qwen25vl-3b-instruct, bf16) loaded aliased, the Go heap in
// use after runtime.GC and debug.FreeOSMemory, against 15% of its f32 projection bytes; the copying load (aliasOffForTest) is read
// beside it for the record. Armed by ALS_G2=1 (a real checkpoint).
func TestALS_G2_heap(t *testing.T) {
	if os.Getenv("ALS_G2") != "1" {
		t.Skip("set ALS_G2=1 (a real checkpoint)")
	}
	home, err := os.UserHomeDir()
	if err != nil {
		t.Fatal(err)
	}
	dir := filepath.Join(home, "models", "qwen25vl-3b-instruct")
	heap := func() uint64 {
		runtime.GC()
		debug.FreeOSMemory()
		var ms runtime.MemStats
		runtime.ReadMemStats(&ms)
		return ms.HeapInuse
	}
	read := func(off bool) (inUse uint64, projBytes int64) {
		aliasOffForTest = off
		defer func() { aliasOffForTest = false }()
		base := heap()
		e, err := LoadQwenVisionEncoder(dir, false)
		if err != nil {
			t.Fatal(err)
		}
		inUse = heap() - base
		for i := range e.blocks {
			b := &e.blocks[i]
			for _, w := range []*linalg.WeightMat{&b.qkvw, &b.projw, &b.gatew, &b.upw, &b.downw} {
				projBytes += int64(w.Rows()) * int64(w.Cols()) * 4
			}
		}
		projBytes += int64(e.merger0w.Rows())*int64(e.merger0w.Cols())*4 + int64(e.merger2w.Rows())*int64(e.merger2w.Cols())*4
		e.Close()
		runtime.KeepAlive(e)
		return inUse, projBytes
	}
	aliased, proj := read(false)
	copied, _ := read(true)
	const MiB = 1 << 20
	fmt.Fprintf(os.Stderr, "[ALS-G2] Qwen2.5-VL-3B tower: f32 projection bytes %.1f MiB; heap in use after load: aliased %.1f MiB (%.1f%%), copied %.1f MiB (%.1f%%)\n",
		float64(proj)/MiB, float64(aliased)/MiB, 100*float64(aliased)/float64(proj), float64(copied)/MiB, 100*float64(copied)/float64(proj))
	if float64(aliased) > 0.15*float64(proj) {
		t.Errorf("aliased heap %.1f MiB is over 15%% of the f32 projection bytes (%.1f MiB)", float64(aliased)/MiB, float64(proj)/MiB)
	}
}
