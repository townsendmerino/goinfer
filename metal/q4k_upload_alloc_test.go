//go:build darwin

package metal

import (
	"math/rand"
	"os"
	"path/filepath"
	"runtime"
	"runtime/debug"
	"slices"
	"testing"
	"unsafe"

	"github.com/townsendmerino/aikit/linalg"
	"github.com/townsendmerino/goinfer/decoder"
)

// heapAllocDuring is the Go heap bytes allocated while f runs (runtime.MemStats.TotalAlloc), with the collector held off
// so nothing is reclaimed mid-count.
func heapAllocDuring(f func()) uint64 {
	defer debug.SetGCPercent(debug.SetGCPercent(-1))
	runtime.GC()
	var a, b runtime.MemStats
	runtime.ReadMemStats(&a)
	f()
	runtime.ReadMemStats(&b)
	return b.TotalAlloc - a.TotalAlloc
}

// TestQ4KLaneBuf_uploadsWithoutStaging: q4kLaneBuf writes each part of a fused projection straight into the device buffer.
// The Go heap it allocates is a small fraction of the bytes it uploads (no per-projection concatenation), and the buffer
// holds the parts in order, bit for bit, for both its forms: Q4_K super-blocks, and int8 rows with their scales.
func TestQ4KLaneBuf_uploadsWithoutStaging(t *testing.T) {
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Skipf("no metal device: %v", err)
	}
	rng := rand.New(rand.NewSource(1))
	const cols = 256
	var q4 []*linalg.WeightMat
	var wantQ4 []byte
	for _, rows := range []int{4096, 1024, 1024} { // a fused Q, K, V
		raw := make([]byte, rows*linalg.Q4KRowBytes(cols))
		rng.Read(raw)
		w, err := linalg.WrapQ4K(raw, rows, cols)
		if err != nil {
			t.Fatal(err)
		}
		q4, wantQ4 = append(q4, &w), append(wantQ4, raw...)
	}
	var i8 []*linalg.WeightMat
	var wantQ []int8
	var wantS []float32
	for _, rows := range []int{4096, 4096} { // a fused gate and up
		q8, sc := make([]int8, rows*cols), make([]float32, rows)
		for i := range q8 {
			q8[i] = int8(rng.Intn(255) - 127)
		}
		for i := range sc {
			sc[i] = rng.Float32()
		}
		w := linalg.WrapInt8(q8, sc, rows, cols, true)
		i8, wantQ, wantS = append(i8, &w), append(wantQ, q8...), append(wantS, sc...)
	}

	var bq4, bq, bs Buffer
	alloc := heapAllocDuring(func() { bq4, _ = q4kLaneBuf(d, q4...) })
	if got := unsafe.Slice((*byte)(unsafe.Pointer(&bq4.Int8s()[0])), len(wantQ4)); !slices.Equal(got, wantQ4) || bq4.Len() != len(wantQ4) {
		t.Errorf("Q4_K: the buffer (%d bytes) does not hold the %d parts' bytes in order", bq4.Len(), len(q4))
	}
	if alloc > uint64(len(wantQ4))/8 {
		t.Errorf("Q4_K: %d bytes of Go heap to upload %d: a staging copy", alloc, len(wantQ4))
	}
	alloc = heapAllocDuring(func() { bq, bs = q4kLaneBuf(d, i8...) })
	if !slices.Equal(bq.Int8s()[:len(wantQ)], wantQ) || !slices.Equal(bs.Floats()[:len(wantS)], wantS) || bq.Len() != len(wantQ) || bs.Len() != len(wantS) {
		t.Error("int8: the buffers do not hold the parts' rows and scales in order")
	}
	if up := uint64(len(wantQ) + 4*len(wantS)); alloc > up/8 {
		t.Errorf("int8: %d bytes of Go heap to upload %d: a staging copy", alloc, up)
	}
}

// TestQ4KLaneBuild_heapBesideTheUpload: a whole q4k-lane resident build of the 0.5B q4_k_m allocates, on the Go heap, a
// small fraction of the weight bytes it uploads, so a build leaves no per-projection staging behind it for the collector.
// Skips without the file.
func TestQ4KLaneBuild_heapBesideTheUpload(t *testing.T) {
	home, _ := os.UserHomeDir()
	path := filepath.Join(home, "models", "qwen2.5-coder-0.5b-instruct-q4_k_m.gguf")
	if _, err := os.Stat(path); err != nil {
		t.Skipf("no model at %s", path)
	}
	m, err := decoder.Load(path, decoder.Options{Quant: "q4k", ResidentContext: 256, ResidentKVSlots: 1})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	defer m.Close()
	var r *resident
	alloc := heapAllocDuring(func() { r, err = buildResident(m) })
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	defer r.Close()
	weights := uint64(m.ResidentWeightBytes())
	t.Logf("build: %d MB of Go heap allocated beside %d MB of resident weights", alloc>>20, weights>>20)
	if !r.q4kLane {
		t.Fatal("not the q4k lane")
	}
	if alloc > weights/4 {
		t.Errorf("the build allocated %d MB of Go heap for %d MB of weights: staging copies", alloc>>20, weights>>20)
	}
}
