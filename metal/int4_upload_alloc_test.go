//go:build darwin

package metal

import (
	"math/rand"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"github.com/townsendmerino/aikit/linalg"
	"github.com/townsendmerino/goinfer/decoder"
)

// stagedW4A8 is the int4 lane's upload as it was before the no-staging change, kept as the reference the new one must
// equal bit for bit: each part's words and f16 scales built in Go slices (int4DirectWords' f32 round trip of the scales,
// or the int8 rows re-packed one at a time), concatenated, then copied to the device.
func stagedW4A8(wms ...*linalg.WeightMat) (words []uint32, scales []uint16) {
	for _, w := range wms {
		if dw, ds, ok := int4DirectWords(w); ok {
			words, scales = append(words, dw...), append(scales, ds...)
			continue
		}
		q8, sc, _, _ := w.Int8()
		N, K := w.Rows(), w.Cols()
		for n := range N {
			row := make([]float32, K)
			for k := range K {
				row[k] = float32(q8[n*K+k]) * sc[n]
			}
			wd, sd := packW4A8Row(row)
			words = append(words, wd...)
			for _, s := range sd {
				scales = append(scales, f32ToF16(s))
			}
		}
	}
	return words, scales
}

// int4LaneParts is a fused projection of int4 and int8 parts with K a multiple of 32.
func int4LaneParts(rng *rand.Rand, cols int, rows []int, int8At map[int]bool) []*linalg.WeightMat {
	var out []*linalg.WeightMat
	for i, r := range rows {
		f := make([]float32, r*cols)
		for j := range f {
			f[j] = float32(rng.NormFloat64())
		}
		var w linalg.WeightMat
		if int8At[i] {
			w = linalg.QuantizeInt8(f, r, cols, true)
		} else {
			w = linalg.QuantizeInt4(f, r, cols, 32)
		}
		out = append(out, &w)
	}
	return out
}

// TestF16ScaleCopy_equalsTheRoundTrip: every finite binary16 scale survives int4DirectWords' f32 round trip unchanged, so
// copying the stored bits (w4a8Into) uploads what the f32 round trip did.
func TestF16ScaleCopy_equalsTheRoundTrip(t *testing.T) {
	f := make([]float32, 1)
	for h := range 1 << 16 {
		if h&0x7c00 == 0x7c00 && h&0x03ff != 0 {
			continue // NaN: no scale is one
		}
		linalg.F16ToF32Slice(f, []uint16{uint16(h)})
		if got := f32ToF16(f[0]); got != uint16(h) {
			t.Fatalf("f16 %#04x round-trips to %#04x", h, got)
		}
	}
}

// TestInt4Upload_noStagingAndBitIdentical: int4Concat and int4Buf write every part straight into the device buffers. The
// Go heap they allocate is a small fraction of what they upload (int8 parts re-pack through one reused row), and the
// buffers equal the staged form's words and scales bit for bit, for int4, int8 and a mix.
func TestInt4Upload_noStagingAndBitIdentical(t *testing.T) {
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Skipf("no metal device: %v", err)
	}
	rng := rand.New(rand.NewSource(3))
	for _, c := range []struct {
		name   string
		rows   []int
		int8At map[int]bool
	}{
		{"int4 Q, K, V", []int{2048, 512, 512}, nil},
		{"int8 gate, up", []int{1024, 1024}, map[int]bool{0: true, 1: true}},
		{"int4 and int8 mixed", []int{1024, 512}, map[int]bool{1: true}},
	} {
		parts := int4LaneParts(rng, 512, c.rows, c.int8At)
		wantW, wantS := stagedW4A8(parts...)
		var wb, sb Buffer
		alloc := heapAllocDuring(func() { wb, sb = int4Concat(d, parts...) })
		if !slices.Equal(wb.U32s()[:len(wantW)], wantW) || !slices.Equal(sb.U16s()[:len(wantS)], wantS) ||
			wb.Len() != len(wantW) || sb.Len() != len(wantS) {
			t.Errorf("%s: int4Concat's buffers differ from the staged form", c.name)
		}
		if up := uint64(4*len(wantW) + 2*len(wantS)); alloc > up/8 {
			t.Errorf("%s: int4Concat allocated %d bytes of Go heap to upload %d: a staging copy", c.name, alloc, up)
		}
		w1, s1 := stagedW4A8(parts[0])
		alloc = heapAllocDuring(func() { wb, sb, err = int4Buf(d, parts[0]) })
		if err != nil {
			t.Fatal(err)
		}
		if !slices.Equal(wb.U32s()[:len(w1)], w1) || !slices.Equal(sb.U16s()[:len(s1)], s1) {
			t.Errorf("%s: int4Buf's buffers differ from the staged form", c.name)
		}
		if up := uint64(4*len(w1) + 2*len(s1)); alloc > up/8 {
			t.Errorf("%s: int4Buf allocated %d bytes of Go heap to upload %d: a staging copy", c.name, alloc, up)
		}
	}
}

// TestInt4LaneBuild_heapBesideTheUpload: a whole int4 resident build of the 0.5B q4_k_m allocates, on the Go heap, a small
// fraction of the weight bytes it uploads. Skips without the file.
func TestInt4LaneBuild_heapBesideTheUpload(t *testing.T) {
	home, _ := os.UserHomeDir()
	path := filepath.Join(home, "models", "qwen2.5-coder-0.5b-instruct-q4_k_m.gguf")
	if _, err := os.Stat(path); err != nil {
		t.Skipf("no model at %s", path)
	}
	m, err := decoder.Load(path, decoder.Options{Quant: "int4", ResidentContext: 256, ResidentKVSlots: 1})
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
	t.Logf("int4 build: %d MB of Go heap allocated beside %d MB of resident weights", alloc>>20, weights>>20)
	if r.q4kLane || r.w8 {
		t.Fatal("not the int4 lane")
	}
	if alloc > weights/4 {
		t.Errorf("the build allocated %d MB of Go heap for %d MB of weights: staging copies", alloc>>20, weights>>20)
	}
}
