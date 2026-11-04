//go:build cuda

package cuda

import (
	"bufio"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"flag"
	"fmt"
	"hash"
	"math"
	"math/rand/v2"
	"os"
	"sort"
	"strings"
	"testing"

	"github.com/townsendmerino/aikit/linalg"
)

var updateWpackHashes = flag.Bool("wpack.update", false, "rewrite testdata/wpack_hashes.txt from this tree's packers")

const wpackHashFile = "testdata/wpack_hashes.txt"

// TestWeightPackers_bitIdentity hashes every weight packer's output on seeded inputs and compares the hashes with
// testdata/wpack_hashes.txt: package cuda's packers must produce the bytes they produced when the file was written.
// It covers packWeight on every kind planWeight admits, packWeightStack and its refusals, packI8, packI8Into,
// packInt4Into, nibblePosFast, permuteFast, and f32tof16 on all 2^32 float32 bit patterns.
func TestWeightPackers_bitIdentity(t *testing.T) {
	got := wpackHashes(t)
	if *updateWpackHashes {
		var b strings.Builder
		names := make([]string, 0, len(got))
		for n := range got {
			names = append(names, n)
		}
		sort.Strings(names)
		for _, n := range names {
			fmt.Fprintf(&b, "%s %s\n", n, got[n])
		}
		if err := os.WriteFile(wpackHashFile, []byte(b.String()), 0o644); err != nil {
			t.Fatal(err)
		}
		t.Logf("wrote %d hashes to %s", len(got), wpackHashFile)
		return
	}
	f, err := os.Open(wpackHashFile)
	if err != nil {
		t.Fatalf("%v (write it with -wpack.update from a tree whose packers are trusted)", err)
	}
	defer f.Close()
	want := map[string]string{}
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		name, h, ok := strings.Cut(sc.Text(), " ")
		if !ok {
			t.Fatalf("%s: malformed line %q", wpackHashFile, sc.Text())
		}
		want[name] = h
	}
	if len(want) == 0 {
		t.Fatalf("%s lists no hashes", wpackHashFile)
	}
	for n, h := range want {
		switch g, ok := got[n]; {
		case !ok:
			t.Errorf("%s: listed in %s, not computed", n, wpackHashFile)
		case g != h:
			t.Errorf("%s: packed bytes changed: hash %s, want %s", n, g, h)
		}
	}
	for n := range got {
		if _, ok := want[n]; !ok {
			t.Errorf("%s: computed, not listed in %s (add it with -wpack.update)", n, wpackHashFile)
		}
	}
	t.Logf("%d packer outputs match %s", len(want), wpackHashFile)
}

// wpackHashes computes every case's hash, keyed by case name.
func wpackHashes(t *testing.T) map[string]string {
	t.Helper()
	out := map[string]string{}
	rng := rand.New(rand.NewPCG(0x5b1, 0xc0da))
	floats := func(n int, scale float32) []float32 {
		f := make([]float32, n)
		for i := range f {
			f[i] = (rng.Float32()*2 - 1) * scale
		}
		return f
	}
	hashPacked := func(name string, h hostW, err error) {
		d := sha256.New()
		if err != nil {
			fmt.Fprintf(d, "error %s", err)
		} else {
			fmt.Fprintf(d, "%s %d %d\n", h.kind, h.N, h.K)
			writeLE(d, h.wpk)
			writeLE(d, h.ws)
			writeLE(d, h.ws16)
		}
		out[name] = hex.EncodeToString(d.Sum(nil))
	}

	// One weight of each kind planWeight admits, at two shapes.
	mats := map[string]func(N, K int) linalg.WeightMat{
		"int4": func(N, K int) linalg.WeightMat { return linalg.QuantizeInt4(floats(N*K, 2), N, K, 32) },
		"int4f16": func(N, K int) linalg.WeightMat {
			q4 := make([]byte, N*K/2)
			for i := range q4 {
				q4[i] = byte(rng.Uint32())
			}
			s16 := make([]uint16, N*K/32)
			for i := range s16 {
				s16[i] = uint16(rng.Uint32()) &^ 0x7c00 // exponent never all ones: finite, subnormals and negatives included
				if i%7 == 0 {
					s16[i] = uint16(rng.Uint32())&0x83ff | 0x0400*uint16(rng.IntN(30)+1)
				}
			}
			return linalg.WrapInt4F16(q4, s16, N, K, 32)
		},
		"int8":     func(N, K int) linalg.WeightMat { return linalg.QuantizeInt8(floats(N*K, 3), N, K, false) },
		"int8w8a8": func(N, K int) linalg.WeightMat { return linalg.QuantizeInt8(floats(N*K, 3), N, K, true) },
		"q4k": func(N, K int) linalg.WeightMat {
			raw := make([]byte, N*K/256*144)
			for i := range raw {
				raw[i] = byte(rng.Uint32())
			}
			w, err := linalg.WrapQ4K(raw, N, K)
			if err != nil {
				t.Fatalf("WrapQ4K: %v", err)
			}
			return w
		},
		"f32":  func(N, K int) linalg.WeightMat { return linalg.WrapF32(floats(N*K, 5), N, K) },
		"bf16": func(N, K int) linalg.WeightMat { return linalg.QuantizeBF16(floats(N*K, 5), N, K) },
	}
	kinds := make([]string, 0, len(mats))
	for k := range mats {
		kinds = append(kinds, k)
	}
	sort.Strings(kinds)
	for _, k := range kinds {
		for _, sh := range [][2]int{{3, 256}, {17, 512}} {
			w := mats[k](sh[0], sh[1])
			h, err := packWeight(&w)
			hashPacked(fmt.Sprintf("packWeight/%s/%dx%d", k, sh[0], sh[1]), h, err)
		}
		a, b, c := mats[k](5, 256), mats[k](2, 256), mats[k](9, 256)
		h, err := packWeightStack(&a, &b, &c)
		hashPacked("packWeightStack/"+k, h, err)
	}
	// The refusals: a mixed-kind stack, a ragged-K stack, a kind planWeight does not admit, a K that is not a multiple of 4.
	i4, i8 := mats["int4"](2, 256), mats["int8"](2, 256)
	h, err := packWeightStack(&i4, &i8)
	hashPacked("packWeightStack/refuse-mixed-kind", h, err)
	k256, k512 := mats["int8"](2, 256), mats["int8"](2, 512)
	h, err = packWeightStack(&k256, &k512)
	hashPacked("packWeightStack/refuse-ragged-K", h, err)
	f16 := linalg.QuantizeF16(floats(2*256, 1), 2, 256)
	h, err = packWeight(&f16)
	hashPacked("packWeight/refuse-f16", h, err)
	odd := linalg.WrapF32(floats(2*6, 1), 2, 6)
	h, err = packWeight(&odd)
	hashPacked("packWeight/refuse-K6", h, err)
	_, err = packWeightStack()
	hashPacked("packWeightStack/refuse-empty", hostW{}, err)

	// The bit-level packers.
	q8 := make([]int8, 64*128)
	for i := range q8 {
		q8[i] = int8(rng.Uint32())
	}
	d := sha256.New()
	writeLE(d, packI8(q8, 64, 128))
	out["packI8"] = hex.EncodeToString(d.Sum(nil))
	dst := make([]uint32, len(q8)/4)
	packI8Into(dst, q8)
	d = sha256.New()
	writeLE(d, dst)
	out["packI8Into"] = hex.EncodeToString(d.Sum(nil))

	q4 := make([]byte, 64*128/2)
	for i := range q4 {
		q4[i] = byte(rng.Uint32())
	}
	s16 := make([]uint16, 1<<16)
	for i := range s16 {
		s16[i] = uint16(i) // every f16 bit pattern, Inf and NaN included
	}
	dW, dS := make([]uint32, len(q4)/4), make([]uint16, len(s16))
	packInt4Into(dW, dS, q4, s16)
	d = sha256.New()
	writeLE(d, dW)
	writeLE(d, dS)
	out["packInt4Into"] = hex.EncodeToString(d.Sum(nil))

	d = sha256.New()
	for i := range 8 {
		fmt.Fprintf(d, "%d ", nibblePosFast(i))
	}
	out["nibblePosFast"] = hex.EncodeToString(d.Sum(nil))

	words := make([]uint32, 0, 1<<22+8*16)
	for i := range 8 {
		for v := range uint32(16) {
			words = append(words, v<<(4*i))
		}
	}
	for range 1 << 22 {
		words = append(words, rng.Uint32())
	}
	for i, w := range words {
		words[i] = permuteFast(w)
	}
	d = sha256.New()
	writeLE(d, words)
	out["permuteFast"] = hex.EncodeToString(d.Sum(nil))

	// f32tof16 on every float32 bit pattern, in chunks so the buffer stays small.
	d = sha256.New()
	buf := make([]uint16, 1<<20)
	for hi := range uint64(1 << 12) {
		base := uint32(hi << 20)
		for i := range buf {
			buf[i] = f32tof16(math.Float32frombits(base | uint32(i)))
		}
		writeLE(d, buf)
	}
	out["f32tof16/all-2^32"] = hex.EncodeToString(d.Sum(nil))
	return out
}

// writeLE writes a slice of packed values to d little-endian, then a separator, so adjacent slices cannot alias.
func writeLE(d hash.Hash, v any) {
	var b []byte
	switch v := v.(type) {
	case []uint32:
		b = make([]byte, 0, 4*len(v))
		for _, x := range v {
			b = binary.LittleEndian.AppendUint32(b, x)
		}
	case []uint16:
		b = make([]byte, 0, 2*len(v))
		for _, x := range v {
			b = binary.LittleEndian.AppendUint16(b, x)
		}
	case []float32:
		b = make([]byte, 0, 4*len(v))
		for _, x := range v {
			b = binary.LittleEndian.AppendUint32(b, math.Float32bits(x))
		}
	default:
		panic(fmt.Sprintf("writeLE: %T", v))
	}
	d.Write(b)
	d.Write([]byte("|"))
}
