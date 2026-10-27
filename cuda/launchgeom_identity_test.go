//go:build cuda

package cuda

import (
	"bufio"
	"crypto/sha256"
	"encoding/hex"
	"flag"
	"fmt"
	"math"
	"math/rand/v2"
	"os"
	"sort"
	"strings"
	"testing"
)

var updateLaunchGeomHashes = flag.Bool("launchgeom.update", false, "rewrite testdata/launchgeom_hashes.txt from this tree")

const launchGeomHashFile = "testdata/launchgeom_hashes.txt"

// TestLaunchGeometry_identity hashes the launch-geometry constants the kernels hard-code, the functions that derive launch
// shapes from them, and the KV cache's f16 converters, and compares the hashes with testdata/launchgeom_hashes.txt. A
// change here changes what a kernel is launched with or reads, so every hash must hold unless the kernels change with it.
func TestLaunchGeometry_identity(t *testing.T) {
	got := launchGeomHashes()
	if *updateLaunchGeomHashes {
		names := make([]string, 0, len(got))
		for n := range got {
			names = append(names, n)
		}
		sort.Strings(names)
		var b strings.Builder
		for _, n := range names {
			fmt.Fprintf(&b, "%s %s\n", n, got[n])
		}
		if err := os.WriteFile(launchGeomHashFile, []byte(b.String()), 0o644); err != nil {
			t.Fatal(err)
		}
		t.Logf("wrote %d hashes to %s", len(got), launchGeomHashFile)
		return
	}
	f, err := os.Open(launchGeomHashFile)
	if err != nil {
		t.Fatalf("%v (write it with -launchgeom.update from a tree whose geometry is trusted)", err)
	}
	defer f.Close()
	want := map[string]string{}
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		name, h, ok := strings.Cut(sc.Text(), " ")
		if !ok {
			t.Fatalf("%s: malformed line %q", launchGeomHashFile, sc.Text())
		}
		want[name] = h
	}
	if len(want) == 0 {
		t.Fatalf("%s lists no hashes", launchGeomHashFile)
	}
	for n, h := range want {
		switch g, ok := got[n]; {
		case !ok:
			t.Errorf("%s: listed in %s, not computed", n, launchGeomHashFile)
		case g != h:
			t.Errorf("%s: changed: hash %s, want %s", n, g, h)
		}
	}
	for n := range got {
		if _, ok := want[n]; !ok {
			t.Errorf("%s: computed, not listed in %s (add it with -launchgeom.update)", n, launchGeomHashFile)
		}
	}
	t.Logf("%d launch-geometry hashes match %s", len(want), launchGeomHashFile)
}

func launchGeomHashes() map[string]string {
	out := map[string]string{}
	sum := func(name string, write func(w func(format string, a ...any))) {
		d := sha256.New()
		write(func(format string, a ...any) { fmt.Fprintf(d, format, a...) })
		out[name] = hex.EncodeToString(d.Sum(nil))
	}
	sum("constants", func(w func(string, ...any)) {
		w("faWarps=%d faMaxRows=%d attnFusedBM=%d attnFusedThreads=%d attnFusedKPAD=%d attnFusedMinRows=%d",
			faWarps, faMaxRows, attnFusedBM, attnFusedThreads, attnFusedKPAD, attnFusedMinRows)
	})
	sum("faRowsPerCTA", func(w func(string, ...any)) {
		for _, hd := range []int{32, 64, 72, 80, 96, 128, 192, 256, 512} {
			for g := 1; g <= 16; g++ {
				w("%d,%d=%d;", hd, g, faRowsPerCTA(hd, g))
			}
		}
	})
	sum("faRowRuns", func(w func(string, ...any)) {
		for _, n0 := range []int{1, 7, 64, 511, 4096} {
			for _, m := range []int{1, 2, 5, 16} {
				for _, window := range []int{0, 1, 32, 512, 4096} {
					for _, s := range []int{1, 4, 16, 32} {
						w("%d,%d,%d,%d=%v;", n0, m, window, s, faRowRuns(n0, m, window, s)) // %v, not %+v: field names are not geometry
					}
				}
			}
		}
	})
	sum("imgBlockMaxNWin", func(w func(string, ...any)) {
		for _, causal := range []int{0, 1, 64, 1000} {
			for _, window := range []int{0, 1, 128, 1024} {
				for _, se := range [][2]int{{0, 0}, {5, 3}, {0, 256}, {10, 266}, {900, 1500}, {2000, 2001}} {
					w("%d,%d,%v=%d;", causal, window, se, imgBlockMaxNWin(causal, window, se[0], se[1]))
				}
			}
		}
	})
	sum("imgBlocksMaxNWin", func(w func(string, ...any)) {
		blocks := [][][2]int{nil, {{0, 256}}, {{10, 266}, {300, 556}}, {{5, 3}, {900, 1500}, {2000, 2001}}}
		for _, causal := range []int{0, 64, 1000} {
			for _, window := range []int{0, 128, 1024} {
				for _, b := range blocks {
					w("%d,%d,%v=%d;", causal, window, b, imgBlocksMaxNWin(causal, window, b))
				}
			}
		}
	})
	// f32ToF16RNE on every float32 bit pattern, in chunks.
	d := sha256.New()
	buf := make([]byte, 2<<20)
	for hi := range uint64(1 << 12) {
		base := uint32(hi << 20)
		for i := range 1 << 20 {
			h := f32ToF16RNE(math.Float32frombits(base | uint32(i)))
			buf[2*i], buf[2*i+1] = byte(h), byte(h>>8)
		}
		d.Write(buf)
	}
	out["f32ToF16RNE/all-2^32"] = hex.EncodeToString(d.Sum(nil))

	rng := rand.New(rand.NewPCG(0x5b1, 0x9e0))
	src := make([]float32, 1<<16)
	for i := range src {
		src[i] = math.Float32frombits(rng.Uint32())
	}
	dst := make([]uint16, len(src))
	f32SliceToF16RNE(dst, src)
	sum("f32SliceToF16RNE", func(w func(string, ...any)) { w("%v", dst) })
	return out
}
