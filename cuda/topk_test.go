//go:build cuda && goinfer_testhooks

package cuda

import (
	"math"
	"math/rand"
	"os"
	"sort"
	"testing"
	"time"

	gc "github.com/eitamring/gocudrv/cuda"
	"github.com/townsendmerino/goinfer/decoder"
)

// TestTopKSelect_matchesReference is the kernel gate for topk_select (R7): for every row shape a decode
// step could produce — and several it never would — the K ids and logits the kernel returns are EXACTLY
// the first K of "sort by (logit desc, id asc)", which is the order decoder.topKByLogit defines and the
// sampler's cumulative draw depends on. The shapes are chosen to break a radix select and an
// id-ordered gather: heavy exact ties across the K boundary, a row of all-equal values, +0/-0 mixed
// (equal to the host, different bit patterns), sorted and reverse-sorted input, K==V, K==1 and a vocab
// smaller than a block. Z is checked against an f64 sum to 1e-6 relative (the device sums exp in f32).
func TestTopKSelect_matchesReference(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") == "" {
		t.Skip("set GOINFER_HEAVY_TESTS=1 (loads a 0.5B model for the compiled topk kernel)")
	}
	if err := gc.Init(); err != nil {
		t.Skipf("cuInit: %v", err)
	}
	if _, err := gc.GetDevice(0); err != nil {
		t.Skipf("no device: %v", err)
	}
	path := modelPath("qwen2.5-coder-0.5b-instruct-q4_k_m.gguf")
	if _, err := os.Stat(path); err != nil {
		t.Skipf("no fixture at %s", path)
	}
	mc, err := decoder.Load(path, decoder.Options{Backend: "cuda", Quant: "int4"})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	defer mc.Close()
	rf, ok := mc.ResidentForwardForTest().(*cudaResident)
	if !ok || rf == nil {
		t.Skip("model did not go resident")
	}

	ref := func(l []float32, k int) ([]int32, []float32) {
		idx := make([]int, len(l))
		for i := range idx {
			idx[i] = i
		}
		sort.SliceStable(idx, func(a, b int) bool {
			if l[idx[a]] != l[idx[b]] {
				return l[idx[a]] > l[idx[b]]
			}
			return idx[a] < idx[b]
		})
		ids, vals := make([]int32, k), make([]float32, k)
		for i := 0; i < k; i++ {
			ids[i], vals[i] = int32(idx[i]), l[idx[i]]
		}
		return ids, vals
	}
	mk := func(kind string, v int, rng *rand.Rand) []float32 {
		l := make([]float32, v)
		for i := range l {
			switch kind {
			case "normal":
				l[i] = float32(rng.NormFloat64() * 3)
			case "quantized":
				l[i] = float32(math.Round(rng.NormFloat64()*3*2) / 2)
			case "coarse": // ~20 distinct values: the K boundary lands deep inside a tie group
				l[i] = float32(rng.Intn(20))
			case "flat":
				l[i] = 1.25
			case "zeros":
				if rng.Intn(2) == 0 {
					l[i] = float32(math.Copysign(0, -1))
				}
			case "ascending":
				l[i] = float32(i)
			case "descending":
				l[i] = float32(v - i)
			case "negative":
				l[i] = -float32(rng.Intn(1000)) - 0.5
			}
		}
		return l
	}

	rng := rand.New(rand.NewSource(3))
	checked := 0
	for _, v := range []int{rf.vocab, 50000, 5000, 1024, 300, 60} {
		for _, kind := range []string{"normal", "quantized", "coarse", "flat", "zeros", "ascending", "descending", "negative"} {
			l := mk(kind, v, rng)
			for _, k := range []int{1, 2, 32, 256, 1000, 1024, v} {
				if k > v || k > topkMaxK {
					continue
				}
				temp := 0.8
				row, err := rf.TopKForTest(l, k, temp, true)
				if err != nil {
					t.Fatalf("v=%d %s k=%d: %v", v, kind, k, err)
				}
				wantIDs, wantVals := ref(l, k)
				for i := 0; i < k; i++ {
					if row.IDs[i] != wantIDs[i] || row.Logits[i] != wantVals[i] {
						t.Fatalf("v=%d %s k=%d: entry %d is (id %d, %v), want (id %d, %v)",
							v, kind, k, i, row.IDs[i], row.Logits[i], wantIDs[i], wantVals[i])
					}
				}
				var z float64
				mx := float64(wantVals[0])
				for _, x := range l {
					z += math.Exp((float64(x) - mx) / temp)
				}
				if rel := math.Abs(row.Z-z) / z; rel > 1e-6 {
					t.Fatalf("v=%d %s k=%d: Z %.9g vs f64 %.9g (rel %.2e)", v, kind, k, row.Z, z, rel)
				}
				checked++
			}
		}
	}
	t.Logf("%d (vocab × shape × K) rows identical to the reference order; Z within 1e-6", checked)

	// Informational: the per-token cost the decode loop pays after a forward — launch + sync + readback,
	// no upload — at the real vocab, on a NORMAL logits row. The row is uploaded first: the launch-only
	// hook reads whatever r.logits holds, and after the correctness loop above that is leftover tie-heavy
	// data that sends the kernel down its slow ordered-gather path (an earlier version of this test timed
	// exactly that and reported ~320 us).
	timingRow := mk("normal", rf.vocab, rng)
	if _, err := rf.TopKForTest(timingRow, 256, 0.8, true); err != nil {
		t.Fatal(err)
	}
	for _, wz := range []bool{false, true} {
		const n = 500
		if err := rf.TopKLaunchForTest(rf.vocab, 256, 0.8, wz); err != nil {
			t.Fatal(err)
		}
		t0 := time.Now()
		for i := 0; i < n; i++ {
			if err := rf.TopKLaunchForTest(rf.vocab, 256, 0.8, wz); err != nil {
				t.Fatal(err)
			}
		}
		t.Logf("launch-only V=%d K=256 wantZ=%v: %.0f µs/call", rf.vocab, wz, float64(time.Since(t0).Microseconds())/n)
	}
	// Informational: per-call cost of the kernel launch + sync + 2 KB readback at the real vocab.
	l := mk("normal", rf.vocab, rng)
	for _, k := range []int{256} {
		for _, wz := range []bool{false, true} {
			const n = 300
			if _, err := rf.TopKForTest(l, k, 0.8, wz); err != nil {
				t.Fatal(err)
			}
			t0 := time.Now()
			for i := 0; i < n; i++ {
				if _, err := rf.TopKForTest(l, k, 0.8, wz); err != nil {
					t.Fatal(err)
				}
			}
			t.Logf("V=%d K=%d wantZ=%v: %.0f µs/call (includes the upload the hook adds)", rf.vocab, k, wz, float64(time.Since(t0).Microseconds())/n)
		}
	}
}
