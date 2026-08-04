//go:build cuda

package cuda

import (
	"os"
	"testing"
	"time"

	gc "github.com/eitamring/gocudrv/cuda"
	"github.com/townsendmerino/goinfer/decoder"
)

// TestSpecDecodeCurve measures D1 (n-gram speculative decode) across KV depth now that the batched
// verify is bit-identical to the decode path (the contraction fix). At each depth it (1) GATES lossless
// against the SEQUENTIAL greedy stream — the accepted tokens must equal token-by-token Forward greedy,
// exactly, the same instrument as TestPrefillDivergenceRate — and (2) reports spec tok/s vs plain
// decode, accept rate, and tokens/round. The win is expected to compress at long context (the M=k
// verify reads KV for all k positions) and is workload-dependent (n-gram favours repetition). Heavy.
//
//	GOINFER_HEAVY_TESTS=1 go test -tags cuda -run TestSpecDecodeCurve -v -timeout 30m
func TestSpecDecodeCurve(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") == "" {
		t.Skip("set GOINFER_HEAVY_TESTS=1 (loads a 1.5B model)")
	}
	const path = "/home/francis/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"
	if err := gc.Init(); err != nil {
		t.Skipf("cuInit: %v", err)
	}
	if _, err := gc.GetDevice(0); err != nil {
		t.Skipf("no device: %v", err)
	}
	if _, err := os.Stat(path); err != nil {
		t.Skipf("no fixture at %s", path)
	}
	mc, err := decoder.Load(path, decoder.Options{Backend: "cuda", Quant: "int4"})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	defer mc.Close()
	rf := mc.ResidentForwardForTest().(*cudaResident)
	_, _, _, _, _, _, vocab := mc.Dims()
	emb := func(id int) []float32 { return mc.EmbedResidentForTest(((id % vocab) + vocab) % vocab) }

	const N = 160 // tokens generated at each depth
	const k, ctxLen = 8, 2

	t.Logf("%-7s %-9s %-9s %-8s %-8s %-9s %-s", "depth", "plain", "spec", "speedup", "accept", "tok/rnd", "lossless")
	for _, depth := range []int{128, 512, 2048} {
		// prime KV to `depth` with pseudo-random tokens (repetitive greedy → realistic n-gram hits)
		seed := make([]int, depth)
		var s uint32 = 0x2545F491
		seedEmb := make([][]float32, depth)
		for i := range seed {
			s = s*1664525 + 1013904223
			seed[i] = int(s>>9) % vocab
			seedEmb[i] = emb(seed[i])
		}
		prime := func() []float32 {
			lg, e := rf.PrefillLast(seedEmb, 0)
			if e != nil {
				t.Fatalf("prime(%d): %v", depth, e)
			}
			return lg
		}

		// --- plain sequential greedy (the ground truth + the timing baseline) ---
		lg := prime()
		gt := make([]int, 0, N)
		pos := depth
		t0 := time.Now()
		for i := 0; i < N; i++ {
			tk := argmaxF(lg)
			gt = append(gt, tk)
			l, e := rf.Forward(emb(tk), pos)
			if e != nil {
				t.Fatalf("plain@%d: %v", depth, e)
			}
			lg = l
			pos++
		}
		plainDur := time.Since(t0)

		// --- speculative (re-prime; verify via PrefillLastN which now == the decode path) ---
		prime()
		hist := append([]int(nil), seed...)
		drafted, accepted, rounds := 0, 0, 0
		t1 := time.Now()
		for len(hist)-depth < N {
			p := len(hist) - 1
			draft := ngramDraft(hist, k, ctxLen)
			feed := append([]int{hist[p]}, draft...)
			embs := make([][]float32, len(feed))
			for i, tk := range feed {
				embs[i] = emb(tk)
			}
			Ls, e := rf.PrefillLastN(embs, p)
			if e != nil {
				t.Fatalf("verify@%d: %v", depth, e)
			}
			rounds++
			drafted += len(draft)
			acc := 0
			for i := 0; i < len(draft); i++ {
				if argmaxF(Ls[i]) == draft[i] {
					acc++
				} else {
					break
				}
			}
			accepted += acc
			for i := 0; i < acc; i++ {
				hist = append(hist, draft[i])
			}
			hist = append(hist, argmaxF(Ls[acc]))
		}
		specDur := time.Since(t1)
		spec := hist[depth:]

		// --- lossless gate vs the SEQUENTIAL stream ---
		for i := 0; i < N; i++ {
			if spec[i] != gt[i] {
				t.Fatalf("depth %d: LOSSLESS VIOLATION at token %d — spec %d vs sequential %d", depth, i, spec[i], gt[i])
			}
		}
		acceptRate := 0.0
		if drafted > 0 {
			acceptRate = float64(accepted) / float64(drafted)
		}
		t.Logf("%-7d %-9.1f %-9.1f %-8.2f %-8.2f %-9.2f %s",
			depth,
			float64(N)/plainDur.Seconds(),
			float64(N)/specDur.Seconds(),
			float64(plainDur)/float64(specDur),
			acceptRate,
			float64(N)/float64(rounds),
			"✓ (==sequential)")
	}
}
