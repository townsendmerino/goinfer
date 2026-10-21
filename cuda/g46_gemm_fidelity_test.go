//go:build cuda && goinfer_testhooks

package cuda

import (
	"fmt"
	"os"
	"testing"
	"time"
)

// g46Depths are the prompt lengths G46 reads, from where the tensor-core GEMM's existing gate stops to the shared-memory bound
// (splitKVRequired is false at every one of them, so the pass runs attn_batched and not the row kernels).
var g46Depths = []int{4096, 6144, 8192, 10240, 12000, 12160}

// g46Seeds are two independent prompts from the corpus stream.
var g46Seeds = []int{4242, 9001}

// TestG46_gemma3GEMMFidelityLadder is G46's gate (docs/measurements/g46-gemma3-gemm-fidelity-2026-10-21.md): Gemma-3-4B at default
// settings, the batched prefill of each depth against the per-token path, last-row logits, cosine 0.99 and equal argmax. One per-token
// pass per prompt supplies the reference at every depth. A levers-off control at the shortest and longest depth must read 0 logits
// differing, so a nonzero reading elsewhere is the tensor-core GEMM's. The verdict line applies the registered rule.
//
//	GOINFER_HEAVY_TESTS=1 G46_REAL=1 go test -count=1 -timeout 30m -tags 'cuda goinfer_testhooks' -run '^TestG46_gemma3GEMMFidelityLadder$' -v ./cuda/
func TestG46_gemma3GEMMFidelityLadder(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") == "" || os.Getenv("G46_REAL") == "" {
		t.Skip("set GOINFER_HEAVY_TESTS=1 and G46_REAL=1")
	}
	path := modelPath("gemma-3-4b-it-Q4_K_M.gguf")
	requireDeviceAndFile(t, path)
	nMax := g46Depths[len(g46Depths)-1]
	g := kv0LoadCtx(t, path, "f32", nMax+600, kv0TextTokens(t, path))
	attn0, gemm0 := g.r.fastAttn, g.r.fastGemm
	defer func() { g.r.fastAttn, g.r.fastGemm = attn0, gemm0 }()
	if !gemm0 {
		t.Fatalf("the tensor-core GEMM is off at load: the default arm would not read it")
	}
	checkpoint := make(map[int]bool, len(g46Depths))
	for _, d := range g46Depths {
		if splitKVRequired(d) {
			t.Fatalf("depth %d needs the row kernels: the ladder is below the bound", d)
		}
		checkpoint[d] = true
	}

	pass, ambiguous := true, false
	start := time.Now()
	for si, seed := range g46Seeds {
		e := g.embs(g.tokens(nMax, seed))
		g.poison(nMax)
		want := make(map[int][]float32, len(g46Depths))
		t0, last := time.Now(), time.Now()
		for i := 0; i < nMax; i++ {
			if err := g.ctx.Err(); err != nil {
				t.Fatalf("seed %d per-token row %d: %v", seed, i, err)
			}
			if checkpoint[i+1] {
				out, err := g.r.Forward(e[i], i)
				if err != nil {
					t.Fatalf("seed %d per-token row %d: %v", seed, i, err)
				}
				want[i+1] = append([]float32(nil), out...)
			} else if err := g.r.ForwardNoLogits(e[i], i); err != nil {
				t.Fatalf("seed %d per-token row %d: %v", seed, i, err)
			}
			if time.Since(last) > 30*time.Second {
				fmt.Fprintf(os.Stderr, "[g46] seed %d (%d of %d) per-token reference: %d of %d rows, %.0f s, run %.0f s\n",
					seed, si+1, len(g46Seeds), i+1, nMax, time.Since(t0).Seconds(), time.Since(start).Seconds())
				last = time.Now()
			}
		}
		fmt.Fprintf(os.Stderr, "[g46] seed %d per-token reference: %d rows in %.1f s\n", seed, nMax, time.Since(t0).Seconds())

		for di, d := range g46Depths {
			if di == 0 || di == len(g46Depths)-1 {
				g.r.fastAttn, g.r.fastGemm = false, false
				g.poison(d)
				p0, m0 := g.r.psLaunches, g.r.fastGemmLaunches
				got := g.prefill(e[:d], 0)
				ld, worst := kv0LogitsDiff(want[d], got)
				fmt.Fprintf(os.Stderr, "[g46] control seed %d depth %d levers off: logits differing %d of %d (worst %.3g), %d row groups, %d tensor-core launches\n",
					seed, d, ld, len(got), worst, g.r.psLaunches-p0, g.r.fastGemmLaunches-m0)
				if ld != 0 || g.r.fastGemmLaunches != m0 {
					t.Errorf("control seed %d depth %d: levers off must equal the per-token path bit for bit and launch no tensor-core GEMM (%d logits differ, %d launches): the harness cannot attribute a difference to the GEMM",
						seed, d, ld, g.r.fastGemmLaunches-m0)
					pass = false
				}
				g.r.fastAttn, g.r.fastGemm = attn0, gemm0
			}
			g.poison(d)
			p0, m0 := g.r.psLaunches, g.r.fastGemmLaunches
			t1 := time.Now()
			got := g.prefill(e[:d], 0)
			groups, gemm := g.r.psLaunches-p0, g.r.fastGemmLaunches-m0
			if gemm == 0 {
				t.Fatalf("seed %d depth %d: no tensor-core GEMM launch: the default arm did not run the path G46 reads", seed, d)
			}
			if groups != 0 {
				t.Errorf("seed %d depth %d: %d row-kernel groups below the bound", seed, d, groups)
				pass = false
			}
			cos, same := kv0Cosine(got, want[d]), kv0Argmax(got) == kv0Argmax(want[d])
			ld, worst := kv0LogitsDiff(want[d], got)
			kv3NoNaN(t, fmt.Sprintf("seed %d depth %d", seed, d), got, kv0KV{})
			verdict := "PASS"
			switch {
			case cos < 0.99 || !same:
				verdict, pass = "FAIL", false
			case cos < 0.992:
				verdict, ambiguous = "AMBIGUOUS", true
			}
			fmt.Fprintf(os.Stderr, "[g46] seed %d depth %5d: batched %.2f s, %d tensor-core launches; logits differing %d of %d (worst %.3g); cosine %.6f, argmax equal %v: %s\n",
				seed, d, time.Since(t1).Seconds(), gemm, ld, len(got), worst, cos, same, verdict)
		}
	}
	switch {
	case !pass:
		t.Errorf("G46 FAIL: a cell read under cosine 0.99 or with a different argmax, or a control did not hold (the cell lines above); run %.0f s", time.Since(start).Seconds())
	case ambiguous:
		t.Logf("G46 AMBIGUOUS: no cell under 0.99, at least one under 0.992; parked, read cell by cell; run %.0f s", time.Since(start).Seconds())
	default:
		t.Logf("G46 PASS: every cell at cosine 0.992 or better with equal argmax; run %.0f s", time.Since(start).Seconds())
	}
}
