package decoder

import (
	"fmt"
	"math"
	"os"
	"runtime"
	"sync"
	"testing"
	"time"

	"github.com/townsendmerino/aikit/linalg"
)

// P19 — does the FUSED (FlashAttention-style) schedule beat the materialized one?
//
// Step 0 established that goinfer materializes: G20 tiles over QUERY ROWS only,
// `scores` is tile x nKeys, and attendBatchedHeads forbids a key-dimension split
// because it "would re-associate the softmax denominator and the AV fold".
// So the N-wide score row makes three trips through memory per tile — written by
// QK^T, read-and-rewritten by the softmax, read again by scores*V. At the
// production tile budget (attnScoreTileBytes = 8 MiB) that block is far past L2
// on either machine, so the traffic is real.
//
// This measures whether removing it wins, WITHOUT touching production code. It is
// a prototype for a decision, not an implementation.
//
// THE COMPARISON IS AT FIXED PRECISION, and that is the item's own decision rule:
// "a win that only appears with f32 enabled is A3's win, not this item's". Both
// arms are f32 over identical inputs, so the delta is the SCHEDULE and nothing
// else. Gathers sit outside the timed region because both arms pay exactly the
// same ones; including them would only dilute the effect being measured.
//
// PRE-REGISTERED BAR (written before the first run, and before the kernel):
//
//	>= 1.30x at K=8192 shapes -> fusion clears its own bar, prototype in production
//	<  1.10x                  -> close the item
//	1.10-1.30x                -> AMBIGUOUS, parks
//
// NOT bit-identical by construction: the running-max rescale re-associates. That
// is the item's stated cost, the same category as --cpu-fast-attention. So
// correctness is a tolerance, declared here rather than after: cosine >= 0.9999
// against the materialized arm.
//
// Masking: neither arm masks (full attention over nKeys). Identical in both, so
// the ratio is unaffected; it means the absolute times are not production TTFT.
func TestP19FusedAttention(t *testing.T) {
	if os.Getenv("GOINFER_P19") == "" {
		t.Skip("set GOINFER_P19=1 to run the fused-attention prototype measurement")
	}
	const (
		hd     = 128  // head dim
		nKeys  = 8192 // the depth the 55%-of-CUDA-prefill share was measured at
		ktProd = 256  // attnRowTile(K=8192, nKeys=8192) — production's query tile
		reps   = 3
	)
	scale := float32(1 / math.Sqrt(float64(hd)))
	qh := randF32(ktProd*hd, 21)
	kh := randF32(nKeys*hd, 22)       // gathered K, [nKeys, hd]
	vt := randF32(hd*nKeys, 23)       // gathered+transposed V, [hd, nKeys]
	vRow := make([]float32, nKeys*hd) // V in [nKeys, hd] order, for the fused AV block
	for s := range nKeys {
		for d := range hd {
			vRow[s*hd+d] = vt[d*nKeys+s]
		}
	}

	// ARM 1 — MATERIALIZED, the production schedule.
	scores := make([]float32, ktProd*nKeys)
	chMat := make([]float32, ktProd*hd)
	materialized := func() {
		linalg.MatmulBT(qh, kh, scores, ktProd, hd, nKeys)
		for i := range ktProd {
			row := scores[i*nKeys : (i+1)*nKeys]
			maxS := float32(math.Inf(-1))
			for j := range row {
				row[j] *= scale
				if row[j] > maxS {
					maxS = row[j]
				}
			}
			var sum float64
			for j := range row {
				e := math.Exp(float64(row[j] - maxS))
				row[j] = float32(e)
				sum += e
			}
			inv := float32(1 / sum)
			for j := range row {
				row[j] *= inv
			}
		}
		linalg.MatmulBT(scores, vt, chMat, ktProd, nKeys, hd)
	}

	// ARM 2 — FUSED. Block over KEYS; keep the score block resident and fold it
	// into the output accumulator with a running max and running sum, so the
	// kt x nKeys matrix never exists.
	chFused := make([]float32, ktProd*hd)
	fused := func(kb int) func() {
		sBlk := make([]float32, ktProd*kb)
		tmp := make([]float32, ktProd*hd)
		acc := make([]float32, ktProd*hd)
		mRun := make([]float32, ktProd)
		lRun := make([]float32, ktProd)
		// V, pre-transposed PER BLOCK and laid out block-major, ONCE. MatmulBT
		// computes a·bᵀ so the AV block needs b as [hd, n], and a column range of
		// the [hd, nKeys] `vt` is not contiguous. The first draft of this built
		// that transpose inside the timed loop, which would have charged the fused
		// arm for a layout cost the schedule does not actually imply — and made
		// the arm being tested lose for the wrong reason.
		vBlk := make([]float32, hd*nKeys)
		for k0 := 0; k0 < nKeys; k0 += kb {
			n := min(kb, nKeys-k0)
			base := k0 * hd
			for j := range n {
				for d := range hd {
					vBlk[base+d*n+j] = vRow[(k0+j)*hd+d]
				}
			}
		}
		return func() {
			for i := range ktProd {
				mRun[i], lRun[i] = float32(math.Inf(-1)), 0
			}
			for i := range acc {
				acc[i] = 0
			}
			for k0 := 0; k0 < nKeys; k0 += kb {
				k1 := min(k0+kb, nKeys)
				n := k1 - k0
				// score block [ktProd, n] — small enough to stay in cache
				linalg.MatmulBT(qh, kh[k0*hd:k1*hd], sBlk[:ktProd*n], ktProd, hd, n)
				for i := range ktProd {
					row := sBlk[i*n : i*n+n]
					blkMax := float32(math.Inf(-1))
					for j := range row {
						row[j] *= scale
						if row[j] > blkMax {
							blkMax = row[j]
						}
					}
					mNew := mRun[i]
					if blkMax > mNew {
						mNew = blkMax
					}
					corr := float32(math.Exp(float64(mRun[i] - mNew)))
					var sum float64
					for j := range row {
						e := math.Exp(float64(row[j] - mNew))
						row[j] = float32(e)
						sum += e
					}
					lRun[i] = lRun[i]*corr + float32(sum)
					mRun[i] = mNew
					// rescale this row's accumulator to the new max
					if corr != 1 {
						a := acc[i*hd : (i+1)*hd]
						for d := range a {
							a[d] *= corr
						}
					}
				}
				// acc += P_block · V_block. MatmulBT overwrites, so fold via tmp —
				// kt*hd is 128 KB, noise beside the block matmul.
				linalg.MatmulBT(sBlk[:ktProd*n], vBlk[k0*hd:k0*hd+hd*n], tmp[:ktProd*hd], ktProd, n, hd)
				for i := range ktProd * hd {
					acc[i] += tmp[i]
				}
			}
			for i := range ktProd {
				inv := 1 / lRun[i]
				a := acc[i*hd : (i+1)*hd]
				o := chFused[i*hd : (i+1)*hd]
				for d := range a {
					o[d] = a[d] * inv
				}
			}
		}
	}

	best := func(f func()) time.Duration {
		f()
		b := time.Duration(1<<62 - 1)
		for range reps {
			t0 := time.Now()
			f()
			if d := time.Since(t0); d < b {
				b = d
			}
		}
		return b
	}

	// SERIAL CONTROL. MatmulBT fans out over its N output columns, and the two arms
	// present very different N: the materialized QK^T has N=nKeys=8192, the fused
	// one N=kb (128..1024). So a fused loss could be a PARALLELISM artifact --
	// fewer columns to shard, more fork/joins -- rather than a property of the
	// schedule. Running both arms serial as well separates those. This repo has
	// already published one ratio that was mostly core count (G24's first pass,
	// 17.6x against a documented ~3.7x), so the control is not optional.
	serial := os.Getenv("GOINFER_P19_SERIAL") == "1"
	if serial {
		orig := linalg.ParallelThreshold()
		linalg.SetParallelThreshold(1 << 62)
		defer linalg.SetParallelThreshold(orig)
	}
	fmt.Fprintf(os.Stderr, "P19 fused vs materialized: kt=%d hd=%d nKeys=%d, f32 both arms, serial=%v\n",
		ktProd, hd, nKeys, serial)
	dMat := best(materialized)
	fmt.Fprintf(os.Stderr, "  materialized (production schedule) %8.1f ms\n", float64(dMat.Microseconds())/1000)

	type res struct {
		kb    int
		d     time.Duration
		cos   float64
		maxAb float64
	}
	var out []res
	for _, kb := range []int{128, 256, 512, 1024} {
		fn := fused(kb)
		d := best(fn)
		// correctness against the materialized arm
		var dot, na, nb, maxAb float64
		for i := range chMat {
			x, y := float64(chMat[i]), float64(chFused[i])
			dot += x * y
			na += x * x
			nb += y * y
			if a := math.Abs(x - y); a > maxAb {
				maxAb = a
			}
		}
		cos := dot / (math.Sqrt(na) * math.Sqrt(nb))
		out = append(out, res{kb, d, cos, maxAb})
		fmt.Fprintf(os.Stderr, "  fused kb=%-5d %8.1f ms   %.3fx   cosine %.9f  max|diff| %.3g\n",
			kb, float64(d.Microseconds())/1000, float64(dMat)/float64(d), cos, maxAb)
	}

	// ---------------------------------------------------------------------
	// ROW-PARALLEL ARMS — the control for this file's own stated caveat.
	//
	// The arms above showed fusion losing 0.690x in parallel and washing (1.031x)
	// serially, and attributed the gap to MatmulBT's fan-out being over N OUTPUT
	// COLUMNS: materialized presents N=8192, fused presents N=kb. That is a
	// property of composing fusion over a column-parallel matmul, NOT of the
	// schedule -- so it left open whether a version parallelised over QUERY ROWS
	// changes the answer.
	//
	// This tests exactly that, and controls the obvious confound: both arms are
	// row-parallel across the same worker count, and both use MatmulBT as a
	// SERIAL inner primitive (per-worker Workspace, threshold pinned). So kernel
	// quality is identical and the parallelism model is identical; the schedule is
	// again the only variable. Writing a hand-rolled SIMD inner loop instead would
	// have measured my scalar Go against aikit's tuned kernel and told us nothing
	// about scheduling.
	rowsPer := func(w, workers int) (int, int) {
		per := (ktProd + workers - 1) / workers
		return w * per, min((w+1)*per, ktProd)
	}
	workers := min(runtime.GOMAXPROCS(0), 8)

	chMatRP := make([]float32, ktProd*hd)
	// PER-WORKER SCRATCH AND WORKSPACES HOISTED OUT OF THE TIMED REGION. The first
	// version allocated the materialized arm's [n, nKeys] score buffer (1.4 MB per
	// worker) INSIDE the timed call while the fused arm allocated far less — which
	// would have charged the losing arm for allocation and inflated exactly the
	// result being claimed. Caught before quoting the number, not after.
	matScr := make([][]float32, workers)
	matWS := make([]*linalg.Workspace, workers)
	for w := range workers {
		r0, r1 := rowsPer(w, workers)
		if r0 < r1 {
			matScr[w] = make([]float32, (r1-r0)*nKeys)
		}
		matWS[w] = serialMMWorkspace()
	}
	matRowPar := func() {
		var wg sync.WaitGroup
		for w := range workers {
			r0, r1 := rowsPer(w, workers)
			if r0 >= r1 {
				continue
			}
			wg.Add(1)
			go func(w, r0, r1 int) {
				defer wg.Done()
				ws := matWS[w]
				n := r1 - r0
				sc := matScr[w]
				ws.MatmulBT(qh[r0*hd:r1*hd], kh, sc, n, hd, nKeys)
				for i := range n {
					row := sc[i*nKeys : (i+1)*nKeys]
					maxS := float32(math.Inf(-1))
					for j := range row {
						row[j] *= scale
						if row[j] > maxS {
							maxS = row[j]
						}
					}
					var sum float64
					for j := range row {
						e := math.Exp(float64(row[j] - maxS))
						row[j] = float32(e)
						sum += e
					}
					inv := float32(1 / sum)
					for j := range row {
						row[j] *= inv
					}
				}
				ws.MatmulBT(sc, vt, chMatRP[r0*hd:r1*hd], n, nKeys, hd)
			}(w, r0, r1)
		}
		wg.Wait()
	}

	chFusedRP := make([]float32, ktProd*hd)
	fusedRowPar := func(kb int) func() {
		vBlk := make([]float32, hd*nKeys)
		for k0 := 0; k0 < nKeys; k0 += kb {
			n := min(kb, nKeys-k0)
			base := k0 * hd
			for j := range n {
				for d := range hd {
					vBlk[base+d*n+j] = vRow[(k0+j)*hd+d]
				}
			}
		}
		// Same hoist for the fused arm, so neither pays allocation inside timing.
		fWS := make([]*linalg.Workspace, workers)
		fS := make([][]float32, workers)
		fTmp := make([][]float32, workers)
		fAcc := make([][]float32, workers)
		fM := make([][]float32, workers)
		fL := make([][]float32, workers)
		for w := range workers {
			r0, r1 := rowsPer(w, workers)
			nr := max(r1-r0, 0)
			fWS[w] = serialMMWorkspace()
			fS[w] = make([]float32, nr*kb)
			fTmp[w] = make([]float32, nr*hd)
			fAcc[w] = make([]float32, nr*hd)
			fM[w] = make([]float32, nr)
			fL[w] = make([]float32, nr)
		}
		return func() {
			var wg sync.WaitGroup
			for w := range workers {
				r0, r1 := rowsPer(w, workers)
				if r0 >= r1 {
					continue
				}
				wg.Add(1)
				go func(w, r0, r1 int) {
					defer wg.Done()
					ws := fWS[w]
					nr := r1 - r0
					sBlk, tmp, acc, mRun, lRun := fS[w], fTmp[w], fAcc[w], fM[w], fL[w]
					for i := range acc {
						acc[i] = 0
					}
					for i := range nr {
						mRun[i], lRun[i] = float32(math.Inf(-1)), 0
					}
					for k0 := 0; k0 < nKeys; k0 += kb {
						k1 := min(k0+kb, nKeys)
						n := k1 - k0
						ws.MatmulBT(qh[r0*hd:r1*hd], kh[k0*hd:k1*hd], sBlk[:nr*n], nr, hd, n)
						for i := range nr {
							row := sBlk[i*n : i*n+n]
							blkMax := float32(math.Inf(-1))
							for j := range row {
								row[j] *= scale
								if row[j] > blkMax {
									blkMax = row[j]
								}
							}
							mNew := mRun[i]
							if blkMax > mNew {
								mNew = blkMax
							}
							corr := float32(math.Exp(float64(mRun[i] - mNew)))
							var sum float64
							for j := range row {
								e := math.Exp(float64(row[j] - mNew))
								row[j] = float32(e)
								sum += e
							}
							lRun[i] = lRun[i]*corr + float32(sum)
							mRun[i] = mNew
							if corr != 1 {
								a := acc[i*hd : (i+1)*hd]
								for d := range a {
									a[d] *= corr
								}
							}
						}
						ws.MatmulBT(sBlk[:nr*n], vBlk[k0*hd:k0*hd+hd*n], tmp[:nr*hd], nr, n, hd)
						for i := range nr * hd {
							acc[i] += tmp[i]
						}
					}
					for i := range nr {
						inv := 1 / lRun[i]
						a := acc[i*hd : (i+1)*hd]
						o := chFusedRP[(r0+i)*hd : (r0+i+1)*hd]
						for d := range a {
							o[d] = a[d] * inv
						}
					}
				}(w, r0, r1)
			}
			wg.Wait()
		}
	}

	cosVs := func(ref, got []float32) float64 {
		var dot, na, nb float64
		for i := range ref {
			x, y := float64(ref[i]), float64(got[i])
			dot += x * y
			na += x * x
			nb += y * y
		}
		return dot / (math.Sqrt(na) * math.Sqrt(nb))
	}

	fmt.Fprintf(os.Stderr, "\n  ROW-PARALLEL control (%d workers, serial MatmulBT inner in BOTH arms)\n", workers)
	dMatRP := best(matRowPar)
	fmt.Fprintf(os.Stderr, "  materialized row-par %8.1f ms   cosine %.9f\n",
		float64(dMatRP.Microseconds())/1000, cosVs(chMat, chMatRP))
	bestRP, bestRPkb := time.Duration(1<<62-1), 0
	for _, kb := range []int{128, 256, 512, 1024} {
		d := best(fusedRowPar(kb))
		fmt.Fprintf(os.Stderr, "  fused row-par kb=%-5d %8.1f ms   %.3fx vs mat-row-par   cosine %.9f\n",
			kb, float64(d.Microseconds())/1000, float64(dMatRP)/float64(d), cosVs(chMat, chFusedRP))
		if d < bestRP {
			bestRP, bestRPkb = d, kb
		}
	}
	rpRatio := float64(dMatRP) / float64(bestRP)
	fmt.Fprintf(os.Stderr, "  ROW-PARALLEL VERDICT: best kb=%d -> %.3fx (bar: >=1.30 clears, <1.10 closes)\n",
		bestRPkb, rpRatio)

	bestF := out[0]
	for _, r := range out[1:] {
		if r.d < bestF.d {
			bestF = r
		}
	}
	ratio := float64(dMat) / float64(bestF.d)
	verdict := "AMBIGUOUS -> parks"
	switch {
	case ratio >= 1.30:
		verdict = "CLEARS ITS BAR -> prototype in production"
	case ratio < 1.10:
		verdict = "CLOSE THE ITEM"
	}
	fmt.Fprintf(os.Stderr, "\n  BEST fused kb=%d: %.3fx over materialized, cosine %.9f -> %s\n",
		bestF.kb, ratio, bestF.cos, verdict)
	if bestF.cos < 0.9999 {
		t.Errorf("fused arm diverges beyond the declared tolerance: cosine %.9f < 0.9999 "+
			"— the schedule is not computing the same attention, so the timing is not a comparison",
			bestF.cos)
	}
}
