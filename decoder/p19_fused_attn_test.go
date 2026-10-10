package decoder

import (
	"fmt"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"sync"
	"testing"
	"time"

	"github.com/townsendmerino/aikit/linalg"
)

// TestP19FusedAttention measures whether the FUSED (FlashAttention-style) schedule beats the materialized
// one, WITHOUT touching production code: a prototype for a decision, not an implementation. The
// materialized path tiles over QUERY ROWS only (`scores` is tile x nKeys, attendBatchedHeads forbids a
// key-dimension split because it would re-associate the softmax denominator and the AV fold), so the
// N-wide score row makes three trips through memory per tile.
//
// THE COMPARISON IS AT FIXED PRECISION: both arms are f32 over identical inputs, so the delta is the
// SCHEDULE and nothing else (a win that only appears with f32 enabled is A3's win, not this item's).
// Gathers sit outside the timed region because both arms pay the same ones.
//
// Pre-registered bar:
//
//	>= 1.30x at K=8192 shapes -> fusion clears its own bar, prototype in production
//	<  1.10x                  -> close the item
//	1.10-1.30x                -> AMBIGUOUS, parks
//
// Not bit-identical by construction (the running-max rescale re-associates, the same category as
// --cpu-fast-attention), so correctness is a declared tolerance: cosine >= 0.9999 against the
// materialized arm. Neither arm masks, which leaves the ratio unaffected but makes the absolute times
// not production TTFT. Provenance and results: docs/code-notes/decoder.md#TestP19FusedAttention.
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
		// V, pre-transposed PER BLOCK and laid out block-major, ONCE: MatmulBT computes a·bᵀ so the AV block needs
		// b as [hd, n], and a column range of the [hd, nKeys] `vt` is not contiguous. The transpose stays outside
		// the timed loop, because the schedule does not imply that layout cost.
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

	// SERIAL CONTROL. MatmulBT fans out over its N output columns and the two arms present very different
	// N (materialized QK^T N=nKeys=8192, fused N=kb), so a fused loss could be a parallelism artifact. Running
	// both arms serial as well separates the two; this repo has already published one ratio that was mostly
	// core count (see docs/code-notes/decoder.md#TestP19FusedAttention.serial).
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

	// ROW-PARALLEL ARMS: the control for the column-parallel arms above, whose MatmulBT fan-out is over N
	// OUTPUT COLUMNS (materialized N=8192, fused N=kb), a property of composing fusion over that matmul and
	// not of the schedule. Both arms here are row-parallel across the same worker count and use MatmulBT as
	// a SERIAL inner primitive (per-worker Workspace, threshold pinned), so kernel quality and parallelism
	// model are identical and the schedule is the only variable. A hand-rolled SIMD inner loop would measure
	// our scalar Go against aikit's tuned kernel and say nothing about scheduling.
	rowsPer := func(w, workers int) (int, int) {
		per := (ktProd + workers - 1) / workers
		return w * per, min((w+1)*per, ktProd)
	}
	workers := min(runtime.GOMAXPROCS(0), 8)

	chMatRP := make([]float32, ktProd*hd)
	// Per-worker scratch and workspaces are hoisted out of the timed region, so neither arm is charged for
	// allocation (the materialized arm's [n, nKeys] score buffer is 1.4 MB per worker).
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

	// CAUSAL ARMS, the condition production runs under. The materialized path computes the FULL QK^T over
	// every key and masks only inside the softmax, so causality buys it nothing, while a key-blocked fused
	// loop can skip fully masked blocks. Rows sit at absolute positions startPos+i, and the sweep covers
	// EVERY tile of an nKeys-token prefill (row 0 attends 1 key, the last attends all): the last tile alone
	// is the least favourable one for fusion and would park the item on the instrument. Production's cost
	// is the SUM over tiles.
	var startPos int
	chMatC := make([]float32, ktProd*hd)
	matCausalTile := func() {
		var wg sync.WaitGroup
		for w := range workers {
			r0, r1 := rowsPer(w, workers)
			if r0 >= r1 {
				continue
			}
			wg.Add(1)
			go func(w, r0, r1 int) {
				defer wg.Done()
				ws, n := matWS[w], r1-r0
				sc := matScr[w]
				ws.MatmulBT(qh[r0*hd:r1*hd], kh, sc, n, hd, nKeys) // full width, as production does
				for i := range n {
					hi := startPos + r0 + i // inclusive causal bound
					row := sc[i*nKeys : (i+1)*nKeys]
					maxS := float32(math.Inf(-1))
					for j := 0; j <= hi; j++ {
						row[j] *= scale
						if row[j] > maxS {
							maxS = row[j]
						}
					}
					var sum float64
					for j := 0; j <= hi; j++ {
						e := math.Exp(float64(row[j] - maxS))
						row[j] = float32(e)
						sum += e
					}
					inv := float32(1 / sum)
					for j := 0; j <= hi; j++ {
						row[j] *= inv
					}
					for j := hi + 1; j < nKeys; j++ {
						row[j] = 0
					}
				}
				ws.MatmulBT(sc, vt, chMatC[r0*hd:r1*hd], n, nKeys, hd)
			}(w, r0, r1)
		}
		wg.Wait()
	}

	chFusedC := make([]float32, ktProd*hd)
	fusedCausalTile := func(kb int) func() {
		vBlk := make([]float32, hd*nKeys)
		for k0 := 0; k0 < nKeys; k0 += kb {
			n := min(kb, nKeys-k0)
			for j := range n {
				for d := range hd {
					vBlk[k0*hd+d*n+j] = vRow[(k0+j)*hd+d]
				}
			}
		}
		fWS := make([]*linalg.Workspace, workers)
		for w := range workers {
			fWS[w] = serialMMWorkspace()
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
					ws, nr := fWS[w], r1-r0
					sBlk := make([]float32, nr*kb)
					tmp := make([]float32, nr*hd)
					acc := make([]float32, nr*hd)
					mRun := make([]float32, nr)
					lRun := make([]float32, nr)
					for i := range nr {
						mRun[i], lRun[i] = float32(math.Inf(-1)), 0
					}
					hiMax := startPos + r1 - 1 // the widest bound any row in this worker needs
					for k0 := 0; k0 < nKeys; k0 += kb {
						if k0 > hiMax {
							break // BLOCK SKIP: every row here is masked past this point
						}
						k1 := min(k0+kb, nKeys)
						n := k1 - k0
						ws.MatmulBT(qh[r0*hd:r1*hd], kh[k0*hd:k1*hd], sBlk[:nr*n], nr, hd, n)
						for i := range nr {
							hi := startPos + r0 + i
							if k0 > hi {
								continue // this row is fully masked in this block
							}
							row := sBlk[i*n : i*n+n]
							lim := min(n, hi-k0+1)
							blkMax := float32(math.Inf(-1))
							for j := range lim {
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
							for j := range lim {
								e := math.Exp(float64(row[j] - mNew))
								row[j] = float32(e)
								sum += e
							}
							for j := lim; j < n; j++ {
								row[j] = 0
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
						o := chFusedC[(r0+i)*hd : (r0+i+1)*hd]
						for d := range a {
							o[d] = a[d] * inv
						}
					}
				}(w, r0, r1)
			}
			wg.Wait()
		}
	}

	nTiles := nKeys / ktProd
	fmt.Fprintf(os.Stderr, "\n  CAUSAL arms, SUMMED over all %d tiles of an %d-token prefill\n", nTiles, nKeys)
	sumOver := func(f func()) time.Duration {
		var tot time.Duration
		for tIdx := range nTiles {
			startPos = tIdx * ktProd
			tot += best(f)
		}
		return tot
	}
	dMatC := sumOver(matCausalTile)
	fmt.Fprintf(os.Stderr, "  materialized causal (all tiles) %8.1f ms\n", float64(dMatC.Microseconds())/1000)
	bestC, bestCkb := time.Duration(1<<62-1), 0
	for _, kb := range []int{256, 512, 1024} {
		fn := fusedCausalTile(kb)
		d := sumOver(fn)
		fmt.Fprintf(os.Stderr, "  fused causal kb=%-5d (all tiles) %8.1f ms   %.3fx   cosine %.9f (last tile)\n",
			kb, float64(d.Microseconds())/1000, float64(dMatC)/float64(d), cosVs(chMatC, chFusedC))
		if d < bestC {
			bestC, bestCkb = d, kb
		}
	}
	fmt.Fprintf(os.Stderr, "  CAUSAL VERDICT (whole prefill): best kb=%d -> %.3fx (bar: >=1.30 clears, <1.10 closes)\n",
		bestCkb, float64(dMatC)/float64(bestC))

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

// TestFusedAttention_matchesMaterialized gates the production fused path. Not bit-identity: the
// running-max rescale re-associates by construction, so the bar is cosine >= 0.9999 against the
// materialized path, through the REAL attendBatchedHeads with masking on. It runs a sliding window and a
// non-zero base so the lo/hi bounds are exercised rather than assumed to be [0, pos].
func TestFusedAttention_matchesMaterialized(t *testing.T) {
	const (
		nH    = 8
		nKV   = 2
		hd    = 16
		kvDim = nKV * hd
		qDim  = nH * hd
	)
	arch := &Architecture{NumHeads: nH, NumKVHeads: nKV, HeadDim: hd, AttnScale: 0.25}
	for _, tc := range []struct {
		name     string
		K, nKeys int
		window   int
		startPos int
	}{
		{"causal, one block", 48, 96, 0, 48},
		{"causal, multi block", 300, 1400, 0, 1100},
		{"sliding window", 300, 1400, 256, 1100},
	} {
		t.Run(tc.name, func(t *testing.T) {
			q := randF32(tc.K*qDim, 41)
			keys := randF32(tc.nKeys*kvDim, 42)
			vals := randF32(tc.nKeys*kvDim, 43)
			cache := &KVCache{}
			if tc.window > 0 {
				cache.window = tc.window
			}
			run := func(on string) []float32 {
				t.Setenv("GOINFER_FUSED_ATTENTION", on)
				// wantFused=true: useAcc64=false and the cache has no treeMask for both the "0" and "1" runs, so the
				// "1" (fused) run exercises the vt/scores elimination.
				pool := newHeadWorkerPool(4, tc.K, tc.nKeys, hd, true)
				ctx := make([]float32, tc.K*qDim)
				attendBatchedHeads(q, ctx, keys, vals, 0, cache, 0, tc.startPos, tc.K,
					tc.window == 0, arch, false, pool)
				return ctx
			}
			mat := run("0")
			fus := run("1")
			var dot, na, nb, maxAb float64
			for i := range mat {
				x, y := float64(mat[i]), float64(fus[i])
				dot += x * y
				na += x * x
				nb += y * y
				if a := math.Abs(x - y); a > maxAb {
					maxAb = a
				}
			}
			cos := dot / (math.Sqrt(na) * math.Sqrt(nb))
			t.Logf("cosine %.9f  max|diff| %.3g", cos, maxAb)
			if cos < 0.9999 {
				t.Fatalf("fused diverges past the declared bar: cosine %.9f (max|diff| %.3g)", cos, maxAb)
			}
			nz := 0
			for _, v := range mat {
				if v != 0 {
					nz++
				}
			}
			if nz < len(mat)/2 {
				t.Fatalf("output mostly zeros (%d/%d) — the arms agree because nothing ran", nz, len(mat))
			}
		})
	}
}

// TestFusedAttention_logitDivergence tells a NEAR-TIE FLIP from a BUG when fusion changes the
// generated tokens (a token diff cannot). It compares the PREFILL LOGITS on a real checkpoint and
// reports the cosine (same computation?) and the top-2 margin at the final position (was the argmax
// decidable?): a high cosine with a margin near the perturbation size is a tie flip, a low cosine is a
// defect. Same shape as a3_divergence_test.go.
func TestFusedAttention_logitDivergence(t *testing.T) {
	if os.Getenv("GOINFER_P19") == "" {
		t.Skip("set GOINFER_P19=1 (loads a real model)")
	}
	m, err := loadBenchModel()
	if err != nil {
		t.Skipf("no model (%v)", err)
	}
	ctx := deadlineCtx(t)
	const K = 768 // the long-prompt golden's depth
	ids := make([]int, K)
	for i := range ids {
		ids[i] = 700 + i%64
	}
	run := func(on string) []float32 {
		setKnob(t, m, knobFusedAttention, on)
		out, err := m.forwardLayersN(ctx, ids, m.NewCache(K+8), true)
		if err != nil {
			t.Fatalf("forward(fused=%s): %v", on, err)
		}
		return out
	}
	// BASELINE ON THE SAME MODEL: how far the f32 flag already moves the hidden states away from acc64.
	// Both arms use one checkpoint at one depth; a number from another model would not compare.
	accRun := func() []float32 {
		setKnob(t, m, knobFusedAttention, "0")
		out, err := m.forwardLayersN(ctx, ids, m.NewCache(K+8), false) // acc64
		if err != nil {
			t.Fatalf("forward(acc64): %v", err)
		}
		return out
	}
	cosOf := func(a, b []float32) (float64, float64) {
		var dot, na, nb, mx float64
		for i := range a {
			x, y := float64(a[i]), float64(b[i])
			dot += x * y
			na += x * x
			nb += y * y
			if d := math.Abs(x - y); d > mx {
				mx = d
			}
		}
		return dot / (math.Sqrt(na) * math.Sqrt(nb)), mx
	}
	acc := accRun()
	off, on := run("0"), run("1")
	c1, m1 := cosOf(acc, off)
	c2, m2 := cosOf(acc, on)
	t.Logf("BASELINE  acc64 vs f32-materialized (what the flag already ships): cosine %.9f  max|diff| %.3g", c1, m1)
	t.Logf("TOTAL     acc64 vs f32-fused        (what shipping fusion means):  cosine %.9f  max|diff| %.3g", c2, m2)
	var dot, na, nb, maxAb float64
	for i := range off {
		x, y := float64(off[i]), float64(on[i])
		dot += x * y
		na += x * x
		nb += y * y
		if a := math.Abs(x - y); a > maxAb {
			maxAb = a
		}
	}
	cos := dot / (math.Sqrt(na) * math.Sqrt(nb))
	// top-2 margin at the LAST row — the position whose argmax picks token 0.
	hidden := len(off) / K
	last := off[(K-1)*hidden:]
	b1, b2 := math.Inf(-1), math.Inf(-1)
	arg1 := -1
	for i, v := range last {
		f := float64(v)
		if f > b1 {
			b2, b1, arg1 = b1, f, i
		} else if f > b2 {
			b2 = f
		}
	}
	lastOn := on[(K-1)*hidden:]
	o1, oarg := math.Inf(-1), -1
	for i, v := range lastOn {
		if float64(v) > o1 {
			o1, oarg = float64(v), i
		}
	}
	t.Logf("hidden-state cosine %.9f   max|diff| %.3g", cos, maxAb)
	t.Logf("last row: top1 idx %d (%.6f), top2 %.6f, MARGIN %.3g", arg1, b1, b2, b1-b2)
	t.Logf("fused    top1 idx %d (%.6f)   %s", oarg, o1,
		map[bool]string{true: "SAME", false: "FLIPPED"}[oarg == arg1])
	// The bar compares fusion's ADDITIONAL divergence against the divergence the f32 flag already
	// accepts on the SAME model; a kernel-level 0.9999 bar is the wrong instrument for a 24-layer forward.
	if c2 < c1*0.999 {
		t.Fatalf("fusion moves the hidden states materially further from acc64 than the f32 flag "+
			"already does (%.9f vs baseline %.9f) — that is a defect, not a tie flip", c2, c1)
	}
}

// TestFusedAttention_endToEnd measures what the fused kernel is worth through a real forward, the
// only number that justifies a user-visible output change: kernel-level ratios do not compose with
// a profile share. It uses a DENSE model on purpose (attention's share of prefill is what the win
// scales with, so a weak result here closes the question for MoE too), paired and interleaved with
// alternating lead; both arms run the f32 path (fastAttn=true) and the only difference is
// GOINFER_FUSED_ATTENTION. Figures: docs/code-notes/decoder.md#TestFusedAttention_endToEnd.
func TestFusedAttention_endToEnd(t *testing.T) {
	if os.Getenv("GOINFER_P19_E2E") == "" {
		t.Skip("set GOINFER_P19_E2E=1")
	}
	path := os.Getenv("GOINFER_P19_MODEL")
	if path == "" {
		path = os.Getenv("HOME") + "/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"
	}
	if _, err := os.Stat(path); err != nil {
		t.Skipf("no model at %s: %v", path, err)
	}
	m, err := Load(path, Options{Quant: "int4"})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	defer m.Close()
	ctx := deadlineCtx(t)
	K := 4096
	if v := os.Getenv("GOINFER_P19_K"); v != "" {
		fmt.Sscanf(v, "%d", &K)
	}
	pairs := 2
	if v := os.Getenv("GOINFER_P19_PAIRS"); v != "" {
		fmt.Sscanf(v, "%d", &pairs)
	}
	if !m.canBatchN(K) {
		t.Skip("no batched prefill")
	}
	ids := longPromptIDs(K)
	start := time.Now()
	fmt.Fprintf(os.Stderr, "P19 e2e: start %s  model=%s K=%d pairs=%d\n",
		start.Format("15:04:05"), filepath.Base(path), K, pairs)

	run := func(on string) time.Duration {
		t.Helper()
		setKnob(t, m, knobFusedAttention, on)
		t0 := time.Now()
		if _, err := m.forwardLayersN(ctx, ids, m.NewCache(K+8), true); err != nil {
			t.Fatalf("forward(fused=%s): %v", on, err)
		}
		return time.Since(t0)
	}
	run("0") // warm, discarded
	var onD, offD []time.Duration
	for p := range pairs {
		var dOn, dOff time.Duration
		if p%2 == 0 {
			dOff, dOn = run("0"), run("1")
		} else {
			dOn, dOff = run("1"), run("0")
		}
		onD, offD = append(onD, dOn), append(offD, dOff)
		fmt.Fprintf(os.Stderr, "  pair %d/%d  materialized %7.1fs  fused %7.1fs  %.3fx  [elapsed %s]\n",
			p+1, pairs, dOff.Seconds(), dOn.Seconds(), float64(dOff)/float64(dOn),
			time.Since(start).Round(time.Second))
	}
	r := medianDur(offD).Seconds() / medianDur(onD).Seconds()
	fmt.Fprintf(os.Stderr, "\n  K=%d END-TO-END: materialized %.1fs  fused %.1fs  %.3fx (%+.1f%%)\n",
		K, medianDur(offD).Seconds(), medianDur(onD).Seconds(), r, 100*(r-1))
}
