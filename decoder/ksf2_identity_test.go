package decoder

import (
	"fmt"
	"math"
	"math/rand"
	"runtime"
	"testing"
)

// KS-F2's G0 (docs/measurements/ks-f2-2026-10/README.md): attendSinksToken against the code it replaced and attendQueryP against its own, kept below
// verbatim as the references. Every ctx word and every KV row is compared by its bits, over shapes that cover the eight-key chains'
// edges (key counts not a multiple of 8), GQA groups, windows, a wrapped ring and softcap, with the cache's scratch present (heads fan
// out) and absent (one worker), at GOMAXPROCS 1 and 8.

func ksf2Floats(r *rand.Rand, n int, amp float64) []float32 {
	out := make([]float32, n)
	for i := range out {
		out[i] = float32((r.Float64()*2 - 1) * amp)
	}
	return out
}

func ksf2SameBits(t *testing.T, what string, got, want []float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: %d words, want %d", what, len(got), len(want))
	}
	for i := range got {
		if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
			t.Fatalf("%s: word %d is %v (%08x), the reference %v (%08x)", what, i, got[i], math.Float32bits(got[i]), want[i], math.Float32bits(want[i]))
		}
	}
}

type ksf2Shape struct{ nH, nKV, hd int }

var ksf2Shapes = []ksf2Shape{{8, 8, 64}, {8, 4, 80}, {16, 2, 128}, {64, 8, 64}}
var ksf2Keys = []int{1, 7, 8, 9, 15, 64, 129, 300}

func TestKSF2_sinksBitIdentical(t *testing.T) {
	cells, fanned := 0, 0
	for _, procs := range []int{1, 8} {
		old := runtime.GOMAXPROCS(procs)
		for _, sh := range ksf2Shapes {
			for _, nKeys := range ksf2Keys {
				for _, window := range []int{0, 40} {
					for _, withScr := range []bool{true, false} {
						name := fmt.Sprintf("procs %d, %+v, %d keys, window %d, scratch %v", procs, sh, nKeys, window, withScr)
						r := rand.New(rand.NewSource(int64(nKeys*131 + sh.hd*7 + window + procs)))
						arch := &Architecture{NumHeads: sh.nH, NumKVHeads: sh.nKV, HeadDim: sh.hd}
						kvDim := sh.nKV * sh.hd
						lw := &LayerWeights{AttnSinks: ksf2Floats(r, sh.nH, 3)}
						p := attentionParams{Window: window, Scale: 1 / math.Sqrt(float64(sh.hd))}
						a, b := NewKVCache(1, sh.nKV, sh.hd, window, nKeys, nil), NewKVCache(1, sh.nKV, sh.hd, window, nKeys, nil)
						if withScr {
							a.scr = newDecodeScratch(arch)
						}
						for range nKeys - 1 {
							k, v := ksf2Floats(r, kvDim, 2), ksf2Floats(r, kvDim, 2)
							a.Append(0, k, v)
							b.Append(0, k, v)
						}
						q, k, v := ksf2Floats(r, sh.nH*sh.hd, 2), ksf2Floats(r, kvDim, 2), ksf2Floats(r, kvDim, 2)
						got, want := make([]float32, sh.nH*sh.hd), make([]float32, sh.nH*sh.hd)
						attendSinksToken(p, q, k, v, got, a, arch, lw, 0, nKeys-1)
						attendSinksTokenRef(p, q, k, v, want, b, arch, lw, 0, nKeys-1)
						ksf2SameBits(t, name+": ctx", got, want)
						ksf2SameBits(t, name+": keys", a.Keys(0), b.Keys(0))
						ksf2SameBits(t, name+": vals", a.Vals(0), b.Vals(0))
						cells++
						if withScr && scalarHeadWorkers(sh.nH, nKeys-p.start(nKeys-1), sh.hd) > 1 {
							fanned++
						}
					}
				}
			}
		}
		runtime.GOMAXPROCS(old)
	}
	t.Logf("%d runs (each cell at GOMAXPROCS 1 and 8), %d of them with the heads fanned out (%d distinct cells), every word equal", cells, fanned, fanned/2)
	if fanned == 0 {
		t.Fatal("no cell fanned the heads out: the parallel path went untested")
	}
}

// The attendQueryP families (Qwen3.5/3.6, Granite-4.0-H, Nemotron-H, LFM2, Llama 4) keep today's schedule until they are timed (the
// owner's call on #239): attendQueryP is bit-identical to its reference and never fans its heads out, so its cache's head worker pool
// stays empty.
func TestKSF2_queryPKeepsTheOldPath(t *testing.T) {
	cells, fanned := 0, 0
	for _, procs := range []int{1, 8} {
		old := runtime.GOMAXPROCS(procs)
		for _, sh := range ksf2Shapes {
			for _, nKeys := range ksf2Keys {
				for _, mode := range []string{"global", "window", "ring"} {
					for _, softcap := range []float64{0, 30} {
						for _, withScr := range []bool{true, false} {
							name := fmt.Sprintf("procs %d, %+v, %d keys, %s, softcap %v, scratch %v", procs, sh, nKeys, mode, softcap, withScr)
							r := rand.New(rand.NewSource(int64(nKeys*977 + sh.nH*13 + len(mode) + procs)))
							arch := &Architecture{NumHeads: sh.nH, NumKVHeads: sh.nKV, HeadDim: sh.hd}
							kvDim := sh.nKV * sh.hd
							p := attentionParams{Scale: 1 / math.Sqrt(float64(sh.hd)), Softcap: softcap}
							const W = 48
							if mode != "global" {
								p.Window = W
							}
							a, b := NewKVCache(1, sh.nKV, sh.hd, p.Window, nKeys, nil), NewKVCache(1, sh.nKV, sh.hd, p.Window, nKeys, nil)
							if mode == "ring" {
								a.enableRings(W, func(int) bool { return false })
								b.enableRings(W, func(int) bool { return false })
							}
							if withScr {
								a.scr = newDecodeScratch(arch)
							}
							for i := range nKeys {
								k, v := ksf2Floats(r, kvDim, 2), ksf2Floats(r, kvDim, 2)
								if mode == "ring" {
									a.rings[0].write(i, k, v)
									b.rings[0].write(i, k, v)
								} else {
									a.Append(0, k, v)
									b.Append(0, k, v)
								}
							}
							if mode == "ring" {
								a.rings[0].count, b.rings[0].count = nKeys, nKeys
							}
							q := ksf2Floats(r, sh.nH*sh.hd, 2)
							got, want := make([]float32, sh.nH*sh.hd), make([]float32, sh.nH*sh.hd)
							attendQueryP(p, q, got, make([]float32, nKeys), a, 0, nKeys-1, arch)
							attendQueryPRef(p, q, want, make([]float32, nKeys), b, 0, nKeys-1, arch)
							ksf2SameBits(t, name+": ctx", got, want)
							cells++
							if withScr {
								if len(a.scr.headPool) != 0 {
									t.Fatalf("%s: attendQueryP grew %d head worker slots; it must keep the one-goroutine path", name, len(a.scr.headPool))
								}
								if scalarHeadWorkers(sh.nH, nKeys-p.start(nKeys-1), sh.hd) > 1 {
									fanned++ // a cell gpt-oss's rule would have fanned out
								}
							}
						}
					}
				}
			}
		}
		runtime.GOMAXPROCS(old)
	}
	t.Logf("%d runs, every word equal and no head worker slot grown, %d of them in cells gpt-oss's rule would fan out", cells, fanned)
	if fanned == 0 {
		t.Fatal("no cell is one gpt-oss's rule would fan out: the check that attendQueryP does not proves nothing")
	}
}

// attendSinksTokenRef and attendQueryPRef are the two functions as they were before KS-F2, verbatim: the references G0 compares against.
// attendQueryP keeps exactly that code, so its reference also pins it against an unreviewed change.

func attendSinksTokenRef(p attentionParams, q, k, v, ctx []float32, cache *KVCache, arch *Architecture, lw *LayerWeights, l, pos int) {
	nH, nKV, hd := arch.NumHeads, arch.NumKVHeads, arch.HeadDim
	kvDim := nKV * hd
	cache.Append(l, k, v)
	keys, vals := cache.Keys(l), cache.Vals(l)
	nKeys := len(keys) / kvDim
	start := p.start(pos)
	scale := p.Scale
	group := nH / nKV

	clear(ctx)
	scores := make([]float32, nKeys)
	for qh := range nH {
		kvh := qh / group
		qHead := q[qh*hd : qh*hd+hd]

		maxS := math.Inf(-1)
		for s := start; s < nKeys; s++ {
			base := s*kvDim + kvh*hd
			kHead := keys[base : base+hd]
			var dot float64
			for d := range hd {
				dot += float64(qHead[d]) * float64(kHead[d])
			}
			sc := dot * scale
			scores[s] = float32(sc)
			if sc > maxS {
				maxS = sc
			}
		}
		sinkLogit := float64(lw.AttnSinks[qh])
		if sinkLogit > maxS {
			maxS = sinkLogit
		}
		var sum float64
		for s := start; s < nKeys; s++ {
			e := math.Exp(float64(scores[s]) - maxS)
			scores[s] = float32(e)
			sum += e
		}
		sum += math.Exp(sinkLogit - maxS)
		inv := 1.0 / sum
		oHead := ctx[qh*hd : qh*hd+hd]
		for s := start; s < nKeys; s++ {
			w := float32(float64(scores[s]) * inv)
			base := s*kvDim + kvh*hd
			vHead := vals[base : base+hd]
			for d := range hd {
				oHead[d] += w * vHead[d]
			}
		}
	}
}

func attendQueryPRef(p attentionParams, q, ctx, scores []float32, cache *KVCache, layer, pos int, arch *Architecture) {
	nH, nKV, hd := arch.headsAt(layer), arch.NumKVHeads, arch.HeadDim
	kvDim := nKV * hd
	keys, vals := cache.Keys(layer), cache.Vals(layer)
	nKeys := len(keys) / kvDim
	// Local (sliding-window) layers store only the last W positions in a ring: absolute key s lives at row s%W, nKeys becomes the
	// logical count, and the window guarantees every read s in [start, nKeys) is resident. wrap=0 means global append-forever,
	// row = s. scores stays indexed by absolute s (its scratch is already context-sized).
	wrap := 0
	if r := cache.rings[layer]; r != nil {
		keys, vals, nKeys, wrap = r.k, r.v, r.count, r.w
	}
	rowBase := func(s int) int {
		if wrap > 0 {
			return (s % wrap) * kvDim
		}
		return s * kvDim
	}
	start := p.start(pos)
	scale := p.Scale  // query_pre_attn_scalar^-0.5 (Gemma) or 1/sqrt(headDim)
	group := nH / nKV // GQA: query heads per KV head

	clear(ctx) // accumulated into below
	for qh := range nH {
		kvh := qh / group
		qHead := q[qh*hd : qh*hd+hd]

		maxS := math.Inf(-1)
		for s := start; s < nKeys; s++ {
			kHead := keys[rowBase(s)+kvh*hd : rowBase(s)+kvh*hd+hd]
			var dot float64
			for d := range hd {
				dot += float64(qHead[d]) * float64(kHead[d])
			}
			sc := softcapScore(dot*scale, p.Softcap)
			scores[s] = float32(sc)
			if sc > maxS {
				maxS = sc
			}
		}

		var sum float64
		for s := start; s < nKeys; s++ {
			e := math.Exp(float64(scores[s]) - maxS)
			scores[s] = float32(e)
			sum += e
		}
		inv := 1.0 / sum
		oHead := ctx[qh*hd : qh*hd+hd]
		for s := start; s < nKeys; s++ {
			w := float32(float64(scores[s]) * inv)
			vHead := vals[rowBase(s)+kvh*hd : rowBase(s)+kvh*hd+hd]
			addScaled(oHead, vHead, w)
		}
	}
}

// Many decode steps sharing one scratch: the fanned path reuses decodeScratch.headScoresPool's buffers across steps, grows them, and
// meets shorter steps after longer ones. Each window (narrower than the span, and wider than it) runs on its own cache, growing a row a
// step to 300 keys; then a second, shorter cache on the same scratch runs 1 to 60 keys, so a slot sized and written for 300 keys serves
// steps that read far fewer. Every step's ctx must equal the reference's: a stale score carried into a shorter step, or a slot grown
// short, shows here and not in the one-call cells. attendQueryP shares the scratch, as a cache's layers do, and must stay on its path.
func TestKSF2_scratchReuseAcrossSteps(t *testing.T) {
	for _, sh := range []ksf2Shape{{64, 8, 64}, {16, 2, 128}} {
		r := rand.New(rand.NewSource(int64(sh.nH)))
		arch := &Architecture{NumHeads: sh.nH, NumKVHeads: sh.nKV, HeadDim: sh.hd}
		kvDim := sh.nKV * sh.hd
		lw := &LayerWeights{AttnSinks: ksf2Floats(r, sh.nH, 3)}
		scr := newDecodeScratch(arch)
		fanned, steps := 0, 0
		run := func(name string, window, n int) {
			p := attentionParams{Window: window, Scale: 1 / math.Sqrt(float64(sh.hd)), Softcap: 30}
			a, b := NewKVCache(1, sh.nKV, sh.hd, window, n, nil), NewKVCache(1, sh.nKV, sh.hd, window, n, nil)
			a.scr = scr
			qa, qb := NewKVCache(1, sh.nKV, sh.hd, 0, n, nil), NewKVCache(1, sh.nKV, sh.hd, 0, n, nil)
			qa.scr = scr
			for pos := range n {
				q, k, v := ksf2Floats(r, sh.nH*sh.hd, 2), ksf2Floats(r, kvDim, 2), ksf2Floats(r, kvDim, 2)
				got, want := make([]float32, sh.nH*sh.hd), make([]float32, sh.nH*sh.hd)
				attendSinksToken(p, q, k, v, got, a, arch, lw, 0, pos)
				attendSinksTokenRef(p, q, k, v, want, b, arch, lw, 0, pos)
				ksf2SameBits(t, fmt.Sprintf("%+v %s sinks step %d: ctx", sh, name, pos), got, want)
				qa.Append(0, k, v)
				qb.Append(0, k, v)
				pq := attentionParams{Window: window, Scale: p.Scale, Softcap: p.Softcap}
				attendQueryP(pq, q, got, make([]float32, pos+1), qa, 0, pos, arch)
				attendQueryPRef(pq, q, want, make([]float32, pos+1), qb, 0, pos, arch)
				ksf2SameBits(t, fmt.Sprintf("%+v %s queryP step %d: ctx", sh, name, pos), got, want)
				if scalarHeadWorkers(sh.nH, pos+1-p.start(pos), sh.hd) > 1 { // the span the step reads, as attendSinksToken counts it
					fanned++
				}
				steps++
			}
		}
		run("window 40 (narrower than the span)", 40, 300)
		run("window 400 (wider than the span)", 400, 300)
		run("shorter, after the long ones", 400, 60)
		if fanned == 0 {
			t.Fatalf("%+v: no step fanned the heads out", sh)
		}
		t.Logf("%+v: %d steps on one scratch (growing, then shorter), gpt-oss's attention fanned out in %d, every word equal", sh, steps, fanned)
	}
}
